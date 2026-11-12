"""Apply the Dragon Q6A changes to an ncnn source tree (tested on tag 20261020):
1. a Vulkan MatMul layer (upstream ncnn runs activation x activation MatMul on the CPU, with GPU<->CPU copies);
2. env NCNN_SUBGROUP_SIZE: override the subgroup size ncnn plans for (used to diagnose the Turnip subgroup bug).
usage: python3 apply_patch.py /path/to/ncnn"""
import shutil, sys
from pathlib import Path

ncnn = Path(sys.argv[1])
here = Path(__file__).resolve().parent / "patch"
for f in here.rglob("*"):
    if f.is_file():
        dst = ncnn / f.relative_to(here)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
        print("added", dst.relative_to(ncnn))

gpu = ncnn / "src" / "gpu.cpp"
s = gpu.read_text()
anchor = """        querySubgroupSizeControlFeatures.computeFullSubgroups = VK_FALSE;
    }
"""
block = """
    // dragon-q6a: NCNN_SUBGROUP_SIZE overrides the subgroup size ncnn plans for (e.g. 64 on turnip a6xx)
    {
        const char* sg = getenv("NCNN_SUBGROUP_SIZE");
        if (sg)
        {
            uint32_t s = (uint32_t)atoi(sg);
            if (s >= querySubgroupSizeControlProperties.minSubgroupSize && s <= querySubgroupSizeControlProperties.maxSubgroupSize && (s & (s - 1)) == 0)
            {
                NCNN_LOGE("NCNN_SUBGROUP_SIZE override %u -> %u", querySubgroupProperties.subgroupSize, s);
                querySubgroupProperties.subgroupSize = s;
            }
        }
    }
"""
if "NCNN_SUBGROUP_SIZE" not in s:
    assert s.count(anchor) == 1, "anchor not found in src/gpu.cpp (different ncnn version?)"
    gpu.write_text(s.replace(anchor, anchor + block))
    print("patched src/gpu.cpp")
