"""Element-wise max of several activation-range files (calibrate_lfm3.py / calibrate2_lfm3.py outputs).
usage: merge_ranges.py OUT.json IN1.json IN2.json [...]"""
import json, sys
out, ins = sys.argv[1], [json.load(open(f)) for f in sys.argv[2:]]
m = json.loads(json.dumps(ins[0]))
for d in ins[1:]:
    for i, layer in enumerate(d["layers"]):
        for k, v in layer.items():
            m["layers"][i][k] = max(m["layers"][i].get(k, 0.0), v)
    for k, v in d.get("head", {}).items():
        m["head"][k] = max(m["head"].get(k, 0.0), v)
json.dump(m, open(out, "w"), indent=1)
print("wrote", out)
