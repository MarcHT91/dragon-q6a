#!/bin/sh
# Download Qualcomm's Adreno userspace (OpenCL, Vulkan, gbm) from Ubuntu's public Qualcomm PPA and unpack it
# into adreno/root WITHOUT installing it (Mesa stays untouched). Writes adreno-env.sh: source it to use these
# drivers in one shell (needs msm_kgsl loaded, see kgsl-load.sh).
set -e
cd "$(dirname "$0")"
P=https://ppa.launchpadcontent.net/ubuntu-qcom-iot/qcom-ppa/ubuntu/pool/main
mkdir -p adreno && cd adreno
for f in q/qcom-adreno/qcom-adreno-common1_1.877.5-1_arm64.deb q/qcom-adreno/qcom-adreno-cl1_1.877.5-1_arm64.deb \
         q/qcom-adreno/qcom-adreno-vulkan1_1.877.5-1_arm64.deb q/qcom-adreno/qcom-adreno-cl-dev_1.877.5-1_arm64.deb \
         libg/libgbm-msm/libgbm-msm_1.2.6-2_arm64.deb; do
    d=$(basename "$f"); [ -f "$d" ] || curl -fL -o "$d" "$P/$f"; dpkg-deb -x "$d" root
done
R=$(pwd)/root; L=$R/usr/lib/aarch64-linux-gnu
mkdir -p "$R/icd"
echo "$L/libOpenCL_adreno.so.1" > "$R/icd/adrenocl.icd"
sed "s#/usr/lib/aarch64-linux-gnu#$L#" "$R/usr/share/vulkan/icd.d/adrenovk.json" > "$R/icd/adrenovk.json"
cat > ../adreno-env.sh <<EOT
# source this file to use Qualcomm's Adreno OpenCL/Vulkan in this shell (msm_kgsl must be loaded)
export LD_LIBRARY_PATH=$L\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export OCL_ICD_VENDORS=$R/icd
export VK_DRIVER_FILES=$R/icd/adrenovk.json
EOT
echo "unpacked into $R; source $(cd .. && pwd)/adreno-env.sh"
