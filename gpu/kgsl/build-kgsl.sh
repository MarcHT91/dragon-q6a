#!/bin/sh
# Build Qualcomm's KGSL GPU driver (msm_kgsl.ko) for the running kernel, without installing anything.
# Source: kgsl-dkms from Ubuntu's public "Qualcomm Addon Packages" PPA (ubuntu-qcom-iot/qcom-ppa, resolute).
# KERNEL_SRC defaults to /lib/modules/$(uname -r)/build. If that tree has no Module.symvers, modpost only warns
# about unresolved symbols (KBUILD_MODPOST_WARN=1); check-symbols below verifies them against the running kernel.
set -e
cd "$(dirname "$0")"
KSRC=${KERNEL_SRC:-/lib/modules/$(uname -r)/build}
DEB=kgsl-dkms_1.0.11-0qli1_arm64.deb
mkdir -p build && cd build
[ -f "$DEB" ] || curl -fL -o "$DEB" "https://ppa.launchpadcontent.net/ubuntu-qcom-iot/qcom-ppa/ubuntu/pool/main/k/kgsl/$DEB"
rm -rf src && dpkg-deb -x "$DEB" pkg && cp -r pkg/usr/src/kgsl-1.0.11 src
# Linux 7.3 dropped qcom_ubwc_cfg_data.ubwc_swizzle: it is now derived from flags by qcom_ubwc_swizzle().
if grep -q 'qcom_ubwc_swizzle(const struct qcom_ubwc_cfg_data' "$KSRC/include/linux/soc/qcom/ubwc.h" 2>/dev/null; then
	sed -i 's/cfg->ubwc_swizzle/qcom_ubwc_swizzle(cfg)/' src/adreno_a6xx.c
	grep -q 'qcom_ubwc_swizzle(cfg)' src/adreno_a6xx.c
fi
make -C src KERNEL_SRC="$KSRC" KBUILD_MODPOST_WARN=1 -j"$(nproc)" > build.log 2>&1 || { tail -30 build.log; exit 1; }
cp src/msm_kgsl.ko .
echo "built $(pwd)/msm_kgsl.ko ($(modinfo -F vermagic msm_kgsl.ko))"
# every undefined symbol of the module must be exported by the running kernel
nm -u msm_kgsl.ko | awk '{print $2}' | LC_ALL=C sort -u > need.txt
grep -oE "__ksymtab_[A-Za-z0-9_]+" /proc/kallsyms | sed 's/__ksymtab_//' | LC_ALL=C sort -u > have.txt
missing=$(LC_ALL=C comm -23 need.txt have.txt | wc -l)
echo "symbols needed: $(wc -l < need.txt), missing from the running kernel: $missing"
