#!/bin/bash
# After booting with msm.skip_gpu=1: stop Kodi (no Mesa GPU in this mode), load msm_kgsl, give the video group
# access to /dev/kgsl-3d0 and save the kernel log. Run as: sudo bash kgsl-load.sh
set -u
D=$(cd "$(dirname "$0")" && pwd)
LOG=$D/load.log
exec > >(tee "$LOG") 2>&1
grep -q "msm.skip_gpu=1" /proc/cmdline || { echo "not booted with msm.skip_gpu=1"; exit 1; }
systemctl stop kodi.service 2>/dev/null || true
dmesg -C
insmod "$D/build/msm_kgsl.ko" || echo "INSMOD FAILED"
sleep 3
[ -e /dev/kgsl-3d0 ] && chgrp video /dev/kgsl-3d0 && chmod 660 /dev/kgsl-3d0
ls -la /dev/kgsl* 2>&1
dmesg > "$D/dmesg-load.txt"
[ -n "${SUDO_USER:-}" ] && chown "$SUDO_USER" "$D/dmesg-load.txt" "$LOG"
grep -iE "kgsl|adreno|gmu|oops|fail|error" "$D/dmesg-load.txt" | tail -20
