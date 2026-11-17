#!/bin/bash
# Boot the KGSL test entry (current entry plus msm.skip_gpu=1) on the next reboot only; the boot after that is
# the normal one again (msm GPU, Mesa, Kodi). Needs EFI variables from Linux: Radxa firmware 6.0.260916 with a
# kernel that allows QSEECOM on this board (Armbian 7.3+). With older firmware use kgsl-boot-default.sh.
# Run as: sudo bash kgsl-boot-once.sh, then reboot.
set -eu
E=/boot/efi/loader/entries
bootctl status 2>/dev/null | grep -q '✓ One-shot entry control' || { echo "no EFI one-shot support: use kgsl-boot-default.sh"; exit 1; }
CUR=$(grep -l "$(uname -r)" "$E"/*.conf | grep -v kgsl-test | head -1)
echo "current entry: $CUR"
sed -e 's/^title.*/& (KGSL test, msm.skip_gpu=1)/' -e '/^options/ s/$/ msm.skip_gpu=1/' "$CUR" > "$E/kgsl-test.conf"
grep -q "^options.*msm.skip_gpu=1" "$E/kgsl-test.conf" || { echo "no options line"; rm -f "$E/kgsl-test.conf"; exit 1; }
bootctl set-oneshot kgsl-test.conf
bootctl status 2>/dev/null | grep -iE 'one-?shot|Default Entry' || true
echo "OK: the next boot (only) uses kgsl-test.conf. kgsl-revert.sh removes the entry."
