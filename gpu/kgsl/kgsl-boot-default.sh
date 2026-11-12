#!/bin/bash
# Add a systemd-boot entry identical to the current one plus msm.skip_gpu=1 (msm keeps the display but does not
# bind the GPU), and make it the default. Undo with kgsl-revert.sh. (bootctl set-oneshot does not work on this
# board: its firmware exposes no EFI variables to Linux.)
# Run as: sudo bash kgsl-boot-default.sh, then reboot.
set -eu
E=/boot/efi/loader/entries; L=/boot/efi/loader/loader.conf
CUR=$(grep -l "$(uname -r)" "$E"/*.conf | grep -v kgsl-test | head -1)
echo "current entry: $CUR"
sed -e 's/^title.*/& (KGSL test, msm.skip_gpu=1)/' -e '/^options/ s/$/ msm.skip_gpu=1/' "$CUR" > "$E/kgsl-test.conf"
grep -q "^options.*msm.skip_gpu=1" "$E/kgsl-test.conf" || { echo "no options line"; rm -f "$E/kgsl-test.conf"; exit 1; }
[ -f $L.pre-kgsl ] || cp -p $L $L.pre-kgsl
grep -v '^default' $L.pre-kgsl > $L
echo "default kgsl-test.conf" >> $L
cat $L
echo "OK: reboot. In the 3 s boot menu you can still pick the normal entry."
