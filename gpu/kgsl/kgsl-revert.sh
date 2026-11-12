#!/bin/bash
# Back to the normal boot (msm drives the GPU, Mesa/Turnip, Kodi): restore loader.conf, drop the test entry.
# Run as: sudo bash kgsl-revert.sh, then reboot.
set -eu
L=/boot/efi/loader/loader.conf
[ -f $L.pre-kgsl ] && mv $L.pre-kgsl $L
rm -f /boot/efi/loader/entries/kgsl-test.conf
cat $L; ls /boot/efi/loader/entries/
