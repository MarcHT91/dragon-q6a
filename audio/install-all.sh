#!/bin/sh
# Full setup used on the author's board. Run with sudo from the repo root, then reboot.
#  1. hifi: DKMS (24-bit headphone path) + DSP topology + modprobe options
#  2. alsa-ctl-filter system-wide + shairport-sync config (S24_LE @ 48 kHz, hardware volume)
#  3. boot service that enables the DSP output routes before players start; save mixer state
# Without a linux-headers package for the running kernel, set KSRC=<prepared tree> (see hifi/README).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
"$here/hifi/install.sh"
"$here/route/install.sh"
"$here/alsa-ctl-filter/install-shairport-hwvol.sh"
alsactl store
echo "Done. Reboot: sudo reboot"
