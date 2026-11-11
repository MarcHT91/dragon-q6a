#!/bin/sh
# qcs6490-hifi: 24-bit headphone path on Radxa Dragon Q6A (QCS6490).
#  - DKMS snd-soc-sc8280xp: headphone BE format/rate as module params
#    (rx_be_format, rx_be_rate; defaults = stock S16_LE @ 48 kHz)
#  - DSP topology override: MultiMedia1 Playback accepts S16/S24, 8-192 kHz
# Run with sudo. Reboot afterwards.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
kver="${KVER:-$(uname -r)}"
ver=0.6
name=qcs6490-hifi

# 1. kernel build tree for DKMS (only if no headers package provides one)
if [ ! -e "/lib/modules/$kver/build/Makefile" ]; then
	[ -n "$KSRC" ] || { echo "No /lib/modules/$kver/build. Install linux-headers or set KSRC=<prepared kernel tree>"; exit 1; }
	ln -sfn "$KSRC" "/lib/modules/$kver/build"
fi

# 2. DKMS module
for v in $(dkms status -m $name 2>/dev/null | sed -E "s#^$name/([^,]+),.*#\1#" | sort -u); do
	dkms remove -m $name -v "$v" --all || true
	rm -rf "/usr/src/$name-$v"
done
cp -r "$here/dkms" "/usr/src/$name-$ver"
dkms add -m $name -v $ver
dkms build -m $name -v $ver -k "$kver"
dkms install -m $name -v $ver -k "$kver"
modinfo -n snd-soc-sc8280xp -k "$kver"

# 3. DSP topology override (firmware loader checks /lib/firmware/updates first)
tp=qcom/qcs6490/QCS6490-Radxa-Dragon-Q6A-tplg.bin
mkdir -p "/lib/firmware/updates/$(dirname $tp)"
python3 -I "$here/patch-tplg.py" "/lib/firmware/$tp" "/lib/firmware/updates/$tp" \
	--formats S16_LE,S24_LE --rates 8000-192000 "MultiMedia1 Playback"

# 4. headphone BE: 24-bit at 48 kHz (tested 2026-11-11: S16/S24/S32 and
#    48/96/192 kHz work; 44.1 kHz family is rejected by the BE)
cat > /etc/modprobe.d/qcs6490-hifi.conf <<'EOM'
# qcs6490-hifi: headphone (codec DMA RX) back end format/rate
options snd_soc_sc8280xp rx_be_format=24 rx_be_rate=48000
EOM

update-initramfs -u -k "$kver" 2>/dev/null || true
echo "Fet. Reinicia la placa."
