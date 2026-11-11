#!/bin/sh
# Undo install.sh. Run with sudo, then reboot.
for v in $(dkms status -m qcs6490-hifi 2>/dev/null | sed -E "s#^qcs6490-hifi/([^,]+),.*#\1#" | sort -u); do
	dkms remove -m qcs6490-hifi -v "$v" --all
	rm -rf "/usr/src/qcs6490-hifi-$v"
done
rm -f /etc/modprobe.d/qcs6490-hifi.conf
rm -f /lib/firmware/updates/qcom/qcs6490/QCS6490-Radxa-Dragon-Q6A-tplg.bin
update-initramfs -u 2>/dev/null || true
echo "Desinstal·lat. Reinicia la placa."
