#!/bin/sh
# Instal·la el plugin a nivell de sistema i fa que shairport-sync controli
# el volum del DSP (maquinari). Executar amb sudo des d'aquest directori.
set -e
cd "$(dirname "$0")"
[ -f libasound_module_ctl_filter.so ] || make
install -D -m 0755 libasound_module_ctl_filter.so /usr/local/lib/alsa-lib/libasound_module_ctl_filter.so
[ -e /etc/asound.conf ] && cp /etc/asound.conf /etc/asound.conf.bak
install -m 0644 examples/asound.conf /etc/asound.conf
cp /etc/shairport-sync.conf /etc/shairport-sync.conf.bak-$(date +%Y%m%d-%H%M%S)
install -m 0644 examples/shairport-sync.conf /etc/shairport-sync.conf
mkdir -p /etc/systemd/system/shairport-sync.service.d
printf '[Service]\nTimeoutStopSec=5\n' > /etc/systemd/system/shairport-sync.service.d/timeout.conf
systemctl daemon-reload
systemctl restart shairport-sync
alsactl store
echo "Fet. Volum: amixer -D airplay sget AirPlay"
