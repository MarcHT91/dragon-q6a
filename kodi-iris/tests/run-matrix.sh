#!/bin/bash
# Executa bitexact.sh per a tota la matriu. Surt 1 si algun cas falla.
set -uo pipefail
T=$(dirname "$0"); M=$T/media; rc=0
for mode in nv12 drm; do
  for s in 640x360 1280x720 1920x1080 3840x2160; do
    "$T/bitexact.sh" "$M/h264_$s.mkv" h264 h264_v4l2m2m $mode || rc=1
    "$T/bitexact.sh" "$M/hevc_$s.mkv" hevc hevc_v4l2m2m $mode || rc=1
    "$T/bitexact.sh" "$M/vp9_$s.mkv"  vp9  vp9_v4l2m2m  $mode || echo "  (VP9 opcional)"
  done
  "$T/bitexact.sh" "$M/h264_reschange.mkv" h264 h264_v4l2m2m $mode || rc=1
done
# 10 bits. Iris amb firmware HFI gen2 ofereix P010 a la captura: el 10 bits es
# descodifica per maquinari i s'ha de comparar en P010. Amb gen1 (sense P010) el
# descodificador ha de refusar el flux a l'obertura perquè Kodi torni a programari.
dec=$(grep -l '^qcom-iris-decoder$' /sys/class/video4linux/video*/name 2>/dev/null | head -1)
if [ -n "$dec" ] && v4l2-ctl -d "/dev/$(basename "$(dirname "$dec")")" --list-formats 2>/dev/null | grep -q "'P010'"; then
  for mode in nv12 drm; do
    "$T/bitexact.sh" "$M/hevc10_1920x1080.mkv" hevc hevc_v4l2m2m $mode p010le || rc=1
    "$T/bitexact.sh" "$M/vp910_1920x1080.mkv"  vp9  vp9_v4l2m2m  $mode p010le || rc=1
  done
  exit $rc
fi
# Gen1: codi no nul, "Error while opening decoder" i menys de 2 s.
s=$(date +%s%N)
out=$(timeout 30 ${PREFIX:-/opt/ffmpeg-iris}/bin/ffmpeg -hide_banner -v error -c:v hevc_v4l2m2m -i "$M/hevc10_1920x1080.mkv" -f null - 2>&1 >/dev/null)
st=$?; ms=$(( ($(date +%s%N) - s) / 1000000 ))
if [ $st -eq 124 ]; then echo "FAIL hevc10: penjat (timeout)"; rc=1
elif [ $st -eq 0 ]; then echo "FAIL hevc10: no ha fallat (codi 0)"; rc=1
elif ! grep -q "Error while opening decoder" <<<"$out"; then echo "FAIL hevc10: no falla a l'obertura (codi $st, ${ms} ms): $(head -1 <<<"$out")"; rc=1
elif [ $ms -ge 2000 ]; then echo "FAIL hevc10: refusat a l'obertura però lent (${ms} ms)"; rc=1
else echo "PASS hevc10: refusat a l'obertura en ${ms} ms (codi $st)"; fi
exit $rc
