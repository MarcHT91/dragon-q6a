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
# 10 bits: iris no en té format de captura; el descodificador ha de refusar
# el flux a l'obertura (avcodec_open2) perquè Kodi torni a programari:
# codi no nul, "Error while opening decoder" i menys de 2 s.
s=$(date +%s%N)
out=$(timeout 30 ${PREFIX:-/opt/ffmpeg-iris}/bin/ffmpeg -hide_banner -v error -c:v hevc_v4l2m2m -i "$M/hevc10_1920x1080.mkv" -f null - 2>&1 >/dev/null)
st=$?; ms=$(( ($(date +%s%N) - s) / 1000000 ))
if [ $st -eq 124 ]; then echo "FAIL hevc10: penjat (timeout)"; rc=1
elif [ $st -eq 0 ]; then echo "FAIL hevc10: no ha fallat (codi 0)"; rc=1
elif ! grep -q "Error while opening decoder" <<<"$out"; then echo "FAIL hevc10: no falla a l'obertura (codi $st, ${ms} ms): $(head -1 <<<"$out")"; rc=1
elif [ $ms -ge 2000 ]; then echo "FAIL hevc10: refusat a l'obertura però lent (${ms} ms)"; rc=1
else echo "PASS hevc10: refusat a l'obertura en ${ms} ms (codi $st)"; fi
exit $rc
