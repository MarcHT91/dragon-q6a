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
# 10 bits: el descodificador ha de fallar ràpid i net (no penjar-se).
s=$(date +%s)
timeout 30 ${PREFIX:-/opt/ffmpeg-iris}/bin/ffmpeg -v error -c:v hevc_v4l2m2m -i "$M/hevc10_1920x1080.mkv" -f null - >/dev/null 2>&1
st=$?; d=$(( $(date +%s) - s ))
if [ $st -eq 124 ]; then echo "FAIL hevc10: penjat (timeout)"; rc=1; else echo "PASS hevc10: acaba en ${d}s (codi $st)"; fi
exit $rc
