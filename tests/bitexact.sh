#!/bin/bash
# bitexact.sh FILE SWDEC HWDEC MODE   (MODE: nv12 | drm)
export LC_ALL=C
set -uo pipefail
F=$1; SW=$2; HW=$3; MODE=$4
FF=${PREFIX:-/opt/ffmpeg-iris}/bin/ffmpeg
L=$(dirname "$0")/logs; mkdir -p "$L"
tag=$(basename "$F" .mkv)_$MODE
DRMOPTS="-no_cvt_hw -hwaccel_output_format drm_prime"
md5s() { grep -v '^#' | awk -F', *' '{print $NF}'; }
ref=$(timeout 120 $FF -v error -c:v "$SW" -i "$F" -an -vf format=nv12 -f framemd5 - | md5s)
if [ "$MODE" = drm ]; then
  hw=$(timeout 120 $FF -v debug $DRMOPTS -c:v "$HW" -i "$F" -an -vf hwdownload,format=nv12 -f framemd5 - 2>"$L/$tag.log" | md5s)
else
  hw=$(timeout 120 $FF -v debug -c:v "$HW" -i "$F" -an -vf format=nv12 -f framemd5 - 2>"$L/$tag.log" | md5s)
fi
nr=$(echo "$ref" | grep -c .); nh=$(echo "$hw" | grep -c .)
if [ "$nh" -eq 0 ]; then echo "FAIL $tag: 0 fotogrames (log: $L/$tag.log)"; exit 1; fi
if [ "$ref" != "$hw" ]; then
  first=$(diff <(echo "$ref") <(echo "$hw") | head -1)
  echo "FAIL $tag: ref=$nr hw=$nh fotogrames, primera diferència: $first"; exit 1
fi
echo "PASS $tag: $nh fotogrames idèntics"
