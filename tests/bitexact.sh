#!/bin/bash
# bitexact.sh FILE SWDEC HWDEC MODE   (MODE: nv12 | drm)
export LC_ALL=C
set -uo pipefail
F=$1; SW=$2; HW=$3; MODE=$4
FF=${PREFIX:-/opt/ffmpeg-iris}/bin/ffmpeg
L=$(dirname "$0")/logs; mkdir -p "$L"
tag=$(basename "$F" .mkv)_$MODE
P=${PREFIX:-/opt/ffmpeg-iris}
# Mode drm: l'eina ffmpeg amb -no_cvt_hw no reinicialitza mai el graf de filtres
# (fftools/ffmpeg_filter.c), i hwdownload rebutja els fotogrames posteriors a un
# canvi de resolució. drmframemd5 consumeix DRM_PRIME com Kodi i descarrega
# cada fotograma amb el seu propi hw_frames_ctx.
PROBE=$(dirname "$0")/bin/drmframemd5
if [ "$MODE" = drm ] && [ ! "$PROBE" -nt "$(dirname "$0")/drmframemd5.c" -o ! "$PROBE" -nt "$P/lib/libavcodec.so" ]; then
  mkdir -p "$(dirname "$PROBE")"
  gcc -O2 -Wall -o "$PROBE" "$(dirname "$0")/drmframemd5.c" -I"$P/include" -L"$P/lib" \
      -Wl,-rpath,"$P/lib" -lavformat -lavcodec -lavutil || { echo "FAIL $tag: no compila drmframemd5"; exit 1; }
fi
md5s() { grep -v '^#' | awk -F', *' '{print $NF}'; }
# -noautoscale: compara cada fotograma a la seva resolució real (sense reescalar
# al primer tamany després d'un canvi de resolució).
ref=$(timeout 120 $FF -v error -c:v "$SW" -i "$F" -an -noautoscale -vf format=nv12 -f framemd5 - | md5s)
if [ "$MODE" = drm ]; then
  hw=$(AV_LOG_FORCE_NOCOLOR=1 timeout 120 "$PROBE" "$F" "$HW" 2>"$L/$tag.log")
else
  hw=$(timeout 120 $FF -v debug -c:v "$HW" -i "$F" -an -noautoscale -vf format=nv12 -f framemd5 - 2>"$L/$tag.log" | md5s)
fi
nr=$(echo "$ref" | grep -c .); nh=$(echo "$hw" | grep -c .)
if [ "$nh" -eq 0 ]; then echo "FAIL $tag: 0 fotogrames (log: $L/$tag.log)"; exit 1; fi
if [ "$ref" != "$hw" ]; then
  first=$(diff <(echo "$ref") <(echo "$hw") | head -1)
  echo "FAIL $tag: ref=$nr hw=$nh fotogrames, primera diferència: $first"; exit 1
fi
echo "PASS $tag: $nh fotogrames idèntics"
