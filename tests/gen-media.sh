#!/bin/bash
# Genera vídeos sintètics de 4 s (100 fotogrames) amb l'FFmpeg del sistema.
set -euo pipefail
D=$(dirname "$0")/media; mkdir -p "$D"
FF="/usr/bin/ffmpeg -hide_banner -loglevel error -y"
SRC="-f lavfi -i testsrc2=rate=25:duration=4:size"
for s in 640x360 1280x720 1920x1080 3840x2160; do
  $FF $SRC=$s -pix_fmt yuv420p -c:v libx264 -profile:v high -g 50 "$D/h264_$s.mkv"
  $FF $SRC=$s -pix_fmt yuv420p -c:v libx265 -x265-params log-level=error:keyint=50 "$D/hevc_$s.mkv"
  $FF $SRC=$s -pix_fmt yuv420p -c:v libvpx-vp9 -b:v 4M -deadline realtime -cpu-used 8 -g 50 "$D/vp9_$s.mkv"
done
# Canvi de resolució a mig flux: 720p i després 1080p en un sol flux H.264.
$FF $SRC=1280x720 -pix_fmt yuv420p -c:v libx264 -bf 0 -g 50 -f h264 "$D/a.h264"
$FF $SRC=1920x1080 -pix_fmt yuv420p -c:v libx264 -bf 0 -g 50 -f h264 "$D/b.h264"
cat "$D/a.h264" "$D/b.h264" > "$D/ab.h264"
$FF -f h264 -framerate 25 -i "$D/ab.h264" -c copy -bsf:v setts=ts=N/25/TB "$D/h264_reschange.mkv"
rm -f "$D"/a.h264 "$D"/b.h264 "$D"/ab.h264
# 10 bits: ha de caure a programari (fora d'abast per a iris).
$FF $SRC=1920x1080 -pix_fmt yuv420p10le -c:v libx265 -x265-params log-level=error "$D/hevc10_1920x1080.mkv"
ls -1 "$D"
