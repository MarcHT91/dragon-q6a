#!/bin/bash
# Comprova que totes les funcions d'FFmpeg que importa kodi.bin existeixen a /opt/ffmpeg-iris.
set -euo pipefail
export LC_ALL=C   # sort i comm han de coincidir en l'ordre
P=${PREFIX:-/opt/ffmpeg-iris}
KODI=/usr/lib/aarch64-linux-gnu/kodi/kodi.bin
need=$(nm -D --undefined-only "$KODI" | awk '{print $2}' | sed 's/@.*//' \
       | grep -E '^(av|avcodec|avformat|avfilter|avutil|sws|swr)_' | sort -u)
have=$(for l in avcodec avformat avutil avfilter swscale swresample; do
         nm -D --defined-only "$P/lib/lib$l.so"; done | awk '{print $3}' | sed 's/@.*//' | sort -u)
missing=$(comm -23 <(echo "$need") <(echo "$have"))
n=$(echo "$need" | wc -l)
if [ -n "$missing" ]; then echo "FAIL: falten símbols:"; echo "$missing"; exit 1; fi
echo "PASS: $n símbols FFmpeg de Kodi presents a $P"
