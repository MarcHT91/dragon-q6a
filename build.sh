#!/bin/bash
# Compila rpi-ffmpeg (branca iris) i l'instal·la a $PREFIX.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM=${UPSTREAM:-https://github.com/jc-kynesim/rpi-ffmpeg}
BASE=2a287060cc6e9024e0d20bc7f61da696546b553b   # test/8.0.1/main (libavcodec 62.11)
SRC=${SRC:-$HOME/src/ffmpeg-iris}
PREFIX=${PREFIX:-/opt/ffmpeg-iris}
BUILD=${BUILD:-$SRC/out/build}
if [ ! -d "$SRC/.git" ]; then
  # Primera vegada: rpi-ffmpeg a la revisió base + els pegats d'aquest repositori.
  git clone --branch test/8.0.1/main --single-branch "$UPSTREAM" "$SRC"
  git -C "$SRC" checkout -b iris "$BASE"
  git -C "$SRC" -c user.name=kodi-iris -c user.email=kodi-iris@invalid am "$HERE"/patches/*.patch
fi
mkdir -p "$BUILD"
cd "$BUILD"
if [ ! -f config.h ] || [ "${RECONF:-0}" = 1 ]; then
  "$SRC/configure" \
    --prefix="$PREFIX" --libdir="$PREFIX/lib" --extra-version=iris \
    --arch=arm64 --enable-gpl --enable-shared --disable-static \
    --disable-stripping --disable-doc \
    --enable-gnutls --enable-libdav1d --enable-libvpx --enable-libxml2 \
    --enable-libzvbi --enable-libbluray --enable-libsoxr --enable-libopus \
    --enable-libvorbis --enable-libmp3lame \
    --enable-libdrm --enable-v4l2-m2m --enable-sand --enable-vout-drm \
    --extra-ldflags="-Wl,-rpath,$PREFIX/lib"
fi
make -j"$(nproc)"
make install
"$PREFIX/bin/ffmpeg" -hide_banner -version | head -1
