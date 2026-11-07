#!/bin/bash
# Compila rpi-ffmpeg (branca iris) i l'instal·la a $PREFIX.
set -euo pipefail
SRC=${SRC:-$HOME/src/ffmpeg-iris}
PREFIX=${PREFIX:-/opt/ffmpeg-iris}
BUILD=${BUILD:-$SRC/out/build}
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
