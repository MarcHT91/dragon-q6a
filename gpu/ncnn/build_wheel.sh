#!/bin/sh
# Build a patched ncnn Python wheel (Vulkan on) for the board: Ubuntu 26.04 arm64, CPython 3.14.
# Runs in Docker on any arm64 host (the board, or an Apple Silicon Mac: native arm64, ~1 min on an M5).
# usage: ./build_wheel.sh [ncnn tag]   -> ncnn-build/dist/ncnn-*-cp314-cp314-linux_aarch64.whl
# On the board, load it without touching the installed ncnn: unzip the wheel into a dir and set PYTHONPATH.
set -e
TAG=${1:-20261020}
cd "$(dirname "$0")"
mkdir -p ncnn-build && cd ncnn-build
[ -d ncnn ] || git clone -q --depth 1 --branch "$TAG" https://github.com/Tencent/ncnn.git
(cd ncnn && git submodule update -q --init --depth 1 glslang python/pybind11)
python3 ../apply_patch.py ncnn
cat > build-in-docker.sh <<'EOT'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential cmake python3-dev python3-pip python3-venv git >/dev/null
cd /src/ncnn
export EXTRA_CMAKE_ARGS="-DNCNN_VULKAN=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
python3 -m pip wheel . --no-deps -w /src/dist
EOT
docker run --rm --platform linux/arm64 -v "$(pwd)":/src ubuntu:26.04 bash /src/build-in-docker.sh
ls dist/
