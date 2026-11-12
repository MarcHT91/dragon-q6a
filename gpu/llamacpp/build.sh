#!/bin/sh
# Build llama.cpp (Vulkan + OpenCL backends, static) for the board in an arm64 Ubuntu 26.04 container
# (on the board or an Apple Silicon Mac). Used only as a reference for the LFM3-VL vision encoder (mmproj) on the GPU.
# usage: ./build.sh -> llama.cpp/build/bin/llama-mtmd-cli
set -e
cd "$(dirname "$0")"
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git
cat > build-in-docker.sh <<'EOT'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential cmake git glslc libvulkan-dev spirv-headers ocl-icd-opencl-dev opencl-headers >/dev/null
cd /src/llama.cpp
cmake -B build -DGGML_VULKAN=ON -DGGML_OPENCL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
      -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-mtmd-cli
EOT
docker run --rm --platform linux/arm64 -v "$(pwd)":/src ubuntu:26.04 bash /src/build-in-docker.sh
ls -la llama.cpp/build/bin/
# Run (encoder time = "mtmd batch encoding done in N ms"; pass the image twice to see the steady-state time):
#   ./llama.cpp/build/bin/llama-mtmd-cli -m LFM3-VL-2B-QAD-Q4_0.gguf --mmproj mmproj-LFM3-VL-2B-F16.gguf \
#       --image img512.jpg --image img512.jpg -p "Describe." -n 1 -ngl 0 -v
