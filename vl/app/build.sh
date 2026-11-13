#!/bin/sh
# Build vl/app/lfm3vl (LFM3-VL chat: vision encoder on the GPU with ncnn/Vulkan, language model on the NPU with QNN).
#
# Needs:
#   NCNN_DIR   ncnn install prefix (lib/libncnn.a, include/ncnn) built with -DNCNN_VULKAN=ON and our MatMul Vulkan
#              patch (gpu/ncnn/apply_patch.py); see README.md.
#   QNN_INCLUDE QNN API headers (default: the ones npu/native/build.sh fetched into npu/native/qnn-headers)
#   QNN_DIR    runtime libs dir (libQnnHtp.so, libQnnSystem.so, libQnnHtpV68Stub/Skel.so), baked in as the default;
#              default: the onnxruntime_qnn package of the python found as $PY (or python3)
#   libpcre2-dev (tokenizer regex)
# third_party/json.hpp (nlohmann/json 3.11.3, MIT) and third_party/stb_image.h (public domain) are fetched if missing.
set -e
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
: "${NCNN_DIR:?set NCNN_DIR to the ncnn install prefix}"
INC=${QNN_INCLUDE:-$REPO/npu/native/qnn-headers/usr/include/QNN}
[ -f "$INC/QnnInterface.h" ] || (cd "$REPO/npu/native" && sh build.sh)
if [ -z "$QNN_DIR" ]; then
    QNN_DIR=$(${PY:-python3} -c "import onnxruntime_qnn, os; print(os.path.dirname(onnxruntime_qnn.__file__))")
fi
mkdir -p third_party
[ -f third_party/json.hpp ] || curl -fsSL -o third_party/json.hpp https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
[ -f third_party/stb_image.h ] || curl -fsSL -o third_party/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

NCNN_LIBS="$NCNN_DIR/lib/libncnn.a"
for l in glslang SPIRV MachineIndependent GenericCodeGen OSDependent glslang-default-resource-limits OGLCompiler; do
    [ -f "$NCNN_DIR/lib/lib$l.a" ] && NCNN_LIBS="$NCNN_LIBS $NCNN_DIR/lib/lib$l.a"
done
g++ -O2 -std=c++17 -Wall -Wno-missing-field-initializers -Wno-sign-compare -fopenmp \
    -I"$NCNN_DIR/include/ncnn" -I"$INC" -DQNN_DIR_DEFAULT="\"$QNN_DIR\"" \
    lfm3vl.cpp "$REPO/npu/native/qnnrun.cpp" \
    -Wl,--start-group $NCNN_LIBS -Wl,--end-group -lpcre2-8 -ldl -lpthread -o lfm3vl
echo "built vl/app/lfm3vl (QNN libs: $QNN_DIR)"
