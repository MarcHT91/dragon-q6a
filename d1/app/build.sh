#!/bin/sh
# Build d1/app/d1 (d1-2B: GPU vision encoder + NPU stage graphs + CPU readout). Same requirements as vl/app/build.sh:
# NCNN_DIR (ncnn install prefix with Vulkan + the MatMul patch), QNN headers, QNN_DIR baked in (onnxruntime_qnn of $PY),
# libpcre2-dev. Reuses vl/app's tokenizer.h, image.h and third_party headers.
set -e
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
: "${NCNN_DIR:?set NCNN_DIR to the ncnn install prefix}"
INC=${QNN_INCLUDE:-$REPO/npu/native/qnn-headers/usr/include/QNN}
[ -f "$INC/QnnInterface.h" ] || (cd "$REPO/npu/native" && sh build.sh)
[ -f "$REPO/vl/app/third_party/json.hpp" ] || (cd "$REPO/vl/app" && NCNN_DIR="$NCNN_DIR" sh build.sh)
if [ -z "$QNN_DIR" ]; then
    QNN_DIR=$(${PY:-python3} -c "import onnxruntime_qnn, os; print(os.path.dirname(onnxruntime_qnn.__file__))")
fi
NCNN_LIBS="$NCNN_DIR/lib/libncnn.a"
for l in glslang SPIRV MachineIndependent GenericCodeGen OSDependent glslang-default-resource-limits OGLCompiler; do
    [ -f "$NCNN_DIR/lib/lib$l.a" ] && NCNN_LIBS="$NCNN_LIBS $NCNN_DIR/lib/lib$l.a"
done
g++ -O3 -std=c++17 -Wall -Wno-missing-field-initializers -Wno-sign-compare -fopenmp \
    -I"$REPO/vl/app" -I"$NCNN_DIR/include/ncnn" -I"$INC" -DQNN_DIR_DEFAULT="\"$QNN_DIR\"" \
    d1.cpp "$REPO/npu/native/qnnrun.cpp" \
    -Wl,--start-group $NCNN_LIBS -Wl,--end-group -lpcre2-8 -ldl -lpthread -o d1
echo "built d1/app/d1 (QNN libs: $QNN_DIR)"
