#!/bin/sh
# Build native/libqnnrun.so (native QNN runner used by npu/lfm3_native.py).
# Needs only the public QNN API headers: by default they are taken from Ubuntu's public "Qualcomm Addon
# Packages" PPA (qairt-headers .deb, no account needed) and unpacked into native/qnn-headers/.
# Set QNN_INCLUDE to use headers you already have (the directory that contains QnnInterface.h).
# The runtime libraries (libQnnHtp.so, libQnnSystem.so, v68 skel) are the ones inside the onnxruntime-qnn wheel.
set -e
cd "$(dirname "$0")"
INC=${QNN_INCLUDE:-qnn-headers/usr/include/QNN}
if [ ! -f "$INC/QnnInterface.h" ]; then
    DEB=qairt-headers_2.49.0-0ubuntu2_arm64.deb
    URL=https://ppa.launchpadcontent.net/ubuntu-qcom-iot/qcom-ppa/ubuntu/pool/main/q/qairt-sdk/$DEB
    echo "fetching QNN headers: $URL"
    curl -fL -o "$DEB" "$URL"
    dpkg-deb -x "$DEB" qnn-headers
    rm -f "$DEB"
    INC=qnn-headers/usr/include/QNN
fi
g++ -O2 -std=c++17 -fPIC -shared -Wall -Wno-missing-field-initializers -I"$INC" qnnrun.cpp -ldl -o libqnnrun.so
echo "built native/libqnnrun.so"
