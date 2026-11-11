# dragon-q6a

*[Català](README.ca.md)*

Everything I got working on the **Radxa Dragon Q6A** (Qualcomm QCS6490) running Ubuntu 26.04.1 LTS
(rafaelwms image) with the Armbian edge kernel `7.2.9-edge-qcs6490`. Each directory is self-contained and has its own README.

| Directory | What it is |
|---|---|
| [`board/`](board) | General setup: GPU acceleration (Adreno 643), CPU/network/zram/writeback tuning, a GPU monitor (`gputop`). |
| [`audio/`](audio) | Audio: 24-bit headphone path (DKMS with two kernel fixes), DSP routes that no longer get lost, a usable `alsamixer`, AirPlay 2 (shairport-sync) with hardware volume. |
| [`kodi-iris/`](kodi-iris) | Hardware video decoding in Kodi with the iris decoder (V4L2 mem2mem, DRM PRIME), 4K through EGL, and the HDMI colour range with the onboard RA620 DP→HDMI converter. |
| [`npu/`](npu) | LFM3-1.7B (Liquid AI) running entirely on the Hexagon HTP v68 NPU with `onnxruntime-qnn`, without the Qualcomm SDK. |

The fixes are done on the board side: kernel modules, configuration and scripts, not settings
on the TV or the monitor. Kernel patches meant for upstream are in
[`audio/hifi/upstream`](audio/hifi/upstream).

## History

This repository merges three earlier repositories, with their full history:
`dragon-q6a-audio` → `audio/`, `kodi-iris` → `kodi-iris/`, `lfm3-q6a-npu` → `npu/`.
The old repositories are archived and point here.

## License

MIT ([LICENSE](LICENSE)). Some parts have their own licence, stated in their directory:
the FFmpeg patches in `kodi-iris/patches` (LGPL-2.1+), `audio/alsa-ctl-filter/ctl_filter.c`
(LGPL-2.1+), and the kernel patches and DKMS-built modules in `audio/hifi` (GPL-2.0).
