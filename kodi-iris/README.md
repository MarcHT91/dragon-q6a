# kodi-iris

*[Català](README.ca.md)*

Hardware video decoding in Kodi on the **Radxa Dragon Q6A** (Qualcomm QCS6490), using the
kernel's **iris** decoder (`qcom_iris`, `/dev/video0`) through V4L2 mem2mem.

Kodi as packaged by Ubuntu cannot use iris with the system FFmpeg. This project builds
[rpi-ffmpeg](https://github.com/jc-kynesim/rpi-ffmpeg) 8.0.1 with a few fixes for iris,
installs it under `/opt/ffmpeg-iris`, and starts Kodi with those libraries through
`LD_LIBRARY_PATH`. rpi-ffmpeg is ABI-compatible with Ubuntu's `libavcodec62`. No system
package is replaced, and plain `kodi` keeps using the system FFmpeg.

## Credits

- **[jc-kynesim/rpi-ffmpeg](https://github.com/jc-kynesim/rpi-ffmpeg)**: the FFmpeg branch
  this work starts from. Its V4L2 mem2mem decoder with DRM PRIME output does almost all
  of the work. The patches here only adapt it to iris.
- **[rafaelwms/ubuntu-26.04.1-dragon-q6a](https://github.com/rafaelwms/ubuntu-26.04.1-dragon-q6a)**:
  the Ubuntu 26.04.1 image for the Dragon Q6A (mainline 7.2 kernel with iris) that this
  was developed and tested on.

## Tested on

- Radxa Dragon Q6A, Ubuntu 26.04.1 LTS (rafaelwms image)
- Kernel `7.3.1-edge-qcs6490` (Armbian) with Radxa firmware `6.0.260916`: iris running the
  **HFI Gen2** firmware (`vpu20_p1_gen2_s6.mbn`), 10-bit in hardware
- Kernel `7.2.9-edge-qcs6490` with firmware `6.0.251230`: iris running the **Gen1** firmware
  (`vpu20_p1.mbn`), 8-bit only
- Kodi 21.3 (Ubuntu package), GBM windowing
- Mesa 26.0.8 (freedreno, Adreno 643)

Other kernels or Kodi versions may behave differently, especially iris, whose behaviour
depends on the driver version.

### Gen1 or Gen2

Same hardware (VPU 2.0), different firmware and host protocol (HFI). Gen2 needs a kernel that
supports it on SC7280 (7.3) and a DT that asks for it (`firmware-name`, `iris-iova`), which
Radxa firmware 260916 provides (DT `v7.0`). With Gen2, iris offers `P010` capture and decodes
HEVC Main10 and VP9 profile 2 up to 4K. Do not mix them: 7.2.9 with the `v7.0` DT loads the
Gen2 firmware but speaks Gen1 (iris does not start), and 7.3.1 with firmware 251230 tries Gen2
without the right DT (`firmware download failed -22`).

## Requirements

```bash
sudo apt install -y git build-essential pkg-config \
    libdrm-dev libdav1d-dev libvpx-dev libxml2-dev libzvbi-dev \
    libbluray-dev libmp3lame-dev libudev-dev libgnutls28-dev libsoxr-dev \
    libopus-dev libvorbis-dev
sudo install -d -o "$USER" -g "$USER" /opt/ffmpeg-iris
```

Your user must be in the `video` and `render` groups. If Kodi is started from SSH, it also
needs to be in `audio` and `input`.

## Build

```bash
./build.sh            # first run: clones rpi-ffmpeg, applies patches/, configures, builds, installs
RECONF=1 ./build.sh   # force a reconfigure
```

On the first run `build.sh` clones `test/8.0.1/main` from rpi-ffmpeg into `$SRC`, checks out
the base commit `2a287060cc6e`, and applies `patches/*.patch` with `git am`.

Variables: `SRC` (default `~/src/ffmpeg-iris`), `PREFIX` (`/opt/ffmpeg-iris`), `BUILD`
(`$SRC/out/build`) and `UPSTREAM` (the rpi-ffmpeg URL).

Then build the EGL shim and install it with the launcher:

```bash
gcc -O2 -shared -fPIC -o /opt/ffmpeg-iris/lib/libegl_linear_shim.so shim/egl_linear_shim.c -ldl
install -m 755 kodi-iris /opt/ffmpeg-iris/bin/
```

The launcher preloads `libegl_linear_shim.so`, switches on the DSP audio routes for HDMI and the
headphone jack (they can be lost on reboot; it does not use `alsaucm`, whose HDMI sequence on
this board disconnects the headphones, see
[audio](../audio)), and turns on the shim's limited-range mode when Kodi's *limited colour range*
setting is on (see [HDMI colour range](docs/hdmi-colour-range.md)).

The shim adds the explicit `DRM_FORMAT_MOD_LINEAR` modifier that Kodi leaves out when it
imports iris frames into EGL. Without it, Mesa refuses every frame (`0x3003`).

## Using it with Kodi

1. Start Kodi with `/opt/ffmpeg-iris/bin/kodi-iris [file]`. Ubuntu's `kodi` wrapper splits
   file arguments on spaces, so pass paths without spaces (a symlink works).
2. In *Settings → Player → Videos* (Expert mode):
   - **Allow using DRM PRIME decoder**: on (`videoplayer.useprimedecoder`).
   - **Allow hardware acceleration with DRM PRIME**: on
     (`videoplayer.useprimedecoderforhw`). This one is required: without it Kodi picks the
     software decoder.
   - **PRIME render method**: **EGL** (`videoplayer.useprimerenderer` = `1`). It needs the
     shim installed above. **Direct to plane** (`0`) also works and uses a little less GPU,
     but cannot show video wider than 2400 px (see Limitations).

   If you edit `~/.kodi/userdata/guisettings.xml` by hand while Kodi is closed, remove the
   `default="true"` attribute from the line. Otherwise Kodi ignores the value.
3. On a TV, turn on *Settings → System → Display → Use limited colour range (16-235)* and
   leave the TV's HDMI black level on Auto. The board's HDMI converter cannot tell the TV
   that it sends full range: see [HDMI colour range](docs/hdmi-colour-range.md).
4. To quit Kodi from a script, use `tests/kodi-stop.sh`. It stops playback through JSON-RPC
   (port 8080, web server enabled), waits until `/dev/video0` is released, quits, and only
   then falls back to `kill -9`.

If iris ever gets stuck (every program fails with `VIDIOC_REQBUFS failed: Invalid argument`
while nothing holds `/dev/video0`), reload the driver:

```bash
sudo modprobe -r qcom_iris && sudo modprobe qcom_iris
```

## Patches (`patches/`)

| Patch | Problem it fixes |
|---|---|
| `0001` | iris returns empty capture buffers flagged as errors. They are now discarded instead of being output as frames. |
| `0002` | If capture cannot be restarted after a resolution change, decoding fails promptly instead of hanging. |
| `0003` | iris has no capture format above 8 bits, so >8-bit streams are refused at open (`FF_V4L2_QUIRK_8BIT_ONLY`) instead of failing after decoding has started. |
| `0004` | When the hardware decoder refuses a stream, Kodi's DRM PRIME path retries with the software decoder. That path only accepts 8-bit formats, so 10-bit HEVC gave a black screen. `hevcdec` now refuses >8-bit streams at open when `DRM_PRIME` is requested, and Kodi falls back to its normal software decoder. |
| `0005` | After a drain (Kodi drains before every seek), iris never decodes again in the same session, and the stalled session could leave the driver unusable for every client. On iris, flush now closes and reopens the decoder (`FF_V4L2_QUIRK_REOPEN_AFTER_DRAIN`). |
| `0006` | With the Gen2 firmware, iris offers `P010` and picks it at source change for 10-bit streams. Adds `P010` ↔ `V4L2_PIX_FMT_P010` and `DRM_FORMAT_P010` for DRM PRIME, and only applies the `FF_V4L2_QUIRK_8BIT_ONLY` from `0003` when the capture queue has no `P010` (Gen1). |

Patches `0001`–`0003`, `0005` and `0006` only take effect when the driver is `iris_driver`. Patch
`0004` only takes effect when the caller asks the software HEVC decoder for `DRM_PRIME`.

## Tests

| Script | What it does |
|---|---|
| `tests/abi-check.sh` | Checks that every FFmpeg symbol `kodi.bin` imports exists in `/opt/ffmpeg-iris`. |
| `tests/gen-media.sh` | Generates the synthetic clips in `tests/media/`: H.264, HEVC and VP9 from 360p to 2160p, a mid-stream resolution change, and 10-bit HEVC and VP9. |
| `tests/run-matrix.sh` | Frame-by-frame comparison (framemd5) of the software decoder against iris, with NV12 and DRM PRIME output (`tests/bin/drmframemd5`). If iris offers `P010` (Gen2), 10-bit is compared as P010; otherwise (Gen1) it must be refused at open. Exits 0 when everything passes. |
| `tests/bin/seektest FILE DEC N` | Does N seeks the way Kodi does them (NULL-packet drain, `avcodec_flush_buffers`, seek) and measures the time to the first frame. |
| `tests/bin/primefallback FILE 0\|1` | Checks that the software HEVC decoder refuses >8-bit streams when `DRM_PRIME` is preset. |
| `tests/kodi-check.sh` | While Kodi is playing: libraries from `/opt`, `/dev/video0` open, V4L2 decoder in the log, CPU < 30 %. |
| `tests/kodi-seek.sh [N]` | While Kodi is playing: N seeks through JSON-RPC, with resume time (fails above 3 s). |
| `tests/kodi-stop.sh` | Quits Kodi without leaving iris stuck. |
| `tests/eglprobe.c` | Diagnostics: prints the DRM descriptor iris returns and tries to import it into EGL. |
| `tests/yuvhint.c` | Prints the RGB Mesa produces from NV12 under each EGL colour-space/range hint (with the shim: checks its limited-range mode). |

Build the C tools against `/opt/ffmpeg-iris`, for example:

```bash
mkdir -p tests/bin
gcc -Wall -O1 tests/seektest.c -I/opt/ffmpeg-iris/include -L/opt/ffmpeg-iris/lib \
    -Wl,-rpath,/opt/ffmpeg-iris/lib -lavformat -lavcodec -lavutil -o tests/bin/seektest
```

For more realistic 8-bit HEVC, re-encode part of any H.264 video:

```bash
ffmpeg -ss 120 -t 60 -i INPUT.mkv -map 0:v:0 -map 0:a:0 -c:v libx265 -preset ultrafast \
    -pix_fmt yuv420p -c:a copy tests/media/real_hevc_1080p.mkv
```

## Results

With the Gen1 firmware (kernel 7.2.9, firmware 251230), Kodi 21.3, Direct to plane, 1080p display:

| Content | Decoder | Kodi CPU | Notes |
|---|---|---|---|
| H.264 1080p | iris | ~15 % | software decode: 100–150 % |
| HEVC 8-bit 1080p | iris | ~15 % | |
| HEVC 8-bit 2160p | iris | ~16 % | decodes fine; see the 4K limitation below |
| HEVC 10-bit 1080p | software | ~100 % (one core) | real time, no errors |
| 10 forward/back seeks | iris | | resumes in ~0.6 s in Kodi; `seektest`: ~55 ms (1080p), ~130 ms (2160p) |
| Switching video, quitting Kodi | iris | | clean, iris stays usable |

`tests/run-matrix.sh`: 27/27 (H.264, HEVC and VP9 from 360p to 2160p, NV12 and DRM PRIME,
resolution change, 10-bit refused at open).

With the Gen2 firmware (kernel 7.3.1, firmware 260916), Kodi 21.3, EGL rendering, 2160p display:

| Content | Decoder | Kodi CPU | Notes |
|---|---|---|---|
| HEVC 8-bit 1080p | iris | ~15–17 % | |
| **HEVC 10-bit 2160p** | **iris (P010)** | **~11 %** | previously software, not smooth |
| 10 seeks back and forth | iris | | resumes in ~0.55 s; with `seektest`, ~50 ms (1080p) and ~108 ms (2160p) |

`tests/run-matrix.sh`: 30/30 (the 26 8-bit cases plus 10-bit HEVC and VP9 as P010, with NV12
and DRM PRIME). 10-bit HEVC 2160p and 10-bit VP9 2160p are also bit-exact against software.

## Known limitations

- **4K with Direct to plane.** The display controller's planes accept at most 2400 px of
  source width (`max_linewidth` on SC7280), so 3840-wide frames fail the atomic test commit
  (`DrmAtomicCommit - test commit failed`) on every frame. Use the **EGL** renderer.
- **HDMI colour range and HDR.** The onboard RA620 DP-to-HDMI converter never tells the TV
  the RGB range, and does not forward HDR metadata. Kodi is set to limited range as a
  workaround; HDR10 is tone-mapped to SDR. Details: [HDMI colour range](docs/hdmi-colour-range.md).
- **10-bit with Gen1** is not decoded by iris (the driver offers no >8-bit capture format)
  and falls back to software. 10-bit 4K in software is not smooth. With Gen2 it is decoded in
  hardware (see [Gen1 or Gen2](#gen1-or-gen2)).
- **10-bit VP9 with Gen1** probably gives a black screen. VP9 has no bit depth information at
  open, and patch `0004` only covers HEVC.
- **Seeking with Gen2.** Now and then a seek made right after opening a video stalls for
  ~10 s, with `OutputPicture - timeout waiting for buffer` in Kodi's log. It can also happen
  when a video restarts in repeat mode. The cause is not known yet; `seektest` (no rendering)
  does not reproduce it.
- **AV1**: VPU 2.0 has no AV1 decoder.
- **8K**: iris refuses it ("current session not supported"). Software decoding manages
  ~32 fps.
- JSON-RPC `Player.Open` answers `Invalid params` for new files in a directory Kodi has
  already listed (directory cache). Use a new directory.

## Uninstall

```bash
rm -rf /opt/ffmpeg-iris
```

Then turn the two DRM PRIME options off again in Kodi.

## Updating

The FFmpeg in `/opt` must keep the same `LIBAVCODEC_VERSION_MAJOR/MINOR` as the
`libavcodec62` that Kodi was built against (62.11 at the time of writing):

```bash
git -C ~/src/ffmpeg-iris fetch origin
git -C ~/src/ffmpeg-iris show origin/test/8.0.x/main:libavcodec/version.h | grep VERSION_MINOR
git -C ~/src/ffmpeg-iris rebase --onto origin/test/8.0.x/main 2a287060cc6e iris
./build.sh && tests/abi-check.sh && tests/run-matrix.sh
```

## Licence

This project's own code (the shim, launcher, build script, tests and tools) is under the
[MIT licence](LICENSE).

The patches in `patches/` modify FFmpeg and are under FFmpeg's licence (LGPL-2.1-or-later).
The build enables `--enable-gpl`, so the resulting libraries are GPL-2.0-or-later.
