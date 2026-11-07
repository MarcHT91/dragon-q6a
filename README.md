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

- Radxa Dragon Q6A, Ubuntu 26.04.1 LTS (rafaelwms image), kernel `7.2.9-edge-qcs6490`
- Kodi 21.3 (Ubuntu package), GBM windowing
- Mesa 26.0.8 (freedreno, Adreno 643)

Other kernels or Kodi versions may behave differently, especially iris, whose behaviour
depends on the driver version.

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

Then install the launcher:

```bash
install -m 755 kodi-iris /opt/ffmpeg-iris/bin/
```

## Using it with Kodi

1. Start Kodi with `/opt/ffmpeg-iris/bin/kodi-iris [file]`. Ubuntu's `kodi` wrapper splits
   file arguments on spaces, so pass paths without spaces (a symlink works).
2. In *Settings → Player → Videos* (Expert mode):
   - **Allow using DRM PRIME decoder**: on (`videoplayer.useprimedecoder`).
   - **Allow hardware acceleration with DRM PRIME**: on
     (`videoplayer.useprimedecoderforhw`). This one is required: without it Kodi picks the
     software decoder.
   - **PRIME render method**: **Direct to plane** (`videoplayer.useprimerenderer` = `0`).
     EGL does not work (see Limitations).

   If you edit `~/.kodi/userdata/guisettings.xml` by hand while Kodi is closed, remove the
   `default="true"` attribute from the line. Otherwise Kodi ignores the value.
3. To quit Kodi from a script, use `tests/kodi-stop.sh`. It stops playback through JSON-RPC
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

Patches `0001`–`0003` and `0005` only take effect when the driver is `iris_driver`. Patch
`0004` only takes effect when the caller asks the software HEVC decoder for `DRM_PRIME`.

## Tests

| Script | What it does |
|---|---|
| `tests/abi-check.sh` | Checks that every FFmpeg symbol `kodi.bin` imports exists in `/opt/ffmpeg-iris`. |
| `tests/gen-media.sh` | Generates the synthetic clips in `tests/media/`: H.264, HEVC and VP9 from 360p to 2160p, a mid-stream resolution change, and 10-bit HEVC. |
| `tests/run-matrix.sh` | Frame-by-frame comparison (framemd5) of the software decoder against iris, with NV12 and DRM PRIME output (`tests/bin/drmframemd5`). 10-bit must be refused at open. Exits 0 when everything passes. |
| `tests/bin/seektest FILE DEC N` | Does N seeks the way Kodi does them (NULL-packet drain, `avcodec_flush_buffers`, seek) and measures the time to the first frame. |
| `tests/bin/primefallback FILE 0\|1` | Checks that the software HEVC decoder refuses >8-bit streams when `DRM_PRIME` is preset. |
| `tests/kodi-check.sh` | While Kodi is playing: libraries from `/opt`, `/dev/video0` open, V4L2 decoder in the log, CPU < 30 %. |
| `tests/kodi-seek.sh [N]` | While Kodi is playing: N seeks through JSON-RPC, with resume time (fails above 3 s). |
| `tests/kodi-stop.sh` | Quits Kodi without leaving iris stuck. |
| `tests/eglprobe.c` | Diagnostics: prints the DRM descriptor iris returns and tries to import it into EGL. |

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

Kodi 21.3, Direct to plane, 1080p display:

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

## Known limitations

- **EGL renderer does not work.** iris returns linear NV12 (`DRM_FORMAT_MOD_LINEAR`). Mesa
  only imports it when the linear modifier is passed explicitly. Kodi leaves the modifier
  out when it is linear, so every frame fails with `0x3003` (EGL_BAD_ALLOC). Use
  **Direct to plane**.
- **4K on a 1080p display.** With Direct to plane, the display controller rejects a
  3840×2160 plane scaled down to 1080p (`DrmAtomicCommit - test commit failed: No such
  device` on every frame), so no picture. A display running at 2160p needs no scaling.
  This has not been tested yet.
- **10-bit** is not decoded by iris (the driver offers no >8-bit capture format) and falls
  back to software. 10-bit 4K in software is not smooth.
- **10-bit VP9** probably gives a black screen. VP9 has no bit depth information at open,
  and patch `0004` only covers HEVC.
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

The patches in `patches/` modify FFmpeg and are under FFmpeg's licence (LGPL-2.1-or-later).
The build enables `--enable-gpl`, so the resulting libraries are GPL-2.0-or-later.
