# hifi: 24-bit headphone path (qcs6490-hifi)

24-bit headphone output on the Radxa Dragon Q6A (Qualcomm QCS6490, Armbian
edge kernel 7.2.9). Stock Linux limits the 3.5 mm jack to S16_LE @ 48 kHz,
but every one of those limits is in software.

## Where the limits are

| Stage | Stock | Real limit (tested) |
|---|---|---|
| DSP front end (`MultiMedia1`, `hw:0,0`) | S16_LE, 48 kHz (topology caps) | S16_LE/S24_LE, 8–192 kHz; DSP resamples, including 44.1 kHz |
| S24_LE into the DSP | crashes ADSP `audio_process` | works with the audioreach fix below |
| Back end codec DMA RX → RX macro → WCD9385 | S16_LE, 48 kHz (`sc8280xp_be_hw_params_fixup`) | S16/S24/S32, 48/96/192 kHz; 44.1 kHz family rejected (-EINVAL, no crash) |
| HDMI/DP (`MultiMedia2`) | S16_LE, 48 kHz | untouched |

Each working combination was confirmed by listening (1 kHz tone, correct pitch, clean).
The WCD9385 receives an already modulated 1-bit stream over SoundWire
(`sconfig.bps = 1`): bit depth only matters up to the LPASS RX macro.
That is also why the codec has `DSD_L/R Switch` controls (native DSD input),
which Linux has no path to use.

AirPlay 2 note: iPhone sends `AAC/48000/F24/2` for both normal and hi-res Tidal,
so S24_LE @ 48 kHz is the best the AirPlay path can carry.

## What it installs

* **DKMS `qcs6490-hifi`** (`dkms/`), rebuilt for each kernel:
  * `snd-soc-sc8280xp`: headphone back end format/rate as module parameters
    `rx_be_format` (16 default, 24, 32, 0 = follow FE) and `rx_be_rate`
    (48000 default, 0 = follow FE). Other links unchanged.
  * `snd-q6apm`:
    * S24_LE uses a 32-bit container (`bits_per_sample = 32`). Upstream sends
      24, the DSP reads packed 3-byte samples and crashes.
      Upstream patch: `upstream/0001-*.patch`.
    * A route mixer write of 0 only disconnects the link if that control owns
      it. Upstream clears the back end's link on any 0 write, so `alsactl restore`
      loses HDMI and the board's UCM HDMI sequence disconnects the headphones.
      Upstream patch: `upstream/0002-*.patch`. See [`../route`](../route).
  * `prepare.sh` takes the sources of the target kernel (tree or kernel.org)
    and refuses to build if the patch context changed.
* **Topology override** `/lib/firmware/updates/qcom/qcs6490/QCS6490-Radxa-Dragon-Q6A-tplg.bin`
  made by `patch-tplg.py` (MultiMedia1 caps only: formats 0x4→0x44, rates 48000→8000–192000).
* `/etc/modprobe.d/qcs6490-hifi.conf`: `rx_be_format=24 rx_be_rate=48000`.

## Install

Needs `flex bison bc libelf-dev dkms`. With an Armbian `linux-headers-*` package
for the running kernel:

    sudo ./install.sh && sudo reboot

Without headers (7.2.9-edge-qcs6490 has none published), prepare a tree first:

    curl -LO https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-7.2.9.tar.xz   # verify sha256sums.asc
    tar xf linux-7.2.9.tar.xz && cd linux-7.2.9
    cp /boot/config-$(uname -r) .config
    scripts/config -d ARM64_LSUI          # newer binutils enables it; stock kernel has it off
    make olddefconfig LOCALVERSION=-edge-qcs6490
    make -j8 modules_prepare LOCALVERSION=-edge-qcs6490   # kernelrelease must equal uname -r
    sudo KSRC=$PWD /path/to/dragon-q6a-audio/hifi/install.sh && sudo reboot

No MODVERSIONS/module signing on this kernel, so only the vermagic must match.
`dkms/gen-symvers.sh` builds a Module.symvers from installed modules so the
`depends:` field is correct.

Remove with `sudo ./uninstall.sh && sudo reboot`.

## Check

    aplay -D hw:CARD=QCS6490RadxaDra,DEV=0 --dump-hw-params -d1 /dev/zero   # FORMAT: S16_LE S24_LE
    cat /sys/module/snd_soc_sc8280xp/parameters/rx_be_*                     # 24, 48000

## Test bench used

`sudo ./lab-setup.sh` (adm group for kernel logs, sudoers rule for a whitelisted
parameter setter, shairport stopped), then `lab/run.sh FE_RATE FE_FMT BE_FMT BE_RATE`
(plays a tone, reports PCM state and kernel log) and
`lab/listen.sh N FE_RATE FE_FMT BE_FMT BE_RATE` (N beeps + 1 kHz, for listening).
`sudo ./lab-teardown.sh` undoes it.

Results (2026-11-11, kernel 7.2.9-edge-qcs6490):

| # | FE | BE | Result |
|---|---|---|---|
| 1 | S16 48k | S16 48k | OK |
| 2 | S16 44.1k | S16 48k | OK (DSP resamples) |
| 3 | S24 48k | S16 48k | ADSP crash without the audioreach fix, OK with it |
| 4 | S24 48k | S24 48k | OK |
| 5 | S24 48k | S32 48k | OK |
| 6 | S24 48k | S24 96k / 192k | OK |
| 7 | S24 44.1k | S24 44.1k | rejected, -EINVAL at link prepare, no crash |
| 8 | S24 192k | S24 192k | OK |

## Caveats

* After an ADSP crash the card does not come back (wcd938x probe -EBUSY): reboot.
* An Armbian firmware update does not refresh the override in `/lib/firmware/updates`;
  re-run `install.sh` if the topology changes.
* Headphone mixer route must be on: `RX_CODEC_DMA_RX_0 Audio Mixer MultiMedia1`
  (UCM `_enadev HDMI` and `_enadev Headphones` disable each other), then `alsactl store`.

Related: [`../alsa-ctl-filter`](../alsa-ctl-filter) (alsamixer whitelist + AirPlay hardware volume).
