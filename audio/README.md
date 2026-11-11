# dragon-q6a-audio

[Català](README.ca.md)

Audio fixes for the **Radxa Dragon Q6A** (Qualcomm QCS6490), on Ubuntu 26.04.1 LTS (rafaelwms image)
with the Armbian edge kernel `7.2.9-edge-qcs6490`:
24-bit headphone output, reliable DSP routes for the headphone jack and HDMI, a usable
`alsamixer`, and an AirPlay 2 receiver (shairport-sync) with hardware volume.

| Directory | What it does |
|---|---|
| [`hifi/`](hifi) | DKMS module with two kernel fixes and a DSP topology override: the headphone path runs at **S24_LE / 48 kHz** instead of S16. Includes the test bench and patches ready for upstream. |
| [`route/`](route) | Why the headphone route disappeared on boot (kernel bug and UCM sequence) and a boot service that enables both routes. |
| [`alsa-ctl-filter/`](alsa-ctl-filter) | ALSA control plugin: `alsamixer` shows only the useful volumes, and shairport-sync gets a correct dB scale for the DSP volume. Includes the shairport-sync config. |
| [`docs/mixer-controls.md`](docs/mixer-controls.md) | What every one of the ~200 mixer controls does. |

## Findings

* **The 16-bit/48 kHz limit is software.** The DSP topology only declares S16/48k, and the
  machine driver (`sc8280xp.c`) forces the back end to S16/48k. The hardware handles
  S16/S24/S32 at 48, 96 and 192 kHz on the headphone path (each one checked by listening).
  44.1 kHz on the back end is rejected cleanly. The DSP resamples any front-end rate from 8 to 192 kHz.
* **S24_LE crashed the ADSP.** `audioreach.c` declared S24_LE as packed 3-byte samples
  (`bits_per_sample = 24`). With a 32-bit container it works. → [`hifi/upstream/0001`](hifi/upstream)
  The DSP then reads full 32-bit words, so **S24_LE data must be sign-extended** into the top byte.
  alsa-lib conversions (`plughw`) do that; shairport-sync's S24_LE output leaves the top byte at
  zero and sounds completely distorted, so shairport uses S16_LE (the volume is applied in the DSP
  and the back end stays S24, so attenuation does not lose resolution).
* **Routes got lost.** In `topology.c`, writing 0 to any route control disconnected the back end,
  whichever stream was connected. `alsactl restore` lost HDMI on every boot, and the board's UCM
  HDMI sequence disconnected the headphones. → [`hifi/upstream/0002`](hifi/upstream), [`route/`](route)
* A PCM started while its route is off never runs (`hw_ptr` stays 0). shairport-sync used to stall
  this way and ignored SIGTERM.
* **AirPlay 2** from an iPhone (Tidal, including hi-res) arrives as `AAC/48000/F24/2`: no
  resampling is needed at 48 kHz.

## Install

Needs `flex bison bc libelf-dev dkms` and either a `linux-headers` package for the running
kernel or a prepared kernel tree (see [`hifi/README.md`](hifi/README.md)).

    sudo KSRC=/path/to/linux-7.2.9 ./install-all.sh
    sudo reboot

Or install the parts separately: `hifi/install.sh`, `route/install.sh`,
`alsa-ctl-filter/install-shairport-hwvol.sh`. Remove the DKMS module with `hifi/uninstall.sh`.

Check:

    aplay -D hw:CARD=QCS6490RadxaDra,DEV=0 --dump-hw-params -d1 /dev/zero | grep FORMAT   # S16_LE S24_LE
    cat /proc/asound/card0/pcm0p/sub0/hw_params                                          # while playing

## Related

* [kodi-iris](../kodi-iris): hardware video decoding in Kodi on the same board.

## License

MIT ([LICENSE](LICENSE)), except:
* `alsa-ctl-filter/ctl_filter.c`: LGPL-2.1-or-later (an alsa-lib plugin).
* `hifi/upstream/*.patch` and the modules built by `hifi/dkms` (from Linux kernel sources fetched at
  build time): GPL-2.0.
