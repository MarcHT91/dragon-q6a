# board: general setup of the Dragon Q6A

[Català](README.ca.md)

Tested on Ubuntu 26.04.1 LTS (rafaelwms image) with the Armbian edge kernel `7.2.9-edge-qcs6490`.

## GPU acceleration (Adreno 643)

The GPU works out of the box with Mesa: freedreno for GL/GLES (the renderer shows as `FD635`)
and Turnip for Vulkan 1.3 (it reports itself as "Adreno (TM) 643"). The render node is
`/dev/dri/renderD128`.

If `glxinfo`/`eglinfo` show **llvmpipe**, the user cannot open `/dev/dri` and Mesa silently
falls back to software rendering:

    sudo usermod -aG render,video $USER     # then log in again

Sessions started before the group change keep the old groups.

## Performance tuning (`tuning/`)

| File | Effect |
|---|---|
| `cpufrequtils` → `/etc/default/cpufrequtils` | Sets the CPU governor to `schedutil` (the kernel default is `ondemand`). Needs the `cpufrequtils` package. |
| `99-rps-enp1s0.rules` | The onboard Ethernet (r8169) has one RX queue and its IRQ runs on CPU0. RPS spreads receive processing over CPUs 1–6 (mask `7e`). |
| `zram-generator.conf` | Compressed swap in RAM: half of the RAM, at most 6 GB, zstd. Needs `systemd-zram-generator`. |
| `90-dragon-q6a.conf` → `/etc/sysctl.d/` | Limits pending writes to 64/256 MB (avoids long stalls on USB storage), and sets `swappiness=100` and `page-cluster=0` for zram. |

    sudo ./tuning/install.sh && sudo reboot

## GPU monitor (`tools/gputop`)

A `top`-style monitor for the Adreno. It shows the frequency (current, average, and time at
each step from devfreq `trans_stat`), busy %, per-process GPU use and memory (from DRM
`fdinfo`), and the `gpuss0/1` temperatures.

    tools/gputop            # refresh every second
    sudo tools/gputop       # also shows processes with capabilities (e.g. Kodi)

Kodi has `CAP_WAKE_ALARM`, so the kernel marks it non-dumpable and even the same user cannot
read its `/proc/<pid>/fdinfo`. That is why `gputop` and `nvtop` need sudo to see it.

The GPU heats up fast: from about 57 °C at idle to about 84 °C after 4 s at 812 MHz full load.

## Related

* [`../audio`](../audio): audio (24-bit headphones, DSP routes, AirPlay).
* [`../kodi-iris`](../kodi-iris): hardware video decoding in Kodi and the HDMI colour range.
