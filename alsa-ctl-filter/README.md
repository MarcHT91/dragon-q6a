# alsa-ctl-filter

ALSA external control plugin that shows only a whitelist of mixer controls,
optionally renamed and range-clamped. Built for the Radxa Dragon Q6A
(QCS6490), whose card exposes ~200 ASoC routing controls that make
`alsamixer` unusable.

* Child ctl defaults to `sysdefault:<card>`, so UCM `LibraryConfig` remaps
  (e.g. the combined `HP Volume` built from `HPHL/HPHR Volume`) are available.
* `max N` clamps the range (Q6 DSP stream volumes: 8192 = unity gain; higher clips).
* `linear_db true` advertises a linear dB scale (min = mute, max = 0 dB) instead
  of the driver's incorrect TLV.
* No root needed: install to `~/.local/lib/alsa-lib` and reference it with
  `ctl_type.filter { lib "..." }`.

## Build / install

    sudo apt install libasound2-dev
    make install            # -> ~/.local/lib/alsa-lib

## ~/.asoundrc

    ctl_type.filter {
        lib "/home/USER/.local/lib/alsa-lib/libasound_module_ctl_filter.so"
    }
    ctl.!default {
        type filter
        card QCS6490RadxaDra
        controls [
            { name "Auriculars Playback Volume" source "HP Volume" }
            { name "Auriculars digital Playback Volume"
              source "stream0.vol_ctrl0 MultiMedia1 Playback Volu" max 8192 linear_db true }
            { name "HDMI Playback Volume"
              source "stream1.vol_ctrl1 MultiMedia2 Playback Volu" max 8192 linear_db true }
        ]
    }

`alsamixer` now shows three sliders; the full mixer is still at
`alsamixer -D sysdefault`.

Supported element types: integer, boolean, enumerated.

## Known limitation

Change notifications for controls that are themselves UCM remaps (`HP Volume`)
are not propagated by alsa-lib's remap plugin, so an open alsamixer won't
refresh that slider when another program changes it. Direct kernel controls
refresh live.
