# Dragon Q6A mixer controls

`alsamixer -D sysdefault` shows ~200 controls. They are one signal chain:

```
app -> ADSP (AudioReach: MultiMedia1/2/3) -> LPASS RX/TX macro (digital) -> WCD9385 (analog) -> 3.5 mm jack
                                          \-> DISPLAY_PORT_RX_0 -> HDMI (via DP)
```

For daily use only **`HP Volume`** matters (the `alsa-ctl-filter` config shows it as
*Auriculars*). Everything else is set by UCM or by this repo.

## 1. DSP routes (AudioReach)

| Control | Meaning |
|---|---|
| `RX_CODEC_DMA_RX_0 Audio Mixer MultiMedia1` | `hw:0,0` → headphones. Must be on. |
| `RX_CODEC_DMA_RX_0 Audio Mixer MultiMedia2` | `hw:0,1` → headphones. Normally off. |
| `DISPLAY_PORT_RX_0 Audio Mixer MultiMedia2` | `hw:0,1` → HDMI. On for a TV. |
| `DISPLAY_PORT_RX_0 Audio Mixer MultiMedia1` | `hw:0,0` → HDMI. Normally off. |
| `MultiMedia3 Mixer TX_CODEC_DMA_TX_3` | jack microphone → `hw:0,2` |
| `stream0/1.vol_ctrl… Playback Volu` | DSP digital volume per stream. 8192 = gain 1.0 (0 dB); the driver's dB scale is wrong and values above 8192 clip. |

Each back end accepts one source at a time; see [`../route`](../route) for the
kernel bug that made writes of 0 disconnect the wrong stream.

## 2. LPASS RX macro (digital playback)

* `RX_MACRO RX0…RX5 MUX`: input of each internal channel (RX0/RX1 = AIF1_PB = headphone L/R).
* `RX INT0/1/2 …`: interpolators (INT0 = HPHL, INT1 = HPHR, INT2 = line/ear).
  `MIX1 INP0-2` mixer inputs, `INTx_2 MUX`/`MIX2 INP` second inputs (sidetone),
  `DEM MUX` modulator mode (`CLSH_DSM_OUT`), `INTERP` filters.
* `RX_RX0/1/2 Digital Volume`: 84 = 0 dB. `… Mix Digital Volume`: second input.
* `RX_COMP1/2 Switch`, `RX_Softclip Switch`: compressor/limiter, off.
* `IIR0/1/2 Band1-5`, `INPx MUX/Volume`: 5-band biquad EQ, unused.
* `RX HPH Mode`: amplifier class. `CLS_H_LOHIFI` (UCM default), `CLS_AB_LOHIFI`
  (fixed rail), `CLS_AB`, low-power `…_LP`, `CLS_H_ULP`. Read at amplifier power-up,
  so change it while nothing plays. **Use `CLS_AB_LOHIFI`**: in class H a sine above
  about -3 dBFS clips (the rail does not follow the peaks), even with low analog gain.
* `RX_HPH PWR Mode`: `LOHIFI` / `ULP`, must match the above.
* `RX_HPH HD2 Mode Switch`: digital 2nd-harmonic predistortion with a fixed
  coefficient tuned for low-impedance headphones. Off; pointless into a line input.
* `RX_EAR Mode Switch`: routes INT0 to the earpiece output, not wired on this board. Keep off.
* `RX MIX TX0-2 MUX`: echo reference.

The RX macro also does the delta-sigma modulation: SoundWire carries a 1-bit
stream to the WCD9385 (`sconfig.bps = 1`). Bit depth only matters up to here.

## 3. WCD9385 codec (analog)

| Control | Meaning |
|---|---|
| `HP Volume` (UCM remap of `HPHL/HPHR Volume`, only on `sysdefault:`) | Real headphone volume, 0–24. |
| `HPHL/HPHR Switch`, `HPHL/HPHR_RDAC Switch` | amplifier / DAC per channel |
| `HPHL/HPHR_COMP Switch` | analog compressor, off |
| `CLSH Switch`, `LO Switch` | class-H supply, line out |
| `HPH Type`, `HPHL/HPHR Impedance` | MBHC detection, read-only. 268435454 = 0x0FFFFFFE = "floating" (e.g. an amplifier input). `HPH Type` has no effect on routing. |
| `EAR_*`, `AUX_*`, `RDAC3_MUX` | earpiece/aux outputs, not wired |
| `DSD_L/R Switch` | native DSD input of the codec; Linux has no path to use it |
| `LDOH Enable`, `MBHC Switch` | supply, jack detection |

## 4. Capture

* `ADC1-4 Switch/Volume/MUX`, `ADCx_MIXER`: analog mic preamps. Only **ADC2 `INP2`**
  (TRRS jack mic, UCM "Headset") is wired. It is a biased mic input, not line in.
* `HDR12/34 MUX`: high dynamic range (pairs of ADCs). `TX0-3 MODE`: ADC power/quality.
* `TX SMIC MUX0-7` (note the off-by-one: `ADC1` here = codec ADC2), `TX DMIC MUX0-7`,
  `TX DEC0-7 MUX/Volume/MODE`, `TX_AIF1/2/3_CAP Mixer DECx`: TX macro routing.
* `DMIC0-8 …`: digital mics, none on this board.

## 5. Jacks and HDMI

`Headphone Jack`, `Mic Jack`, `DP0 Jack`; `ELD` (sink audio capabilities),
`IEC958 Playback Default/Mask`, `Playback Channel Map`.
