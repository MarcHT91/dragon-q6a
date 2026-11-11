# route: DSP output routes at boot

The Dragon Q6A plays through AudioReach "route" mixers:

| Control | Meaning |
|---|---|
| `RX_CODEC_DMA_RX_0 Audio Mixer MultiMedia1` | `hw:0,0` → headphone jack |
| `DISPLAY_PORT_RX_0 Audio Mixer MultiMedia2` | `hw:0,1` → HDMI/DP |

If a PCM is started while its route is off, it never runs: `state: RUNNING`,
`hw_ptr: 0`, buffer full, and the player stays stuck until it reopens the device.

## Why the routes were lost

1. **Kernel** (`sound/soc/qcom/qdsp6/topology.c`): each back end stores one
   connected source. A write of 0 to *any* route control of that back end
   clears it, even if another stream is connected.
   * `alsactl restore` writes `DP<-MM2 = 1` and then `DP<-MM1 = 0`, so HDMI is lost on every boot.
   * The UCM `HDMI` enable sequence writes `RX<-MM2 = 0`, which disconnects the headphones.
2. **UCM** (`ucm2/Qualcomm/qcs6490/QCS6490-Radxa-Dragon-Q6A/HiFi.conf`): the
   `HiFi` verb connects `RX<-MM2` (taking the headphones away from MM1) and
   the `HDMI` device then clears it. Even with the kernel fixed, `alsaucm set
   _verb HiFi set _enadev HDMI` leaves the headphones disconnected.
3. A Kodi launcher ran exactly that `alsaucm` line at every start (fixed in
   [kodi-iris](../../kodi-iris)).

Found with a boot trace (`diag/`) that logs the route state every 0.5 s and
which processes hold `/dev/snd/*`.

## Fix

* Kernel: `../hifi` DKMS includes the topology fix (`hifi/upstream/0002-*.patch`).
* `q6a-audio-route` + `q6a-audio-route.service`: switch both routes on after
  `alsa-restore` and before `shairport-sync`/`kodi`. Writes of 1 never
  disconnect anything, so this is safe on an unpatched kernel too.
* Do not use `alsaucm ... _enadev HDMI` on this board.

Install: `sudo ./install.sh`.
