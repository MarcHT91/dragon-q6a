# upstream patches

Sent to the ALSA/ASoC maintainers on 2026-11-11 as
"[PATCH 0/2] ASoC: qcom: audioreach: fix S24_LE playback and route mixer disconnects"
(based on `broonie/sound.git` `for-next`; search lore.kernel.org for the subject).

* `0001`: S24_LE uses a 32-bit container (`bits_per_sample = 32`).
  Fixes: 25ab80db6b13 ("ASoC: qdsp6: audioreach: add module configuration command helpers")
* `0002`: a route mixer write of 0 only disconnects the link that control owns.
  Fixes: e4977b91cff8 ("ASoC: qdsp6: audioreach: Simplify handing FE and BE graph connections")

* `0003`: volume control changes are sent to the DSP while the graph runs.
  Sent separately on 2026-11-11 as "[PATCH] ASoC: qcom: audioreach: apply volume changes to a running graph".
  Fixes: 36ad9bf1d93d ("ASoC: qdsp6: audioreach: add topology support")

The DKMS in `../dkms` applies the same changes to the running kernel's sources.
