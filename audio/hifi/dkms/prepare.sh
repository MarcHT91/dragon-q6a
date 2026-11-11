#!/bin/sh
# Fetch sound/soc/qcom/sc8280xp.c (+ local headers) matching the target kernel
# and make the headphone (codec-DMA RX) BE format/rate configurable via module
# parameters rx_be_format / rx_be_rate (defaults = stock S16_LE @ 48 kHz).
set -e
kver="$1"; ksrc="$2"
here="$(cd "$(dirname "$0")" && pwd)"
files="sc8280xp.c common.h sdw.h qdsp6/q6afe.h qdsp6/q6apm.c qdsp6/audioreach.c qdsp6/topology.c qdsp6/audioreach.h qdsp6/q6apm.h"
rm -rf "$here/src"; mkdir -p "$here/src/qdsp6"

if [ -f "$ksrc/sound/soc/qcom/sc8280xp.c" ]; then
	for f in $files; do cp "$ksrc/sound/soc/qcom/$f" "$here/src/$f"; done
else
	# upstream version: 7.2.9-edge-qcs6490 -> v7.2.9 ; 7.3-rc1-... -> v7.3-rc1
	uver="v$(echo "$kver" | sed -E 's/^([0-9]+\.[0-9]+(\.[0-9]+)?(-rc[0-9]+)?).*/\1/; s/\.0$//')"
	base="https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain/sound/soc/qcom"
	for f in $files; do
		curl -fsSL --retry 3 -o "$here/src/$f" "$base/$f?h=$uver"
	done
fi

c="$here/src/sc8280xp.c"
if ! grep -q qcs6490-hifi "$c"; then
old='	rate->min = rate->max = 48000;
	snd_mask_set_format(fmt, SNDRV_PCM_FORMAT_S16_LE);'
grep -qzF "$old" "$c" || { echo "qcs6490-hifi: fixup pattern not found in sc8280xp.c" >&2; exit 1; }
perl -0pi -e '
s{\trate->min = rate->max = 48000;\n\tsnd_mask_set_format\(fmt, SNDRV_PCM_FORMAT_S16_LE\);\n}{\t/* qcs6490-hifi: configurable codec DMA RX (headphones) BE format/rate */
\tif (sc8280xp_is_codec_dma_rx(cpu_dai->id)) {
\t\tif (rx_be_rate > 0)
\t\t\trate->min = rate->max = rx_be_rate;
\t\tswitch (rx_be_format) {
\t\tcase 0:
\t\t\tbreak;
\t\tcase 24:
\t\t\tsnd_mask_none(fmt);
\t\t\tsnd_mask_set_format(fmt, SNDRV_PCM_FORMAT_S24_LE);
\t\t\tbreak;
\t\tcase 32:
\t\t\tsnd_mask_none(fmt);
\t\t\tsnd_mask_set_format(fmt, SNDRV_PCM_FORMAT_S32_LE);
\t\t\tbreak;
\t\tdefault:
\t\t\tsnd_mask_none(fmt);
\t\t\tsnd_mask_set_format(fmt, SNDRV_PCM_FORMAT_S16_LE);
\t\t\tbreak;
\t\t}
\t\tdev_dbg(rtd->dev, "qcs6490-hifi: %s BE rate %u-%u fmt mask 0x%x\\n",
\t\t\t cpu_dai->name, rate->min, rate->max, fmt->bits[0]);
\t} else {
\t\trate->min = rate->max = 48000;
\t\tsnd_mask_set_format(fmt, SNDRV_PCM_FORMAT_S16_LE);
\t}
};
s{(static int sc8280xp_be_hw_params_fixup\()}{static int rx_be_format = 16;
module_param(rx_be_format, int, 0644);
MODULE_PARM_DESC(rx_be_format, "qcs6490-hifi: headphone BE format: 16 (default), 24, 32, 0 = follow FE");
static int rx_be_rate = 48000;
module_param(rx_be_rate, int, 0644);
MODULE_PARM_DESC(rx_be_rate, "qcs6490-hifi: headphone BE rate (default 48000), 0 = follow FE");

static bool sc8280xp_is_codec_dma_rx(int id)
{
\tswitch (id) {
\tcase RX_CODEC_DMA_RX_0:
\tcase RX_CODEC_DMA_RX_1:
\tcase RX_CODEC_DMA_RX_2:
\tcase RX_CODEC_DMA_RX_3:
\tcase RX_CODEC_DMA_RX_4:
\tcase RX_CODEC_DMA_RX_5:
\tcase RX_CODEC_DMA_RX_6:
\tcase RX_CODEC_DMA_RX_7:
\t\treturn true;
\tdefault:
\t\treturn false;
\t}
}

$1};
' "$c"
grep -q 'sc8280xp_is_codec_dma_rx(cpu_dai->id)' "$c"
fi

# audioreach: S24_LE is 24 valid bits in a 32-bit container. Upstream sends
# bits_per_sample = 24 (packed 3-byte samples), which crashes the ADSP
# audio_process on QCS6490. Use the container size for bits_per_sample.
a="$here/src/qdsp6/audioreach.c"
if ! grep -q qcs6490-hifi "$a"; then
n=$(grep -cE '^\s*(cfg|media_cfg)->bits_per_sample = mcfg->bit_width;' "$a")
[ "$n" = 2 ] || { echo "qcs6490-hifi: audioreach bits_per_sample pattern found $n times (want 2)" >&2; exit 1; }
perl -pi -e 's{^(\s*)(cfg|media_cfg)->bits_per_sample = mcfg->bit_width;}{$1/* qcs6490-hifi: container size, S24_LE = 24 in 32 */\n$1$2->bits_per_sample = mcfg->bit_width == 24 ? 32 : mcfg->bit_width;}' "$a"
fi

# topology: a route mixer write of 0 must only drop the sub-graph link if that
# link belongs to this control. Upstream clears the device graph's link for any
# 0 write, so restoring "DP<-MM1 off" after "DP<-MM2 on" (alsactl order) or the
# UCM HDMI sequence ("RX<-MM2 0") silently disconnects the other stream.
t="$here/src/qdsp6/topology.c"
if ! grep -q qcs6490-hifi "$t"; then
grep -qzF '	} else {
		audioreach_connect_sub_graphs(data, scontrol, dapm_scontrol, false);' "$t" ||
	{ echo "qcs6490-hifi: topology put_audio_mixer pattern not found" >&2; exit 1; }
perl -0pi -e 's#\t\} else \{\n(\t\taudioreach_connect_sub_graphs\(data, scontrol, dapm_scontrol, false\);)#\t} else if (audioreach_is_vmixer_connected(data, scontrol, dapm_scontrol)) {\n\t\t/* qcs6490-hifi: only drop the link if it belongs to this control */\n$1#' "$t"
grep -q 'else if (audioreach_is_vmixer_connected(data, scontrol, dapm_scontrol))' "$t"
fi

# topology: the stream volume control only stored the new gain; the DSP got it
# from the widget's POST_PMU event, i.e. only when a stream starts. Changes
# while playing (e.g. the AirPlay volume) were ignored. Send it right away when
# the volume widget is powered.
if ! grep -q 'qcs6490-hifi: apply volume' "$t"; then
grep -qzF '	mod->gain = ucontrol->value.integer.value[0];

	return 1;
}' "$t" || { echo "qcs6490-hifi: put_vol_ctrl pattern not found" >&2; exit 1; }
perl -0pi -e 's#\tmod->gain = ucontrol->value.integer.value\[0\];\n\n\treturn 1;\n\}#\tlong gain = ucontrol->value.integer.value[0];\n\n\tif (mod->gain == gain)\n\t\treturn 0;\n\n\tmod->gain = gain;\n\n\t/* qcs6490-hifi: apply volume now if the graph is running */\n\tif (dw->power) {\n\t\tstruct snd_soc_component *c = snd_soc_dapm_to_component(dw->dapm);\n\n\t\taudioreach_gain_set_vol_ctrl(dev_get_drvdata(c->dev), mod, gain);\n\t}\n\n\treturn 1;\n}#' "$t"
grep -q 'qcs6490-hifi: apply volume' "$t"
fi

# Kernel headers packages (e.g. Armbian linux-headers-*) ship Module.symvers;
# for a hand-prepared tree, derive module exports from the installed modules.
if [ -s "$ksrc/Module.symvers" ]; then
	: > "$here/installed.symvers"
else
	"$here/gen-symvers.sh" "$kver" "$here/installed.symvers"
fi
