#!/bin/sh
# Fetch sound/soc/qcom/sc8280xp.c + the snd-q6apm sources (+ local headers)
# matching the target kernel, then:
#  - sc8280xp: make the headphone (codec-DMA RX) BE format/rate configurable via
#    module parameters rx_be_format / rx_be_rate (defaults = stock S16_LE @ 48 kHz)
#  - q6apm/audioreach: apply the v2 of the upstream series in patches/
# When the sources come from kernel.org (headers packages ship no .c files), the
# Armbian patches for those files (armbian/<major.minor>/) are applied first, so
# the module does not lose the fixes the Armbian kernel carries.
set -e
kver="$1"; ksrc="$2"
here="$(cd "$(dirname "$0")" && pwd)"
files="sc8280xp.c common.h sdw.h qdsp6/q6afe.h qdsp6/q6prm.h qdsp6/q6dsp-common.h qdsp6/q6apm.c qdsp6/audioreach.c qdsp6/topology.c qdsp6/audioreach.h qdsp6/q6apm.h"
mm="$(echo "$kver" | sed -E 's/^([0-9]+\.[0-9]+).*/\1/')"
rm -rf "$here/src"; mkdir -p "$here/src/qdsp6"

# apply_patches DIR: patches are against the kernel tree (a/sound/soc/qcom/...).
# DIR/<major.minor>/<name> replaces DIR/<name> (backports for older kernels).
apply_patches() {
	for p in "$1"/*.patch; do
		[ -e "$p" ] || continue
		[ -e "$1/$mm/$(basename "$p")" ] && p="$1/$mm/$(basename "$p")"
		if patch -d "$here/src" -p4 -s -f -N --dry-run < "$p" >/dev/null 2>&1; then
			patch -d "$here/src" -p4 -s -f -N --no-backup-if-mismatch < "$p"
			echo "qcs6490-hifi: applied $(basename "$p")"
		elif patch -d "$here/src" -p4 -s -f -R --dry-run < "$p" >/dev/null 2>&1; then
			echo "qcs6490-hifi: already in source $(basename "$p")"
		else
			echo "qcs6490-hifi: $(basename "$p") does not apply" >&2
			exit 1
		fi
	done
}

if [ -f "$ksrc/sound/soc/qcom/sc8280xp.c" ]; then
	for f in $files; do cp "$ksrc/sound/soc/qcom/$f" "$here/src/$f"; done
else
	# upstream version: 7.2.9-edge-qcs6490 -> v7.2.9 ; 7.3-rc1-... -> v7.3-rc1
	uver="v$(echo "$kver" | sed -E 's/^([0-9]+\.[0-9]+(\.[0-9]+)?(-rc[0-9]+)?).*/\1/; s/\.0$//')"
	base="https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain/sound/soc/qcom"
	for f in $files; do
		curl -fsSL --retry 3 -o "$here/src/$f" "$base/$f?h=$uver"
	done
	case "$kver" in
	*-qcs6490)
		[ -d "$here/armbian/$mm" ] ||
			echo "qcs6490-hifi: WARNING: no Armbian patches for $mm, check armbian/build patch/kernel/archive/qcs6490-$mm" >&2
		apply_patches "$here/armbian/$mm" ;;
	esac
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

# audioreach/q6apm, v2 of the upstream series:
#  0001 S24_LE is 24 valid bits in a 32-bit container (packed 24/24 crashes the
#       ADSP audio_process on QCS6490)
#  0002 a route mixer write of 0 only drops the sub-graph link this control
#       owns (alsactl restore / UCM HDMI sequence silently cut the other stream)
#  0003 apply stream volume changes to a running graph (AirPlay volume)
apply_patches "$here/patches"

# Kernel headers packages (e.g. Armbian linux-headers-*) ship Module.symvers;
# for a hand-prepared tree, derive module exports from the installed modules.
if [ -s "$ksrc/Module.symvers" ]; then
	: > "$here/installed.symvers"
else
	"$here/gen-symvers.sh" "$kver" "$here/installed.symvers"
fi
