#!/usr/bin/env python3
"""Widen the stream caps of PCMs in an ASoC topology binary.

usage: patch-tplg.py IN OUT [--formats S16_LE,S24_LE] [--rates 8000-192000] PCM_NAME...
Only struct snd_soc_tplg_stream_caps (size 104) entries whose name matches are
touched; anything unexpected aborts without writing."""
import argparse, struct, sys

FMT = {'S16_LE': 2, 'S24_LE': 6, 'S32_LE': 10}
RATES = [5512, 8000, 11025, 16000, 22050, 32000, 44100, 48000, 64000,
         88200, 96000, 176400, 192000]          # SNDRV_PCM_RATE_* bits 0..12

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--formats', default='S16_LE,S24_LE')
ap.add_argument('--rates', default=None, help='MIN-MAX, e.g. 8000-192000')
ap.add_argument('names', nargs='+')
a = ap.parse_args()

fmask = 0
for f in a.formats.split(','):
    fmask |= 1 << FMT[f]
d = bytearray(open(a.src, 'rb').read())
for nm in a.names:
    hits, off = [], 0
    while (i := d.find(nm.encode() + b'\0', off)) >= 0:
        off = i + 1
        if struct.unpack_from('<I', d, i - 4)[0] == 104:
            hits.append(i)
    if len(hits) != 1:
        sys.exit(f'{nm}: expected 1 stream_caps, found {len(hits)}')
    fo = hits[0] + 44                      # formats u64, rates u32, rate_min, rate_max
    old = struct.unpack_from('<QIII', d, fo)
    if not old[0] & (1 << FMT['S16_LE']):
        sys.exit(f'{nm}: unexpected formats {old[0]:#x}')
    struct.pack_into('<Q', d, fo, fmask)
    if a.rates:
        lo, hi = map(int, a.rates.split('-'))
        rmask = sum(1 << b for b, r in enumerate(RATES) if lo <= r <= hi)
        struct.pack_into('<III', d, fo + 8, rmask, lo, hi)
    new = struct.unpack_from('<QIII', d, fo)
    print(f'{nm}: formats {old[0]:#x}->{new[0]:#x} rates {old[1]:#x} {old[2]}-{old[3]} -> {new[1]:#x} {new[2]}-{new[3]}')
open(a.dst, 'wb').write(d)
