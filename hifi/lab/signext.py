#!/usr/bin/env python3
# signext.py OUT MODE : 8 s 440 Hz at -12 dBFS, S24_LE in 32-bit words.
# MODE "sext": upper byte = sign extension; "zero": upper byte = 0 (24-bit masked)
import math, struct, sys
out, mode = sys.argv[1], sys.argv[2]
r = 48000; a = 0.25 * 8388607
buf = bytearray()
for i in range(r * 8):
    v = int(a * math.sin(2 * math.pi * 440 * i / r))
    if mode == 'zero':
        v &= 0xFFFFFF
        buf += struct.pack('<II', v, v)
    else:
        buf += struct.pack('<ii', v, v)
open(out, 'wb').write(buf)
