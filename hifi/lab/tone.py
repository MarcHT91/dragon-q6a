#!/usr/bin/env python3
# tone.py RATE FMT(S16_LE|S24_LE|S32_LE) SECONDS OUT  -> raw stereo 1 kHz sine at -12 dBFS
import math, struct, sys
rate, fmt, secs, out = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
full = {'S16_LE': 32767, 'S24_LE': 8388607, 'S32_LE': 2147483647}[fmt]
pk = '<hh' if fmt == 'S16_LE' else '<ii'
amp = 0.25 * full
with open(out, 'wb') as f:
    buf = bytearray()
    for i in range(int(rate * secs)):
        v = int(amp * math.sin(2 * math.pi * 1000 * i / rate))
        buf += struct.pack(pk, v, v)
    f.write(buf)
