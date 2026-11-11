#!/usr/bin/env python3
# listen.py N RATE FMT OUT : N short 2 kHz blips, pause, 3 s 1 kHz at -20 dBFS
import math, struct, sys
n, rate, fmt, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
full = {'S16_LE': 32767, 'S24_LE': 8388607, 'S32_LE': 2147483647}[fmt]
pk = '<hh' if fmt == 'S16_LE' else '<ii'
a = 0.1 * full
buf = bytearray()
def tone(f, secs):
    global buf
    N = int(rate * secs); ramp = int(rate * 0.005)
    for i in range(N):
        g = min(1, i / ramp, (N - i) / ramp)
        v = int(a * g * math.sin(2 * math.pi * f * i / rate))
        buf += struct.pack(pk, v, v)
def silence(secs):
    global buf
    buf += bytes(int(rate * secs) * struct.calcsize(pk))
silence(0.3)
for _ in range(n):
    tone(2000, 0.08); silence(0.15)
silence(0.4); tone(1000, 3); silence(0.3)
open(out, 'wb').write(buf)
