#!/usr/bin/env python3
"""dpcd-dump.py: read the DPCD of the DP sink behind the msm DP controller (on the Dragon Q6A,
the onboard RA620 DP->HDMI converter) through the kernel's DP AUX character device.

Needs read access to /dev/drm_dp_aux0 (root only by default):
    sudo chmod o+r /dev/drm_dp_aux0      # until the next reboot
Usage: dpcd-dump.py [/dev/drm_dp_aux0]
"""
import os
import sys

dev = sys.argv[1] if len(sys.argv) > 1 else '/dev/drm_dp_aux0'
fd = os.open(dev, os.O_RDONLY)


def rd(addr, n):
    os.lseek(fd, addr, os.SEEK_SET)
    try:
        return os.read(fd, n)
    except OSError:
        return None


def hx(b):
    return ' '.join('%02x' % x for x in b)


r = rd(0x000, 16)
print('DPCD 000-00F:', hx(r))
print('  DPCD rev %x.%x, max link rate 0x%02x, max lanes %d' % (r[0] >> 4, r[0] & 15, r[1], r[2] & 31))
print('  DOWNSTREAMPORT_PRESENT 0x%02x (port present %d, detailed caps %d)' % (r[5], r[5] & 1, (r[5] >> 4) & 1))

d = rd(0x080, 4)
print('DFP caps 080-083:', hx(d), '(byte1 0x%02x = %g MHz if read as max TMDS)' % (d[1], d[1] * 2.5))

b = rd(0x500, 12)
print('Branch OUI %s, id %r, hw rev %02x, fw %d.%d' % (hx(b[0:3]), bytes(b[3:9]), b[9], b[10], b[11]))

v = rd(0x2210, 1)
print('DPRX_FEATURE_ENUMERATION 2210: 0x%02x (VSC SDP extended colorimetry supported: %d)' % (v[0], (v[0] >> 3) & 1))

pc = rd(0x3050, 3)
if pc is not None:
    print('PROTOCOL_CONVERTER_CONTROL 3050-3052:', hx(pc))

print('Non-zero bytes in the branch vendor area 0x500-0x5FF:')
for off in range(0x500, 0x600, 16):
    row = rd(off, 16)
    if row and any(row):
        print('  %04x: %s' % (off, hx(row)))
