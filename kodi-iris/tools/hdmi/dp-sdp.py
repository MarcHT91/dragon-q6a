#!/usr/bin/env python3
# dp-sdp.py: DIAGNOSTIC tool. Programs secondary-data packets (SDP) in the msm DP controller of the
# QCS6490 by writing its registers through /dev/mem, to test what the RA620 DP->HDMI converter
# does with them. This is how we found that the RA620 ignores range information from the source.
#
# Register layout and sequence mirror drivers/gpu/drm/msm/dp/dp_panel.c (msm_dp_panel_enable_vsc_sdp)
# and the DPU peripheral flush in drivers/gpu/drm/msm/disp/dpu1/dpu_hw_ctl.c (kernel 7.2).
#
#   show            print DP MISC/SDP registers
#   go-full         VSC SDP: RGB, 8 bpc, VESA (full) range, then set MISC1 "use VSC"
#   go-limited      same with CTA (limited) range
#   go-ycc          VSC SDP that falsely claims YCbCr 4:4:4 (checks whether the sink reads DB16)
#   go-avi-full     CTA-861 AVI InfoFrame SDP (type 0x82) with Q = full range (env VIC=, HB2=)
#   go-avi-limited  same with Q = limited
#   off             disable the generic SDP and the MISC1 VSC bit
#
# WARNING: needs root and a kernel without CONFIG_STRICT_DEVMEM. Run it only while the display is
# on (the registers are clock-gated otherwise). The driver rewrites these registers on every mode
# change or audio start, so the effect is temporary. Setting the MISC1 VSC bit without a latched
# VSC packet makes the RA620 drop the picture; "off" restores it.
import mmap, os, struct, sys

DP_BASE = 0x0AE90000            # displayport-controller@ae90000 (kodiak.dtsi)
AHB, LINK = 0x000, 0x400        # reg regions: ahb @+0x000, aux @+0x200, link @+0x400
REG_HW_VERSION = AHB + 0x000
MAINLINK_CTRL = LINK + 0x000
MISC1_MISC0 = LINK + 0x02C
SDP_CFG = LINK + 0x228
SDP_CFG2 = LINK + 0x22C
SDP_CFG3 = LINK + 0x24C
GENERIC0 = LINK + 0x300         # GENERIC0_0..9 (2 header words + 8 data words = 32 bytes)
GEN0_SDP_EN = 0x00020000
GENERIC0_SDPSIZE_VALID = 0x00010000
DP_MISC1_VSC_SDP = 0x00004000
UPDATE_SDP = 0x1


def g0(d):
    c = [(d >> i) & 1 for i in range(4)]
    g = [c[3], c[0] ^ c[3], c[1], c[2]]
    return sum(b << i for i, b in enumerate(g))


def g1(d):
    c = [(d >> i) & 1 for i in range(4)]
    g = [c[0] ^ c[3], c[0] ^ c[1] ^ c[3], c[1] ^ c[2], c[2] ^ c[3]]
    return sum(b << i for i, b in enumerate(g))


def parity(data):
    x0 = x1 = 0
    nb = 8 if data & 0xFF00 else 2
    for i in range(nb):
        ci = ((data >> (i * 4)) & 0xF) ^ x1
        x1 = x0 ^ g1(ci)
        x0 = g0(ci)
    return x1 | (x0 << 4)


def header_words(hb):
    w0 = hb[0] | parity(hb[0]) << 8 | hb[1] << 16 | parity(hb[1]) << 24
    w1 = hb[2] | parity(hb[2]) << 8 | hb[3] << 16 | parity(hb[3]) << 24
    return w0, w1


fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
m = mmap.mmap(fd, 0x2000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=DP_BASE)
rd = lambda o: struct.unpack_from('<I', m, o)[0]
def wr(o, v): struct.pack_into('<I', m, o, v & 0xFFFFFFFF)


def show(tag=''):
    misc = rd(MISC1_MISC0)
    print(f'{tag}HW_VERSION=0x{rd(REG_HW_VERSION):08x} MAINLINK_CTRL=0x{rd(MAINLINK_CTRL):08x}')
    print(f'{tag}MISC1_MISC0=0x{misc:08x} (MISC0 0x{misc & 0xff:02x}: colorimetry {(misc >> 1) & 0xf:#x}, '
          f'bpc code {(misc >> 5) & 7}; MISC1 VSC bit {bool(misc & DP_MISC1_VSC_SDP)})')
    print(f'{tag}SDP_CFG=0x{rd(SDP_CFG):08x} (GEN0 en {bool(rd(SDP_CFG) & GEN0_SDP_EN)}) '
          f'SDP_CFG2=0x{rd(SDP_CFG2):08x} SDP_CFG3=0x{rd(SDP_CFG3):08x}')
    print(tag + 'GENERIC0: ' + ' '.join(f'{rd(GENERIC0 + 4 * i):08x}' for i in range(10)))


CTL0_BASE = 0x0AE01000 + 0x15000   # DPU mdp @ae01000, ctl_0 @+0x15000 (dpu_7_2_sc7280.h); DP uses ctl 0 / intf 0
CTL_FLUSH, CTL_PERIPH_FLUSH, PERIPH_IDX = 0x018, 0x128, 30


def periph_flush():
    # what dpu_hw_ctl does for a 420 DP mode: latch DP SDP/MSA updates on the next vsync
    import time
    cfd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    c = mmap.mmap(cfd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=CTL0_BASE)
    struct.pack_into('<I', c, CTL_PERIPH_FLUSH, 1 << 0)   # INTF_0
    struct.pack_into('<I', c, CTL_FLUSH, 1 << PERIPH_IDX)
    time.sleep(0.1)
    print('periph flush done, CTL_FLUSH now 0x%08x' % struct.unpack_from('<I', c, CTL_FLUSH)[0])


cmd = sys.argv[1] if len(sys.argv) > 1 else 'show'
if cmd in ('go-full', 'go-limited', 'go-ycc'):
    import subprocess
    mode = cmd[3:]
    subprocess.run([sys.executable, sys.argv[0], 'stage-' + mode], check=True)
    periph_flush()
    subprocess.run([sys.executable, sys.argv[0], 'arm'], check=True)
    periph_flush()
    sys.exit(0)
if cmd in ('go-avi-full', 'go-avi-limited'):
    # CTA-861 AVI InfoFrame carried as a DP InfoFrame SDP (packing as i915 intel_dp.c does for type 0x87)
    q = 2 if cmd.endswith('full') else 1             # AVI Q1:Q0 = 10 full, 01 limited
    vic = int(os.environ.get('VIC', '95'))           # 95 = 3840x2160@30, 94 = @25, 93 = @24, 16 = 1080p60
    pb = bytearray(13)
    pb[0] = 0x00                                     # PB1: Y=RGB, no AFD/bar/scan info
    pb[1] = (2 << 4) | 8                             # PB2: 16:9 picture, AFD same as picture
    pb[2] = q << 2                                   # PB3: quantization range
    pb[3] = vic                                      # PB4: VIC
    db = bytearray(32)
    db[0], db[1] = 2, 13                             # CTA header: version 2, length 13
    db[2:15] = pb
    hb2 = int(os.environ.get('HB2', str(4 + 13 - 1)))  # i915 style: infoframe size (hdr+payload) - 1
    w0, w1 = header_words([0x00, 0x82, hb2, 0x13 << 2])
    show('before: ')
    wr(SDP_CFG, rd(SDP_CFG) | GEN0_SDP_EN)
    wr(SDP_CFG2, rd(SDP_CFG2) | GENERIC0_SDPSIZE_VALID)
    wr(GENERIC0, w0)
    wr(GENERIC0 + 4, w1)
    for i in range(0, 32, 4):
        wr(GENERIC0 + 8 + i, struct.unpack_from('<I', db, i)[0])
    periph_flush()
    show('after:  ')
    sys.exit(0)
if cmd == 'flush':
    periph_flush()
    sys.exit(0)
if cmd == 'show':
    show()
    sys.exit(0)

show('before: ')
stage = cmd.startswith('stage-')
if stage:
    cmd = cmd[6:]
if cmd == 'arm':
    wr(MISC1_MISC0, rd(MISC1_MISC0) | DP_MISC1_VSC_SDP)
elif cmd in ('full', 'limited', 'ycc'):
    db = bytearray(32)
    db[16] = (0x0 << 4) | 0x0                       # pixel format RGB, colorimetry default (sRGB)
    if cmd == 'ycc':                                 # deliberate lie: claim YCbCr 4:4:4 BT.709 (diagnostic only)
        db[16] = (0x1 << 4) | 0x1
    db[17] = 0x01 | (0x80 if cmd == 'limited' else 0)  # 8 bpc; bit7: 0 = VESA (full), 1 = CTA (limited)
    db[18] = 0x1                                     # content type: graphics (same as the 420 path)
    w0, w1 = header_words([0x00, 0x07, 0x05, 0x13])  # VSC SDP, rev 5, 0x13 bytes
    wr(SDP_CFG, rd(SDP_CFG) | GEN0_SDP_EN)
    wr(SDP_CFG2, rd(SDP_CFG2) | GENERIC0_SDPSIZE_VALID)
    wr(GENERIC0, w0)
    wr(GENERIC0 + 4, w1)
    for i in range(0, 32, 4):
        wr(GENERIC0 + 8 + i, struct.unpack_from('<I', db, i)[0])
    if not stage:                                    # stage-*: send the packet but keep MSA in charge
        wr(MISC1_MISC0, rd(MISC1_MISC0) | DP_MISC1_VSC_SDP)
elif cmd == 'off':
    wr(SDP_CFG, rd(SDP_CFG) & ~GEN0_SDP_EN)
    wr(SDP_CFG2, rd(SDP_CFG2) & ~GENERIC0_SDPSIZE_VALID)
    wr(MISC1_MISC0, rd(MISC1_MISC0) & ~DP_MISC1_VSC_SDP)
else:
    sys.exit('usage: dpvsc.py show|go-full|go-limited|go-ycc|go-avi-full|go-avi-limited|flush|off  (env VIC=, HB2=)')
# latch: the <1.2 method (SDP_CFG3 pulse); >=1.2 latches on the DPU peripheral flush (next commit)
wr(SDP_CFG3, UPDATE_SDP)
wr(SDP_CFG3, 0)
show('after:  ')
