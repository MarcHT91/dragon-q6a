# HDMI colour range on the Dragon Q6A (RA620)

*[Català](hdmi-colour-range.ca.md)*

## Symptom

On a TV with HDMI black level on **Auto** (tested: Samsung UE BU8500), the picture is too dark:
shadows are crushed and colours look heavy. Setting the TV's black level to **Normal**/**High**
by hand fixes it. A PC monitor usually looks fine.

## Cause

The Dragon Q6A has no native HDMI. The QCS6490 drives DisplayPort, and an onboard **Radxa
RA620** converts it to HDMI:

```
msm_dp (DP, 2 lanes @ 5.4 Gbps) -> QMP PHY -> RA620 (simple-bridge) -> HDMI connector
```

- The board sends **RGB 0–255** (full range). The DP driver already marks it as full range
  (`MISC0` = "RGB VESA").
- The RA620 builds the HDMI AVI InfoFrame itself and does not mark the range. For a TV
  (CTA) mode such as 2160p30 or 1080p60, a TV on Auto then assumes **limited range (16–235)**:
  everything at or below 16 shows as black and everything at or above 235 as white.
- Linux offers no way to change this on this board. The connector has no `Broadcast RGB`
  property, and the RA620 has no driver: upstream declares it as a passive
  `simple-bridge` ("can be used without involving software").

Same chip on the ROCK 5 ITX and ROCK 5A, so those boards very likely behave the same.

### How we checked it

With `tools/hdmi/testcard` and the TV on Auto, the 5 leftmost near-black steps (0, 4, 8, 12,
16) merge into one black block. That is exactly the limited-range reading. With the TV on
Normal, every step is visible.

Things that **do not** work, tested on kernel 7.2.9:

| Attempt | Result |
|---|---|
| CRTC `GAMMA_LUT` compressing to 16–235 | Ignored by the hardware on this output (an inverting LUT changed nothing), even with a `CTM` set. |
| DP `MISC0` range bit | The driver already sends "VESA" (full); the RA620 ignores it. |
| DP VSC SDP with RGB + VESA range (`tools/hdmi/dp-sdp.py go-full`) | The RA620 needs the packet to be present (setting the MISC1 "use VSC" bit without a latched packet makes it drop the picture), but ignores its content. Claiming YCbCr in it changes nothing. |
| CTA-861 AVI InfoFrame SDP with Q = full (`go-avi-full`) | Not forwarded to the TV. |

The VSC SDP only reaches the wire after a DPU *peripheral flush*, which the driver only does
for YUV 4:2:0 modes; `dp-sdp.py` does it by hand.

### HDR

The TV's EDID announces HDR10 (PQ) and HLG, but the board cannot send HDR: the DP driver has
no `HDR_OUTPUT_METADATA` support, and HDR metadata travels in the same kind of InfoFrame SDP
that the RA620 does not forward. HDR10 video in Kodi is tone-mapped to SDR.

## Fix used here: output limited range from Kodi

The board cannot tell the TV it sends full range, so it sends what the TV expects: **16–235**.

1. In Kodi, *Settings → System → Display → Use limited colour range (16-235)*
   (`videoscreen.limitedrange` = `true`). This covers the GUI and video that Kodi renders
   with its own shaders (software decoding, e.g. 10-bit HEVC).
2. Kodi's **DRM PRIME EGL** renderer (the one used for iris hardware decoding) ignores that
   setting, so hardware-decoded video stays 0–255. The shim fixes this: with
   `EGL_SHIM_LIMITED_OUT=1` it tells Mesa that the (limited-range) YUV is full range. Mesa
   then passes luma through untouched, so video black and white land on 16 and 235, the same as
   the GUI. Chroma comes out about 2 % more saturated than an exact conversion (224/255
   instead of 219/255), which is not visible in practice.
3. The `kodi-iris` launcher sets `EGL_SHIM_LIMITED_OUT=1` automatically when
   `videoscreen.limitedrange` is `true` in `guisettings.xml`.
4. TV: HDMI black level on **Auto**. On a Samsung, do not set the input icon to **PC**: in PC
   mode only *Standard* and *Entertainment* picture modes are offered, and Entertainment
   crushes shadows further.

Measured on a Kodi screenshot of a paused 720p H.264 frame against the same frame decoded in
software: video = `0.856 × ref + 18` (ideal limited: `0.859 × ref + 16`); GUI = exactly
16–235. `tests/yuvhint` checks the shim's relabel without Kodi.

### Why limited range is not a loss here

Almost all video is already encoded in limited range. With the shim, its luma values reach the
TV unchanged. Full range would expand them on the GPU, then the TV maps them back. The GUI
loses some levels (220 instead of 256 per channel); enabling Kodi's dithering
(`videoscreen.dither`) hides any banding.

### Combinations

| Board | TV black level | Result |
|---|---|---|
| limited (this fix) | Auto / Low | correct |
| full (default) | Normal / High | correct, but only on TVs set by hand |
| full (default) | Auto | too dark, crushed shadows (the original problem) |
| limited | Normal | washed out, grey blacks |

The fix covers Kodi only. The console and any other program still send 0–255.

## Tools

| Tool | What it does |
|---|---|
| `tools/hdmi/testcard.c` | Range test card on the active CRTC (stop Kodi first; it needs DRM master). |
| `tools/hdmi/dpcd-dump.py` | Reads the RA620's DPCD (needs `sudo chmod o+r /dev/drm_dp_aux0`). |
| `tools/hdmi/dp-sdp.py` | Diagnostic: programs VSC / AVI InfoFrame SDPs in the DP controller through `/dev/mem` (root). |
| `tests/yuvhint.c` | Checks how Mesa converts NV12 under each EGL colour-space/range hint, with or without the shim. |

Do not start Kodi while `testcard` holds the display: Kodi then gets no DRM master and leaks
EGL fence file descriptors until it is restarted.

## RA620 details (as read from DPCD)

```
DPCD rev 1.4, max link 5.4 Gbps, 4 lanes; DOWNSTREAMPORT_PRESENT 0x10
Branch OUI 00-60-AD, id "cHDMIb", hw 01, fw 20.146
DFP caps byte 1 = 0x88 (340 MHz max TMDS), PROTOCOL_CONVERTER_CONTROL 3050-3052 = 00 00 00
DPRX_FEATURE_ENUMERATION (0x2210) = 0x08: VSC SDP extended colorimetry supported
```

The range could only be fixed in the RA620's firmware. That is a question for Radxa.
