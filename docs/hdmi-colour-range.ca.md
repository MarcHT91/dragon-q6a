# Rang de color HDMI al Dragon Q6A (RA620)

*[English](hdmi-colour-range.md)*

## Símptoma

En un televisor amb el nivell de negre HDMI en **Automàtic** (provat amb un Samsung UE BU8500),
la imatge es veu massa fosca: les ombres queden aixafades i els colors, carregats. Posar el
nivell de negre a **Normal**/**Alt** a mà ho arregla. Un monitor de PC normalment es veu bé.

## Causa

El Dragon Q6A no té HDMI natiu. La QCS6490 treu DisplayPort, i un **Radxa RA620** de la placa
el converteix a HDMI:

```
msm_dp (DP, 2 carrils a 5,4 Gbps) -> PHY QMP -> RA620 (simple-bridge) -> connector HDMI
```

- La placa envia **RGB 0–255** (rang complet). El driver DP ja l'indica com a rang complet
  (`MISC0` = "RGB VESA").
- L'RA620 genera ell mateix l'infoframe AVI de l'HDMI i no hi indica el rang. Per a un mode
  de televisió (CTA), com 2160p30 o 1080p60, un televisor en Automàtic suposa **rang limitat
  (16–235)**: tot el que està a 16 o per sota es veu negre, i tot el que està a 235 o per
  sobre, blanc.
- Linux no ofereix cap manera de canviar-ho en aquesta placa. El connector no té la propietat
  `Broadcast RGB`, i l'RA620 no té driver: el kernel el declara com un `simple-bridge` passiu
  ("can be used without involving software").

És el mateix xip del ROCK 5 ITX i del ROCK 5A, així que aquestes plaques molt probablement es
comporten igual.

### Com ho hem comprovat

Amb `tools/hdmi/testcard` i el televisor en Automàtic, els 5 graons gairebé negres de
l'esquerra (0, 4, 8, 12 i 16) es veuen com un sol bloc negre. És exactament la lectura en rang
limitat. Amb el televisor a Normal, es veuen tots els graons.

El que **no** funciona (provat amb el kernel 7.2.9):

| Prova | Resultat |
|---|---|
| `GAMMA_LUT` del CRTC comprimint a 16–235 | El maquinari l'ignora en aquesta sortida (una taula que inverteix la imatge no canvia res), fins i tot amb una `CTM`. |
| Bit de rang del `MISC0` del DP | El driver ja envia "VESA" (complet); l'RA620 l'ignora. |
| VSC SDP amb RGB i rang VESA (`tools/hdmi/dp-sdp.py go-full`) | L'RA620 necessita que el paquet hi sigui (si s'activa el bit MISC1 "fes servir VSC" sense paquet, talla la imatge), però n'ignora el contingut. Fins i tot si s'hi diu que és YCbCr, no canvia res. |
| Infoframe AVI CTA-861 per SDP amb Q = complet (`go-avi-full`) | No arriba al televisor. |

El VSC SDP només surt pel cable després d'un *peripheral flush* de la DPU, que el driver
només fa en els modes YUV 4:2:0. `dp-sdp.py` el fa a mà.

### HDR

L'EDID del televisor declara HDR10 (PQ) i HLG, però la placa no pot enviar HDR. El driver DP
no implementa `HDR_OUTPUT_METADATA`, i les metadades HDR viatgen en el mateix tipus de paquet
SDP d'infoframe que l'RA620 no reenvia. El Kodi converteix el vídeo HDR10 a SDR (*tone
mapping*).

## Solució que fem servir: rang limitat des del Kodi

Com que la placa no pot dir al televisor que envia rang complet, li envia el que el televisor
espera: **16–235**.

1. Al Kodi, *Configuració → Sistema → Pantalla → Fer servir rang de color limitat (16-235)*
   (`videoscreen.limitedrange` = `true`). Això cobreix la interfície i el vídeo que el Kodi
   dibuixa amb els seus shaders (descodificació per software, per exemple HEVC de 10 bits).
2. El renderitzador **DRM PRIME EGL** del Kodi (el que es fa servir amb la descodificació per
   maquinari d'iris) ignora aquesta opció, i el vídeo descodificat per maquinari continua a
   0–255. El shim ho corregeix: amb `EGL_SHIM_LIMITED_OUT=1`, diu a Mesa que el YUV (de rang
   limitat) és de rang complet. Així Mesa deixa passar la lluminositat sense tocar-la, i el
   negre i el blanc del vídeo queden a 16 i 235, igual que la interfície. El color surt un 2 %
   més saturat que amb una conversió exacta (224/255 en lloc de 219/255), cosa que a la
   pràctica no es nota.
3. El llançador `kodi-iris` activa `EGL_SHIM_LIMITED_OUT=1` automàticament quan
   `videoscreen.limitedrange` és `true` a `guisettings.xml`.
4. Al televisor, nivell de negre HDMI en **Automàtic**. En un Samsung, no marquis l'entrada
   com a **PC**: en mode PC només deixa triar els modes d'imatge *Estàndard* i *Oci*, i el
   mode Oci aixafa encara més les ombres.

Mesurat amb una captura del Kodi d'un fotograma en pausa (H.264 720p), comparada amb el mateix
fotograma descodificat per software: vídeo = `0,856 × ref + 18` (l'ideal en rang limitat és
`0,859 × ref + 16`); interfície = exactament 16–235. `tests/yuvhint` comprova la conversió del
shim sense el Kodi.

### Per què el rang limitat no fa perdre res aquí

Gairebé tot el vídeo ja ve codificat en rang limitat. Amb el shim, els valors de lluminositat
arriben al televisor tal com són. En rang complet, la GPU primer els expandiria i després el
televisor els tornaria a convertir. La interfície perd alguns nivells (220 per canal en lloc
de 256); si s'activa el *dithering* del Kodi (`videoscreen.dither`) no es veuen bandes.

### Combinacions

| Placa | Nivell de negre del TV | Resultat |
|---|---|---|
| limitat (aquesta solució) | Automàtic / Baix | correcte |
| complet (per defecte) | Normal / Alt | correcte, però cal ajustar cada televisor a mà |
| complet (per defecte) | Automàtic | massa fosc, ombres aixafades (el problema original) |
| limitat | Normal | esvaït, negres grisos |

La solució només cobreix el Kodi. La consola i qualsevol altre programa continuen enviant 0–255.

## Eines

| Eina | Què fa |
|---|---|
| `tools/hdmi/testcard.c` | Carta d'ajust de rang al CRTC actiu (cal aturar el Kodi abans, perquè necessita ser DRM master). |
| `tools/hdmi/dpcd-dump.py` | Llegeix el DPCD de l'RA620 (cal `sudo chmod o+r /dev/drm_dp_aux0`). |
| `tools/hdmi/dp-sdp.py` | Diagnòstic: programa paquets SDP de VSC i d'infoframe AVI al controlador DP a través de `/dev/mem` (com a root). |
| `tests/yuvhint.c` | Comprova com Mesa converteix NV12 amb cada indicació d'espai de color i de rang d'EGL, amb el shim o sense. |

No engeguis el Kodi mentre `testcard` ocupi la pantalla: el Kodi no aconseguirà ser DRM
master i anirà acumulant descriptors de *fence* d'EGL fins que el reiniciïs.

## Dades de l'RA620 (llegides del DPCD)

```
DPCD rev 1.4, enllaç màxim 5,4 Gbps, 4 carrils; DOWNSTREAMPORT_PRESENT 0x10
OUI de branca 00-60-AD, id "cHDMIb", hw 01, fw 20.146
DFP caps byte 1 = 0x88 (340 MHz de TMDS màxim), PROTOCOL_CONVERTER_CONTROL 3050-3052 = 00 00 00
DPRX_FEATURE_ENUMERATION (0x2210) = 0x08: admet colorimetria ampliada per VSC SDP
```

El rang només es podria corregir al firmware de l'RA620, i això ho ha de respondre Radxa.
