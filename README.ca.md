# dragon-q6a

*[English](README.md)*

Tot el que he fet funcionar a la **Radxa Dragon Q6A** (Qualcomm QCS6490) amb Ubuntu 26.04.1 LTS
(imatge de rafaelwms). Ara la placa funciona amb el kernel edge d'Armbian `7.3.1-edge-qcs6490` i el
firmware de Radxa `6.0.260916` (efivars, SPI NOR i fTPM des de Linux; iris amb el firmware HFI
gen2). La major part de la feina es va fer amb `7.2.9-edge-qcs6490` i el firmware `6.0.251230`;
cada README diu amb quin kernel s'han obtingut els resultats. Cada directori és independent i té
el seu propi README.

| Directori | Què hi ha |
|---|---|
| [`board/`](board) | Configuració general: acceleració de la GPU (Adreno 643), ajustos de CPU, xarxa, zram i escriptura a disc, i un monitor de la GPU (`gputop`). |
| [`audio/`](audio) | Àudio: sortida d'auriculars a 24 bits (DKMS amb tres correccions del kernel), rutes del DSP que ja no es perden, un `alsamixer` que es pot fer servir i AirPlay 2 (shairport-sync) amb volum de maquinari. |
| [`kodi-iris/`](kodi-iris) | Descodificació de vídeo per maquinari a Kodi amb el descodificador iris (V4L2 mem2mem, DRM PRIME), 4K amb EGL, HEVC/VP9 de 10 bits amb el firmware gen2, i el rang de color HDMI amb el convertidor DP→HDMI RA620 de la placa. |
| [`npu/`](npu) | LFM3-1.7B (Liquid AI) executant-se del tot a la NPU Hexagon HTP v68, sense el SDK de Qualcomm: grafs compilats amb `onnxruntime-qnn`, executats per un runner natiu de QNN (11,7 tok/s) o per ONNX Runtime. |
| [`vl/`](vl) | LFM3-VL-2B (visió i llenguatge): l'encoder de visió a la GPU Adreno (ncnn/Vulkan) i el model de llenguatge a la NPU, en un sol programa de xat en C++. |
| [`d1/`](d1) | d1-2B (el model de decisió de Liquid AI): respostes calibrades de sí/no, elecció i puntuació sobre text i imatges en una passada, encoder a la GPU + NPU amb pesos int8, programa en C++ i mode servei JSON. |
| [`gpu/`](gpu) | Proves de càlcul a la GPU: ncnn sobre Turnip (error dels subgroups, capa MatMul per a Vulkan), el driver KGSL de Qualcomm compilat per a la 7.2.9, OpenCL/CLML/QNN GPU, llama.cpp; resultats i scripts. |

Els arranjaments es fan des de la placa: mòduls del kernel, configuració i scripts, i no amb
ajustos a la TV o al monitor. Els pedaços del kernel enviats als desenvolupadors del
kernel (v1 el 2026-11-11, v2 el 2026-11-16) són a [`audio/hifi/upstream`](audio/hifi/upstream).

## Historial

Aquest repositori ajunta tres repositoris anteriors, amb tot el seu historial:
`dragon-q6a-audio` → `audio/`, `kodi-iris` → `kodi-iris/` i `lfm3-q6a-npu` → `npu/`.
Els repositoris antics estan arxivats i apunten aquí.

## Llicència

MIT ([LICENSE](LICENSE)). Algunes parts tenen una llicència pròpia, indicada al seu directori:
els pedaços d'FFmpeg de `kodi-iris/patches` (LGPL-2.1+), `audio/alsa-ctl-filter/ctl_filter.c`
(LGPL-2.1+), i els pedaços del kernel i els mòduls que compila el DKMS a `audio/hifi` (GPL-2.0).
