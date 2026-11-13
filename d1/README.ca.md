# Decisions amb d1-2B a la GPU + NPU

*[English](README.md)*

[d1-2B](https://huggingface.co/LiquidAI/d1-2B) (Liquid AI) és un **model de decisió** construït sobre LFM3-VL-2B: a
partir d'un context (text, JSON i/o una imatge) i unes preguntes amb nom, retorna respostes calibrades en **una sola
passada**, sense generar tokens. Aquí corre a la Dragon Q6A amb l'encoder de visió a la **GPU Adreno** (ncnn) i el
model de llenguatge a la **NPU Hexagon** (QNN, pesos int8), en un sol procés C++.

```
$ d1 --build npu --enc enc consulta.json
{"answers": {"refund": {"type": "noul", "noul": 0.996},
             "team": {"type": "choice", "choice": "billing", ...},
             "topics": {"type": "choice", "choices": ["billing", "delivery"], ...},
             "urgency": {"type": "score", "score": 1.0, ...}}, ...}
```

Tipus de pregunta: `noul` (sí/no: P(sí)), `choice` (una opció, o `"multiple": true`), `score` (2-10 nivells
ordenats). Encara no: les extraccions (fragments de text, números, dates), les estimacions i les caixes.

## Resultats

Mesurat amb 60 consultes no vistes (24 fotos de COCO, 20 passatges de Wikipedia, 16 ordres tipus Home Assistant;
652 respostes), contra d1 en fp32 amb les fotos a la seva mida:

| | Mateixa resposta | Variació mitjana de probabilitat | 95% per sota de |
|---|---|---|---|
| NPU, dades d'imatge fp32 | **98,9%** | 0,008 | 0,034 |
| NPU + encoder a la GPU (mantenint la forma de la foto, 192 tokens d'imatge) | **97,1%** | 0,027 | 0,110 |

Els exemples de la fitxa del model surten exactes (`1.0 billing ['billing', 'delivery'] medium`, `shipping`,
`technical`).

Velocitat (`--serve`, escalfat):

| Consulta | Tokens | Temps |
|---|---|---|
| Ordre de casa + 2 preguntes (intenció, habitació) | 132 | 1,3 s |
| Foto + 3 preguntes | 274 | 3,9 s (GPU 1,7 + NPU 2,2) |
| Foto + 13 preguntes | 673 | 6,5 s (GPU 1,7 + NPU 4,8) |

La part de la NPU està limitada per la lectura dels pesos int8 (cada bloc de 64 tokens llegeix els ~250 MB de cada
etapa); cada pregunta hi afegeix uns 45 tokens.

## Com funciona

- **Atenció `set`.** d1 posa fins a 16 preguntes després del context. A les 6 capes d'atenció completa, cada token
  del context veu tota la seqüència, i cada token d'una pregunta veu el context i la seva pregunta, en tots dos
  sentits (les capes de finestra lliscant i de convolució continuen sent causals). Les gràfiques de la NPU són blocs
  de prefill, així que el model s'executa **per etapes**: una etapa és una capa d'atenció completa, les capes
  causals que la segueixen i només les claus/valors de la capa d'atenció completa següent. Cada etapa passa per
  tots els blocs abans que comenci la següent, de manera que totes les claus existeixen quan una consulta les
  necessita. 7 etapes, blocs de 64 tokens, context de 1024 tokens.
- **Pesos int8 a tot arreu**, activacions de 16 bits i projeccions reescalades per token abans de les normes
  posteriors (`PT_NORM`, vegeu [`../npu/docs/HTP_V68.md`](../npu/docs/HTP_V68.md)).
- **Lectura de respostes a la CPU.** Només calen els logits dels tokens de resposta (sí/no, codis d'opció, dígits):
  la norma final i uns quants productes escalars, en lloc d'una capa final de 263 MB a la NPU. La normalització
  sobre tot el vocabulari es cancel·la al softmax entre opcions. Després, la calibració de d1 (temperatures, la
  taula de l'elecció múltiple) com a la seva publicació.
- **Imatges**: l'encoder exportat per a 512x384, 384x512, 576x320, 320x576 i 448x448; cada foto fa servir la forma
  més propera a la seva proporció.

**Decisions i les proves que les justifiquen** (sempre contra d1 en fp32 amb consultes no vistes):

- *Pesos*: es van comparar en simulació int4 (RTN, GPTQ, AWQ+GPTQ), int4 amb les capes sensibles en int8, i int8;
  int8 manté el 99,7% de les respostes amb una variació mitjana de 0,003; la millor combinació (GPTQ int4 + capes
  sensibles en int8), el 98,9% i 0,012.
  AWQ no va ajudar.
- *Precisió de la NPU*: al principi la NPU donava 97,1% / 0,030 amb dades fp32. La pèrdua es va localitzar,
  operació per operació, a la sortida de les multiplicacions W8A16 (~12 bits efectius); el reescalat per token la
  va portar a 98,9% / 0,008. Retallar els rangs d'activació en lloc d'això ho empitjorava.
- *Forma de la imatge*: el quadrat de 384 px perd 3,4 punts fins i tot en fp32; mantenir la proporció amb ~200k
  píxels en perd 1,7.
- *Blocs*: 64 tokens amb context de 1024 és la variant més ràpida de les provades (els blocs de 128 anaven més lents).

## Fitxers

| Fitxer | On | Què |
|---|---|---|
| `d1_prompt.py` | on sigui | prompt, empaquetament `set` i lectura de respostes (sense torch), compartit per la referència i la NPU |
| `d1_ref.py` | PC | referència fp32 (torch); `python d1_ref.py MODEL` executa els exemples de la fitxa |
| `d1_quant.py` | PC | la comparació de quantitzacions de pesos d'aquí dalt |
| `d1_data.py`, `d1_calib.py` | PC | consultes de calibració i avaluació; rangs d'activació + conjunt d'avaluació amb respostes fp32 |
| `build_d1.py` | PC | construeix i compila les gràfiques per etapes (compilació QNN offline) |
| `configs/ranges.json` | | els rangs d'activació de la compilació de referència |
| `d1_npu.py`, `d1_validate.py` | placa | runtime en Python i validació contra les respostes fp32 |
| `gpu_feats.py` | placa | dades d'imatge de la GPU per al conjunt d'avaluació |
| [`app/`](app) | placa | `d1`: el programa en C++ (encoder a la GPU + etapes a la NPU + lectura), `validate_cpp.py` |

## Compilar i executar

1. **Gràfiques de la NPU** (PC amb GPU NVIDIA; [`../npu/docs/PC_TOOLS.md`](../npu/docs/PC_TOOLS.md)), amb el
   checkpoint de Hugging Face a `d1-2B/`:
   ```
   python d1/build_d1.py d1-2B d1/configs/ranges.json build 1024 64
   ```
   (`d1_calib.py d1-2B COCO_PARQUET WIKI_PARQUET work` torna a fer els rangs i el conjunt d'avaluació.) Copia a la
   placa: `stage*_ctx_qnn.bin`, `embed_f16.npy`, `fnorm.npy`, `manifest.json`, `tokenizer.json`,
   `calibration.json`, `config.json`.
2. **Encoder** (placa): `python vl/encoder/split_checkpoint.py d1-2B out/` i, per a cada forma,
   `VL_WEIGHTS=out/vision.safetensors python vl/encoder/vl_enc.py export 384 512` (i 512 384, 320 576, 576 320,
   448 448) a la carpeta de l'encoder.
3. **Programa** (placa, mateixos requisits que [`../vl/app`](../vl/app)):
   `NCNN_DIR=... PY=.../python d1/app/build.sh`.
4. **Executar**: `d1 --build BUILD --enc ENC consulta.json`, o `d1 --build BUILD --enc ENC --serve` per mantenir el
   model carregat i respondre una consulta JSON per línia de l'entrada estàndard. Una consulta:
   ```json
   {"state": "I was charged twice and my parcel never arrived.", "image": "foto.jpg",
    "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}}
   ```
   `state` pot ser text, qualsevol valor JSON, o `null` si només hi ha una imatge.

Cal que `/dev/dma_heap/system` sigui accessible pel grup `fastrpc` i que només hi hagi un procés a la NPU alhora
([`../npu/docs/TROUBLESHOOTING.md`](../npu/docs/TROUBLESHOOTING.md)). Els pesos són de Liquid AI (LFM Open License
v1.0) i no es redistribueixen aquí.
