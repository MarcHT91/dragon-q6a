# LFM3-VL-2B a la GPU + NPU

*[English](README.md)*

[LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) (Liquid AI) amb cada part al seu accelerador de la Dragon
Q6A: l'**encoder de visió** (14 capes, 400M paràmetres amb el projector) a la **GPU Adreno 643** amb ncnn sobre Vulkan
(Mesa Turnip), i el **model de llenguatge** (LFM3, 24 capes) a la **NPU Hexagon** amb les mateixes gràfiques i el
mateix runner natiu que [`../npu`](../npu). Les dades de la imatge entren al model de llenguatge com a files
d'embedding extra a les posicions `<|image|>`, així que les gràfiques de la NPU no necessiten res específic per a
imatges.

## Resultats

| | |
|---|---|
| Encoder a la GPU (fp16) | 256 / 384 / 512 px: 0,65 / 1,2 / 2,4 s; cosinus per token respecte a fp32 0,99-0,998 |
| Model de llenguatge a la NPU | generació a 11,6 tok/s; prefill en blocs de 16 tokens |
| Temps fins al primer token (una imatge) | 256 / 384 / 512 px: 1,2 / 2,4 / 4,3 s |
| Qualitat respecte a fp32 (fotos de COCO no vistes, teacher-forced) | top-1 85,8%, KL 0,22 (dades d'imatge fp32); 86,8%, KL 0,22 amb les de la GPU |

Les respostes són majoritàriament les mateixes que en fp32 (*"The image shows three giraffes reaching up to eat
green leaves from a mesh feeder hanging on a tree."*). Comprovacions darrere d'aquests números: la referència fp32
de l'encoder reprodueix el model de Hugging Face; el tokenitzador en C++ és idèntic al de `tokenizers` de Hugging
Face en 85k tokens de text variat; el redimensionat d'imatge és idèntic bit a bit al de Pillow.

## Contingut

| Ruta | Què |
|---|---|
| `encoder/vl_enc.py` | referència fp32 de l'encoder + projector, i la versió channels-first exportada a ncnn (`export H W`) |
| `encoder/split_checkpoint.py` | separa el checkpoint de Hugging Face en `vision.safetensors` i un model de llenguatge LFM3 normal |
| `encoder/gpu_feats.py` | dades d'imatge de la GPU per al conjunt d'avaluació (per validar la NPU amb elles) |
| [`app/`](app) | `lfm3vl`: el programa de xat en C++ (encoder a la GPU + model de llenguatge a la NPU en un sol procés) |
| `../npu/npu/vl_pipeline.bat` | part del PC de la compilació per a la NPU: dades de calibració amb imatges, avaluació, GPTQ, rangs, compilació |
| `../npu/npu/validate_vl.py` | NPU contra fp32 al conjunt d'avaluació |

## Compilació

1. **Separar el checkpoint** (qualsevol màquina): `python vl/encoder/split_checkpoint.py LFM3-VL-2B/ out/` dona
   `out/vision.safetensors` i `out/lm/`.
2. **Gràfiques de la NPU** (PC amb GPU NVIDIA, compilació offline com a
   [`../npu/docs/PC_TOOLS.md`](../npu/docs/PC_TOOLS.md)): carpeta de treball amb `models\LFM3-VL-LM` (= `out/lm`),
   `vl\vision.safetensors`, `data\coco_val0.parquet` (un fragment de validació de COCO, per exemple
   `detection-datasets/coco` a Hugging Face), el parquet de wikitext i `npu\` = `../npu/npu`; executar
   `npu\vl_pipeline.bat` (~30 min amb una RTX 3060). Fa 128 prompts amb imatges i preguntes que respon el model
   fp32, hi afegeix la calibració de xat de text, fa GPTQ, mesura els rangs d'activació i compila (int4 + capes 0-5
   en int8, context de 4096). Copia `build\npu` a la placa.
3. **Encoder** (placa, venv amb torch, pnnx i el ncnn amb el pedaç de [`../gpu/ncnn`](../gpu/ncnn)):
   `VL_WEIGHTS=out/vision.safetensors python vl/encoder/vl_enc.py export 384 384` (unes quantes fotos qualssevol a
   `calib/*.jpg` per al reescalat fp16).
4. **Programa**: vegeu [`app/README.md`](app/README.md).

## Notes

- L'Adreno posa a zero els subnormals fp16, i les activacions de l'encoder són petites: `vl_enc.py` reescala el
  flux residual i les entrades de les normes per potències de dos, i calcula en fp32 els productes de l'atenció,
  el softmax, les operacions element a element i les reduccions (emmagatzematge fp16). Sense això, els tokens a
  512 px baixen a cosinus 0,36.
- Cal `net.opt.lightmode = False` amb emmagatzematge fp16 a Vulkan (el reciclatge de blobs corromp els resultats).
- El programa fa servir exportacions quadrades; el model de decisió de [`../d1`](../d1) tria l'exportació més
  propera a la proporció de cada foto, que és més precisa.

Els pesos són de Liquid AI (LFM Open License v1.0) i no es redistribueixen aquí.
