# lfm3-q6a-npu

*[English](README.md)*

Executa **[LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)** (Liquid AI) a la **NPU**
de la **Radxa Dragon Q6A** (Qualcomm QCS6490, Hexagon HTP v68), tot a la NPU, sense l'SDK de
Qualcomm i sense compte de Qualcomm (el runner natiu opcional només necessita les capçaleres
públiques de l'API de QNN, que el seu script de compilació descarrega del PPA públic de
Qualcomm a Ubuntu).

El model es reconstrueix a mà com a grafs ONNX quantitzats (pesos int4/int8, activacions
de 16 bits) que el proveïdor d'execució QNN d'`onnxruntime-qnn` compila per a la HTP. Un
bucle a la CPU guarda les memòries cau de K/V i els estats de les convolucions, i crida els
grafs token a token, o de 16 en 16 per al prompt. Hi ha dos runtimes per executar els grafs
compilats: un **runner natiu** (C++ que crida directament l'API de QNN i guarda l'estat a
memòria compartida amb la NPU; és el predeterminat) i ONNX Runtime. Tots dos donen sortides
idèntiques bit a bit.

| | |
|---|---|
| Generació | **11,7 tok/s** (context 4096, runtime natiu); ONNX Runtime: 7,8 tok/s (4096), 8,9 (2048), 9,4 (1024) |
| Prefill | blocs de 16 tokens, context 4096: **150 tok/s** en un prompt de 723 tokens amb el runtime natiu (ONNX Runtime: 94 tok/s; ~85 tok/s en un prompt de 2,8k tokens) |
| Càrrega | ~2-4 s amb el runtime natiu, ~6-9 s amb ONNX Runtime (contextos compilats) |
| Memòria | 2,1 GB de grafs compilats + 0,5 GB d'embedding, ~3,6 GB de RSS (context 4096) |
| Qualitat respecte a fp32 | coincidència top-1 del 84,1%, KL mitjana 0,156 (teacher-forced) |
| Bench de tasques | 36/36 tasques bàsiques (crides a eines, JSON, instruccions, coneixement, mates); 22/28 de difícils (fp32: ~26/28) |

Tot està mesurat a la placa. Detalls a [docs/RESULTS.md](docs/RESULTS.md).

## Contingut

| Ruta | Què hi ha |
|---|---|
| `npu/` | Constructor de grafs, runtimes de la NPU, quantització (GPTQ, calibratge, sensibilitat), avaluació, profiling, xat per terminal |
| `native/` | Runner natiu de QNN (`qnnrun.cpp`, es compila amb `native/build.sh`) que fa servir el runtime natiu |
| `configs/` | Rangs d'activació i llistes de matrius int8 del build de referència |
| `experiments/` | AWQ + GPTQ (provat, no millora; es conserva com a referència) |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Pas a pas: del checkpoint de Hugging Face al model funcionant |
| [docs/HTP_V68.md](docs/HTP_V68.md) | Què funciona i què falla a la HTP v68, i com s'hi adapten els grafs |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | Decisions de quantització i els experiments que les justifiquen |
| [docs/PC_TOOLS.md](docs/PC_TOOLS.md) | Un PC x86-64 amb GPU NVIDIA: GPTQ, avaluació i **compilació per a la NPU sense la placa** |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Penjades del DSP, errors de QNN i com recuperar-se'n |
| [docs/MODELS.md](docs/MODELS.md) | Altres models útils per a aquesta NPU |
| [docs/RESULTS.md](docs/RESULTS.md) | Velocitat, qualitat i benchmarks |

## Inici ràpid

Cal la placa preparada com a [Requisits](#requisits), el checkpoint del model, els pesos
GPTQ i un build. El procediment complet és a [docs/PIPELINE.md](docs/PIPELINE.md); en resum:

```bash
# 1. entorn a la placa
python3 -m venv .venv && .venv/bin/pip install -r requirements-board.txt
.venv/bin/hf download LiquidAI/LFM3-1.7B --local-dir models/LFM3-1.7B

# 2. dades de calibratge + pesos GPTQ int4 (millor en un PC amb GPU, vegeu docs/PC_TOOLS.md)
python npu/gen_calib_lfm3.py models/LFM3-1.7B build/calib_chat.jsonl
CHAT=build/calib_chat.jsonl python npu/gptq_lfm3.py models/LFM3-1.7B data/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet build/gptq 128 512

# 3. construir i compilar els grafs (placa: ~13 min; en un PC sense la NPU: ~6 min)
export INT8_LAYERS=0,1,2,3,4,5 INT8_MATS=configs/int8_mats_100M.json
.venv/bin/python npu/build_all_lfm3.py models/LFM3-1.7B configs/lfm3_act_ranges.json build/gptq build/npu 4096 6
.venv/bin/python npu/build_prefill_lfm3.py models/LFM3-1.7B configs/lfm3_act_ranges.json build/gptq build/npu 16

# 4. runner natiu (opcional, generació ~1,5x més ràpida; cal g++)
native/build.sh

# 5. executar
.venv/bin/python npu/chat.py            # --runtime native|ort
```

Per fer-lo servir des del teu codi, `load_model(build_dir)` de `npu/runtime.py` retorna el
runtime natiu (`npu/lfm3_native.py`, `LFM3Native`) si `native/libqnnrun.so` està compilat, i si no
el d'ONNX Runtime (`npu/lfm3_npu.py`, `LFM3NPU`); `LFM3_RUNTIME=native|ort` en força un. Tots dos
tenen la mateixa interfície: `feed(ids)` retorna els logits després d'un prompt i `step(token)`
genera un token.

## Requisits

- Radxa Dragon Q6A. Provat amb Ubuntu 26.04.1 LTS, kernel `7.2.9-edge-qcs6490`, amb els
  paquets `fastrpc` / `libcdsprpc1` (1.0.7) de la distribució i el firmware del CDSP de la
  imatge.
- L'usuari dins del grup `fastrpc` (`/dev/fastrpc-cdsp`).
- Python 3.14 amb `onnxruntime==1.27.0` i `onnxruntime-qnn==2.3.0` (QNN 2.47). Les
  biblioteques de QNN per a la HTP, inclòs l'*skel* v68 que s'executa al DSP, venen dins el
  wheel d'`onnxruntime-qnn`. El runner natiu fa servir aquestes mateixes biblioteques; per
  compilar-lo cal `g++`, `curl` i `dpkg-deb`.
- Placa de ~12 GB de RAM: construir els grafs i executar el model hi caben. El model de
  referència fp32 (només per validar) necessita ~7 GB i va millor en un PC.
- Opcional però recomanat: un PC x86-64 amb GPU NVIDIA (8 GB o més) per al GPTQ i
  l'avaluació, que també pot compilar els grafs de la NPU
  ([docs/PC_TOOLS.md](docs/PC_TOOLS.md)).

## Com funciona (resum)

- **Grafs.** Les 24 capes (12 de convolució curta amb porta, 6 d'atenció completa i 6
  d'atenció de finestra lliscant) es reparteixen en 4 grafs de 6 capes, més un cap (LM
  head) en int8. Uns altres 4 grafs processen 16 tokens del prompt alhora (prefill per
  blocs).
- **Precisió.** Activacions uint16 (QDQ); pesos int4 simètrics per canal (GPTQ). Les 6
  primeres capes i 26 matrius sensibles fan servir int8 (vegeu a sota).
- **Bucle a la CPU.** L'embedding, les memòries cau K/V (atenció completa: fins al
  context; finestra lliscant: un anell de 255 posicions), els estats de les convolucions, les
  taules de RoPE i les màscares viuen a la CPU, i entren i surten dels grafs com a tensors
  uint16/uint8. El runtime natiu els guarda tots en buffers rpcmem registrats una sola vegada
  a QNN, de manera que no es copia res a cada token (ONNX Runtime copia ~53 MB d'estat als
  buffers de QNN a cada token), i fa tot el pas de generació en C++. On va el temps:
  [docs/RESULTS.md](docs/RESULTS.md#decode-profile).
- **Limitacions de la v68.** No hi ha RMSNorm (es fa amb LpNorm × γ√N); les
  multiplicacions activació × activació de 16×16 bits perden precisió sense avisar (es
  parteixen en byte alt i baix); els pesos int4 per blocs donen resultats incorrectes (només
  per canal). Vegeu [docs/HTP_V68.md](docs/HTP_V68.md).

### Per què hi ha pesos en int8

Amb GPTQ int4 a tot arreu, el model tenia una tendència forta a cridar una eina per a
preguntes trivials quan hi havia eines disponibles: p(`<tool_call>`) per a "What is the
capital of Italy?" passava de 0,000 (fp32) a 0,994. Injectant els estats ocults de la NPU al
model fp32 es va veure que el biaix ve de l'error int4 acumulat a les 6 primeres capes.
Passar aquestes capes a int8 ho resol (0,010) amb un ~10% de cost de velocitat. Després, un
estudi de sensibilitat per matriu va triar 26 matrius més (sobretot `k_proj`/`v_proj`
d'atenció i `out_proj` de les convolucions) que redueixen un 17% el KL respecte a fp32 per
un ~1,5% de velocitat. Detalls a [docs/QUANTIZATION.md](docs/QUANTIZATION.md).

## Problemes coneguts

- **Fuita de memòria al DSP en arrencar o tancar processos.** Amb aquest kernel, fastrpc
  sovint no pot alliberar un buffer del DSP quan un procés que fa servir la NPU arrenca o
  surt (`unmmap ... ERROR` a `dmesg`). Al cap de diverses fuites, el CDSP deixa de respondre
  i totes les sessions de la NPU fallen fins que es reinicia. Mantingues el model en un sol
  procés de llarga durada i evita molts processos curts seguits. Atura tots els processos de la NPU abans de reiniciar el CDSP. Com recuperar-ho:
  [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
- **Un sol runtime de QNN per procés.** No carreguis les sessions d'ONNX Runtime i el runner
  natiu en el mateix procés: el DSP es queda sense espai d'adreces, el procés es pot penjar i
  la memòria que tenia mapada es perd fins que reinicies.
- **No llegeixis `/sys/kernel/debug/dma_buf/bufinfo`** amb aquest kernel: provoca un error
  intern (*oops*) i deixa bloquejats els dma-buf, i totes les sessions de la NPU posteriors
  es pengen fins que reinicies.

## Crèdits

- **[Liquid AI](https://huggingface.co/LiquidAI)** per LFM3-1.7B. Els pesos del model no
  s'inclouen aquí i tenen la seva pròpia llicència.
- **[PrismPhi/radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu](https://github.com/PrismPhi/radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu)**,
  que va demostrar que el wheel `onnxruntime-qnn` de PyPI és suficient per fer servir la NPU
  de la Q6A.
- **onnxruntime** i el seu proveïdor d'execució QNN (`onnxruntime-qnn`).
- **GPTQ** (Frantar et al., 2022) per a la quantització int4 dels pesos.
- **[rafaelwms/ubuntu-26.04.1-dragon-q6a](https://github.com/rafaelwms/ubuntu-26.04.1-dragon-q6a)**,
  la imatge d'Ubuntu on s'ha desenvolupat i provat.

## Llicència

MIT per al codi d'aquest repositori (vegeu [LICENSE](LICENSE)). Els pesos, el tokenitzador i
la plantilla de xat del model pertanyen als seus autors i mantenen les seves llicències.
