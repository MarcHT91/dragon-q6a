# Càlcul a la GPU Adreno 643

*[English](README.md)*

Proves (12-11-2026) per fer feina de xarxes neuronals a la GPU Adreno 643 al costat de la NPU, amb el
driver obert (Mesa: driver de kernel msm, Vulkan Turnip, OpenCL rusticl) i amb la pila propietària de
Qualcomm (driver de kernel KGSL, OpenCL/Vulkan d'Adreno, QNN GPU). La feina objectiu era l'encoder de
visió del LFM3-VL-2B (164M paràmetres: 8 capes de convolució depthwise amb porta i 6 d'atenció
bidireccional, hidden 1024, patches de 16 px), amb el model de llenguatge a la NPU ([`../npu`](../npu)).

**Resultat:** la placa es queda amb el driver obert. ncnn sobre Turnip, amb les correccions d'aquí sota,
iguala el millor resultat de la pila de Qualcomm per a l'encoder i manté Kodi funcionant. El compilador
de shaders de Qualcomm és millor (~1,5x en una GEMM escrita a mà), però cap de les seves biblioteques
d'aprenentatge automàtic hi funciona millor.

## Resultats

### Encoder de visió (LFM3-VL, 14 capes), en règim estable

| Runtime | Driver | 256 px | 384 px | 512 px |
|---|---|---|---|---|
| ncnn + la nostra MatMul Vulkan (encoder sintètic) | Turnip (obert) | **0,37 s** | **0,93 s** | 1,87 s |
| ncnn original (MatMul a la CPU) | Turnip (obert) | 0,58 s | 1,47 s | 2,94 s |
| ncnn + la nostra MatMul Vulkan | Vulkan de Qualcomm | 0,37 s | 0,94 s | **1,60 s** |
| llama.cpp, mmproj real F16 | Turnip (obert) | 0,79 s | 1,80 s | 3,36 s |
| llama.cpp, mmproj real F16 | OpenCL de Qualcomm | 0,51 s | 1,01 s | 1,80 s |
| llama.cpp, mmproj real F16 | Vulkan de Qualcomm | 2,0 s | 4,7 s | falla |

Tokens per al LLM: 64 / 144 / 256 (un per cada 32x32 px). L'encoder "sintètic" té les formes i els tipus
de capa reals amb pesos aleatoris i sense RoPE (`ncnn/enc_cf.py`). La seva sortida en fp16 coincideix
amb la referència fp32 de PyTorch (cos 0,9998-1,0000; error relatiu del 0,4-2% amb la MatMul que suma
en fp16).

### Multiplicació de matrius (fp16, 1024x1024x1024 si no s'indica)

| Kernel | Driver | GFLOPS |
|---|---|---|
| GEMM OpenCL a mà, B com a imatge (`opencl/gemm.c img 8 16 4`) | OpenCL de Qualcomm | **265** |
| la mateixa, sumes en fp32 | OpenCL de Qualcomm | 176 |
| la mateixa, B com a buffer | OpenCL de Qualcomm | 227 |
| el mateix kernel | rusticl (obert) | 114 |
| Convolució 1x1 d'ncnn (= Linear), 1024x1024x3072 (`ncnn/tiny3.py`) | Turnip | 177 |
| Convolució 1x1 d'ncnn | Vulkan de Qualcomm | 105 |
| Gemm / InnerProduct d'ncnn, sense subgroups (`ncnn/tiny2.py`) | Turnip | 38-43 |
| Gemm d'ncnn amb subgroups, NCNN_SUBGROUP_SIZE=64 | Turnip | ~1 |
| Gemm d'ncnn amb subgroups | Vulkan de Qualcomm | 27 |
| Convolució d'ncnn amb pesos dinàmics (q·kᵀ, `ncnn/dynconv.py`) | Turnip | 24 |
| CPU (InnerProduct d'ncnn, fp32) | - | 48 |

Amb tots dos drivers la GPU arriba als 812 MHz amb el governor per defecte (`simple_ondemand`), així que
la diferència entre drivers ve del compilador de shaders (ir3 contra el de Qualcomm), no de la freqüència.

## Driver obert (Mesa)

### ncnn sobre Turnip

- **Les operacions amb subgroups trenquen la capa Gemm.** Amb `use_subgroup_ops` activat (el valor per
  defecte), el shader `gemm_sg` d'ncnn penja la GPU a 1024x1024 (`vkWaitForFences failed -4`,
  VK_ERROR_DEVICE_LOST; la GPU es recupera sola) i dona resultats incorrectes, diferents a cada execució, a
  128x128. Ni `TU_DEBUG=flushall|syncdraw` ni set opcions d'`IR3_SHADER_DEBUG` ho canvien. ncnn planifica
  el shader per a la mida de subgroup que es reporta (128) amb blocs grans (16x8x8). Forçar-ne 64
  (`NCNN_SUBGROUP_SIZE=64`, el nostre pedaç) fa que els resultats siguin correctes, però llavors el shader
  va a ~1 GFLOPS i a 1024 continua excedint el temps. Cal fer servir `net.opt.use_subgroup_ops = False`.
- **Escriu les capes Linear com a convolucions 1x1** (format channels-first). El camí conv1x1 de Vulkan
  d'ncnn va a 150-180 GFLOPS. Gemm i InnerProduct van a ~40 GFLOPS, que no és més ràpid que la CPU.
- **ncnn no té capa MatMul per a Vulkan.** Els productes activació x activació (q·kᵀ i scores·v de
  l'atenció) es calculen a la CPU, amb còpies GPU↔CPU a cada capa d'atenció. `ncnn/patch` afegeix
  `MatMul_vulkan` (per lots, elempack 1, 4x4 sortides per fil, lectures vec4). Un bloc d'atenció passa de
  386 a 190 ms amb 1024 patches, i l'encoder sencer va un ~35% més ràpid. Suma en fp16 (error relatiu
  de l'1-2% a 512 px); sumar en fp32 queda pendent.
- Exportar l'atenció com a SDPA va 2 vegades més lent que MatMul + Softmax explícits.

### llama.cpp

llama.cpp (Vulkan) ja executa el mmproj real del LFM3-VL (`llamacpp/build.sh`) i la sortida de l'encoder
és correcta (descripcions d'imatge coherents). Va unes 2 vegades més lent que ncnn amb la nostra MatMul.
El seu backend OpenCL rebutja rusticl ("unsupported GPU 'FD643'"): només accepta el driver de Qualcomm.

### rusticl (OpenCL)

`apt install mesa-opencl-icd` i després executar amb `RUSTICL_ENABLE=freedreno`. Reporta OpenCL 3.0 amb
fp16, sense subgroups ni extensions `cl_qcom_*`. La GEMM escrita a mà arriba a 114 GFLOPS.

## Pila propietària de Qualcomm (KGSL)

El PPA públic "Qualcomm Addon Packages" d'Ubuntu (`ppa:ubuntu-qcom-iot/qcom-ppa`, sèrie resolute) l'empaqueta
per a Ubuntu 26.04: `kgsl-dkms` 1.0.11, `qcom-adreno-*` 1.877.5 (OpenCL, Vulkan, EGL/GLES) i `qairt-*`
2.49 (QNN). No cal portar res:

- `kgsl/build-kgsl.sh` compila `msm_kgsl.ko` contra l'arbre del kernel 7.2.9 sense canvis. El kernel
  exporta els 490 símbols que el mòdul importa. Reconeix el device tree estàndard (`qcom,adreno-635.0` =
  A643, chip id 0x06030500; `qcom,adreno-gmu-635.0`) i fa servir el firmware `a660_sqe.fw` / `a660_gmu.bin`
  que ja hi ha a la placa.
- msm ha de deixar lliure la GPU. El nostre kernel té `msm.skip_gpu` (msm és dins del kernel, així que és
  una opció de la línia d'arrencada): `kgsl/kgsl-boot-default.sh` afegeix una entrada de systemd-boot amb
  l'opció i la posa per defecte. `bootctl set-oneshot` no funciona perquè el firmware no exposa variables
  EFI.
- Desvincular la GPU de msm en calent (`echo 3d00000.gpu > .../adreno/unbind`) **no** funciona. msm deixa
  recursos devres sobre la GMU (`3d6a000.gmu: Resources present before probing`) i KGSL no la pot agafar;
  surten WARN de msm_gem i la pantalla queda apagada fins que reinicies.
- Després del reinici, `sudo kgsl/kgsl-load.sh` carrega el mòdul, que crea `/dev/kgsl-3d0`.
  `kgsl/fetch-adreno.sh` desempaqueta els drivers d'Adreno en una carpeta privada i escriu
  `kgsl/adreno-env.sh`, de manera que no s'instal·la res amb apt i Mesa queda intacte. `kgsl/kgsl-revert.sh`
  i un reinici tornen al driver obert.
- En aquest mode **Kodi no pot dibuixar**: necessita el GL de Mesa sobre la GPU de msm, i hauria de passar
  al GLES de Qualcomm, que no està provat.

Què aporta aquesta pila (números a les taules de dalt):

- **OpenCL** es presenta com a "QUALCOMM Adreno(TM) 635" i té subgroups, `cl_qcom_dot_product8`,
  `cl_qcom_ml_ops` i camins d'imatge. Ha donat el millor resultat: el backend OpenCL per a Adreno de
  llama.cpp, i 265 GFLOPS en la GEMM escrita a mà.
- **Vulkan** va més lent que Turnip amb ncnn: conv1x1 a 105 GFLOPS contra 177. La Gemm amb subgroups hi
  dona resultats correctes, però és lenta.
- **CLML** (Adreno OpenCL ML SDK v4.1, `opencl/clml_bench.cpp`) no es pot fer servir. Totes les crides
  retornen -30 (CL_INVALID_VALUE). Amb gdb es veu que el shim de `libOpenCL_adreno` obre `libCB.so.1` i no
  hi troba el backend d'ML: aquesta compilació de `libCB` no exporta cap símbol `mlb*`. L'extensió
  s'anuncia, però no s'inclou.
- **QNN GPU** (`qnn-gpu/qnn_gpu_test.py`, ONNX Runtime 1.27 + onnxruntime-qnn 2.3, `libQnnGpu.so`): un
  bloc conv triga 225 ms en fp32 (ncnn: 115 ms) i el bloc d'atenció dona resultats incorrectes (cos 0,71).
  Amb el model exportat en fp16, el bloc conv triga 149 ms però és incorrecte (cos 0,70), i l'atenció dona
  NaN. Necessita KGSL; amb el driver obert no funciona.

## Fitxers

| Ruta | Què hi ha |
|---|---|
| `ncnn/enc_cf.py` | Encoder sintètic del LFM3-VL en PyTorch (channels-first, Linear com a conv 1x1), exportat amb pnnx |
| `ncnn/bench_ncnn.py` | Mesura un encoder exportat a la GPU i el compara amb la sortida de PyTorch |
| `ncnn/tiny2.py`, `tiny3.py`, `dynconv.py` | Microbenchmarks d'una sola capa (Gemm/InnerProduct, conv1x1, conv amb pesos dinàmics) |
| `ncnn/patch/`, `ncnn/apply_patch.py` | Capa MatMul per a Vulkan + variable `NCNN_SUBGROUP_SIZE` per a ncnn 20261020 |
| `ncnn/build_wheel.sh` | Compila la wheel d'ncnn amb els pedaços (cp314, aarch64) en Docker; ~1 min en un Mac amb Apple Silicon |
| `kgsl/` | Compilar KGSL, desempaquetar els drivers d'Adreno, arrencar amb `msm.skip_gpu=1`, carregar, desfer |
| `opencl/gemm.c` | GEMM OpenCL fp16 escrita a mà (variants buffer / imatge) |
| `opencl/clml_bench.cpp` | Benchmark de GEMM/conv amb CLML (necessita la capçalera del SDK; documenta la fallada -30) |
| `qnn-gpu/qnn_gpu_test.py` | Blocs de l'encoder amb ONNX Runtime + el backend GPU de QNN |
| `llamacpp/build.sh` | llama.cpp amb Vulkan + OpenCL, fet servir com a encoder de referència |

Python per a les proves d'ncnn: `pip install ncnn pnnx torch numpy`. El ncnn 1.0.20261020 de PyPI funciona
sobre Turnip si es desactiven els subgroups; la wheel amb els pedaços hi afegeix la MatMul per a Vulkan.

Llicències: el nostre codi és MIT com la resta del repositori; `ncnn/patch` està fet per a ncnn (BSD-3-Clause).
No s'inclou res de Qualcomm: els scripts descarreguen el codi de KGSL, els drivers d'Adreno i les capçaleres de QNN
del PPA públic d'Ubuntu, i el SDK de CLML s'ha de descarregar de Qualcomm.
