# kodi-iris

*[English](README.md)*

Descodificació de vídeo per maquinari a Kodi a la **Radxa Dragon Q6A** (Qualcomm QCS6490),
fent servir el descodificador **iris** del kernel (`qcom_iris`, `/dev/video0`) a través de
V4L2 mem2mem.

El Kodi empaquetat per Ubuntu no pot fer servir iris amb l'FFmpeg del sistema. Aquest
projecte compila [rpi-ffmpeg](https://github.com/jc-kynesim/rpi-ffmpeg) 8.0.1 amb unes
quantes correccions per a iris, l'instal·la a `/opt/ffmpeg-iris` i obre Kodi amb aquestes
biblioteques mitjançant `LD_LIBRARY_PATH`. rpi-ffmpeg és compatible a nivell d'ABI amb el
`libavcodec62` d'Ubuntu. No es substitueix cap paquet del sistema, i el `kodi` normal
continua fent servir l'FFmpeg del sistema.

## Crèdits

- **[jc-kynesim/rpi-ffmpeg](https://github.com/jc-kynesim/rpi-ffmpeg)**: la branca d'FFmpeg
  d'on surt aquesta feina. El seu descodificador V4L2 mem2mem amb sortida DRM PRIME en fa
  gairebé tota la part important. Els pegats d'aquí només l'adapten a iris.
- **[rafaelwms/ubuntu-26.04.1-dragon-q6a](https://github.com/rafaelwms/ubuntu-26.04.1-dragon-q6a)**:
  la imatge d'Ubuntu 26.04.1 per a la Dragon Q6A (kernel 7.2 principal, amb iris) on s'ha
  desenvolupat i provat.

## Provat a

- Radxa Dragon Q6A, Ubuntu 26.04.1 LTS (imatge de rafaelwms), kernel `7.2.9-edge-qcs6490`
- Kodi 21.3 (paquet d'Ubuntu), sistema de finestres GBM
- Mesa 26.0.8 (freedreno, Adreno 643)

Altres kernels o versions de Kodi poden comportar-se diferent, sobretot iris, que depèn de
la versió del controlador.

## Requisits

```bash
sudo apt install -y git build-essential pkg-config \
    libdrm-dev libdav1d-dev libvpx-dev libxml2-dev libzvbi-dev \
    libbluray-dev libmp3lame-dev libudev-dev libgnutls28-dev libsoxr-dev \
    libopus-dev libvorbis-dev
sudo install -d -o "$USER" -g "$USER" /opt/ffmpeg-iris
```

L'usuari ha de ser als grups `video` i `render`. Si Kodi s'obre per SSH, també ha de ser a
`audio` i `input`.

## Compilació

```bash
./build.sh            # primer cop: clona rpi-ffmpeg, aplica patches/, configura, compila i instal·la
RECONF=1 ./build.sh   # força tornar a configurar
```

El primer cop, `build.sh` clona `test/8.0.1/main` de rpi-ffmpeg a `$SRC`, es posa a la
revisió base `2a287060cc6e` i aplica `patches/*.patch` amb `git am`.

Variables: `SRC` (per defecte `~/src/ffmpeg-iris`), `PREFIX` (`/opt/ffmpeg-iris`), `BUILD`
(`$SRC/out/build`) i `UPSTREAM` (l'URL de rpi-ffmpeg).

Després, instal·la el llançador:

```bash
install -m 755 kodi-iris /opt/ffmpeg-iris/bin/
```

## Ús amb Kodi

1. Obre Kodi amb `/opt/ffmpeg-iris/bin/kodi-iris [fitxer]`. El script `kodi` d'Ubuntu talla
   els arguments pels espais, així que cal passar rutes sense espais (un enllaç simbòlic
   serveix).
2. A *Paràmetres → Reproductor → Vídeos* (mode Expert):
   - **Permet usar el descodificador DRM PRIME**: activat (`videoplayer.useprimedecoder`).
   - **Permet l'acceleració per maquinari amb DRM PRIME**: activat
     (`videoplayer.useprimedecoderforhw`). Cal tenir-lo activat: si no, Kodi tria el
     descodificador per programari.
   - **Mètode de renderitzat PRIME**: **Direct to plane** (`videoplayer.useprimerenderer`
     = `0`). EGL no funciona (vegeu les limitacions).

   Si edites `~/.kodi/userdata/guisettings.xml` a mà amb Kodi tancat, treu l'atribut
   `default="true"` de la línia. Si hi és, Kodi no fa cas del valor.
3. Per tancar Kodi des d'un script, fes servir `tests/kodi-stop.sh`. Atura la reproducció
   per JSON-RPC (port 8080, amb el servidor web activat), espera que s'alliberi
   `/dev/video0`, surt i només com a últim recurs fa `kill -9`.

Si iris es queda penjat (qualsevol programa rep `VIDIOC_REQBUFS failed: Invalid argument`
i ningú no té `/dev/video0` obert), recarrega el controlador:

```bash
sudo modprobe -r qcom_iris && sudo modprobe qcom_iris
```

## Pegats (`patches/`)

| Pegat | Problema que resol |
|---|---|
| `0001` | iris retorna buffers de captura buits marcats amb error. Ara es descarten en lloc de donar-los com a fotogrames. |
| `0002` | Si la captura no es pot tornar a engegar després d'un canvi de resolució, la descodificació falla de seguida en lloc de quedar-se penjada. |
| `0003` | iris no té cap format de captura de més de 8 bits, així que els fluxos de més de 8 bits es refusen en obrir (`FF_V4L2_QUIRK_8BIT_ONLY`) en lloc de fallar quan la descodificació ja ha començat. |
| `0004` | Quan el descodificador per maquinari refusa un flux, el camí DRM PRIME de Kodi torna a provar amb el descodificador per programari. Aquest camí només admet formats de 8 bits, i l'HEVC de 10 bits donava pantalla negra. Ara `hevcdec` refusa els fluxos de més de 8 bits en obrir si se li demana `DRM_PRIME`, i Kodi passa al seu descodificador per programari normal. |
| `0005` | Després d'un buidatge (Kodi en fa un abans de cada salt), iris no torna a descodificar mai amb la mateixa sessió, i la sessió encallada podia deixar el controlador inservible per a tothom. A iris, el flush ara tanca el descodificador i n'obre un de nou (`FF_V4L2_QUIRK_REOPEN_AFTER_DRAIN`). |

Els pegats `0001`–`0003` i `0005` només tenen efecte quan el controlador és
`iris_driver`. El `0004` només té efecte quan qui crida demana `DRM_PRIME` al
descodificador HEVC per programari.

## Proves

| Script | Què fa |
|---|---|
| `tests/abi-check.sh` | Comprova que tots els símbols d'FFmpeg que importa `kodi.bin` existeixen a `/opt/ffmpeg-iris`. |
| `tests/gen-media.sh` | Genera els vídeos sintètics de `tests/media/`: H.264, HEVC i VP9 de 360p a 2160p, un canvi de resolució a mig flux i HEVC de 10 bits. |
| `tests/run-matrix.sh` | Compara fotograma a fotograma (framemd5) el descodificador per programari amb iris, amb sortida NV12 i DRM PRIME (`tests/bin/drmframemd5`). El 10 bits s'ha de refusar en obrir. Surt amb 0 si tot passa. |
| `tests/bin/seektest FITXER DEC N` | Fa N salts tal com els fa Kodi (buidatge amb un paquet NULL, `avcodec_flush_buffers`, salt) i mesura el temps fins al primer fotograma. |
| `tests/bin/primefallback FITXER 0\|1` | Comprova que el descodificador HEVC per programari refusa els fluxos de més de 8 bits quan se li demana `DRM_PRIME`. |
| `tests/kodi-check.sh` | Amb Kodi reproduint: biblioteques de `/opt`, `/dev/video0` obert, descodificador V4L2 al registre i CPU < 30 %. |
| `tests/kodi-seek.sh [N]` | Amb Kodi reproduint: N salts per JSON-RPC, amb el temps de represa (falla si passa de 3 s). |
| `tests/kodi-stop.sh` | Tanca Kodi sense deixar iris penjat. |
| `tests/eglprobe.c` | Diagnòstic: mostra el descriptor DRM que dona iris i prova d'importar-lo a EGL. |

Els programes en C es compilen contra `/opt/ffmpeg-iris`. Per exemple:

```bash
mkdir -p tests/bin
gcc -Wall -O1 tests/seektest.c -I/opt/ffmpeg-iris/include -L/opt/ffmpeg-iris/lib \
    -Wl,-rpath,/opt/ffmpeg-iris/lib -lavformat -lavcodec -lavutil -o tests/bin/seektest
```

Per tenir HEVC de 8 bits més realista, pots recodificar un tros de qualsevol vídeo H.264:

```bash
ffmpeg -ss 120 -t 60 -i ENTRADA.mkv -map 0:v:0 -map 0:a:0 -c:v libx265 -preset ultrafast \
    -pix_fmt yuv420p -c:a copy tests/media/real_hevc_1080p.mkv
```

## Resultats

Kodi 21.3, Direct to plane, pantalla de 1080p:

| Contingut | Descodificador | CPU de Kodi | Notes |
|---|---|---|---|
| H.264 1080p | iris | ~15 % | per programari: 100–150 % |
| HEVC 8 bits 1080p | iris | ~15 % | |
| HEVC 8 bits 2160p | iris | ~16 % | es descodifica bé; vegeu la limitació del 4K |
| HEVC 10 bits 1080p | programari | ~100 % (un nucli) | velocitat real, sense errors |
| 10 salts endavant i enrere | iris | | represa en ~0,6 s a Kodi; amb `seektest`, ~55 ms (1080p) i ~130 ms (2160p) |
| Canvi de vídeo i sortida de Kodi | iris | | net; iris continua funcionant |

`tests/run-matrix.sh`: 27/27 (H.264, HEVC i VP9 de 360p a 2160p, NV12 i DRM PRIME, canvi de
resolució i refús del 10 bits en obrir).

## Limitacions conegudes

- **El renderitzat EGL no funciona.** iris dona NV12 lineal (`DRM_FORMAT_MOD_LINEAR`). Mesa
  només l'importa si el modificador lineal s'hi passa explícitament. Kodi no el passa
  quan és lineal, i tots els fotogrames fallen amb `0x3003` (EGL_BAD_ALLOC). Cal fer
  servir **Direct to plane**.
- **4K en una pantalla de 1080p.** Amb Direct to plane, el controlador de pantalla rebutja
  un pla de 3840×2160 reduït a 1080p (`DrmAtomicCommit - test commit failed: No such
  device` a cada fotograma), i no es veu imatge. Una pantalla que treballi a 2160p no ha
  de reduir. Encara no s'ha provat.
- **El 10 bits** no el descodifica iris (el controlador no ofereix cap format de captura de
  més de 8 bits) i passa a programari. El 10 bits 4K per programari no és fluid.
- **El VP9 de 10 bits** probablement dona pantalla negra. El VP9 no porta informació de bits
  en obrir, i el pegat `0004` només cobreix HEVC.
- **8K**: iris el refusa ("current session not supported"). Per programari arriba a ~32 fps.
- `Player.Open` per JSON-RPC respon `Invalid params` amb fitxers nous dins d'una carpeta que
  Kodi ja ha llistat (memòria cau de directoris). Cal fer servir una carpeta nova.

## Desinstal·lació

```bash
rm -rf /opt/ffmpeg-iris
```

I torna a desactivar les dues opcions de DRM PRIME a Kodi.

## Actualització

L'FFmpeg de `/opt` ha de mantenir la mateixa `LIBAVCODEC_VERSION_MAJOR/MINOR` que el
`libavcodec62` contra el qual està compilat Kodi (62.11 en el moment d'escriure això):

```bash
git -C ~/src/ffmpeg-iris fetch origin
git -C ~/src/ffmpeg-iris show origin/test/8.0.x/main:libavcodec/version.h | grep VERSION_MINOR
git -C ~/src/ffmpeg-iris rebase --onto origin/test/8.0.x/main 2a287060cc6e iris
./build.sh && tests/abi-check.sh && tests/run-matrix.sh
```

## Llicència

Els pegats de `patches/` modifiquen l'FFmpeg i estan sota la seva llicència
(LGPL-2.1-or-later). La compilació activa `--enable-gpl`, així que les biblioteques
resultants són GPL-2.0-or-later.
