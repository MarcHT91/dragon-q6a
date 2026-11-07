# kodi-iris

Descodificació de vídeo per maquinari a Kodi a la Radxa Dragon Q6A (QCS6490), fent servir
el descodificador **iris** del kernel (`/dev/video0`, `qcom_iris`) a través de V4L2 mem2mem.

El Kodi d'Ubuntu (21.3) no pot fer servir iris amb el seu FFmpeg. Aquest projecte compila
[rpi-ffmpeg](https://github.com/jc-kynesim/rpi-ffmpeg) 8.0.1, que és compatible a nivell
d'ABI amb el `libavcodec62` d'Ubuntu, amb unes quantes correccions per a iris. L'instal·la a
`/opt/ffmpeg-iris`, i Kodi el carrega amb `LD_LIBRARY_PATH`. No es toca cap paquet del
sistema.

- Codi de l'FFmpeg: `~/src/ffmpeg-iris`, branca `iris` (sobre `2a287060cc6e` de
  `test/8.0.1/main`).
- Aquest repositori: script de compilació, llançador i proves.

## Requisits

```bash
sudo apt install -y libdrm-dev libdav1d-dev libvpx-dev libxml2-dev libzvbi-dev \
    libbluray-dev libmp3lame-dev libudev-dev
sudo install -d -o radxa -g radxa /opt/ffmpeg-iris
```

L'usuari ha de ser als grups `video` i `render`, i també a `audio` i `input` si Kodi
s'obre per SSH.

## Compilació

```bash
./build.sh            # configura (només el primer cop), compila i instal·la a /opt/ffmpeg-iris
RECONF=1 ./build.sh   # força tornar a configurar
```

Variables: `SRC` (per defecte `~/src/ffmpeg-iris`), `PREFIX` (`/opt/ffmpeg-iris`) i
`BUILD` (`$SRC/out/build`).

## Ús amb Kodi

1. Llançador: `/opt/ffmpeg-iris/bin/kodi-iris [fitxer]`. Obre `kodi` amb les
   biblioteques de `/opt`. El script `kodi` d'Ubuntu talla els arguments pels espais, així
   que cal passar rutes sense espais (o un enllaç simbòlic).
2. Configuració (*Paràmetres → Reproductor → Vídeos*, mode Expert):
   - **Permet usar el descodificador DRM PRIME**: activat (`videoplayer.useprimedecoder`).
   - **Permet l'acceleració per maquinari amb DRM PRIME**: activat
     (`videoplayer.useprimedecoderforhw`). Cal tenir-lo activat: si no, Kodi tria el
     descodificador per programari.
   - **Mètode de renderitzat PRIME**: **Direct to plane** (`videoplayer.useprimerenderer`
     = `0`). EGL no funciona (vegeu les limitacions).

   Si s'edita `~/.kodi/userdata/guisettings.xml` a mà amb Kodi tancat, cal treure
   l'atribut `default="true"` de la línia, perquè si hi és Kodi no fa cas del valor.
3. Per tancar Kodi, fer servir `tests/kodi-stop.sh`. Atura la reproducció per JSON-RPC,
   espera que s'alliberi `/dev/video0`, surt i, com a últim recurs, fa `kill -9` (el
   servei de The Crew a vegades penja la sortida).

Si iris es queda penjat (qualsevol programa rep `VIDIOC_REQBUFS failed: Invalid argument`
i ningú no té `/dev/video0` obert), es recupera així:

```bash
sudo modprobe -r qcom_iris && sudo modprobe qcom_iris
```

## Proves

| Script | Què fa |
|---|---|
| `tests/abi-check.sh` | Comprova que tots els símbols d'FFmpeg que importa `kodi.bin` existeixen a `/opt/ffmpeg-iris`. |
| `tests/gen-media.sh` | Genera els vídeos sintètics de `tests/media/` (H.264, HEVC i VP9 de 360p a 2160p, canvi de resolució, HEVC 10 bits). |
| `tests/run-matrix.sh` | Compara fotograma a fotograma (framemd5) el descodificador per programari amb iris, tant en NV12 com en DRM PRIME (`tests/bin/drmframemd5`). El 10 bits ha de ser refusat en obrir. Surt amb 0 si tot passa. |
| `tests/bin/seektest FITXER DEC N` | Fa N salts tal com els fa Kodi (buidatge amb un paquet NULL, `avcodec_flush_buffers`, salt) i mesura el temps fins al primer fotograma. |
| `tests/bin/primefallback FITXER 0\|1` | Comprova que el descodificador HEVC per programari refusa els fluxos de més de 8 bits quan el camí PRIME de Kodi li demana `DRM_PRIME`. |
| `tests/kodi-check.sh` | Amb Kodi reproduint: biblioteques de `/opt`, `/dev/video0` obert, descodificador V4L2 i CPU < 30 %. |
| `tests/kodi-seek.sh [N]` | Amb Kodi reproduint: N salts per JSON-RPC i temps de represa (falla si és > 3 s). |
| `tests/eglprobe.c` | Diagnòstic: mostra el descriptor DRM d'iris i prova d'importar-lo a EGL. |

Els programes en C es compilen contra `/opt/ffmpeg-iris`. Per exemple:

```bash
gcc -Wall -O1 tests/seektest.c -I/opt/ffmpeg-iris/include -L/opt/ffmpeg-iris/lib \
    -Wl,-rpath,/opt/ffmpeg-iris/lib -lavformat -lavcodec -lavutil -o tests/bin/seektest
```

Els clips HEVC 8 bits "reals" (`tests/media/real_hevc_*.mkv`) es van fer a partir d'un
capítol en H.264, perquè no hi ha cap fitxer HEVC de 8 bits a la biblioteca:

```bash
ffmpeg -ss 120 -t 60 -i CAPITOL.mkv -map 0:v:0 -map 0:a:0 -c:v libx265 -preset ultrafast \
    -pix_fmt yuv420p -c:a copy tests/media/real_hevc_1080p.mkv
# 4K: el mateix amb -t 30 -vf scale=3840:2160
```

## Correccions a ffmpeg-iris

Branca `iris`, sobre `2a287060cc6e`:

| Commit | Problema |
|---|---|
| `6979743670` | iris retorna buffers de captura buits marcats amb error: es descarten en lloc de donar-los com a fotogrames. |
| `148088d7f1` | Si la captura no es pot tornar a engegar després d'un canvi de resolució, falla de seguida en lloc de quedar-se penjat. |
| `bec47a0b52` | iris no té cap format de captura de més de 8 bits: els fluxos de 10 bits es refusen en obrir (quirk `FF_V4L2_QUIRK_8BIT_ONLY`). |
| `6c54aafb59` | El camí DRM PRIME de Kodi torna a provar amb el descodificador per programari, que només admet 8 bits, i el 10 bits donava pantalla negra. Ara `hevcdec` refusa el 10 bits en obrir si se li demana `DRM_PRIME`, i Kodi passa al seu descodificador per programari normal. |
| `199288807b` | Després d'un buidatge (Kodi en fa un a cada salt), iris no torna a descodificar mai amb la mateixa sessió, i la sessió encallada podia penjar el controlador. Ara, per a iris, el flush tanca el descodificador i n'obre un de nou (quirk `FF_V4L2_QUIRK_REOPEN_AFTER_DRAIN`). |

## Resultats (2026-11-07, Dell U2413 a 1920×1200, Direct to plane)

| Contingut | Descodificador | CPU de Kodi | Notes |
|---|---|---|---|
| H.264 1080p (capítol real) | iris | ~15 % | per programari, 100–150 %; imatge correcta |
| HEVC 8 bits 1080p | iris | ~15 % | |
| HEVC 8 bits 2160p | iris | ~16 % | es descodifica bé, però la pantalla rebutja cada fotograma (vegeu les limitacions) |
| HEVC 10 bits 1080p | programari | ~98 % (1 nucli) | velocitat real, 0 errors |
| Salts (10 endavant i enrere) | iris | | represa en ~0,6 s a Kodi; amb `seektest`, ~55 ms (1080p) i ~130 ms (2160p) |
| Canvi de capítol i sortida de Kodi | iris | | sortida neta en ~6 s; iris continua funcionant |

La matriu (`tests/run-matrix.sh`) passa 27/27: H.264, HEVC i VP9 de 360p a 2160p en NV12 i
en DRM PRIME, canvi de resolució i refús del 10 bits.

## Limitacions conegudes

- **Renderitzat EGL**: no funciona. Iris dona NV12 lineal amb `DRM_FORMAT_MOD_LINEAR`, i Mesa
  (26.0.8, freedreno) només l'importa si el modificador lineal s'hi passa explícitament.
  Kodi no el passa quan és lineal, i tots els fotogrames fallen amb `0x3003`
  (EGL_BAD_ALLOC). Cal fer servir **Direct to plane**.
- **4K en una pantalla de 1080p/1200p**: amb Direct to plane, la pantalla no pot reduir
  una font de 3840×2160 (`DrmAtomicCommit - test commit failed: No such device` a cada
  fotograma). En una tele 4K que treballi a 2160p no cal reduir; **falta provar-ho**.
- **10 bits**: iris no el descodifica (el controlador no té format de captura de més de 8
  bits) i passa a programari. El 10 bits 4K per programari no és fluid.
- **VP9 de 10 bits**: probablement dona pantalla negra. El VP9 no porta prou informació en
  obrir per saber els bits, i la correcció `6c54aafb59` només cobreix HEVC. Encara no s'ha
  vist en fitxers reals.
- **8K**: iris el refusa ("current session not supported") i per programari va a ~32 fps.
- **Parpelleig del menú al U2413**: el menú de Kodi va a 1920×1200@60, el mode en què
  aquest monitor ja perdia el senyal abans d'aquest projecte. El vídeo (1080p23.976) no
  parpelleja. Falta comprovar-ho a la tele.
- `Player.Open` per JSON-RPC respon `Invalid params` amb fitxers nous dins d'una carpeta
  que Kodi ja ha llistat (memòria cau de directoris). Cal fer servir una carpeta nova.

## Com desfer-ho

```bash
rm -rf /opt/ffmpeg-iris
```

I a Kodi, desactivar les dues opcions de DRM PRIME (o restaurar
`~/.kodi/userdata/guisettings.xml.bak-prime-*`). El `kodi` normal continua fent servir
l'FFmpeg del sistema.

## Com actualitzar

L'FFmpeg de `/opt` ha de tenir sempre la mateixa versió `LIBAVCODEC_VERSION_MAJOR/MINOR`
que el `libavcodec62` d'Ubuntu contra el qual està compilat Kodi (ara 62.11):

```bash
git -C ~/src/ffmpeg-iris fetch origin
# triar una branca test/8.0.x/main nova i comprovar la versió:
git -C ~/src/ffmpeg-iris show origin/test/8.0.x/main:libavcodec/version.h | grep VERSION_MINOR
git -C ~/src/ffmpeg-iris rebase --onto origin/test/8.0.x/main 2a287060cc6e iris
./build.sh && tests/abi-check.sh && tests/run-matrix.sh
```
