# dragon-q6a-audio

[English](README.md)

Arranjaments d'àudio per a la **Radxa Dragon Q6A** (Qualcomm QCS6490), amb Ubuntu 26.04.1 LTS (imatge de
rafaelwms) i el kernel edge d'Armbian `7.2.9-edge-qcs6490`.
Inclou la sortida d'auriculars a 24 bits, unes rutes del DSP fiables per al jack d'auriculars i
per a l'HDMI, un `alsamixer` que es pot fer servir i un receptor AirPlay 2 (shairport-sync) amb
volum de maquinari.

| Directori | Què fa |
|---|---|
| [`hifi/`](hifi) | Mòdul DKMS amb dues correccions del kernel i una topologia del DSP modificada. La sortida d'auriculars treballa en **S24_LE / 48 kHz** en lloc de S16. Inclou el banc de proves i els pedaços preparats per enviar als desenvolupadors del kernel. |
| [`route/`](route) | Per què es perdia la ruta dels auriculars en arrencar (error del kernel i seqüència UCM), i un servei que activa les dues rutes en arrencar. |
| [`alsa-ctl-filter/`](alsa-ctl-filter) | Connector de controls d'ALSA. Fa que `alsamixer` mostri només els volums útils i dona a shairport-sync una escala de dB correcta per al volum del DSP. Inclou la configuració de shairport-sync. |
| [`docs/mixer-controls.md`](docs/mixer-controls.md) | Què fa cadascun dels ~200 controls del mesclador (en anglès). |

## Què hem trobat

* **El límit de 16 bits i 48 kHz és de software.** La topologia del DSP només declara S16 a 48 kHz,
  i el driver de la targeta (`sc8280xp.c`) força la sortida cap al còdec a S16 a 48 kHz. El maquinari
  admet S16, S24 i S32 a 48, 96 i 192 kHz a la sortida d'auriculars, i ho hem comprovat escoltant
  cada combinació. La sortida a 44,1 kHz es rebutja amb un error net. El DSP converteix qualsevol
  freqüència d'entrada entre 8 i 192 kHz.
* **El S24_LE feia caure el DSP.** `audioreach.c` declarava el S24_LE com a mostres de 3 bytes
  (`bits_per_sample = 24`). Amb un contenidor de 32 bits funciona. → [`hifi/upstream/0001`](hifi/upstream)
  Llavors el DSP llegeix paraules senceres de 32 bits, i per això **les dades S24_LE han de portar el
  signe estès** al byte de dalt. Les conversions d'alsa-lib (`plughw`) ho fan bé. En canvi,
  shairport-sync deixa el byte de dalt a zero en treure S24_LE, i se sent amb una distorsió total.
  Per això shairport treu S16_LE. El volum s'aplica al DSP i la sortida cap al còdec continua sent
  S24, de manera que baixar el volum no fa perdre resolució.
* **Es perdien les rutes.** A `topology.c`, escriure 0 en qualsevol control de ruta desconnectava la
  sortida, fos quin fos el flux connectat. Per això `alsactl restore` perdia l'HDMI a cada arrencada,
  i la seqüència UCM d'HDMI de la placa desconnectava els auriculars.
  → [`hifi/upstream/0002`](hifi/upstream), [`route/`](route)
* Un flux que comença amb la seva ruta desactivada no avança mai (`hw_ptr` es queda a 0). Així és
  com s'encallava shairport-sync, que a més ignorava el SIGTERM.
* **AirPlay 2** des d'un iPhone (Tidal, fins i tot en hi-res) arriba com a `AAC/48000/F24/2`. Per
  tant, a 48 kHz no cal cap conversió de freqüència.

## Instal·lació

Cal `flex bison bc libelf-dev dkms` i, a més, el paquet `linux-headers` del kernel en marxa o un
arbre del kernel preparat (vegeu [`hifi/README.md`](hifi/README.md)).

    sudo KSRC=/camí/a/linux-7.2.9 ./install-all.sh
    sudo reboot

També es pot instal·lar per parts: `hifi/install.sh`, `route/install.sh` i
`alsa-ctl-filter/install-shairport-hwvol.sh`. El mòdul DKMS es desinstal·la amb `hifi/uninstall.sh`.

Comprovació:

    aplay -D hw:CARD=QCS6490RadxaDra,DEV=0 --dump-hw-params -d1 /dev/zero | grep FORMAT   # S16_LE S24_LE
    cat /proc/asound/card0/pcm0p/sub0/hw_params                                          # mentre sona

## Relacionat

* [kodi-iris](../kodi-iris): descodificació de vídeo per maquinari a Kodi a la mateixa placa.

## Llicència

MIT ([LICENSE](LICENSE)), excepte:
* `alsa-ctl-filter/ctl_filter.c`: LGPL-2.1-or-later (és un connector d'alsa-lib).
* `hifi/upstream/*.patch` i els mòduls que compila `hifi/dkms` (a partir del codi del kernel de Linux,
  que es descarrega en compilar): GPL-2.0.
