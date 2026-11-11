# board: configuració general de la Dragon Q6A

[English](README.md)

Provat amb Ubuntu 26.04.1 LTS (imatge de rafaelwms) i el kernel edge d'Armbian `7.2.9-edge-qcs6490`.

## Acceleració de la GPU (Adreno 643)

La GPU funciona directament amb Mesa: freedreno per a GL/GLES (el renderitzador surt com a
`FD635`) i Turnip per a Vulkan 1.3 (s'anuncia com a "Adreno (TM) 643"). El node de renderitzat
és `/dev/dri/renderD128`.

Si `glxinfo`/`eglinfo` mostren **llvmpipe**, vol dir que l'usuari no pot obrir `/dev/dri`, i Mesa
passa a renderitzar per software sense avisar:

    sudo usermod -aG render,video $USER     # després torna a iniciar la sessió

Les sessions que ja estaven obertes abans del canvi de grups mantenen els grups antics.

## Ajustos de rendiment (`tuning/`)

| Fitxer | Què fa |
|---|---|
| `cpufrequtils` → `/etc/default/cpufrequtils` | Posa el governador de la CPU a `schedutil` (el kernel fa servir `ondemand` per defecte). Cal el paquet `cpufrequtils`. |
| `99-rps-enp1s0.rules` | L'Ethernet de la placa (r8169) té una sola cua de recepció, i la seva interrupció va a la CPU0. RPS reparteix el processament de recepció entre les CPU 1–6 (màscara `7e`). |
| `zram-generator.conf` | Swap comprimida a la RAM: la meitat de la RAM, com a màxim 6 GB, amb zstd. Cal `systemd-zram-generator`. |
| `90-dragon-q6a.conf` → `/etc/sysctl.d/` | Limita les escriptures pendents a 64/256 MB (evita aturades llargues amb discos USB), i posa `swappiness=100` i `page-cluster=0`, adequats per a zram. |

    sudo ./tuning/install.sh && sudo reboot

## Monitor de la GPU (`tools/gputop`)

És un monitor a l'estil de `top` per a l'Adreno. Mostra:
* la freqüència: l'actual, la mitjana i el temps a cada esglaó, a partir de `trans_stat` de devfreq;
* el percentatge d'ocupació;
* l'ús i la memòria de GPU de cada procés, a partir de l'`fdinfo` de DRM;
* les temperatures `gpuss0/1`.

    tools/gputop            # s'actualitza cada segon
    sudo tools/gputop       # també mostra processos amb capacitats especials (p. ex. Kodi)

Kodi té `CAP_WAKE_ALARM`, i per això el kernel el marca com a no inspeccionable: ni el mateix
usuari pot llegir el seu `/proc/<pid>/fdinfo`. Per això `gputop` i `nvtop` necessiten sudo per
veure'l.

La GPU s'escalfa de pressa: passa d'uns 57 °C en repòs a uns 84 °C després de 4 s a 812 MHz a
plena càrrega.

## Relacionat

* [`../audio`](../audio): àudio (auriculars a 24 bits, rutes del DSP, AirPlay).
* [`../kodi-iris`](../kodi-iris): descodificació de vídeo per maquinari a Kodi i rang de color HDMI.
