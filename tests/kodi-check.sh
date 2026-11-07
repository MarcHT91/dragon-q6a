#!/bin/bash
# Comprova un Kodi en reproducció: biblioteques de /opt, /dev/video0, descodificador al log, CPU.
set -uo pipefail
P=${PREFIX:-/opt/ffmpeg-iris}; LOG=$HOME/.kodi/temp/kodi.log; rc=0
pid=$(pgrep -n kodi.bin) || { echo "FAIL: Kodi no està en marxa"; exit 1; }
if grep -q "$P/lib/libavcodec" /proc/$pid/maps; then echo "OK   libs: $P"; else echo "FAIL libs: Kodi no usa $P"; rc=1; fi
if ls -l /proc/$pid/fd 2>/dev/null | grep -q /dev/video0; then echo "OK   video0 obert"; else echo "FAIL video0 no obert"; rc=1; fi
dec=$(grep -oE 'CDVDVideoCodecDRMPRIME::Open - using decoder [^ ]+' "$LOG" | tail -1)
case "$dec" in *v4l2m2m*) echo "OK   $dec";; *) echo "FAIL decoder: '${dec:-cap}'"; rc=1;; esac
cpu=$(top -bn2 -d5 -p $pid | awk -v p=$pid '$1==p {c=$9} END {print c}')
if awk -v c="$cpu" 'BEGIN{exit !(c<30)}'; then echo "OK   cpu=${cpu}%"; else echo "FAIL cpu=${cpu}% (>=30)"; rc=1; fi
errs=$(grep -cE 'DRMPRIME.*(failed|error)' "$LOG")
echo "INFO errors DRMPRIME al log: $errs"
exit $rc
