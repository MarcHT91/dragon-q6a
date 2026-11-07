#!/bin/bash
# kodi-seek.sh [N]: fa N salts (alternant endavant/enrere) al vídeo en reproducció i mesura
# quant triga a tornar a avançar el temps. Falla si algun salt triga > 3 s o queda aturat.
N=${1:-10}; LOG=$HOME/.kodi/temp/kodi.log
rpc() { curl -s -m 5 -H 'Content-Type: application/json' -d "$1" http://127.0.0.1:8080/jsonrpc; }
pid=$(rpc '{"jsonrpc":"2.0","id":1,"method":"Player.GetActivePlayers"}' | grep -oE '"playerid":[0-9]+' | cut -d: -f2)
[ -n "$pid" ] || { echo "FAIL: no hi ha res en reproducció"; exit 1; }
ms() { rpc '{"jsonrpc":"2.0","id":1,"method":"Player.GetProperties","params":{"playerid":'$pid',"properties":["time"]}}' |
       python3 -c 'import json,sys; t=json.load(sys.stdin)["result"]["time"]; print(((t["hours"]*60+t["minutes"])*60+t["seconds"])*1000+t["milliseconds"])'; }
L0=$(wc -l < "$LOG"); rc=0
for i in $(seq 1 "$N"); do
  if [ $((i % 2)) = 1 ]; then s=120; else s=-60; fi
  rpc '{"jsonrpc":"2.0","id":1,"method":"Player.Seek","params":{"playerid":'$pid',"value":{"seconds":'$s'}}}' >/dev/null
  t0=$(date +%s%N); a=$(ms); ok=0
  while [ $(( ($(date +%s%N) - t0) / 1000000 )) -lt 10000 ]; do
    sleep 0.2; b=$(ms)
    if [ "$b" -gt "$((a + 300))" ]; then ok=1; break; fi
  done
  dt=$(( ($(date +%s%N) - t0) / 1000000 ))
  if [ $ok = 1 ] && [ $dt -le 3000 ]; then r=OK; else r=FAIL; rc=1; fi
  printf '%-4s salt %2d (%+4ds) -> %ds, reprèn en %d ms\n' $r $i $s $((a / 1000)) $dt
done
echo "INFO errors DRMPRIME nous: $(tail -n +$L0 "$LOG" | grep -cE 'DRMPRIME.*(failed|error)')"
tail -n +$L0 "$LOG" | grep -E 'DRMPRIME.*(failed|error)' | cut -c30- | sed 's/(-\?[0-9]*)//' | sort | uniq -c
exit $rc
