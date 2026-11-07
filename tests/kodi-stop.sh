#!/bin/bash
# Atura Kodi sense deixar iris penjat: primer Player.Stop (allibera /dev/video0), després Quit, i kill -9 només al final.
rpc() { curl -s -m 3 -H 'Content-Type: application/json' -d "$1" http://127.0.0.1:8080/jsonrpc; echo; }
pid=$(pgrep -n kodi.bin) || exit 0
for id in $(rpc '{"jsonrpc":"2.0","id":0,"method":"Player.GetActivePlayers"}' | grep -oE '"playerid":[0-9]+' | cut -d: -f2); do
  rpc '{"jsonrpc":"2.0","id":1,"method":"Player.Stop","params":{"playerid":'"$id"'}}'
done
for i in $(seq 1 10); do ls -l /proc/$pid/fd 2>/dev/null | grep -q /dev/video0 || break; sleep 1; done
ls -l /proc/$pid/fd 2>/dev/null | grep -q /dev/video0 && echo "AVÍS: /dev/video0 encara obert"
rpc '{"jsonrpc":"2.0","id":2,"method":"Application.Quit"}'
for i in $(seq 1 10); do pgrep kodi.bin >/dev/null || exit 0; sleep 1; done
echo "Kodi penjat en sortir (The Crew?): kill -9"; pkill -9 kodi.bin
