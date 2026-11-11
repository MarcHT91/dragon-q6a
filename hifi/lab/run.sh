#!/bin/sh
# run.sh FE_RATE FE_FMT BE_FMT BE_RATE  -> plays 4 s tone on hw:0,0, reports
fr=$1 ff=$2 bf=$3 br=$4
here=$(dirname "$0"); tmp=${TMPDIR:-/tmp}/hifi-tone-$fr-$ff.raw
[ -f "$tmp" ] || python3 -I "$here/tone.py" "$fr" "$ff" 4 "$tmp"
sudo -n /usr/local/sbin/qcs6490-hifi-param rx_be_format "$bf" || exit 1
sudo -n /usr/local/sbin/qcs6490-hifi-param rx_be_rate "$br" || exit 1
since=$(date '+%Y-%m-%d %H:%M:%S')
echo "== FE $ff@$fr -> BE fmt=$bf rate=$br"
aplay -q -D hw:CARD=QCS6490RadxaDra,DEV=0 -f "$ff" -r "$fr" -c2 "$tmp" 2>&1 &
p=$!
for i in 1 2 3; do
	sleep 1
	st=/proc/asound/card0/pcm0p/sub0
	if [ -e $st/status ]; then
		echo "  t=$i $(grep -E 'state' $st/status) $(grep hw_ptr $st/status | tr -s ' ') $(grep -E '^(format|rate):' $st/hw_params 2>/dev/null | tr '\n' ' ')"
	else
		echo "  t=$i NO CARD"
	fi
done
wait $p; echo "  aplay exit=$?"
journalctl -k --since "$since" --no-pager -o cat | grep -iE 'qcs6490-hifi|timeout|error|crash|fail|PDR' | head -15 | sed 's/^/  log: /'
grep -q QCS6490 /proc/asound/cards && echo "  card: OK" || echo "  card: GONE (reboot needed)"
