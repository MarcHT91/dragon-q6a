#!/bin/sh
# listen.sh N FE_RATE FE_FMT BE_FMT BE_RATE
n=$1 fr=$2 ff=$3 bf=$4 br=$5
f=${TMPDIR:-/tmp}/listen-$n.raw
python3 -I "$(dirname "$0")/listen.py" "$n" "$fr" "$ff" "$f"
sudo -n /usr/local/sbin/qcs6490-hifi-param rx_be_format "$bf"
sudo -n /usr/local/sbin/qcs6490-hifi-param rx_be_rate "$br"
echo "Prova $n: FE $ff@$fr -> BE S$bf@$br"
aplay -q -D hw:CARD=QCS6490RadxaDra,DEV=0 -f "$ff" -r "$fr" -c2 "$f"
