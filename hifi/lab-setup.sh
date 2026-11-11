#!/bin/sh
# Temporary test-bench setup for the qcs6490-hifi investigation. Run with sudo.
#  - the invoking user in group adm: lets the assistant read kernel logs (journalctl -k)
#  - /usr/local/sbin/qcs6490-hifi-param + sudoers rule: change only the two
#    module parameters (whitelisted values) without a password
#  - shairport-sync stopped/disabled so nothing opens the headphones in S24 at boot
# Undo with lab-teardown.sh.
set -e
u="${SUDO_USER:-radxa}"
usermod -aG adm "$u"
cat > /usr/local/sbin/qcs6490-hifi-param <<'EOS'
#!/bin/sh
case "$1" in
rx_be_format) case "$2" in 0|16|24|32) ;; *) echo "bad value" >&2; exit 2;; esac ;;
rx_be_rate) case "$2" in 0|44100|48000|88200|96000|176400|192000) ;; *) echo "bad value" >&2; exit 2;; esac ;;
*) echo "usage: $0 rx_be_format|rx_be_rate VALUE" >&2; exit 2 ;;
esac
echo "$2" > "/sys/module/snd_soc_sc8280xp/parameters/$1"
EOS
chmod 0755 /usr/local/sbin/qcs6490-hifi-param
echo "$u ALL=(root) NOPASSWD: /usr/local/sbin/qcs6490-hifi-param" > /etc/sudoers.d/qcs6490-hifi
chmod 0440 /etc/sudoers.d/qcs6490-hifi
visudo -cf /etc/sudoers.d/qcs6490-hifi
systemctl disable --now shairport-sync
echo "Banc de proves a punt."
