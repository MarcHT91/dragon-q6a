#!/bin/sh
# Arm one traced boot without the route fix. Run with sudo, then reboot.
# Afterwards: cat /var/log/q6a-route-trace.log ; undo with trace-done.sh
set -e
cd "$(dirname "$0")"
install -m 0755 q6a-route-trace /usr/local/sbin/q6a-route-trace
install -m 0644 q6a-route-trace.service /etc/systemd/system/q6a-route-trace.service
systemctl daemon-reload
systemctl enable q6a-route-trace.service
systemctl disable q6a-audio-route.service 2>/dev/null || true
echo "Armat. Reinicia: sudo reboot"
