#!/bin/sh
# Install the boot-time route service. Run with sudo.
set -e
cd "$(dirname "$0")"
install -m 0755 q6a-audio-route /usr/local/sbin/q6a-audio-route
install -m 0644 q6a-audio-route.service /etc/systemd/system/q6a-audio-route.service
systemctl daemon-reload
systemctl enable --now q6a-audio-route.service
