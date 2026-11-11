#!/bin/sh
# Remove the trace service and re-enable the route fix. Run with sudo.
systemctl disable q6a-route-trace.service
rm -f /etc/systemd/system/q6a-route-trace.service /usr/local/sbin/q6a-route-trace
systemctl daemon-reload
systemctl enable --now q6a-audio-route.service
