#!/bin/sh
# Undo lab-setup.sh. Run with sudo.
rm -f /etc/sudoers.d/qcs6490-hifi /usr/local/sbin/qcs6490-hifi-param
gpasswd -d "${SUDO_USER:-radxa}" adm || true
systemctl enable --now shairport-sync
