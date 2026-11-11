#!/bin/sh
# Install the Dragon Q6A tuning. Run with sudo, then reboot.
#  - CPU governor schedutil (needs the cpufrequtils package)
#  - RPS for the onboard Ethernet
#  - zram swap (needs systemd-zram-generator)
#  - writeback limits and zram-friendly VM settings
set -e
cd "$(dirname "$0")"
install -m 0644 cpufrequtils /etc/default/cpufrequtils
install -m 0644 99-rps-enp1s0.rules /etc/udev/rules.d/99-rps-enp1s0.rules
install -m 0644 zram-generator.conf /etc/systemd/zram-generator.conf
install -m 0644 90-dragon-q6a.conf /etc/sysctl.d/90-dragon-q6a.conf
sysctl -q --system
echo "Done. Reboot to apply the governor, RPS and zram."
