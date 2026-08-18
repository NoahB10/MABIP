#!/bin/bash
# MABIP installer for Raspberry Pi. Run from inside the extracted bundle dir:
#   tar xzf mabip-YYYYMMDD.tar.gz && cd mabip-dist && sudo ./install.sh
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST=/opt/mabip

[ "$(id -u)" -eq 0 ] || { echo "Run with sudo: sudo ./install.sh"; exit 1; }
if [ -z "${SUDO_USER:-}" ] || [ "$SUDO_USER" = root ]; then
    echo "Run via sudo from the normal user account, not a root shell."; exit 1
fi
U=$SUDO_USER
UHOME=$(getent passwd "$U" | cut -d: -f6)

echo "== Preflight =="
[ "$(uname -m)" = aarch64 ] || { echo "This bundle is aarch64-only (got $(uname -m))."; exit 1; }
. /etc/os-release; echo "OS: $PRETTY_NAME"
case "${VERSION_ID:-0}" in 12|13|14) ;; *) echo "WARNING: untested OS version (${VERSION_ID:-unknown}); continuing." ;; esac
[ -x "$SRC/MABIP/MABIP" ] || { echo "MABIP/ bundle not found next to install.sh."; exit 1; }

echo "== Install to $DEST =="
rm -rf "$DEST.new"
mkdir -p "$DEST.new"
cp -a "$SRC/MABIP/." "$DEST.new/"
cp -a "$SRC/mabip.sh" "$DEST.new/"
chmod +x "$DEST.new/mabip.sh" "$DEST.new/MABIP" "$DEST.new/mabip-selftest"
if [ -d "$DEST" ]; then rm -rf "$DEST.old"; mv "$DEST" "$DEST.old"; fi
mv "$DEST.new" "$DEST"
echo "Installed. (Previous version, if any, kept at $DEST.old for rollback.)"

echo "== udev rule (Fluigent hidraw access) =="
cp "$SRC/99-fluigent.rules" /etc/udev/rules.d/99-fluigent.rules
udevadm control --reload
udevadm trigger
echo "Rule installed. Replug the Fluigent USB if it was already connected."

echo "== Group membership =="
need_relogin=0
for g in plugdev dialout; do
    if ! id -nG "$U" | tr ' ' '\n' | grep -qx "$g"; then
        usermod -aG "$g" "$U"; echo "Added $U to $g."; need_relogin=1
    fi
done
[ "$need_relogin" -eq 1 ] && echo "NOTE: $U must log out and back in before the hardware is accessible."

echo "== Bluetooth =="
systemctl is-active --quiet bluetooth || { systemctl enable --now bluetooth; echo "Bluetooth service started."; }
if bluetoothctl devices 2>/dev/null | grep -q "FC90"; then
    echo "AMUZA (FC90-*) already known to BlueZ."
else
    echo "AMUZA not paired yet. Pair it once:"
    echo "  bluetoothctl   ->  scan on  ->  wait for FC90-XXXX  ->  pair <MAC>  ->  trust <MAC>"
fi
ldconfig -p | grep -q libbluetooth.so.3 || apt-get install -y bluez libbluetooth3

echo "== Desktop integration =="
sudo -u "$U" mkdir -p "$UHOME/.local/share/icons/hicolor/256x256/apps" \
    "$UHOME/.local/share/applications" "$UHOME/MABIP_Data"
sudo -u "$U" cp "$SRC/mabip.png" "$UHOME/.local/share/icons/hicolor/256x256/apps/mabip.png"
sudo -u "$U" cp "$SRC/mabip.desktop" "$UHOME/.local/share/applications/mabip.desktop"
if [ -d "$UHOME/Desktop" ]; then
    sudo -u "$U" cp "$SRC/mabip.desktop" "$UHOME/Desktop/mabip.desktop"
    sudo -u "$U" chmod +x "$UHOME/Desktop/mabip.desktop"
fi

echo "== Self test =="
if sudo -u "$U" "$DEST/mabip-selftest"; then
    echo
    echo "INSTALL OK — launch MABIP from the menu (Science) or the desktop icon."
    echo "Data will be written to $UHOME/MABIP_Data; logs to $UHOME/.mabip/logs."
else
    echo
    echo "INSTALL COMPLETED BUT SELF TEST FAILED — see output above."
    echo "The GUI may start with features missing (e.g. the Flow Control tab)."
    exit 1
fi
