#!/bin/bash
# Build the standalone MABIP bundle (PyInstaller onedir) and gate-check it.
# Memory-hungry (~1+ GB peak on analysis): run when the MABIP GUI is NOT
# running, ideally with VS Code / browser closed.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/pi/miniconda3/envs/mabip/bin

if pgrep -f "gui_async.py" >/dev/null; then
    echo "MABIP GUI is running — close it before building."; exit 1
fi
avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
[ "$avail" -lt 1200 ] && echo "WARNING: only ${avail}MB available; the build may swap. Closing VS Code/browser speeds it up."

nice -n 19 ionice -c3 "$PY/pyinstaller" --noconfirm --clean --log-level WARN mabip.spec

echo "== Gate checks =="
D=dist/MABIP/_internal
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  OK  $1"; else echo "FAIL  $1"; fail=1; fi; }
chk "GUI exe"                "test -f dist/MABIP/MABIP"
chk "selftest exe"           "test -f dist/MABIP/mabip-selftest"
chk "qt xcb platform plugin" "find $D -path '*platforms/libqxcb.so' | grep -q ."
chk "xcbglintegrations"      "find $D -type d -name xcbglintegrations | grep -q ."
chk "fluigent native .so"    "test -f $D/hardware/Fluigent/SDK/shared/linux/arm64/libfgt_SDK.so"
chk "dual_syringe.py"        "test -f $D/hardware/dual_syringe.py"
chk "burst_calibration.json" "test -f $D/burst_calibration.json"
chk "libbluetooth traced"    "find $D -name 'libbluetooth.so*' | grep -q ."

echo "== Selftest (headless) =="
dist/MABIP/mabip-selftest || fail=1

if [ "$fail" -eq 0 ]; then
    echo "BUILD + GATES OK: dist/MABIP ($(du -sh dist/MABIP | cut -f1))"
else
    echo "BUILD GATES FAILED — see above."; exit 1
fi
