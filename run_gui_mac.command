#!/bin/bash
# MABIP launcher for macOS (double-clickable from Finder).
#
# run_gui.sh is the Pi's launcher: it activates the conda env `mabip` and forces
# QT_QPA_PLATFORM=xcb, neither of which exists here. This one uses the local
# .venv and lets Qt pick its native cocoa backend.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PY="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "No virtualenv at .venv — create it first:" >&2
    echo "    uv venv --python 3.11 .venv" >&2
    echo "    uv pip install --python .venv/bin/python -r requirements.txt" >&2
    exit 1
fi

LOG_DIR="$HOME/.mabip/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gui_$(date +%Y%m%d_%H%M%S).log"
# keep only the 15 most recent logs, same as launch_mabip.sh on the Pi
ls -1t "$LOG_DIR"/gui_*.log 2>/dev/null | tail -n +16 | xargs rm -f 2>/dev/null

"$PY" gui_async.py "$@" 2>&1 | tee "$LOG"
