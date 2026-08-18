#!/bin/bash
# MABIP desktop launcher: conda env, xcb, logging, single-instance, visible errors.
# Dev runs can keep using run_gui.sh; this is what the desktop icon invokes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HOME/.mabip/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/gui_$(date +%Y%m%d_%H%M%S).log"
# keep only the 15 most recent logs
ls -1t "$LOG_DIR"/gui_*.log 2>/dev/null | tail -n +16 | xargs -r rm -f

fail() {
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --width=500 --title="MABIP failed to start" \
            --text="$1\n\nLog: $LOG\n\n$(tail -n 15 "$LOG" 2>/dev/null)" &
    fi
    exit 1
}

# Single instance: two GUIs would fight over the pump serial port and the
# AMUZA bluetooth socket. Lock FD is held by this shell (and inherited by
# python) for the app's lifetime.
exec 200>"$HOME/.mabip/gui.lock"
flock -n 200 || fail "MABIP is already running."

source "$HOME/miniconda3/etc/profile.d/conda.sh" >>"$LOG" 2>&1 \
    || fail "Could not load conda."
conda activate mabip >>"$LOG" 2>&1 || fail "Could not activate the mabip env."
export QT_QPA_PLATFORM=xcb
cd "$SCRIPT_DIR" || fail "App directory missing."
python gui_async.py >>"$LOG" 2>&1
rc=$?
# 130/143 = SIGINT/SIGTERM, which the app handles as a clean shutdown.
[ $rc -ne 0 ] && [ $rc -ne 130 ] && [ $rc -ne 143 ] && fail "MABIP exited with error (code $rc)."
exit $rc
