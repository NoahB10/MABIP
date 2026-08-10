#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mabip
export QT_QPA_PLATFORM=xcb
cd "$SCRIPT_DIR"
exec python gui_async.py "$@"
