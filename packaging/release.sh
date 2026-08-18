#!/bin/bash
# Stage the built bundle + installer collateral into a shippable tarball.
set -euo pipefail
cd "$(dirname "$0")"
[ -d dist/MABIP ] || { echo "No dist/MABIP — run build.sh first."; exit 1; }

OUT="mabip-$(date +%Y%m%d).tar.gz"
rm -rf stage/mabip-dist
mkdir -p stage/mabip-dist
cp -a dist/MABIP stage/mabip-dist/
echo "MABIP $(date +%Y.%m.%d) ($(git -C .. rev-parse --short HEAD 2>/dev/null || echo unversioned))" > stage/mabip-dist/MABIP/VERSION
cp mabip.sh install.sh mabip.desktop mabip.png SMOKE_TEST.md stage/mabip-dist/
# Ship the live udev rule from this Pi (falls back to the repo copy; they are
# byte-identical today).
cp /etc/udev/rules.d/99-fluigent.rules stage/mabip-dist/ 2>/dev/null \
    || cp ../hardware/linux-fluigent-udev.rules stage/mabip-dist/99-fluigent.rules
chmod +x stage/mabip-dist/install.sh stage/mabip-dist/mabip.sh
tar -C stage -czf "$OUT" mabip-dist
du -sh "$OUT"
echo "Ship $OUT; on the target Pi: tar xzf $OUT && cd mabip-dist && sudo ./install.sh"
