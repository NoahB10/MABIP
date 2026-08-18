"""PyInstaller runtime hook for MABIP. Runs before any app code."""
import os
import sys

# The conda Qt build has no wayland plugin; the Pi desktop is labwc/Xwayland.
# setdefault (not assignment) so the selftest can force "offscreen".
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# The vendored hardware tree (dual_syringe, chemyx_pump, fluigent_sensor and
# the Fluigent SDK) is shipped as plain data files, never frozen into the PYZ:
# Fluigent/SDK/low_level.py resolves its native .so with
# pkg_resources.resource_filename, which needs the package to be REAL files on
# disk with the shared/linux/arm64/ layout intact.
_hw = os.path.join(sys._MEIPASS, "hardware")
if os.path.isdir(_hw) and _hw not in sys.path:
    sys.path.insert(0, _hw)
