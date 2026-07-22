# hardware/ — vendored pump & flow-sensor drivers

Self-contained copies of the flow-control backend so MABIP runs on any machine
without the Pi's `~/pumpcontrol-project` checkout:

- `chemyx_pump.py` — Chemyx syringe-pump serial driver (9600 baud, `\r`
  terminator; handles this pump's firmware quirks).
- `fluigent_sensor.py` — `FluigentSensor` wrapper around the Fluigent SDK
  (read/poll/CSV logging; degrades gracefully with no device attached).
- `dual_syringe.py` — `DualSyringeLine`: two 20 mL syringes teed into one
  sensed line; the backend `flow_control_tab.py` drives.
- `Fluigent/` — the official Fluigent Python SDK package (from
  github.com/Fluigent/fgt-SDK) with bundled native libs for Windows
  (x86/x64), Linux (x64/arm/arm64), and macOS — no pip install needed
  (the PyPI `fluigent-sdk` package is an empty placeholder; don't use it).

`refactored/flow_control_tab.py`, `headless_rig.py`, and
`timing_calibration.py` add this folder to `sys.path` automatically. On the Pi
rig, live copies in `~/pumpcontrol-project` take precedence when present — if
you change the drivers there, re-copy them here so other machines get the fix.

Linux only: install the Fluigent udev rules (`linux-udev.sh` from the fgt-SDK
repo) so `/dev/hidraw*` is accessible without root. The pump serial port is
found via `/dev/serial/by-id/` (FTDI FT232R, serial A100ZWTS on the rig);
adjust in the GUI/driver if your other system enumerates it differently.
