# MABIP — testing / development copy

Instrument-control GUI for the AMUZA microsampler (Bluetooth RFCOMM) with
Fluigent flow sensing, Chemyx pump control, and potentiostat serial readout.
PyQt5 + qasync, Python 3.11, conda env `mabip`.

**This checkout is the TESTING version** — make changes here, run them with
the dev launcher, and when a version is good, build + install it as the
production app. The **production version** users launch from the desktop icon
is a standalone bundle installed at `/opt/mabip` (no Python/conda needed);
see `packaging/`.

## Layout

| Path | What |
|---|---|
| `gui_async.py` | GUI entry point |
| `config.py` | All settings (hardware IDs, timings, file locations) |
| `amuza_async.py`, `sensor_reader_async.py`, `flow_control_tab.py` | Device + UI modules |
| `mac_ui.py` | macOS-only font/toolbar/window fixes (no-op on the Pi) |
| `hardware/` | Vendored drivers: Fluigent SDK (native `.so`), chemyx_pump, dual_syringe |
| `experiments/` | Experiment definition files (tracked) |
| `packaging/` | Standalone-build tooling: `build.sh` → `release.sh` → `install.sh` |
| `test_*.py` | Unit tests (`pytest`) |
| `Sensor_Readings/`, `Amuza_Logs/`, `exports/` | Data written by TESTING runs (git-ignored) |

The production app writes its data to `~/MABIP_Data` instead, so testing and
production runs never mix files.

## Run (testing)

```bash
./run_gui.sh            # terminal, or
./launch_mabip.sh       # what the "MABIP Testing" desktop icon runs
                        # (logs to ~/.mabip/logs, single-instance lock)
```

Only one GUI can run at a time — testing and production share a lock on
purpose, because they'd otherwise fight over the pump serial port and the
AMUZA bluetooth socket.

## Run on a Mac (development only)

The rig is a Pi; a Mac is for editing the code, running the tests, and seeing
the UI. Two devices cannot be reached from macOS, so the app degrades instead
of failing:

- **AMUZA** — PyBluez has no working macOS build, so `connect()` falls back to
  a simulated socket. The status line then reads `AMUZA: Machine 1
  [SIMULATED]`; every command is swallowed. Never read a green status on a Mac
  as a live rig.
- **Fluigent flow sensor** — the vendored SDK ships an x86-64 `.dylib` only, so
  it will not load against an arm64 Python. Flow Control opens, plots nothing.

The potentiostat serial readout works over any USB-serial adapter, and every
other tab is fully functional.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./run_gui_mac.command          # or double-click it in Finder
```

`run_gui_mac.command` exists because `run_gui.sh` activates the conda env
`mabip` and forces `QT_QPA_PLATFORM=xcb` — both Pi-only. Cocoa also differs
from the Pi's X11 in font size (13 pt vs ~10 pt), toolbar icon size (32 px vs
24 px) and pixel ratio; `mac_ui.py` corrects those so the same layouts fit
their window. It is a no-op off Darwin, so nothing here changes the Pi.

## Ship a new production version

```bash
packaging/build.sh      # PyInstaller bundle + gate checks + selftest
packaging/release.sh    # -> packaging/mabip-YYYYMMDD.tar.gz
# this Pi:      cd packaging/stage/mabip-dist && sudo ./install.sh
# another Pi:   copy the tarball, tar xzf, cd mabip-dist, sudo ./install.sh
```

`install.sh` handles the udev rule, group membership, bluetooth checks, the
desktop icon, and runs a self-test. `packaging/SMOKE_TEST.md` is the manual
checklist before trusting a build for real experiments.

## One-time machine setup (already done on this Pi)

conda env `mabip` (Python 3.11, `pip install -r requirements.txt`, keep
`setuptools<81` for the Fluigent SDK), udev rule from
`hardware/linux-fluigent-udev.rules`, user in `plugdev` + `dialout`, AMUZA
paired via bluetoothctl. Details: `hardware/LOCAL_SETUP_TD.md`.
