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
