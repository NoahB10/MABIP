# MABIP standalone bundle — smoke test checklist

Run after `install.sh` on any Pi (including this one) before trusting the
build for real experiments. Items marked **[SAFETY]** are release-blocking.

## Scripted (already run by install.sh / build.sh)
- [ ] `/opt/mabip/mabip-selftest` prints `SELFTEST OK` — covers the Flow
      Control tab import chain (its failure is otherwise SILENT: the GUI
      starts and the tab just disappears), the Fluigent native lib, PyBluez,
      serial, qasync, and the `~/MABIP_Data` redirect.

## First launch
- [ ] Double-click the desktop icon. PCManFM may ask "Execute?" the first
      time — answer "Execute" (normal on Raspberry Pi OS).
- [ ] GUI opens; **Flow Control tab is present** in the tab bar.
- [ ] Double-click the icon again while running → "MABIP is already running"
      dialog (single-instance lock).
- [ ] A new log appears in `~/.mabip/logs/`.

## Hardware
- [ ] Pump: serial ports enumerate (`/dev/ttyUSB*`); pump connects.
      If ports are missing, replug USB (udev rule) and confirm you re-logged
      in after install (group membership).
- [ ] Fluigent flow sensor connects (fgt_init) and shows a live flow reading.
- [ ] AMUZA: discovery finds FC90-XXXX and connects. First time on a new Pi:
      pair + trust via `bluetoothctl` (install.sh prints the steps).

## Data
- [ ] Start a short run: CSVs appear under `~/MABIP_Data/Sensor_Readings/`
      and logs under `~/MABIP_Data/Amuza_Logs/` (NOT inside /opt/mabip).
- [ ] Settings persist across restarts (`~/.mabip/settings.json`).

## Shutdown [SAFETY]
- [ ] With the pump running, close the window (red ✕): pump STOPS, app exits
      cleanly.
- [ ] With the pump running, `kill -TERM <pid of MABIP>`: pump STOPS, app
      exits cleanly. (Verifies signal handlers survived freezing.)

## Known quirks
- If a dev checkout exists at `/home/rpi/pumpcontrol-project` on the target
  machine, it shadows the bundled hardware drivers (dev override by design).
- The app runs via Xwayland (`QT_QPA_PLATFORM=xcb`); this is set by the
  bundle itself and by the launcher.
