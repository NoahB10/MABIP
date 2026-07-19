# TD local-machine setup (Raspberry Pi, user `pi`)

Machine-specific setup for running MABIP flow control on this Pi. The code lives
on the **TD-amuza** branch; these are the host/env steps that live outside git.

## Hardware device map (this Pi, aarch64)
- **Chemyx syringe pump** — FTDI FT232R (VID 0403), enumerates as `/dev/ttyUSB*`.
  Auto-detected via `/dev/serial/by-id` (matches "FTDI"/"FT232"). Serial A100ZWTY.
- **Fluigent flow sensor** — USB-HID under **Microchip VID 04d8** → `/dev/hidraw0`.
  `lsusb` mislabels it "PICkit 2 Microcontroller Programmer" — that is the sensor,
  not a programmer. Read via the `Fluigent.SDK` package (HID, not serial).
- **CP2102 (VID 10c4, `ttyUSB0`)** — the "SIX" sensor; the pump auto-detect skips it.

## 1. Conda env (`mabip`)
Fluigent SDK imports the legacy `pkg_resources`, which setuptools >= 81 removed:

    conda activate mabip
    pip install "setuptools<81"

## 2. Fluigent HID udev rule (sensor readable without root)
Without this, `/dev/hidraw0` is root-only and the SDK reports 0 sensor channels.

    sudo cp hardware/linux-fluigent-udev.rules /etc/udev/rules.d/99-fluigent.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger --action=add /sys/class/hidraw/hidraw0   # or replug the sensor

Verify: `ls -l /dev/hidraw0` shows `crw-rw---- root plugdev` (user `pi` is in `plugdev`).

## 3. Launch
    ./refactored/run_gui.sh        # Flow Control is a tab in the GUI

## Notes
- The pump has intermittently dropped off USB with `disabled by hub (EMI?)` in
  dmesg — use a good/powered USB connection if it keeps disconnecting.
- Bluetooth AMUZA identity on this Pi is FC90-0037 (see config.py / AMUZA_Master.py).
