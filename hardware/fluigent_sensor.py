"""
fluigent_sensor — read Fluigent flow sensors (Flow Units / IPS) on the Pi.

Sensor-only wrapper around the official Fluigent SDK (Fluigent.SDK), which
ships native aarch64 libraries and is officially supported on the Raspberry
Pi. We use the Fluigent purely for READINGS here — no pressure regulation.

Install notes (already done on this Pi):
  * Package copied into site-packages from github.com/Fluigent/fgt-SDK
    (do NOT `pip install fluigent-sdk` — that PyPI name is an empty placeholder).
  * udev rules installed via linux-udev.sh so the USB-HID device is reachable
    without root (VIDs 04d8 / 0483 -> /dev/hidraw*).

Typical use alongside the pump:

    from chemyx_pump import ChemyxPump
    from fluigent_sensor import FluigentSensor

    with FluigentSensor() as sensor, ChemyxPump("/dev/ttyUSB1", 9600) as pump:
        pump.infuse(volume=0.5, rate=3)
        while pump.is_running():
            print(sensor.read(), sensor.unit())
"""

from __future__ import annotations

import csv
import time
import threading

try:
    import Fluigent.SDK as fgt
except Exception:            # SDK missing / import failure
    fgt = None


class FluigentError(RuntimeError):
    pass


class FluigentSensor:
    """Read one or more Fluigent flow sensors.

    Parameters
    ----------
    channel : int
        Default sensor channel to read (0-based).
    require_device : bool
        If True (default), raise if the SDK reports no sensor channels.
    """

    def __init__(self, channel=0, require_device=True, auto_open=True):
        self.channel = channel
        self.require_device = require_device
        self.n_channels = 0
        self._open = False
        self._lock = threading.RLock()

        # Background polling / volume integration state.
        self._poll_thread = None
        self._poll_stop = threading.Event()
        self.dispensed = 0.0          # integrated volume, in sensor unit * min
        self._last_t = None

        if auto_open:
            self.open()

    # ------------------------------------------------------------------ conn
    def open(self):
        """Initialise the Fluigent session and detect sensor channels."""
        if fgt is None:
            raise FluigentError(
                "Fluigent SDK not importable. Ensure the 'Fluigent' package "
                "from github.com/Fluigent/fgt-SDK is on the Python path.")
        with self._lock:
            if self._open:
                return self
            fgt.fgt_init()
            self.n_channels = int(fgt.fgt_get_sensorChannelCount() or 0)
            if self.require_device and self.n_channels < 1:
                try:
                    fgt.fgt_close()
                except Exception:
                    pass
                raise FluigentError(
                    "No Fluigent sensor channel detected (readings would be 0). "
                    "Check the Flow Unit cable into the Flowboard/LINK, the USB "
                    "cable, and that no other Fluigent program holds the device.")
            self._open = True
        return self

    def close(self):
        with self._lock:
            self.stop_polling()
            if self._open and fgt is not None:
                try:
                    fgt.fgt_close()
                except Exception:
                    pass
            self._open = False

    @property
    def is_open(self):
        return self._open

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # --------------------------------------------------------------- readings
    def read(self, channel=None):
        """Instantaneous sensor value (e.g. flow in µL/min) for a channel."""
        ch = self.channel if channel is None else channel
        with self._lock:
            self._require_open()
            return float(fgt.fgt_get_sensorValue(ch))

    def read_all(self):
        """List of current values for every detected sensor channel."""
        with self._lock:
            self._require_open()
            return [float(fgt.fgt_get_sensorValue(i))
                    for i in range(self.n_channels)]

    def unit(self, channel=None):
        """Measurement unit string for a channel (e.g. 'µl/min')."""
        ch = self.channel if channel is None else channel
        with self._lock:
            self._require_open()
            return fgt.fgt_get_sensorUnit(ch)

    def range(self, channel=None):
        """(min, max) measurable value for a channel in its current unit."""
        ch = self.channel if channel is None else channel
        with self._lock:
            self._require_open()
            return tuple(fgt.fgt_get_sensorRange(ch))

    def air_bubble(self, channel=None):
        """True if the sensor currently flags an air bubble (flow sensors)."""
        ch = self.channel if channel is None else channel
        with self._lock:
            self._require_open()
            try:
                return bool(fgt.fgt_get_sensorAirBubbleFlag(ch))
            except Exception:
                return False

    def channels_info(self):
        """(infoArray, typeArray) as returned by the SDK, for diagnostics."""
        with self._lock:
            self._require_open()
            return fgt.fgt_get_sensorChannelsInfo()

    # ------------------------------------------------------------ calibration
    def get_calibration(self, channel=None):
        """Current calibration name (e.g. 'H2O', 'IPA') for a channel."""
        ch = self.channel if channel is None else channel
        with self._lock:
            self._require_open()
            cur = fgt.fgt_get_sensorCalibration(ch)
            try:
                return fgt.fgt_SENSOR_CALIBRATION.reverse_mapping.get(
                    int(cur), str(cur))
            except Exception:
                return str(cur)

    def set_calibration(self, name, channel=None):
        """Set the sensor's calibration table by name (e.g. 'H2O' or 'IPA')."""
        ch = self.channel if channel is None else channel
        with self._lock:
            self._require_open()
            cal = getattr(fgt.fgt_SENSOR_CALIBRATION, name)
            fgt.fgt_set_sensorCalibration(ch, cal)

    # ------------------------------------------------------- background poller
    def reset_dispensed(self):
        """Zero the integrated dispensed-volume counter."""
        with self._lock:
            self.dispensed = 0.0
            self._last_t = None

    def start_polling(self, interval=1.0, channel=None, on_reading=None,
                      csv_path=None, integrate=True):
        """Poll the sensor in a background thread.

        interval : seconds between reads.
        on_reading : optional callback(dict) with keys
            {t, flow, unit, dispensed, air_bubble}.
        csv_path : if given, append readings to this CSV file.
        integrate : accumulate dispensed volume as flow * dt.

        The dispensed counter is in (sensor unit)·minute — e.g. a flow unit of
        µL/min integrates to µL. Call reset_dispensed() to start from zero.
        """
        if self._poll_thread and self._poll_thread.is_alive():
            raise FluigentError("Polling already running.")
        self._require_open()
        ch = self.channel if channel is None else channel
        self._poll_stop.clear()
        self.reset_dispensed()

        def worker():
            csv_file = writer = None
            t0 = time.monotonic()
            if csv_path:
                csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                writer = csv.writer(csv_file)
                writer.writerow(["elapsed_s", "flow", "unit",
                                 "dispensed", "air_bubble"])
                csv_file.flush()
            try:
                unit = self.unit(ch)
                while not self._poll_stop.is_set():
                    now = time.monotonic()
                    flow = self.read(ch)
                    if integrate:
                        with self._lock:
                            if self._last_t is not None:
                                self.dispensed += flow * (now - self._last_t) / 60.0
                            self._last_t = now
                    bubble = self.air_bubble(ch)
                    reading = {"t": now - t0, "flow": flow, "unit": unit,
                               "dispensed": self.dispensed, "air_bubble": bubble}
                    if writer is not None:
                        writer.writerow([f"{reading['t']:.2f}", f"{flow:.4f}",
                                         unit, f"{self.dispensed:.4f}",
                                         int(bubble)])
                        csv_file.flush()
                    if on_reading is not None:
                        try:
                            on_reading(reading)
                        except Exception:
                            pass
                    self._poll_stop.wait(interval)
            finally:
                if csv_file is not None:
                    csv_file.close()

        self._poll_thread = threading.Thread(target=worker, daemon=True)
        self._poll_thread.start()
        return self._poll_thread

    def stop_polling(self):
        self._poll_stop.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None

    # ------------------------------------------------------------------ util
    def _require_open(self):
        if not self._open:
            raise FluigentError("Sensor not connected (call open()).")


def probe():
    """Print what the SDK sees. Safe to run; returns channel count."""
    if fgt is None:
        print("Fluigent SDK not importable.")
        return -1
    fgt.fgt_init()
    n = int(fgt.fgt_get_sensorChannelCount() or 0)
    print(f"sensor channels: {n}")
    if n:
        info, types = fgt.fgt_get_sensorChannelsInfo()
        for i in range(n):
            print(f"  ch{i}: type={types[i]} unit={fgt.fgt_get_sensorUnit(i)} "
                  f"range={tuple(fgt.fgt_get_sensorRange(i))} "
                  f"value={fgt.fgt_get_sensorValue(i):.3f}")
    fgt.fgt_close()
    return n


if __name__ == "__main__":
    probe()
