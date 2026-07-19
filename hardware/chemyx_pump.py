"""
chemyx_pump — a small, precise control library for Chemyx syringe pumps.

Talks to the pump over USB serial using the Chemyx ASCII protocol:
each command is sent terminated with a single '\\r'; the pump echoes the
command, returns its reply line(s), then a '>' prompt.

Verified on this rig: Chemyx (FTDI FT232R, serial A100ZWTS) on
/dev/ttyUSB1 @ 9600 baud, replying in clean ASCII.

Design goals
------------
* Precise, synchronous calls: every method waits for the pump's reply and
  returns the parsed text — no blind sleeps guessing whether it's done.
* Safe to embed: all serial access is guarded by a lock, so this can be
  driven from a GUI or background thread in the mabip project.
* Direction convention: a NEGATIVE volume withdraws, POSITIVE infuses.
  Rate is always positive.

Quick start
-----------
    from chemyx_pump import ChemyxPump

    with ChemyxPump("/dev/ttyUSB1", 9600) as pump:
        pump.set_diameter(4.78)          # syringe inner diameter, mm
        pump.set_units("uL/min")
        pump.infuse(volume=200, rate=50) # 200 uL at 50 uL/min, then start
        pump.wait_until_done()
"""

from __future__ import annotations

import os
import glob
import time
import threading

try:
    import serial
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pyserial is required: pip install pyserial") from e


def find_chemyx_port(preferred=None):
    """Locate the Chemyx pump's serial port via the STABLE /dev/serial/by-id
    links so it survives ttyUSB0/1 renumbering on replug/reboot.

    The Chemyx uses an FTDI FT232R, so we match 'FTDI'/'FT232' in by-id (and
    deliberately avoid the CP210x/Silicon Labs adapter, which is the SIX sensor).
    Returns the by-id symlink path (stable). Falls back to `preferred`, then to
    any present /dev/ttyUSB*.
    """
    byid = "/dev/serial/by-id"
    if os.path.isdir(byid):
        links = sorted(glob.glob(os.path.join(byid, "*")))
        for link in links:
            name = os.path.basename(link)
            if "FTDI" in name or "FT232" in name:
                return link
    if preferred and preferred not in ("auto", "", None) and os.path.exists(preferred):
        return preferred
    for p in sorted(glob.glob("/dev/ttyUSB*")):
        return p
    return preferred if preferred not in ("auto", "", None) else "/dev/ttyUSB0"


# Unit codes the pump understands via 'set units N'.
UNITS = {
    "mL/min": 0, "ml/min": 0,
    "mL/hr": 1,  "ml/hr": 1,
    "uL/min": 2, "μL/min": 2, "uL/hour": 2,
    "uL/hr": 3,  "μL/hr": 3,
}

class PumpError(RuntimeError):
    """Raised when the pump reports an error or a command cannot complete."""


class ChemyxPump:
    """Serial controller for a Chemyx syringe pump.

    Parameters
    ----------
    port : str
        Serial device, e.g. "/dev/ttyUSB1".
    baud : int
        Must match the pump's System Settings baud rate (commonly 9600).
    timeout : float
        Per-command read timeout in seconds.
    verbose : bool
        Print every command/response exchange (handy while integrating).
    multipump : bool
        Set True for dual-channel pumps so 'set' commands are prefixed with
        the active pump number.
    auto_open : bool
        Open the serial port immediately (default True).
    """

    def __init__(self, port="auto", baud=9600, timeout=1.0,
                 verbose=False, multipump=False, auto_open=True):
        # "auto"/None -> resolve the FTDI (Chemyx) via stable by-id links so a
        # ttyUSB0/1 swap can't point us at the SIX sensor by mistake.
        self.port = find_chemyx_port(port) if port in ("auto", "", None) else port
        self.baud = baud
        self.timeout = timeout
        self.verbose = verbose
        self.multipump = multipump
        self.current_pump = 1
        self._lock = threading.RLock()
        self.ser = None
        # Last run time (minutes) the pump reported after a set_rate/set_volume.
        # Used to know when a run is done, since this firmware has no
        # 'pump status' command.
        self._last_run_min = None
        if auto_open:
            self.open()

    # ------------------------------------------------------------------ conn
    # Seconds to wait after writing before reading, so the pump's full echo +
    # reply have arrived. Reading too eagerly drops characters (Custom_Code's
    # driver sleeps 0.4 for the same reason).
    SETTLE = 0.2

    def open(self):
        """Open the serial connection (idempotent)."""
        with self._lock:
            if self.ser and self.ser.is_open:
                return
            self.ser = serial.Serial(self.port, self.baud,
                                     timeout=self.timeout, write_timeout=2.0)
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        return self

    def close(self):
        """Close the serial connection."""
        with self._lock:
            if self.ser and self.ser.is_open:
                self.ser.close()

    @property
    def is_open(self):
        return bool(self.ser and self.ser.is_open)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    # --------------------------------------------------------------- low level
    def command(self, cmd, timeout=None, _retry=True):
        """Send one raw command, return the pump's parsed text reply.

        This is the single choke point for all serial I/O — thread-safe. It
        waits for the line to go quiet before writing (the pump drops leading
        characters if written to mid-reply), settles, then reads until the
        pump's '>' prompt. A garbled reply is retried once for commands where
        that is safe (everything except start/restart).
        """
        with self._lock:
            if not self.is_open:
                raise PumpError("Pump is not connected (call open()).")

            full = cmd
            if self.multipump and full.lstrip().startswith("set"):
                full = f"{self.current_pump} {full}"

            self._drain_quiet()              # wait until the pump stops talking
            self.ser.reset_input_buffer()
            self.ser.write((full + "\r").encode("ascii"))
            self.ser.flush()                 # push all bytes out before reading
            time.sleep(self.SETTLE)          # let the full echo + reply arrive
            raw = self._read_until_prompt(timeout or self.timeout)
            reply = self._parse(raw, full)
            if self.verbose:
                print(f">>> {full!r}  ->  {reply!r}")

            garbled = "not recognized" in reply.lower()
            if garbled or reply.lower().startswith("error"):
                if garbled and _retry and self._safe_to_retry(cmd):
                    time.sleep(0.3)
                    return self.command(cmd, timeout=timeout, _retry=False)
                raise PumpError(f"Pump rejected {full!r}: {reply}")

            # The pump echoes a computed 'time = X' (minutes) when rate/volume
            # are set. Capture it so wait_until_done() knows the run length.
            t = self._extract_number(reply, "time")
            if t is not None:
                self._last_run_min = t
            return reply

    @staticmethod
    def _safe_to_retry(cmd):
        """Retrying is safe for everything except commands that begin motion."""
        c = cmd.strip().lower()
        return not (c.startswith("start") or c.startswith("restart"))

    def _drain_quiet(self, quiet=0.06, cap=1.0):
        """Block until no bytes have arrived for `quiet` seconds (or `cap`).

        Ensures the pump has finished transmitting its previous reply before we
        write the next command, which otherwise loses its leading characters.
        """
        end = time.monotonic() + cap
        while time.monotonic() < end:
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
                time.sleep(quiet)
            else:
                time.sleep(quiet)
                if not self.ser.in_waiting:
                    return

    def _read_until_prompt(self, overall_timeout):
        end = time.monotonic() + overall_timeout
        buf = bytearray()
        while time.monotonic() < end:
            n = self.ser.in_waiting
            if n:
                buf.extend(self.ser.read(n))
                if bytes(buf).rstrip().endswith(b">"):
                    break
            else:
                time.sleep(0.02)
        return bytes(buf)

    @staticmethod
    def _parse(raw, cmd):
        text = raw.decode("ascii", errors="replace")
        # Normalize CR/LF and split into non-empty lines.
        lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
        lines = [ln for ln in lines if ln]
        # Drop the trailing '>' prompt.
        lines = [ln for ln in lines if ln != ">"]
        # Drop the echoed command if the pump echoed it back.
        if lines and lines[0].lower() == cmd.strip().lower():
            lines = lines[1:]
        return "\n".join(lines).strip()

    @staticmethod
    def _fmt(value):
        """Format a scalar or list for a 'set' command (lists -> multistep)."""
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _extract_number(reply, key=None):
        """Pull a float out of a reply.

        With key: match 'key = <number>' (e.g. 'elapsed time = 1.234').
        Without key: return the first number found. Returns None if none.
        """
        import re
        if key is not None:
            m = re.search(rf"{re.escape(key)}\s*=\s*(-?\d+(?:\.\d+)?)",
                          reply, re.IGNORECASE)
            if m:
                return float(m.group(1))
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", reply)
        return float(m.group(0)) if m else None

    # ------------------------------------------------------------- parameters
    def set_units(self, units):
        """Set flow units. Accepts 'mL/min','mL/hr','uL/min','uL/hr' or a code."""
        code = UNITS[units] if isinstance(units, str) else int(units)
        return self.command(f"set units {code}")

    def set_diameter(self, mm):
        """Syringe inner diameter in millimetres."""
        return self.command(f"set diameter {self._fmt(mm)}")

    def set_rate(self, rate):
        """Flow rate (scalar) or list of rates for a multistep run."""
        return self.command(f"set rate {self._fmt(rate)}")

    def set_volume(self, volume):
        """Target volume. NEGATIVE volume withdraws, positive infuses.

        A list runs a multistep sequence paired with set_rate/set_delay lists.
        """
        return self.command(f"set volume {self._fmt(volume)}")

    def set_delay(self, delay):
        """Start delay in minutes (scalar or list)."""
        return self.command(f"set delay {self._fmt(delay)}")

    def set_time(self, minutes):
        return self.command(f"set time {self._fmt(minutes)}")

    # ---------------------------------------------------------------- motion
    def start(self, mode=0, multistep=False):
        """Begin the run configured by the most recent set_* commands."""
        cmd = "start"
        if self.multipump and mode > 0:
            cmd = f"{mode} {cmd}"
        if multistep:
            cmd = f"{cmd} 1"
        return self.command(cmd)

    def stop(self, mode=0):
        cmd = "stop"
        if self.multipump and mode > 0:
            cmd = f"{mode} {cmd}"
        return self.command(cmd)

    def pause(self, mode=0):
        cmd = "pause"
        if self.multipump and mode > 0:
            cmd = f"{mode} {cmd}"
        return self.command(cmd)

    def restart(self):
        return self.command("restart")

    # ---------------------------------------------------------------- queries
    # NOTE: this firmware does NOT support 'pump status' or 'view parameter'.
    # Progress is tracked via 'elapsed time' and 'dispensed volume', and the
    # pump's own reported run time (self._last_run_min).

    def elapsed_time_raw(self):
        return self.command("elapsed time")

    def elapsed_time(self):
        """Elapsed run time in minutes (float). 0 when idle/reset."""
        n = self._extract_number(self.elapsed_time_raw(), "elapsed time")
        return n if n is not None else 0.0

    def dispensed_volume_raw(self):
        return self.command("dispensed volume")

    def dispensed_volume(self):
        """Volume dispensed so far in the pump's current units (float)."""
        n = self._extract_number(self.dispensed_volume_raw(), "dispensed volume")
        return n if n is not None else 0.0

    def get_parameter_limits(self):
        """Raw limits reply: 'maxVol minRate maxRate ...' as a list of floats."""
        raw = self.command("read limit parameter")
        import re
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", raw)]

    @property
    def last_run_time(self):
        """Run time (minutes) the pump computed for the last set rate/volume."""
        return self._last_run_min

    def is_running(self, tol=1e-4):
        """Best-effort: True while elapsed time is below the planned run time.

        This firmware has no status command, so 'running' means the pump has
        been started and elapsed time has not yet reached the computed run
        time. Use wait_until_done() for blocking waits.
        """
        if self._last_run_min is None:
            return False
        return self.elapsed_time() < self._last_run_min - tol

    def help(self):
        return self.command("help", timeout=1.5)

    # ------------------------------------------------------------ high level
    def infuse(self, volume, rate, diameter=None, units=None,
               delay=0, start=True):
        """Configure and (optionally) start an infusion.

        volume, rate are magnitudes (positive). Direction is handled here.
        """
        return self._run(abs(volume), abs(rate), diameter, units, delay, start)

    def withdraw(self, volume, rate, diameter=None, units=None,
                 delay=0, start=True):
        """Configure and (optionally) start a withdrawal (negative volume)."""
        return self._run(-abs(volume), abs(rate), diameter, units, delay, start)

    def _run(self, signed_volume, rate, diameter, units, delay, start):
        replies = {}
        if diameter is not None:
            replies["diameter"] = self.set_diameter(diameter)
            time.sleep(0.05)
        if units is not None:
            replies["units"] = self.set_units(units)
            time.sleep(0.05)
        replies["rate"] = self.set_rate(rate)
        time.sleep(0.05)
        replies["volume"] = self.set_volume(signed_volume)
        time.sleep(0.05)
        replies["delay"] = self.set_delay(delay)
        time.sleep(0.05)
        if start:
            replies["start"] = self.start()
        return replies

    def run_sequence(self, volumes, rates, delays=None, diameter=None,
                     units=None, start=True):
        """Load a multistep program (paired lists) and optionally start it.

        volumes: list, signed (negative entries withdraw).
        rates:   list of positive rates, same length.
        delays:  optional list of start delays (minutes).
        """
        if len(volumes) != len(rates):
            raise ValueError("volumes and rates must be the same length")
        if delays is not None and len(delays) != len(volumes):
            raise ValueError("delays must match volumes length")
        replies = {}
        if diameter is not None:
            replies["diameter"] = self.set_diameter(diameter)
            time.sleep(0.05)
        if units is not None:
            replies["units"] = self.set_units(units)
            time.sleep(0.05)
        replies["rate"] = self.set_rate(list(rates))
        time.sleep(0.05)
        replies["volume"] = self.set_volume(list(volumes))
        time.sleep(0.05)
        if delays is not None:
            replies["delay"] = self.set_delay(list(delays))
            time.sleep(0.05)
        if start:
            replies["start"] = self.start(multistep=True)
        return replies

    def wait_until_done(self, expected_minutes=None, poll=0.25, timeout=None):
        """Block until the current run finishes, tracking 'elapsed time'.

        expected_minutes : run length to wait for. Defaults to the pump's own
            computed run time from the last set rate/volume (last_run_time).
        timeout : hard cap in seconds (None = wait as long as needed).
        Returns True if it finished, False if the timeout was hit.
        """
        target = expected_minutes if expected_minutes is not None else self._last_run_min
        # If we know the run length, cap the wait so a plateau just short of
        # target can't hang forever.
        if timeout is None and target is not None:
            timeout = target * 60 + 5
        end = None if timeout is None else time.monotonic() + timeout
        time.sleep(0.2)  # let the run actually begin
        last = -1.0
        stalls = 0
        while True:
            if end is not None and time.monotonic() > end:
                return False
            elapsed = self.elapsed_time()
            if target is not None and elapsed >= target - 1e-4:
                return True
            # Fallback if we have no target: stop when elapsed stops advancing.
            if target is None:
                if elapsed <= last:
                    stalls += 1
                    if stalls >= 3 and elapsed > 0:
                        return True
                else:
                    stalls = 0
                last = elapsed
            time.sleep(poll)

    def burst(self, pulses, volume, rate, gap=1.0, diameter=None, units=None,
              withdraw=False, on_pulse=None, should_abort=None):
        """Fire N quick pulses of `volume` at `rate`, waiting `gap` s between.

        Ported from the GUI's burst feature. Blocks until all pulses finish.

        pulses : number of pulses.
        volume, rate : magnitude per pulse (units per the current setting).
        gap : seconds between pulses (not after the last).
        withdraw : pull back instead of infuse.
        on_pulse : optional callback(i, pulses) after each pulse starts.
        should_abort : optional callable -> True to stop early.
        Returns the number of pulses actually completed.
        """
        if pulses < 1:
            raise ValueError("pulses must be >= 1")
        if volume <= 0 or rate <= 0:
            raise ValueError("volume and rate must be > 0")
        signed = -abs(volume) if withdraw else abs(volume)
        if diameter is not None:
            self.set_diameter(diameter)
        if units is not None:
            self.set_units(units)

        done = 0
        for i in range(1, pulses + 1):
            if should_abort and should_abort():
                break
            self.set_rate(abs(rate))
            time.sleep(0.05)
            self.set_volume(signed)
            time.sleep(0.05)
            self.set_delay(0)
            time.sleep(0.05)
            self.start()
            pulse_min = abs(volume) / abs(rate)   # units/min -> minutes
            self.wait_until_done(expected_minutes=pulse_min,
                                 timeout=pulse_min * 60 + 5)
            self.stop()
            done = i
            if on_pulse:
                on_pulse(i, pulses)
            if i < pulses:
                # Interruptible gap.
                waited = 0.0
                while waited < gap:
                    if should_abort and should_abort():
                        return done
                    time.sleep(min(0.1, gap - waited))
                    waited += 0.1
        return done


if __name__ == "__main__":
    # Read-only smoke test — connects and reads progress counters.
    # Does not move the pump.
    with ChemyxPump("/dev/ttyUSB1", 9600, verbose=True) as p:
        print("elapsed:", p.elapsed_time(), "min")
        print("dispensed:", p.dispensed_volume())
        print("limits:", p.get_parameter_limits())
