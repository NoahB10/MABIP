"""
dual_syringe — drive TWO syringes as one combined output line, verified by the
Fluigent flow sensor.

Physical setup this models
--------------------------
* ONE Chemyx pump, ONE pusher block, carrying TWO identical syringes
  (here: 2 x 20 mL). Because both barrels sit on the same pusher they move at
  the SAME linear speed, so each delivers the same volumetric flow.
* Both syringe outputs are teed together into a SINGLE line (Y/T connector).
* The Fluigent Flow Unit sits on that one combined line, so it reads the SUM
  of the two syringes:  line_flow  ==  2 x per_syringe_flow.
* Goal: 2x the volume/throughput of a single syringe, with the sensor telling
  us the true combined flow so we can trim the pump to match.

The key relationship
--------------------
    measured_line_flow  =  cal_factor * n_syringes * per_syringe_pump_rate

`cal_factor` starts at 1.0 (ideal: line = 2 x per-syringe) and absorbs every
real-world error at once — syringe-diameter guess, tubing compliance, small
leaks, sensor calibration. You don't need the syringe diameter to be exact:
run `calibrate()` once and the factor makes the sensor read the rate you ask
for. Everything you pass in/out of this class is in COMBINED LINE terms
(what the sensor sees), in the sensor's unit, µL/min.

    from dual_syringe import DualSyringeLine

    with DualSyringeLine(diameter_mm=19.13) as line:   # 2 x 20 mL, BD Plastipak
        line.calibrate(60)                 # trim so the sensor reads 60 µL/min
        line.deliver(line_volume=2000, line_rate=60)   # 2000 µL down the line
        line.wait_until_done()

Notes
-----
* Sensor is a Flow Unit M: range +/-120 µL/min. Keep the COMBINED target under
  that (per-syringe stays under 60). The class guards this.
* Direction: "infuse" pushes fluid out toward the line/sensor (the normal case
  for delivering volume); "withdraw" pulls back.
* Set `diameter_mm` to your real 20 mL syringe bore. Common values:
  BD Plastipak 20 mL ~= 19.13 mm, Terumo 20 mL ~= 20.15 mm. When unsure, pick
  the closest and let calibrate() correct the rest.
"""

from __future__ import annotations

import os
import sys
import time
import statistics

# Make sibling modules importable when run straight from this folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chemyx_pump import ChemyxPump          # noqa: E402
from fluigent_sensor import FluigentSensor  # noqa: E402


class DualSyringeError(RuntimeError):
    pass


class DualSyringeLine:
    """Two identical syringes on one pump, combined into one sensed line.

    Parameters
    ----------
    diameter_mm : float
        Inner diameter of ONE syringe (both are identical). Default 19.13
        (BD Plastipak 20 mL). calibrate() corrects any error in this.
    n_syringes : int
        How many syringes share the pusher. Default 2.
    cal_factor : float
        Empirical (actual line flow) / (n x per-syringe pump rate). Default 1.0;
        calibrate() updates it. Persisted/loaded via save/load_calibration().
    direction : {"infuse", "withdraw"}
        Default motion. "infuse" delivers toward the line/sensor.
    sensor_max : float
        Sensor full-scale magnitude, µL/min (Flow Unit M = 120). Targets above
        ~95% of this are refused so the sensor can still verify them.
    pump, sensor :
        Pre-built ChemyxPump / FluigentSensor. If None, they're created from
        the pump_* / sensor_channel args. sensor may stay None for open-loop
        (no calibration/verification) use.
    """

    UNIT = "uL/min"  # working unit for BOTH pump and sensor (Flow Unit M unit)

    def __init__(self, diameter_mm=19.13, n_syringes=1, cal_factor=1.0,
                 direction="infuse", sensor_max=120.0,
                 pump=None, sensor=None,
                 pump_port="auto", pump_baud=9600,
                 sensor_channel=0, require_sensor=True, verbose=True):
        if n_syringes < 1:
            raise ValueError("n_syringes must be >= 1")
        if direction not in ("infuse", "withdraw"):
            raise ValueError("direction must be 'infuse' or 'withdraw'")
        self.diameter_mm = float(diameter_mm)
        self.n_syringes = int(n_syringes)
        self.cal_factor = float(cal_factor)
        self.direction = direction
        self.sensor_max = float(sensor_max)
        self.sensor_channel = sensor_channel
        self.verbose = verbose

        self._own_pump = pump is None
        self._own_sensor = sensor is None and require_sensor
        self.pump = pump or ChemyxPump(pump_port, pump_baud, auto_open=True)
        if sensor is not None:
            self.sensor = sensor
        elif require_sensor:
            self.sensor = FluigentSensor(channel=sensor_channel, auto_open=True)
        else:
            self.sensor = None

        self._configured = False

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def connect_sensor(self, channel=0):
        """Attach a Fluigent flow sensor to an already-running pump line (the
        sensor is OPTIONAL — the pump works fine without one)."""
        self.disconnect_sensor()
        self.sensor = FluigentSensor(channel=channel, auto_open=True)
        self.sensor_channel = channel
        self._own_sensor = True
        return self.sensor

    def disconnect_sensor(self):
        """Detach/close the flow sensor; pump control continues unaffected."""
        if getattr(self, "sensor", None) is not None:
            try:
                self.sensor.close()
            except Exception:
                pass
        self.sensor = None

    def close(self):
        """Stop motion and release hardware we created."""
        try:
            self.stop()
        except Exception:
            pass
        if self._own_sensor and self.sensor is not None:
            try:
                self.sensor.close()
            except Exception:
                pass
        if self._own_pump:
            try:
                self.pump.close()
            except Exception:
                pass

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ----------------------------------------------------------------- maths
    def per_syringe_rate(self, line_rate):
        """Pump rate to command per syringe to achieve `line_rate` on the line."""
        return abs(line_rate) / (self.n_syringes * self.cal_factor)

    def per_syringe_volume(self, line_volume):
        """Pump volume per syringe to put `line_volume` down the combined line."""
        return abs(line_volume) / (self.n_syringes * self.cal_factor)

    def expected_line_rate(self, per_syringe_rate):
        """Inverse of per_syringe_rate(): line flow expected for a pump rate."""
        return per_syringe_rate * self.n_syringes * self.cal_factor

    def plan(self, line_volume, line_rate):
        """Return the exact pump settings for a run WITHOUT moving anything."""
        self._guard_line_rate(line_rate)
        ps_rate = self.per_syringe_rate(line_rate)
        ps_vol = self.per_syringe_volume(line_volume)
        runtime_min = ps_vol / ps_rate if ps_rate else float("inf")
        return {
            "line_rate": abs(line_rate),
            "line_volume": abs(line_volume),
            "n_syringes": self.n_syringes,
            "cal_factor": self.cal_factor,
            "diameter_mm": self.diameter_mm,
            "per_syringe_rate": ps_rate,
            "per_syringe_volume": ps_vol,
            "runtime_min": runtime_min,
            "direction": self.direction,
            "unit": self.UNIT,
        }

    def _guard_line_rate(self, line_rate):
        if abs(line_rate) > 0.95 * self.sensor_max:
            raise DualSyringeError(
                f"Combined line rate {abs(line_rate):g} µL/min exceeds 95% of the "
                f"sensor range ({self.sensor_max:g} µL/min). The Flow Unit M would "
                f"saturate and could not verify it. Lower the target or use a "
                f"wider-range Flow Unit.")

    # ------------------------------------------------------------- pump setup
    def configure(self):
        """Push syringe diameter + working unit to the pump (idempotent)."""
        self.pump.set_units(self.UNIT)
        time.sleep(0.05)
        self.pump.set_diameter(self.diameter_mm)
        time.sleep(0.05)
        self._configured = True
        return self

    # ------------------------------------------------------------- delivery
    def deliver(self, line_volume, line_rate, direction=None, start=True):
        """Deliver `line_volume` (µL) down the combined line at `line_rate`
        (µL/min, what the sensor should read). Returns the plan dict.

        Both figures are COMBINED/line quantities; per-syringe values are
        derived and sent to the pump. Direction defaults to self.direction.
        """
        direction = direction or self.direction
        p = self.plan(line_volume, line_rate)
        if not self._configured:
            self.configure()
        ps_rate, ps_vol = p["per_syringe_rate"], p["per_syringe_volume"]
        self._log(
            f"[deliver] line {p['line_volume']:g} µL @ {p['line_rate']:g} µL/min "
            f"-> per-syringe {ps_vol:g} µL @ {ps_rate:g} µL/min "
            f"({direction}, ~{p['runtime_min']:.2f} min, cal={self.cal_factor:.4f})")
        if direction == "infuse":
            self.pump.infuse(volume=ps_vol, rate=ps_rate, start=start)
        else:
            self.pump.withdraw(volume=ps_vol, rate=ps_rate, start=start)
        return p

    def hold(self, line_rate, seconds, direction=None, start=True):
        """Run at a steady `line_rate` for ~`seconds` (used by calibrate()).

        Sizes a per-syringe volume just large enough to cover the duration
        (plus 20% margin), capped so it never over-travels a 20 mL syringe.
        """
        direction = direction or self.direction
        self._guard_line_rate(line_rate)
        ps_rate = self.per_syringe_rate(line_rate)
        needed = ps_rate * (seconds / 60.0) * 1.2
        ps_vol = min(needed, 19000.0)  # keep clear of the 20 mL barrel end
        if not self._configured:
            self.configure()
        self._log(f"[hold] {line_rate:g} µL/min line for ~{seconds:g}s "
                  f"(per-syringe {ps_rate:g} µL/min, {ps_vol:g} µL, {direction})")
        if direction == "infuse":
            self.pump.infuse(volume=ps_vol, rate=ps_rate, start=start)
        else:
            self.pump.withdraw(volume=ps_vol, rate=ps_rate, start=start)
        return ps_rate

    def start_flow(self, line_rate, direction=None, max_volume=18000.0):
        """Start (or re-apply) CONTINUOUS flow at `line_rate` on the line.

        Stops any current run first, then runs a large volume so flow continues
        until stopped. Call again with a new rate to change the rate live — that
        is how the GUI's 'Apply rate' works. Returns the per-syringe rate sent.
        """
        direction = direction or self.direction
        self._guard_line_rate(line_rate)
        try:
            self.pump.stop()
        except Exception:
            pass
        time.sleep(0.12)
        if not self._configured:
            self.configure()
        ps_rate = self.per_syringe_rate(line_rate)
        ps_vol = min(abs(max_volume), 19000.0)
        self._log(f"[flow] line {line_rate:g} µL/min -> per-syringe {ps_rate:g} "
                  f"µL/min continuous ({direction})")
        if direction == "infuse":
            self.pump.infuse(volume=ps_vol, rate=ps_rate, start=True)
        else:
            self.pump.withdraw(volume=ps_vol, rate=ps_rate, start=True)
        return ps_rate

    def ramp(self, start_rate, max_rate, step, dwell_s, target,
             tol_frac=0.05, direction=None, measure_s=3.0,
             on_step=None, should_stop=None):
        """Step the LINE rate from `start_rate` up to `max_rate` in `step`
        increments, watching the sensor, and STOP as soon as the measured flow
        reaches `target` (within tol_frac), or at max_rate, or if should_stop().

        At each step: apply the rate, wait `dwell_s`, average the sensor for
        `measure_s`, call on_step({rate, measured, target}). Returns a report
        {reached, stop_rate, measured, history}. Great for finding the pump
        setting that actually produces your target flow at the sensor.
        """
        if self.sensor is None:
            raise DualSyringeError("No sensor attached; cannot ramp.")
        self._guard_line_rate(max_rate)
        if step <= 0:
            raise ValueError("step must be > 0")
        history = []
        reached = False
        rate = float(start_rate)
        stop_rate = rate
        while rate <= max_rate + 1e-9:
            if should_stop and should_stop():
                break
            self.start_flow(rate, direction=direction)
            t_end = time.monotonic() + max(0.0, dwell_s)
            while time.monotonic() < t_end:
                if should_stop and should_stop():
                    break
                time.sleep(0.1)
            stats = self.read_flow(seconds=measure_s)
            measured = stats["mean"]
            rec = {"rate": rate, "measured": measured, "target": target,
                   "std": stats["stdev"]}
            history.append(rec)
            stop_rate = rate
            if on_step:
                on_step(rec)
            if target and abs(measured) >= abs(target) * (1.0 - tol_frac):
                reached = True
                break
            rate += step
        self.stop()
        return {"reached": reached, "stop_rate": stop_rate,
                "measured": history[-1]["measured"] if history else None,
                "history": history}

    # -------- direct MACHINE-term control (values go straight to the pump) ----
    # These bypass the combined/÷n conversion: `machine_rate` is EXACTLY what
    # the pump runs each syringe at, and `single_volume` is per-syringe. The
    # sensor still reads the COMBINED line, so expected_combined() is what it
    # should show (~ n × machine_rate).
    def expected_combined(self, machine_rate):
        return abs(machine_rate) * self.n_syringes * self.cal_factor

    def _run_single(self, single_volume, machine_rate, direction, start):
        # NO sensor-range cap on the pump rate — the pump may run faster than the
        # flow sensor can read (e.g. to blast through a clog). Those flows just
        # aren't measurable, which is fine.
        if not self._configured:
            self.configure()
        if direction == "infuse":
            self.pump.infuse(volume=abs(single_volume), rate=abs(machine_rate), start=start)
        else:
            self.pump.withdraw(volume=abs(single_volume), rate=abs(machine_rate), start=start)

    def deliver_single(self, single_volume, machine_rate, direction=None, start=True):
        """Deliver `single_volume` per syringe at `machine_rate` — the exact
        rate/volume sent to the pump. Sensor reads the combined line."""
        direction = direction or self.direction
        self._run_single(single_volume, machine_rate, direction, start)
        return {"single_volume": abs(single_volume), "machine_rate": abs(machine_rate),
                "expected_combined": self.expected_combined(machine_rate),
                "runtime_min": abs(single_volume) / abs(machine_rate) if machine_rate else float("inf"),
                "direction": direction}

    def start_flow_single(self, machine_rate, direction=None, max_volume=18000.0):
        """Continuous flow at the exact `machine_rate`. Call again with a new
        rate to change it live (that is the GUI's 'apply rate')."""
        direction = direction or self.direction
        # NO sensor-range cap: the pump can run FASTER than the flow sensor reads
        # (needed to push through clogs); you just can't measure those flows.
        # Configure BEFORE stopping so a valid rate change can never strand the
        # pump halted with nothing to restart it.
        if not self._configured:
            self.configure()
        ps_vol = min(abs(max_volume), 19000.0)
        try:
            self.pump.stop()
        except Exception:
            pass
        time.sleep(0.12)
        if direction == "infuse":
            self.pump.infuse(volume=ps_vol, rate=abs(machine_rate), start=True)
        else:
            self.pump.withdraw(volume=ps_vol, rate=abs(machine_rate), start=True)
        return machine_rate

    def ramp_single(self, start_rate, max_rate, step, dwell_s, sensor_target,
                    tol_frac=0.05, direction=None, measure_s=3.0,
                    on_step=None, should_stop=None):
        """Step the MACHINE rate start→max, watch the sensor (combined line),
        and STOP when the sensor reaches `sensor_target` (within tol_frac), or
        at max_rate, or on should_stop(). Finds the machine rate that produces
        the target combined flow."""
        if self.sensor is None:
            raise DualSyringeError("No sensor attached; cannot ramp.")
        if step <= 0:
            raise ValueError("step must be > 0")
        history = []; reached = False
        rate = float(start_rate); stop_rate = rate
        while rate <= max_rate + 1e-9:
            if should_stop and should_stop():
                break
            self.start_flow_single(rate, direction=direction)
            t_end = time.monotonic() + max(0.0, dwell_s)
            while time.monotonic() < t_end:
                if should_stop and should_stop():
                    break
                time.sleep(0.1)
            stats = self.read_flow(seconds=measure_s)
            measured = stats["mean"]
            history.append({"rate": rate, "measured": measured,
                            "target": sensor_target, "std": stats["stdev"]})
            stop_rate = rate
            if on_step:
                on_step(history[-1])
            if sensor_target and abs(measured) >= abs(sensor_target) * (1.0 - tol_frac):
                reached = True
                break
            rate += step
        self.stop()
        return {"reached": reached, "stop_rate": stop_rate,
                "measured": history[-1]["measured"] if history else None,
                "history": history}

    def calibrate_single(self, machine_rate, settle_s=30.0, measure_s=15.0):
        """Run at `machine_rate`, measure the sensor, and set cal_factor so
        expected_combined() matches the real reading."""
        if self.sensor is None:
            raise DualSyringeError("No sensor attached; cannot calibrate.")
        self.start_flow_single(machine_rate)
        time.sleep(settle_s)
        stats = self.read_flow(seconds=measure_s)
        self.stop()
        measured = abs(stats["mean"])
        if measured < 1e-6:
            raise DualSyringeError("Measured ~0 flow while pumping — check the line/plumbing.")
        self.cal_factor = measured / (self.n_syringes * abs(machine_rate))
        return {"cal_factor": self.cal_factor, "measured": stats["mean"],
                "machine_rate": machine_rate}

    def verify_single(self, machine_rate, settle_s=30.0, measure_s=15.0):
        if self.sensor is None:
            raise DualSyringeError("No sensor attached; cannot verify.")
        self.start_flow_single(machine_rate)
        time.sleep(settle_s)
        stats = self.read_flow(seconds=measure_s)
        self.stop()
        exp = self.expected_combined(machine_rate)
        err = stats["mean"] - exp
        return {"machine_rate": machine_rate, "expected_combined": exp,
                "measured_mean": stats["mean"], "error": err,
                "error_pct": 100 * err / exp if exp else 0.0, "unit": stats["unit"]}

    def prime(self, volume=1000.0, rate=100.0, direction="infuse"):
        """Push `volume` µL at `rate` to fill the line and flush bubbles before a
        run. ALWAYS pushes (infuse) by default — priming wets the line, so it must
        NOT follow a 'withdraw' run direction."""
        return self.deliver_single(volume, rate, direction=direction, start=True)

    def burst(self, baseline_rate, high_rate, trigger_flow, high_seconds,
              stop_seconds, read_flow, direction=None, max_total_s=60.0,
              on_phase=None, should_abort=None,
              backflow_rate=0.0, backflow_seconds=0.0, backflow_direction=None,
              backflow_trigger_flow=None):
        """Pressure/flow BURST: jump the pump to `high_rate`; as soon as the
        measured flow (via read_flow()) exceeds `trigger_flow` — or `high_seconds`
        elapses — STOP the syringe for `stop_seconds` to level out, optionally run
        a short BACKFLOW pulse to actively relieve the stored line pressure, then
        resume `baseline_rate`. The whole pulse is hard-capped at `max_total_s`
        (<=60).

        The high-rate boost pressurises the compliant line; on STOP the flow decays
        only slowly (it overshoots baseline for a while). A brief reverse pulse
        (`backflow_rate` for `backflow_seconds`, opposite `direction` by default)
        bleeds that stored pressure off so the flow drops back to `baseline_rate`
        promptly instead of settling passively — key when the burst must finish and
        re-stabilise inside a fixed buffer window.

        read_flow  : callable returning the current measured combined flow.
        on_phase   : optional callback("high"|"stop"|"backflow"|"baseline").
        should_abort : optional callable -> True to bail out (STOP).
        backflow_rate : combined-line rate for the relief pulse (0 = skip backflow).
        backflow_seconds : MAX duration of the relief pulse (0 = skip).
        backflow_direction : defaults to the opposite of `direction`.
        backflow_trigger_flow : optional sensor threshold — the backflow ends
            EARLY as soon as read_flow() drops to/below this (e.g. baseline+8),
            so the reverse runs exactly as long as needed and no longer.
        Returns {triggered, peak_flow, total_s, backflow, ...}.
        """
        direction = direction or self.direction
        rev = backflow_direction or ("infuse" if direction == "withdraw" else "withdraw")
        cap = min(abs(max_total_s), 60.0)          # never exceed 60 s total
        t0 = time.monotonic()
        elapsed = lambda: time.monotonic() - t0
        aborted = lambda: bool(should_abort and should_abort())
        peak = 0.0; triggered = False

        # phase 1 — high rate
        if on_phase:
            on_phase("high")
        self.start_flow_single(high_rate, direction=direction)
        while elapsed() < high_seconds and elapsed() < cap:
            if aborted():
                break
            try:
                f = abs(float(read_flow()))
            except Exception:
                f = 0.0
            peak = max(peak, f)
            if trigger_flow and f >= abs(trigger_flow):
                triggered = True
                break
            time.sleep(0.1)

        # phase 2 — stop to level out
        if on_phase:
            on_phase("stop")
        self.stop()
        ts = time.monotonic()
        while (time.monotonic() - ts) < stop_seconds and elapsed() < cap:
            if aborted():
                break
            time.sleep(0.1)

        # phase 2.5 — backflow: reverse briefly to bleed off the stored pressure so
        # the line drops back to baseline fast (kills the post-boost overshoot).
        did_backflow = False
        if (backflow_rate and backflow_seconds > 0
                and not aborted() and elapsed() < cap):
            if on_phase:
                on_phase("backflow")
            self.start_flow_single(abs(backflow_rate), direction=rev)
            tb = time.monotonic()
            while (time.monotonic() - tb) < backflow_seconds and elapsed() < cap:
                if aborted():
                    break
                if backflow_trigger_flow is not None:
                    try:
                        f = abs(float(read_flow()))
                    except Exception:
                        f = None
                    if f is not None and f <= abs(backflow_trigger_flow):
                        break            # flow is back near baseline — stop reversing
                time.sleep(0.1)
            self.stop()
            did_backflow = True

        # phase 3 — resume baseline
        if on_phase:
            on_phase("baseline")
        if not aborted():
            self.start_flow_single(baseline_rate, direction=direction)
        return {"triggered": triggered, "peak_flow": peak,
                "high_rate": high_rate, "baseline_rate": baseline_rate,
                "backflow": did_backflow, "total_s": elapsed()}

    def stop(self):
        return self.pump.stop()

    def wait_until_done(self, **kw):
        return self.pump.wait_until_done(**kw)

    # -------------------------------------------------------------- sensing
    def read_flow(self, seconds=3.0, interval=0.1):
        """Average the combined line flow over `seconds`. Returns a stats dict."""
        if self.sensor is None:
            raise DualSyringeError("No sensor attached; cannot read line flow.")
        vals = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            vals.append(self.sensor.read(self.sensor_channel))
            time.sleep(interval)
        if not vals:
            vals = [self.sensor.read(self.sensor_channel)]
        return {
            "mean": statistics.mean(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals),
            "unit": self.sensor.unit(self.sensor_channel),
        }

    # ------------------------------------------------------------ calibration
    def calibrate(self, line_rate=60.0, settle_s=30.0, measure_s=15.0,
                  iterations=1, stop_after=True):
        """Trim `cal_factor` so the sensor reads the requested combined rate.

        For each iteration: command `line_rate`, let flow settle, average the
        sensor, then update cal_factor *= measured/target. One pass is usually
        enough (the system is close to linear). Returns a report dict.

        settle_s : seconds to wait for steady flow before measuring. Low flow
            through compliant tubing settles slowly — increase if the reading
            is still drifting. measure_s : averaging window after settling.
        """
        if self.sensor is None:
            raise DualSyringeError("No sensor attached; cannot calibrate.")
        self._guard_line_rate(line_rate)
        history = []
        for i in range(1, iterations + 1):
            self.hold(line_rate, seconds=settle_s + measure_s + 2)
            self._log(f"[calibrate {i}/{iterations}] settling {settle_s:g}s…")
            time.sleep(settle_s)
            stats = self.read_flow(seconds=measure_s)
            self.stop()
            measured = abs(stats["mean"])
            if measured < 1e-6:
                raise DualSyringeError(
                    "Measured ~0 flow while pumping — check the line isn't "
                    "blocked/open, the Flow Unit is in-line, and the syringes "
                    "actually move. cal_factor left unchanged.")
            old = self.cal_factor
            self.cal_factor = old * (measured / abs(line_rate))
            rec = {
                "iteration": i, "target": abs(line_rate),
                "measured_mean": stats["mean"], "measured_std": stats["stdev"],
                "sign": "+" if stats["mean"] >= 0 else "-",
                "old_cal_factor": old, "new_cal_factor": self.cal_factor,
            }
            history.append(rec)
            self._log(
                f"[calibrate {i}/{iterations}] target {abs(line_rate):g}, "
                f"measured {stats['mean']:+.3f} ±{stats['stdev']:.3f} {stats['unit']} "
                f"-> cal_factor {old:.4f} → {self.cal_factor:.4f}")
            if stats["mean"] < 0:
                self._log("  note: sensor read NEGATIVE while infusing — the Flow "
                          "Unit's arrow is opposite your flow. Readings are fine "
                          "(magnitude used); flip the unit if you want +ve values.")
            if abs(measured - abs(line_rate)) / abs(line_rate) < 0.02:
                break  # within 2% — good enough, stop early
        if stop_after:
            self.stop()
        return {"final_cal_factor": self.cal_factor, "history": history}

    def verify(self, line_rate=60.0, settle_s=30.0, measure_s=15.0,
               stop_after=True):
        """Run at `line_rate` and report measured vs target WITHOUT changing
        cal_factor. Use after calibrate() to confirm, or any time to check."""
        if self.sensor is None:
            raise DualSyringeError("No sensor attached; cannot verify.")
        self.hold(line_rate, seconds=settle_s + measure_s + 2)
        self._log(f"[verify] settling {settle_s:g}s…")
        time.sleep(settle_s)
        stats = self.read_flow(seconds=measure_s)
        if stop_after:
            self.stop()
        err = stats["mean"] - abs(line_rate)
        rec = {"target": abs(line_rate), "measured_mean": stats["mean"],
               "measured_std": stats["stdev"], "error": err,
               "error_pct": 100.0 * err / abs(line_rate), "unit": stats["unit"]}
        self._log(f"[verify] target {abs(line_rate):g} {stats['unit']}, "
                  f"measured {stats['mean']:+.3f} ±{stats['stdev']:.3f} "
                  f"({rec['error_pct']:+.1f}%)")
        return rec

    # ------------------------------------------------------- persist cal
    def save_calibration(self, path="dual_syringe_cal.json"):
        import json
        with open(path, "w") as f:
            json.dump({"cal_factor": self.cal_factor,
                       "diameter_mm": self.diameter_mm,
                       "n_syringes": self.n_syringes}, f, indent=2)
        self._log(f"[cal] saved -> {path}")
        return path

    def load_calibration(self, path="dual_syringe_cal.json"):
        import json
        with open(path) as f:
            d = json.load(f)
        self.cal_factor = float(d.get("cal_factor", self.cal_factor))
        self._log(f"[cal] loaded cal_factor={self.cal_factor:.4f} from {path}")
        return self.cal_factor


if __name__ == "__main__":
    # SAFE by default: prints the computed pump plan for an example target and
    # does NOT move the pump. Pass --calibrate to actually run a calibration.
    import argparse
    ap = argparse.ArgumentParser(description="Dual-syringe combined-line helper")
    ap.add_argument("--diameter", type=float, default=19.13,
                    help="single 20 mL syringe inner Ø (mm)")
    ap.add_argument("--rate", type=float, default=60.0,
                    help="target COMBINED line rate (µL/min)")
    ap.add_argument("--volume", type=float, default=2000.0,
                    help="COMBINED volume to plan for (µL)")
    ap.add_argument("--calibrate", action="store_true",
                    help="actually run calibrate() against the sensor (MOVES the pump)")
    args = ap.parse_args()

    if not args.calibrate:
        # Dry run: no hardware needed to just show the arithmetic.
        line = DualSyringeLine(diameter_mm=args.diameter, require_sensor=False,
                               pump=object(), sensor=None, verbose=False)
        import json
        print("Dry-run plan (nothing moved):")
        print(json.dumps(line.plan(args.volume, args.rate), indent=2))
        print("\nRun with --calibrate to trim against the Fluigent sensor.")
    else:
        with DualSyringeLine(diameter_mm=args.diameter) as line:
            print("Calibrating…")
            print(line.calibrate(args.rate))
            print("Verify:", line.verify(args.rate))
            line.save_calibration()
