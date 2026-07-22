#!/usr/bin/env python3
"""Headless MABIP rig — run experiments from the command line, no GUI.

Drives the same backends as the GUI (dual_syringe pump+sensor, AMUZA robot) so
experiments can be scripted, logged and plotted autonomously:

    python headless_rig.py sensor [seconds]          # live flow readout (sanity)
    python headless_rig.py settle 40                 # closed-loop settle to a rate
    python headless_rig.py calibrate [rate]          # trim cal_factor at rate (def 40)
    python headless_rig.py calib ../experiments/Burst_Calibration.txt
    python headless_rig.py wells ../experiments/Burst_experiment.txt

Outputs land in Sensor_Readings/ as timestamped CSVs + PNG plots.

Correctness rules (see .claude/skills/write-experiment):
* every measurement starts only after the SENSOR confirms the target rate —
  closed loop; keeps waiting while flow is still approaching; a genuine STALL
  (no progress toward target) HALTS the experiment, it never "starts anyway".
* everything that must be *measured* stays inside the Flow Unit M ±120 range.
* one thread owns the sensor (the sampler); everyone else reads its cache.
* every phase is tagged in the trace CSV; Ctrl-C stops pump + robot cleanly.
"""

import os
import sys
import csv
import json
import time
import signal
import threading
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_HW = os.path.join(_HERE, os.pardir, "hardware")   # vendored drivers in the repo
_PC = "/home/rpi/pumpcontrol-project"              # live dev copies on the Pi win
for p in (_HW, os.path.join(_PC, "fgt-SDK", "Python"), _PC, _HERE):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")               # headless plotting
import matplotlib.pyplot as plt

from experiment_parse import parse_experiment, num_list

OUT_DIR = os.path.join(_HERE, "Sensor_Readings")
SETTINGS = os.path.expanduser("~/.mabip/flow_settings.json")
SENSOR_MAX = 120.0                  # Flow Unit M full scale (µL/min)


class Stall(RuntimeError):
    """Flow stopped approaching the target — air/blockage. NEVER measure through it."""


class Rig:
    """Pump + flow sensor with a single sampler thread, closed-loop settling and
    tagged high-rate CSV logging."""

    def __init__(self, hz=20.0):
        cfg = {}
        try:
            cfg = json.load(open(SETTINGS)).get("cfg", {})
        except Exception:
            pass
        self.n = max(1, int(cfg.get("n", 2)))
        self.diameter = float(cfg.get("diameter", 19.13))
        self.direction = cfg.get("direction", "withdraw")
        self.rev = "infuse" if self.direction == "withdraw" else "withdraw"
        self.hz = float(hz)
        self.flow = None            # latest sensor value (sampler-owned)
        self.cmd_ratio = 1.0        # converged command/target ratio (line resistance)
        self.segment = "idle"       # current tag written to every trace row
        self.stop_flag = False
        self.line = None
        self._sampler = None
        self._trace = None
        self._trace_lock = threading.Lock()
        # ONE mutex for every pump command. Multiple threads (servo, move timers,
        # main loop) talking to the Chemyx serial port concurrently interleave
        # writes and garble the protocol ("Command not recognized"), which can
        # strand the pump STOPPED mid-well. All pump access goes through this.
        self._pump_lock = threading.Lock()

    # ------------------------------------------------------------ connection
    def connect(self, need_sensor=True):
        from dual_syringe import DualSyringeLine
        self.line = DualSyringeLine(
            diameter_mm=self.diameter, n_syringes=self.n,
            direction=self.direction, pump_port="auto", require_sensor=False)
        print(f"pump connected (n={self.n}, Ø{self.diameter}, {self.direction})")
        if need_sensor:
            self.line.connect_sensor(channel=0)
            print("flow sensor connected")

    def start_trace(self, name):
        os.makedirs(OUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
        path = os.path.join(OUT_DIR, f"{name}_{ts}.csv")
        f = open(path, "w", newline="")
        w = csv.writer(f)
        w.writerow(["t_s", "flow_uL_min", "air", "segment"])
        t0 = time.monotonic()

        def sample():
            period = 1.0 / self.hz
            while not self.stop_flag:
                try:
                    v = float(self.line.sensor.read(self.line.sensor_channel))
                except Exception:
                    v = float("nan")
                try:
                    air = int(bool(self.line.sensor.air_bubble(self.line.sensor_channel)))
                except Exception:
                    air = 0
                self.flow = v
                with self._trace_lock:
                    w.writerow([f"{time.monotonic()-t0:.3f}", f"{v:.3f}", air, self.segment])
                time.sleep(period)
            f.flush(); f.close()

        self._trace = path
        self._sampler = threading.Thread(target=sample, daemon=True)
        self._sampler.start()
        print(f"trace → {path}  ({self.hz:g} Hz)")
        time.sleep(0.5)             # first samples in
        return path

    def shutdown(self):
        self.stop_flag = True
        if self.line is not None:
            try:
                self.line.stop()
            except Exception:
                pass
        if self._sampler is not None:
            self._sampler.join(timeout=2)

    # ------------------------------------------------------------ primitives
    def set_rate(self, line_rate, direction=None):
        """Command a combined line rate (per-syringe = rate/n). Serialized via the
        pump mutex; a rejected command is retried once after a beat, so a firmware
        hiccup can't strand the pump stopped (start_flow_single stops before it
        starts)."""
        for attempt in (1, 2):
            with self._pump_lock:
                try:
                    self.line.start_flow_single(abs(line_rate) / self.n,
                                                direction=direction or self.direction)
                    return True
                except Exception as e:
                    err = e
            if attempt == 1:
                time.sleep(0.4)              # let the firmware settle, then retry
        print(f"  ⚠ set_rate({line_rate:g}) failed twice: {err}")
        return False

    def pump_stop(self):
        with self._pump_lock:
            try:
                self.line.stop()
                return True
            except Exception as e:
                print(f"  ⚠ pump stop failed: {e}")
                return False

    def settle_to(self, target, tol=None, hold_s=2.0, adjust_s=5.0,
                  max_cmd=400.0, tag=None):
        """Closed-loop settle: WAIT until the SENSOR holds within ±tol of `target`
        for hold_s — servoing the pump command to make that happen.

        The measured flow often differs from the command (line resistance,
        calibration): a fixed command can plateau below target forever. So every
        `adjust_s` without progress the command is adjusted PROPORTIONALLY
        (cmd *= target/measured, damped, capped at `max_cmd`) — it escalates hard
        when nothing moves and eases off as the sensor approaches, so the measured
        flow converges on the target instead of ping-ponging and never runs past
        it. The converged ratio is remembered (self.cmd_ratio) so later settles
        start near the right command. Stall is declared only if even `max_cmd`
        produces essentially no flow (a real blockage/air) — then HALT, never
        measure."""
        target = abs(target)
        tol = tol if tol is not None else max(3.0, 0.08 * target)
        self.segment = tag or f"settle_{target:g}"
        cmd = min(max_cmd, max(target * 0.5, target * self.cmd_ratio))
        self.set_rate(cmd)
        best_err, last_adj, in_band = float("inf"), time.monotonic(), None
        no_flow_since = None
        t0 = time.monotonic()
        while not self.stop_flag:
            f = self.flow
            if f is None or f != f:          # no reading yet / NaN
                time.sleep(0.05); continue
            m = abs(f)
            err = abs(m - target)
            if err <= tol:
                in_band = in_band or time.monotonic()
                if time.monotonic() - in_band >= hold_s:
                    self.cmd_ratio = cmd / target      # remember line resistance
                    print(f"  settled at {f:.1f} (target {target:g}) in "
                          f"{time.monotonic()-t0:.1f}s — cmd {cmd:.1f} "
                          f"(ratio {self.cmd_ratio:.2f})")
                    return True
            else:
                in_band = None
            # fast guard: measured shot PAST the target → pull back immediately
            if m > target + 3 * tol:
                cmd = max(target * 0.5, cmd * max(0.5, target / m))
                self.set_rate(cmd); last_adj = time.monotonic()
                print(f"  over target ({m:.1f} > {target:g}) → cmd down to {cmd:.1f}")
                time.sleep(0.05); continue
            if err < best_err - max(1.0, 0.25 * tol):
                best_err = err               # progressing — no adjustment needed
                last_adj = time.monotonic()
            elif time.monotonic() - last_adj >= adjust_s:
                # plateaued off-target → proportional command adjustment
                if m < max(2.0, 0.1 * target):          # essentially no flow: kick hard
                    factor = 1.5
                    no_flow_since = no_flow_since or time.monotonic()
                    if cmd >= max_cmd and time.monotonic() - no_flow_since > 30:
                        self.line.stop()
                        raise Stall(f"no flow even at {max_cmd:g} µL/min (reads "
                                    f"{m:.1f}, target {target:g}) — real blockage/"
                                    f"air. Prime the line.")
                else:
                    no_flow_since = None
                    factor = max(0.6, min(1.8, target / m))   # damped proportional
                new_cmd = min(max_cmd, max(target * 0.5, cmd * factor))
                if abs(new_cmd - cmd) >= 0.5:
                    cmd = new_cmd
                    self.set_rate(cmd)
                    print(f"  at {m:.1f} (target {target:g}) → cmd {cmd:.1f}")
                last_adj = time.monotonic()
            time.sleep(0.05)
        return False

    def hold_servo(self, target, active, tol=None, adjust_s=3.0, max_cmd=400.0):
        """Closed-loop HOLD: keep the sensor at `target` for as long as active()
        is true — the well-long version of settle_to's servo. Proportional command
        adjustment (damped, capped); remembers the converged ratio. Never gives up
        while active; the caller decides when it ends (well ends / move starts)."""
        target = abs(target)
        tol = tol if tol is not None else max(3.0, 0.05 * target)
        cmd = min(max_cmd, max(target * 0.5, target * self.cmd_ratio))
        self.set_rate(cmd)
        last_adj = time.monotonic()
        while active() and not self.stop_flag:
            f = self.flow
            if f is None or f != f:
                time.sleep(0.05); continue
            m = abs(f)
            err = abs(m - target)
            if err <= tol:
                self.cmd_ratio = cmd / target          # good operating point
            elif time.monotonic() - last_adj >= adjust_s:
                if m < max(2.0, 0.1 * target):
                    factor = 1.4                        # kick a dead line
                else:
                    factor = max(0.7, min(1.5, target / m))
                new_cmd = min(max_cmd, max(target * 0.3, cmd * factor))
                if abs(new_cmd - cmd) >= 0.5:
                    cmd = new_cmd
                    self.set_rate(cmd)
                last_adj = time.monotonic()
            time.sleep(0.05)

    def kick_hold(self, target, active, tol=None, max_cmd=400.0):
        """KICK + MODEL + TRIM — the fast well controller.

        KICK : hard overdrive (calibrated up_cmd) until the sensor crosses
               ~60% of target (or 4 s), snapping a discharged/air-soft line
               into motion instead of crawling.
        MODEL: immediately command target × cmd_ratio — the continuously
               maintained estimate of what this line needs right now.
        TRIM : fast closed loop (1 s) with damped proportional correction and
               an instant pull-down if the sensor runs past the target.

        Runs while active(); keeps cmd_ratio fresh for the next well."""
        target = abs(target)
        tol = tol if tol is not None else max(3.0, 0.05 * target)
        kick_cmd = max_cmd
        try:
            cal = json.load(open(os.path.join(_HERE, "burst_calibration.json"))).get(str(self.n))
            if cal:
                kick_cmd = min(max_cmd, float(cal.get("up_cmd", max_cmd)))
        except Exception:
            pass
        # --- KICK (sensor-terminated) ---
        self.set_rate(kick_cmd)
        t0 = time.monotonic()
        while active() and not self.stop_flag and time.monotonic() - t0 < 4.0:
            f = self.flow
            if f is not None and f == f and abs(f) >= 0.6 * target:
                break
            time.sleep(0.05)
        # --- MODEL ---
        cmd = min(max_cmd, max(target * 0.5, target * self.cmd_ratio))
        self.set_rate(cmd)
        # --- TRIM (fast loop) ---
        last_adj = time.monotonic()
        while active() and not self.stop_flag:
            f = self.flow
            if f is None or f != f:
                time.sleep(0.05); continue
            m = abs(f)
            err = abs(m - target)
            if m > target + 3 * tol:                     # ran past → cut now
                cmd = min(max_cmd, max(target * 0.3, cmd * max(0.5, target / m)))
                self.set_rate(cmd); last_adj = time.monotonic()
            elif err <= tol:
                self.cmd_ratio = cmd / target            # model update
            elif time.monotonic() - last_adj >= 1.0:     # fast trim
                factor = max(0.7, min(1.4, target / m)) if m > 2.0 else 1.4
                new_cmd = min(max_cmd, max(target * 0.3, cmd * factor))
                if abs(new_cmd - cmd) >= 0.5:
                    cmd = new_cmd
                    self.set_rate(cmd)
                last_adj = time.monotonic()
            time.sleep(0.05)

    def wait_flow(self, cond, timeout_s, tag=None):
        """Wait until cond(flow) or timeout. Returns (met, seconds, extreme)."""
        if tag:
            self.segment = tag
        t0 = time.monotonic(); ext = None
        while not self.stop_flag:
            f = self.flow
            t = time.monotonic() - t0
            if f is not None and f == f:
                ext = f if ext is None else max(ext, f, key=abs)
                if cond(f):
                    return True, t, ext
            if t >= timeout_s:
                return False, t, ext
            time.sleep(0.02)
        return False, time.monotonic() - t0, ext


# ================================================================ burst calib
def run_burst_calibration(rig, p):
    """RISE: baseline→ceiling across up-rates. FALL: ceiling→baseline across
    reverse rates. Summary CSV + PNG curves. Escalates an up-rate ×1.5 if the
    ceiling isn't reached within rise_timeout_s."""
    base = float(p.get("baseline", 40))
    ceil = float(p.get("ceiling", 115))
    highs = num_list(p.get("high_mults", "4,5,6,7,8"))
    revs = num_list(p.get("rev_mults", "4,5,6,7,8"))
    reps = int(float(p.get("repeats", 3)))
    rise_to = float(p.get("rise_timeout_s", 20))
    max_rate = float(p.get("max_rate", 400))
    if ceil >= SENSOR_MAX:
        raise SystemExit(f"ceiling {ceil} ≥ sensor max {SENSOR_MAX} — unmeasurable")

    trace = rig.start_trace("Burst_Calib")
    summ_path = trace.replace(".csv", "_summary.csv")
    sf = open(summ_path, "w", newline="")
    sw = csv.writer(sf)
    sw.writerow(["sweep", "trial", "mult", "cmd_rate", "time_s",
                 "extreme_flow", "escalated_rate", "reached"])
    print(f"burst calibration: base {base} → ceil {ceil}, up ×{highs}, rev ×{revs}, "
          f"{reps} reps")

    for trial in range(1, reps + 1):
        if rig.stop_flag:
            break
        # ---- RISE ----
        for m in highs:
            if rig.stop_flag:
                break
            # long hold → comparable line-pressure state at every trial start.
            # NOTE: the command↔flow ratio is a DYNAMIC pressure state, not a
            # constant — do not scale boost commands by it (a post-boost settle can
            # converge at ratio<1 and would cripple the next boost). Command the
            # nominal rate; the escalation ladder finds what actually moves the line,
            # and the 20 Hz trace yields (command, dF/dt) pairs for the real curve.
            rig.settle_to(base, hold_s=6.0, tag=f"settle_t{trial}")
            rate = esc = min(max_rate, m * base)
            rig.set_rate(rate)
            met, t, peak = rig.wait_flow(lambda f: abs(f) >= ceil, rise_to,
                                         tag=f"rise_t{trial}_x{m:g}")
            while not met and not rig.stop_flag and esc < max_rate:
                esc = min(max_rate, round(esc * 1.5, 1))
                print(f"  rise x{m:g}: {ceil} not reached in {rise_to}s → up to {esc}")
                rig.set_rate(esc)
                met, t, peak = rig.wait_flow(lambda f: abs(f) >= ceil, rise_to,
                                             tag=f"rise_t{trial}_x{m:g}_esc")
            sw.writerow(["rise", trial, m, rate, f"{t:.3f}",
                         f"{(peak or 0):.2f}", esc, int(met)]); sf.flush()
            print(f"  rise t{trial} x{m:g} ({rate:g}): "
                  f"{'reached' if met else 'MISSED'} {ceil} in {t:.2f}s")
            # pull back toward baseline before the next point
            rig.set_rate(min(max_rate, max(revs) * base), direction=rig.rev)
            rig.wait_flow(lambda f: abs(f) <= base, 15.0, tag=f"return_t{trial}")
            rig.set_rate(base)
        # ---- FALL ----
        ref = max_rate            # hard overdrive is what reliably snaps to the ceiling
        for m in revs:
            if rig.stop_flag:
                break
            rig.settle_to(base, hold_s=6.0, tag=f"settle_t{trial}")
            rig.set_rate(ref)
            up_ok, _, _ = rig.wait_flow(lambda f: abs(f) >= ceil, 2 * rise_to,
                                        tag=f"upref_t{trial}")
            if not up_ok:
                print(f"  fall x{m:g}: couldn't reach {ceil} — skipped"); continue
            revrate = min(max_rate, m * base)
            rig.set_rate(revrate, direction=rig.rev)
            met, t, _ = rig.wait_flow(lambda f: abs(f) <= base, 20.0,
                                      tag=f"fall_t{trial}_x{m:g}")
            rig.set_rate(base)                      # cut reverse, level at baseline
            _, _, mn = rig.wait_flow(lambda f: False, 2.0,
                                     tag=f"level_t{trial}_x{m:g}")
            sw.writerow(["fall", trial, m, revrate, f"{t:.3f}",
                         f"{(mn or 0):.2f}", "", int(met)]); sf.flush()
            print(f"  fall t{trial} x{m:g} ({revrate:g}): {ceil}→{base} in {t:.2f}s "
                  f"(min {mn if mn is not None else float('nan'):.0f})")
    rig.settle_to(base, tag="final_baseline")
    sf.close()
    print(f"summary → {summ_path}")
    plot_calibration(trace, summ_path)
    return summ_path


def plot_calibration(trace_path, summ_path):
    """Rise/fall time-vs-rate curves + the full tagged trace."""
    rows = list(csv.DictReader(open(summ_path)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, sweep, title in ((axes[0], "rise", "rise: baseline→ceiling"),
                             (axes[1], "fall", "fall: ceiling→baseline (reverse)")):
        pts = {}
        for r in rows:
            if r["sweep"] == sweep and r["reached"] == "1":
                pts.setdefault(float(r["mult"]), []).append(float(r["time_s"]))
        if pts:
            ms = sorted(pts)
            mean = [sum(pts[m]) / len(pts[m]) for m in ms]
            ax.plot(ms, mean, "o-")
            for m in ms:
                ax.scatter([m] * len(pts[m]), pts[m], alpha=0.35, s=14)
        ax.set_xlabel("rate multiplier (× baseline)")
        ax.set_ylabel("time (s)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    png = summ_path.replace("_summary.csv", "_curves.png")
    fig.savefig(png, dpi=130)
    print(f"curves → {png}")

    t, v = [], []
    for r in csv.DictReader(open(trace_path)):
        t.append(float(r["t_s"])); v.append(float(r["flow_uL_min"]))
    fig2, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(t, v, lw=0.6)
    ax.set_xlabel("t (s)"); ax.set_ylabel("flow (µL/min)")
    ax.set_title("full trace"); ax.grid(alpha=0.3)
    fig2.tight_layout()
    png2 = trace_path.replace(".csv", "_trace.png")
    fig2.savefig(png2, dpi=130)
    print(f"trace plot → {png2}")


# ================================================================ wells (robot)
async def run_wells(rig, exp):
    """Multi-run well-plate experiment with per-buffer auto-burst — headless port
    of the GUI orchestration (settle before each run per THE RULE)."""
    import asyncio
    from amuza_async import AsyncAmuzaConnection, Sequence, Method

    runs = exp.get("_runs") or [dict(exp, name=exp.get("name", "run"))]
    conn = AsyncAmuzaConnection(use_mock=False)
    if not await conn.connect():
        raise SystemExit("AMUZA connect failed")
    print("AMUZA connected")
    try:
        for i, run in enumerate(runs, 1):
            if rig.stop_flag:
                break
            name = run.get("name", f"run{i}")
            wells = [w.strip().upper() for w in
                     str(run.get("wells", "")).replace(";", ",").split(",") if w.strip()]
            rate = float(run.get("flow_rate", 40))
            t_smp = int(float(run.get("sample_time", 110)))
            t_buf = int(float(run.get("buffer_time", 60)))
            auto_b = str(run.get("auto_burst", "no")).lower() in ("yes", "true", "1")
            print(f"===== run {i}/{len(runs)} {name}: wells {wells}, "
                  f"{rate} µL/min, burst={'on' if auto_b else 'off'} =====")
            rig.settle_to(rate, tag=f"run_{name}_settle")     # THE RULE
            rig.segment = f"run_{name}"

            # ---- feed-forward move strategy (per run) ------------------------
            # pause_on_move: stop the pump `pause_after` s after each move command
            # (tip exits liquid) and resume `resume_after` s later (tip back in).
            # Resume styles: step (default) | ramp over resume_ramp s | BOOST at
            # resume_boost_rate for resume_boost_s then baseline (bang-bang, from
            # the burst calibration). A no-pause run with a boost set fires just
            # the boost at resume_after — recover from the air surge instead of
            # avoiding it.
            pause_on = str(run.get("pause_on_move", "no")).lower() in ("yes", "true", "1")
            pause_after = float(run.get("pause_after", 0.45))
            resume_after = float(run.get("resume_after", 10.0))
            ramp_s = float(run.get("resume_ramp", 0.0) or 0.0)
            boost_rate = float(run.get("resume_boost_rate", 0.0) or 0.0)
            boost_s = float(run.get("resume_boost_s", 0.0) or 0.0)
            servo_on = str(run.get("resume_servo", "no")).lower() in ("yes", "true", "1")
            kick_on = str(run.get("resume_kick_servo", "no")).lower() in ("yes", "true", "1")
            ff_gen = [0]
            servo_gen = [0]

            def on_move(well_id, _name=name, _rate=rate):
                ff_gen[0] += 1
                servo_gen[0] += 1                     # kill any well servo first
                gen = ff_gen[0]
                def do_pause():
                    if gen != ff_gen[0] or rig.stop_flag: return
                    rig.pump_stop()
                    rig.segment = f"run_{_name}_transit"
                def do_resume():
                    if gen != ff_gen[0] or rig.stop_flag: return
                    rig.segment = f"run_{_name}_resume"
                    if boost_rate > 0 and boost_s > 0:          # bang-bang resume
                        rig.set_rate(boost_rate)
                        threading.Timer(boost_s, lambda: (gen == ff_gen[0]
                                        and not rig.stop_flag and rig.set_rate(_rate))).start()
                    elif ramp_s > 0:                            # linear ramp 0→rate
                        def ramp():
                            steps = max(2, int(ramp_s * 2))
                            for k in range(1, steps + 1):
                                if gen != ff_gen[0] or rig.stop_flag: return
                                rig.set_rate(_rate * k / steps)
                                time.sleep(ramp_s / steps)
                        threading.Thread(target=ramp, daemon=True).start()
                    else:                                       # step
                        rig.set_rate(_rate)
                if pause_on:
                    threading.Timer(pause_after, do_pause).start()
                    threading.Timer(resume_after, do_resume).start()
                elif boost_rate > 0 and boost_s > 0:
                    threading.Timer(resume_after, do_resume).start()

            def on_progress(msg, cur, total, _run=run, _name=name, _rate=rate):
                if msg.startswith("Buffer:"):
                    rig.segment = f"run_{_name}_buffer"
                    servo_gen[0] += 1                 # leaving the well ends its servo
                    if auto_b:
                        threading.Thread(target=_burst, args=(rig, _run, _rate, _name),
                                         daemon=True).start()
                elif msg.startswith("Sampling:"):
                    rig.segment = f"run_{_name}_well"
                    if servo_on or kick_on:
                        # closed-loop hold of the baseline through the WHOLE well —
                        # adapts to whatever air state the line is in. kick_on adds
                        # the calibrated hard-kick + model feed-forward + fast trim.
                        servo_gen[0] += 1
                        gen = servo_gen[0]
                        seg = rig.segment
                        ctrl = rig.kick_hold if kick_on else rig.hold_servo
                        threading.Thread(
                            target=ctrl,
                            args=(_rate, lambda: (servo_gen[0] == gen
                                                  and rig.segment == seg)),
                            daemon=True).start()

            seq = Sequence(f"Run {name}")
            for w in wells:
                seq.add_method(Method(pos=w, wait=t_smp, buffer_time=t_buf,
                                      eject=False, insert=False))
            stop_evt = asyncio.Event()
            await conn.execute_sequence(seq, stop_evt, progress_callback=on_progress,
                                        move_callback=on_move)
    finally:
        await conn.disconnect()
        rig.pump_stop()


def _burst(rig, run, base_rate, name):
    """Fire one burst. Uses the CALIBRATED absolute protocol from
    burst_calibration.json when this syringe count has been measured (no
    multipliers); falls back to the run's multiplier settings otherwise."""
    mbase = base_rate / rig.n
    cal = None
    try:
        cal = json.load(open(os.path.join(_HERE, "burst_calibration.json"))).get(str(rig.n))
    except Exception:
        pass
    rig.segment = f"run_{name}_burst"
    if cal:
        rep = rig.line.burst(
            mbase, float(cal.get("up_cmd", 400.0)) / rig.n,
            float(cal.get("up_trigger", 115.0)),
            float(cal.get("up_max_s", 25.0)), 0.0,
            read_flow=lambda: rig.flow or 0.0, direction=rig.direction,
            backflow_rate=float(cal.get("rev_cmd", 280.0)) / rig.n,
            backflow_seconds=float(cal.get("rev_max_s", 25.0)),
            backflow_trigger_flow=base_rate + float(cal.get("rev_stop_offset", 8.0)),
            should_abort=lambda: rig.stop_flag)
    else:
        mult = float(run.get("b_mult", 1.7))
        hs = float(run.get("b_high_s", 10)); ss = float(run.get("b_stop_s", 8))
        bf = float(run.get("b_backflow_s", 0))
        rep = rig.line.burst(mbase, mult * mbase, rig.line.expected_combined(mbase),
                             hs, ss, read_flow=lambda: rig.flow or 0.0,
                             direction=rig.direction,
                             backflow_rate=(base_rate / rig.n if bf > 0 else 0),
                             backflow_seconds=bf,
                             should_abort=lambda: rig.stop_flag)
    print(f"  burst {name}{' [calibrated]' if cal else ''}: "
          f"peak {rep['peak_flow']:.0f}, {rep['total_s']:.1f}s")
    rig.segment = f"run_{name}_buffer"


# ================================================================ CLI
def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    rig = Rig()
    signal.signal(signal.SIGINT, lambda *_: setattr(rig, "stop_flag", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(rig, "stop_flag", True))
    try:
        if cmd == "sensor":
            rig.connect(); rig.start_trace("Sensor_Check")
            secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10
            t0 = time.monotonic()
            while time.monotonic() - t0 < secs and not rig.stop_flag:
                print(f"  flow = {rig.flow if rig.flow is not None else '—'} µL/min")
                time.sleep(1)
        elif cmd == "settle":
            rig.connect(); rig.start_trace("Settle_Check")
            rig.settle_to(float(sys.argv[2]))
        elif cmd == "calibrate":
            rig.connect()
            rate = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
            rep = rig.line.calibrate(line_rate=rate)
            print(f"cal_factor → {rep}")
        elif cmd == "calib":
            exp = parse_experiment(open(sys.argv[2]).read())
            rig.connect(); run_burst_calibration(rig, exp)
        elif cmd == "wells":
            import asyncio
            exp = parse_experiment(open(sys.argv[2]).read())
            rig.connect(); rig.start_trace("Wells")
            asyncio.run(run_wells(rig, exp))
        else:
            print(__doc__)
    except Stall as e:
        print(f"✗ STALLED — experiment halted, nothing measured: {e}")
        sys.exit(2)
    finally:
        rig.shutdown()


if __name__ == "__main__":
    main()
