#!/usr/bin/env python3
"""
timing_calibration.py — one-time AMUZA motion-timing calibration.

The probe motion is deterministic, so instead of a live liquid sensor we
measure, once, WHEN the tip crosses the air/water line relative to each move
command. You watch the probe and tap SPACE every time it LEAVES or ENTERS
liquid (a well or the buffer). The tool logs each tap with full machine
context (time since the move command, is_moving, countdown, current well), so
afterwards we can extrapolate the leave/enter times for every well and drive
the pump feed-forward off the move command — no sensor, no human, in the real run.

It is deliberately AGNOSTIC about the buffer choreography: whatever crossings
happen during a move, you tap them and they're all recorded. The first run
reveals the crossing pattern (e.g. well→buffer→well = 2 leaves + 2 enters);
we label LEAVE/ENTER from that.

Run (real robot):
    cd ~/Documents/MABIP/refactored
    python timing_calibration.py                 # default corner itinerary, 5 reps
    python timing_calibration.py --reps 3 --wells A1,A2,A12,H1,H12
    python timing_calibration.py --dwell 6 --machine "Machine 1"

Controls while running:
    SPACE  = mark a water-line crossing (leave OR enter)
    u      = undo the last tap (mis-click)
    s      = skip to the next move
    q      = quit early (still writes the log)

Output: Timing_Calibration_<ts>.csv  (+ a per-leg summary printed at the end).
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import termios
import tty
from datetime import datetime
from time import perf_counter

# amuza_async lives beside this file; pump/sensor backend is vendored in
# <repo>/hardware, with the Pi's pumpcontrol-project taking precedence.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HW = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "hardware")
_PC = "/home/rpi/pumpcontrol-project"
for _p in (_HW, os.path.join(_PC, "fgt-SDK", "Python"), _PC):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from amuza_async import (AsyncAmuzaConnection, CommandTiming, BLUETOOTH_AVAILABLE,  # noqa: E402
                         Method)

try:
    from config import HARDWARE
    DEVICES = HARDWARE.AMUZA_DEVICES
except Exception:
    DEVICES = {"Machine 1": "FC90-0034"}


# ----------------------------------------------------------------- geometry
def well_rc(well: str):
    """(row, col) 0-based for a well id like 'A1'/'H12'."""
    return ord(well[0].upper()) - ord("A"), int(well[1:]) - 1


def distance(a: str, b: str) -> float:
    """Euclidean plate distance between two wells (same metric the firmware model uses)."""
    (ra, ca), (rb, cb) = well_rc(a), well_rc(b)
    return math.hypot(ra - rb, ca - cb)


# ------------------------------------------------------------- keystroke I/O
class RawKeys:
    """Non-blocking single-key reader on stdin via asyncio (cbreak mode)."""

    def __init__(self, loop, on_key):
        self.loop = loop
        self.on_key = on_key
        self.fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.loop.add_reader(self.fd, self._read)
        return self

    def _read(self):
        try:
            ch = os.read(self.fd, 1).decode(errors="ignore")
        except Exception:
            return
        if ch:
            self.on_key(ch)

    def __exit__(self, *exc):
        try:
            self.loop.remove_reader(self.fd)
        finally:
            if self._old is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)


# ------------------------------------------------------------------- runner
class Calibrator:
    def __init__(self, conn, wells, reps, well_dwell, buffer_dwell, anchor, csv_path,
                 line=None, flow_rate=0.0, n_syringes=2, direction="withdraw"):
        self.conn = conn
        self.wells = wells
        self.reps = reps
        self.well_dwell = well_dwell      # dwell in a target sample well (s)
        self.buffer_dwell = buffer_dwell  # dwell on the anchor/buffer return (s)
        self.anchor = anchor          # well the probe returns to between targets
        self.csv_path = csv_path
        self.rows = []                # CSV rows (dicts)
        self.taps_this_leg = []       # perf_counter times of taps in the current leg
        self.leg = None               # dict describing the active move
        self._quit = False
        self._skip = False
        self._last_move_wells = None
        # optional fluidics (pump withdrawing + flow sensor)
        self.line = line              # DualSyringeLine or None
        self.flow_rate = flow_rate    # line rate (µL/min); machine = rate / n
        self.n_syringes = max(1, int(n_syringes))
        self.direction = direction
        self.latest_flow = None
        self.latest_air = 0
        self._flow_log_last = 0.0

    # ---- logging
    def _log(self, event, **kw):
        now = perf_counter()
        rel = (now - self.leg["t0"]) if self.leg else 0.0
        st = self.conn.status
        self.rows.append({
            "event": event,
            "perf_time": f"{now:.4f}",
            "rel_to_move_s": f"{rel:.3f}",
            "leg": self.leg["idx"] if self.leg else "",
            "from_well": self.leg["frm"] if self.leg else "",
            "to_well": self.leg["to"] if self.leg else "",
            "distance": f"{self.leg['dist']:.3f}" if self.leg else "",
            "tap_in_leg": kw.get("tap_in_leg", ""),
            "is_moving": 1 if getattr(st, "is_moving", False) else 0,
            "countdown": getattr(st, "countdown", 0),
            "current_well": getattr(st, "current_well", ""),
            "flow_uL_min": (f"{self.latest_flow:.3f}" if self.latest_flow is not None else ""),
            "air": self.latest_air,
            "note": kw.get("note", ""),
        })

    # ---- fluidics
    def _read_flow(self):
        """Update latest flow / air-bubble from the sensor (no-op if no sensor)."""
        if self.line is None or getattr(self.line, "sensor", None) is None:
            return
        ch = getattr(self.line, "sensor_channel", 0)
        try:
            self.latest_flow = float(self.line.sensor.read(ch))
        except Exception:
            self.latest_flow = None
        try:
            self.latest_air = 1 if self.line.sensor.air_bubble(ch) else 0
        except Exception:
            self.latest_air = 0

    # ---- key handling (called from the event loop)
    def on_key(self, ch):
        if ch == " ":
            if not self.leg:
                return
            self.taps_this_leg.append(perf_counter())
            n = len(self.taps_this_leg)
            self._log("KEY_SPACE", tap_in_leg=n)
            rel = perf_counter() - self.leg["t0"]
            kind = "LEAVE" if n % 2 == 1 else "ENTER"  # provisional (alternation)
            fl = f"  flow={self.latest_flow:.1f}" if self.latest_flow is not None else ""
            sys.stdout.write(f"\r  ⏱  tap #{n} @ {rel:5.2f}s  (prov. {kind}){fl}            \n")
            sys.stdout.flush()
        elif ch in ("u", "U"):
            if self.taps_this_leg:
                self.taps_this_leg.pop()
                self._log("UNDO_TAP", note="removed last tap")
                sys.stdout.write("\r  ↩  undid last tap                         \n")
                sys.stdout.flush()
        elif ch in ("s", "S"):
            self._skip = True
        elif ch in ("q", "Q"):
            self._quit = True

    # ---- status line printer
    async def _status_printer(self):
        prev_moving = None
        while not self._quit:
            st = self.conn.status
            mv = getattr(st, "is_moving", False)
            self._read_flow()
            if mv != prev_moving and self.leg:
                self._log("MOVING_EDGE", note=("up" if mv else "down"))
                prev_moving = mv
            # periodic flow sample (~2 Hz) so we can align sensor reaction to taps
            if self.line is not None and self.leg:
                now = perf_counter()
                if now - self._flow_log_last >= 0.5:
                    self._flow_log_last = now
                    self._log("FLOW_SAMPLE")
            if self.leg:
                rel = perf_counter() - self.leg["t0"]
                fl = f"{self.latest_flow:6.1f}" if self.latest_flow is not None else "  --  "
                air = "AIR" if self.latest_air else "   "
                sys.stdout.write(
                    f"\r  [{self.leg['frm']}→{self.leg['to']}] "
                    f"t={rel:5.1f}s  moving={int(mv)}  cd={getattr(st,'countdown',0):>3}  "
                    f"flow={fl} {air}  taps={len(self.taps_this_leg)}   (SPACE  s=skip  q=quit) "
                )
                sys.stdout.flush()
            await asyncio.sleep(0.1)

    # ---- one move
    async def _do_move(self, idx, frm, to):
        self.taps_this_leg = []
        self._skip = False
        dwell = self.buffer_dwell if to == self.anchor else self.well_dwell
        kind = "buffer" if to == self.anchor else "well"
        self.leg = {"idx": idx, "frm": frm, "to": to,
                    "dist": distance(frm, to), "t0": perf_counter()}
        print(f"\n── leg {idx}: {frm} → {to}  ({kind}, dist {distance(frm,to):.2f}, dwell {dwell}s)  "
              f"tap SPACE on every water-line crossing ──")
        self._log("MOVE_CMD", note=f"dwell={dwell} kind={kind}")
        # Drive the move the same way the GUI does: execute_method() does
        # wait_for_busy() then monitors is_ready() — so it actually blocks until
        # the robot has moved + dwelled (move_to_well returns too early and the
        # legs collapse). Interruptible via the stop_event on skip/quit.
        stop_event = asyncio.Event()
        method = Method(pos=to, wait=dwell, buffer_time=0)
        move_task = asyncio.create_task(self.conn.execute_method(method, stop_event))
        while not move_task.done():
            if self._quit or self._skip:
                stop_event.set()
                break
            await asyncio.sleep(0.05)
        try:
            await move_task
        except (Exception, asyncio.CancelledError):
            pass
        self._log("MOVE_DONE", note="skipped" if self._skip else "ok")
        # per-leg echo
        rels = [f"{t - self.leg['t0']:.2f}" for t in self.taps_this_leg]
        print(f"\n   leg {idx} taps (s since cmd): {rels if rels else '—'}")
        self._last_move_wells = (frm, to)

    # ---- itinerary
    def _itinerary(self):
        """Return a list of (from, to) moves. Anchor↔target pairs across reps,
        so each target distance is sampled repeatedly, plus the long diagonal."""
        legs = []
        cur = self.anchor
        for _ in range(self.reps):
            for w in self.wells:
                if w == cur:
                    continue
                legs.append((cur, w)); cur = w
                if cur != self.anchor:
                    legs.append((cur, self.anchor)); cur = self.anchor
        return legs

    async def run(self):
        print(f"\nItinerary anchor = {self.anchor};  targets = {', '.join(self.wells)};  "
              f"reps = {self.reps};  well dwell = {self.well_dwell}s;  buffer dwell = {self.buffer_dwell}s")
        print("Make sure the tray is inserted and the probe is at/above the anchor well.")
        # home to anchor first (also inserts tray)
        print("Inserting tray / homing to anchor…")
        try:
            await self.conn.insert()
        except Exception as e:
            print(f"(insert skipped: {e})")
        # use execute_method (waits for busy + monitors ready) so we really arrive
        await self.conn.execute_method(Method(pos=self.anchor, wait=3, buffer_time=0),
                                       asyncio.Event())

        # start the withdraw flow (tip is now submerged in the anchor well)
        if self.line is not None and self.flow_rate > 0:
            machine = self.flow_rate / self.n_syringes
            print(f"Starting {self.direction} flow: line {self.flow_rate:g} µL/min "
                  f"({machine:g}/syringe × {self.n_syringes}).")
            try:
                await asyncio.to_thread(self.line.start_flow_single, machine, self.direction)
            except Exception as e:
                print(f"(flow start failed: {e})")

        legs = self._itinerary()
        printer = asyncio.create_task(self._status_printer())
        try:
            for i, (frm, to) in enumerate(legs, 1):
                if self._quit:
                    break
                await self._do_move(i, frm, to)
        finally:
            self._quit = True
            printer.cancel()
            try:
                await printer
            except (Exception, asyncio.CancelledError):
                pass
            if self.line is not None:
                try:
                    await asyncio.to_thread(self.line.stop)
                    print("Flow stopped.")
                except Exception:
                    pass
        self._write_csv()

    # ---- output
    def _write_csv(self):
        import csv
        cols = ["event", "perf_time", "rel_to_move_s", "leg", "from_well", "to_well",
                "distance", "tap_in_leg", "is_moving", "countdown", "current_well",
                "flow_uL_min", "air", "note"]
        with open(self.csv_path, "w", newline="") as f:
            f.write(f"# AMUZA timing calibration — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"# move model: measured MOVE_TIME_TABLE={CommandTiming.MOVE_TIME_TABLE} "
                    f"(piecewise-linear by dist from A1, flat/saturating beyond)\n")
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        print(f"\n✅ wrote {self.csv_path}")
        self._summary()

    def _summary(self):
        # crossings per leg + rel times, so the crossing pattern is visible at a glance
        print("\n=== per-leg crossing summary (rel seconds since move cmd) ===")
        legs = {}
        for r in self.rows:
            if r["event"] == "KEY_SPACE":
                legs.setdefault(int(r["leg"]), {"info": (r["from_well"], r["to_well"], r["distance"]),
                                                "taps": []})["taps"].append(float(r["rel_to_move_s"]))
        if not legs:
            print("  (no taps recorded)")
            return
        for idx in sorted(legs):
            frm, to, d = legs[idx]["info"]
            taps = legs[idx]["taps"]
            labeled = "  ".join(f"{'LEAVE' if i % 2 == 0 else 'ENTER'}={t:.2f}"
                                for i, t in enumerate(taps))
            print(f"  leg {idx:>2} {frm:>3}→{to:<3} d={float(d):4.2f} | {len(taps)} taps | {labeled}")
        print("\nNext: share this CSV and I'll fit leave/enter-time vs distance and "
              "produce the per-well feed-forward table.")


async def amain():
    ap = argparse.ArgumentParser(description="AMUZA one-time motion-timing calibration")
    ap.add_argument("--wells", default="A1,A2,A12,H1,H12",
                    help="comma-separated target wells (corners recommended)")
    ap.add_argument("--anchor", default="A1", help="well the probe returns to between targets")
    ap.add_argument("--reps", type=int, default=5, help="repeats of the full itinerary")
    ap.add_argument("--well-dwell", type=int, default=25, help="seconds to dwell in each target well")
    ap.add_argument("--buffer-dwell", type=int, default=20, help="seconds to dwell on the anchor/buffer return")
    ap.add_argument("--machine", default="Machine 1", help="machine name from config")
    ap.add_argument("--mock", action="store_true", help="use the mock AMUZA (dry run of the tool)")
    # optional fluidics — pump withdrawing + flow sensor logging
    ap.add_argument("--flow", action="store_true",
                    help="also connect the pump+sensor, withdraw, and log flow/air (BONUS)")
    ap.add_argument("--flow-rate", type=float, default=80.0, help="line flow rate µL/min (--flow)")
    ap.add_argument("--diameter", type=float, default=19.13, help="syringe Ø mm (--flow)")
    ap.add_argument("--n", type=int, default=2, help="# syringes (--flow)")
    ap.add_argument("--pump-port", default="auto", help="Chemyx port or 'auto' (--flow)")
    ap.add_argument("--push", action="store_true", help="infuse instead of withdraw (--flow)")
    args = ap.parse_args()

    # Guard: never silently fall back to the mock Bluetooth socket (that looks
    # exactly like "connection failed 3 times"). This happens when run with a
    # Python that lacks PyBluez — e.g. the miniconda default instead of MABIP's .venv.
    if not args.mock and not BLUETOOTH_AVAILABLE:
        print("❌ PyBluez is NOT available in this Python — a real AMUZA connection is impossible.")
        print(f"   You're running: {sys.executable}")
        print("   Run it with MABIP's venv (the same one the GUI uses):")
        print("     source ~/Documents/MABIP/.venv/bin/activate")
        print("     cd ~/Documents/MABIP/refactored && python timing_calibration.py …")
        print("   (Or pass --mock to dry-run the tool without hardware.)")
        return

    wells = [w.strip().upper() for w in args.wells.split(",") if w.strip()]
    device_name = DEVICES.get(args.machine, "FC90-0034")
    ts = datetime.now().strftime("%d_%m_%y_%H_%M")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"Timing_Calibration_{ts}.csv")

    # optional pump+sensor (do this FIRST so a device conflict fails fast before homing)
    line = None
    if args.flow:
        print("Connecting pump + flow sensor (close the MABIP GUI first — devices are single-owner)…")
        try:
            from dual_syringe import DualSyringeLine
            line = await asyncio.to_thread(
                DualSyringeLine, diameter_mm=args.diameter, n_syringes=args.n,
                direction=("infuse" if args.push else "withdraw"),
                pump_port=args.pump_port, require_sensor=True, sensor=None, verbose=False)
            print(f"  pump + sensor OK (flow unit range {line.sensor.range()}).")
        except Exception as e:
            print(f"❌ pump/sensor connect failed: {e}\n   (Is the GUI still holding the pump/USB?)")
            return

    print(f"Connecting to {args.machine} ({device_name}), mock={args.mock} …")
    conn = AsyncAmuzaConnection(device_name=device_name, use_mock=args.mock)
    if not await conn.connect():
        print("❌ AMUZA connection failed (is the MABIP GUI still connected to it?)")
        if line is not None:
            await asyncio.to_thread(line.close)
        return

    cal = Calibrator(conn, wells, args.reps, args.well_dwell, args.buffer_dwell,
                     args.anchor, csv_path, line=line, flow_rate=args.flow_rate,
                     n_syringes=args.n, direction=("infuse" if args.push else "withdraw"))
    loop = asyncio.get_running_loop()
    try:
        with RawKeys(loop, cal.on_key):
            await cal.run()
    except KeyboardInterrupt:
        cal._quit = True
        cal._write_csv()
    finally:
        try:
            await conn.disconnect()
        except Exception:
            pass
        if line is not None:
            try:
                await asyncio.to_thread(line.stop)
                await asyncio.to_thread(line.close)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass
