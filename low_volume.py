"""Low-volume mode: stay in each well until it runs dry, then move on.

Why this exists
---------------
The normal sequence gives every well a fixed dwell. With small sample volumes
that either wastes time (the well emptied long ago and the needle is sucking
air) or cuts the sample short. Here the robot is told to dwell for a long
safety cap, and the run leaves the well as soon as the sensor shows the AIR
that follows the well running dry.

Per well:

1. Move in with ``@P`` and a long dwell cap (``max_dwell_s``).
2. Watch the metabolite channels for the well-empty air spike. The detector is
   told when the needle entered (``note_move``) so it can ignore the air gap
   that every lift-out sends down the line, which arrives one dead volume later.
3. Air seen -> ``@T``. Verified from the AMUZA logs (22 Oct 2026): ``@T``
   mid-dwell aborts the dwell and returns the needle home (state 05 for 6-8 s,
   then 01 READY at position 000), and a following ``@P`` works normally.
4. After a grace period (``escalate_after_s``, long enough for the well's
   fluid to reach the sensor), check the signal every ``step_every_s``. If the
   metabolite channels are still MOVING (fresh fluid is arriving) leave the
   pump alone: the well is draining fine. Only a STALE signal, flat over the
   last ``fresh_window_s``, raises the pump rate by ``step``, up to
   ``max_rate``. (Elapsed time alone is not a reason to push harder.)
5. Stale for ``hold_at_max_s`` at ``max_rate`` -> treat it as a CLOG.
   Go home (``@T``), run the Flow Control tab's clog clearing there, where
   bursts and reverse pushes cannot disturb a well's reading, then go back into
   the same well at the base rate and keep watching. Give up after
   ``max_clears`` clearing attempts and halt the run.

The controller only decides; it talks to the robot, the pump and the detector
through small interfaces so it can be tested without hardware. The robot side
is ``AsyncAmuzaConnection`` itself; the pump side is ``FlowTabPump`` below.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict, fields
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["LowVolumeConfig", "LowVolumeController", "FlowTabPump"]


@dataclass
class LowVolumeConfig:
    max_dwell_s: int = 900
    """Safety cap sent to the robot as the dwell. The run normally leaves long
    before this; reaching it means the well never showed air."""

    escalate_after_s: float = 90.0
    """Grace period after entering before the signal is judged. Must exceed
    the dead-volume lag (65 s at 120 µL/min), or the flat buffer signal still
    in the line reads as stale."""

    step: float = 20.0
    """Pump rate increase per escalation step, µL/min (combined line)."""

    step_every_s: float = 30.0
    """Seconds between signal checks (and so between escalation steps)."""

    fresh_window_s: float = 30.0
    """The signal is compared with its level this long ago."""

    fresh_k: float = 6.0
    """Change needed to count as moving, in units of the channel's own sample
    noise (median absolute first difference)."""

    fresh_min_channels: int = 2
    """Channels that must be moving for the signal to count as fresh."""

    max_rate: float = 300.0
    """Ceiling for the escalation, µL/min. The pump has no pressure sensor, so
    this rate is the stand-in for "maximum pressure"."""

    hold_at_max_s: float = 60.0
    """How long to run at max_rate without air before calling it a clog."""

    leave_on_dip: bool = True
    """Leave the well on the suction-break dip, the moment the tip meets air
    (no tube delay). Switches on the detector's fast-dip path (fast_dip),
    chosen 22 Oct 2026 from a 10-method comparison over 30 runs: it caught
    all 7 real dry wells ~5 s after the dip (the spike-confirming detector
    caught 2/7 at ~33 s) at ~1.6 false alarms per in-well hour, about one
    early exit every 33 wells. Off = wait for the air spike."""

    plateau_check: bool = True
    """Accept a well-empty call as a GOOD sample only if, just before the tip
    met air, at least 2 channels sat on a flat plateau clearly above the
    buffer baseline (Noah, 25 Oct 2026: "the flatline beforehand is
    essential"). Scored on 15 labelled plates: false alarms 27 -> 20 with no
    loss of bubbles in wells that had a plateau; on 22 Oct all 6 dry wells
    with a plateau pass and the re-entry into an already-empty well fails.
    A call WITHOUT a plateau is not ignored (the needle could be sitting in
    an empty well): the run waits plateau_confirm_s for a plateau to appear
    (then it was a glitch and listening continues), otherwise it leaves and
    flags the well as having no valid plateau."""

    plateau_window_s: float = 15.0
    """Length of the window, ending plateau_gap_s before the air, that must be flat."""

    plateau_gap_s: float = 2.0
    """Gap left between the plateau window and the moment the tip met air."""

    plateau_k: float = 8.0
    """Plateau must sit this many noise units above the baseline."""

    baseline_window_s: float = 600.0
    """Baseline = the low (10th percentile of a 5-sample median) of the last
    this-many seconds. Not the level just before the move: with a 60 s
    buffer the previous well's tail is still there (22 Oct A2, A4)."""

    plateau_confirm_s: float = 20.0
    """After a call with no plateau, wait this long before leaving."""

    dip_stale_s: float = 180.0
    """A suction-break dip (the tip met air) followed by a stale signal within
    this long means the well is DRY, not clogged. Seen 22 Oct 2026 well A2:
    dip at 122 s, then the reading froze at its plateau with no air spike for
    over two minutes; with air in the needle the pump's suction expands the air
    instead of moving liquid past the sensor. A clog never puts the tip in air,
    so it has no dip."""

    max_clears: int = 2
    """Clog-clearing trips home per well before the run halts."""

    clear_timeout_s: float = 300.0
    """Longest a clearing attempt may take before returning anyway."""

    @classmethod
    def from_dict(cls, d: dict) -> "LowVolumeConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


class FlowTabPump:
    """Adapter from the controller to the Flow Control tab.

    Every method is safe without a pump: the controller then simply cannot
    escalate or clear, and only the dwell cap applies."""

    def __init__(self, tab):
        self.tab = tab

    @property
    def available(self) -> bool:
        return self.tab is not None and getattr(self.tab, "line", None) is not None

    def base_rate(self) -> float:
        try:
            return float(self.tab._num(self.tab.f_rate, 0.0))
        except Exception:
            return 0.0

    def set_rate(self, rate: float, label: str = "") -> None:
        if self.available and hasattr(self.tab, "set_override_rate"):
            self.tab.set_override_rate(rate, label)

    def request_clear(self) -> bool:
        if not self.available or not hasattr(self.tab, "request_clog_clear"):
            return False
        return bool(self.tab.request_clog_clear("low-volume"))

    def clearing(self) -> bool:
        return bool(getattr(self.tab, "_clearing", False))

    def abort_clear(self) -> None:
        if hasattr(self.tab, "abort_clog_clear"):
            self.tab.abort_clog_clear()


class LowVolumeController:
    """Decides when to leave a well. One instance per run.

    ``feed(t, channels)`` is called from the sensor callback for every reading;
    ``run_in_well(conn, method, port, stop_event, ...)`` is awaited by
    ``AsyncAmuzaConnection.execute_method`` right after the move is confirmed.
    """

    def __init__(self, config: LowVolumeConfig, detector, pump=None,
                 clock: Callable[[], float] = time.monotonic,
                 poll_s: float = 0.5):
        self.cfg = config
        self.detector = detector
        self.pump = pump
        self.clock = clock
        self.poll_s = poll_s
        self._in_well = False
        self._empty_at: Optional[float] = None
        self._last_dip: Optional[float] = None
        self._manual_next = False
        self._empty_reason = ""
        self._hist = deque()                      # (t, channels) for freshness
        self._long = deque()                      # (t, 5-sample median) for the baseline
        self._entered_t: Optional[float] = None
        self._pending_noplat: Optional[float] = None
        self._dabs = [deque(maxlen=300) for _ in range(6)]
        self._prev = None
        self.last_error: Optional[str] = None
        self.log: list = []          # (well, outcome, seconds_in_well, clears)
        self.flagged: list = []      # wells left with no valid plateau

    # ---- operator ---------------------------------------------------------
    def request_next(self) -> None:
        """The operator says this well is done: leave it (or stop trying to go
        back into it) and move to the next well. Honoured at every stage."""
        self._manual_next = True
        logger.info("Low-volume: operator marked the current well done")

    def _set_rate(self, rate, label):
        self.pump.set_rate(rate, label)
        if hasattr(self.detector, "set_flow_rate"):
            try:
                self.detector.set_flow_rate(rate)
            except Exception:
                pass

    # ---- sensor side ------------------------------------------------------
    def feed(self, t: float, channels) -> None:
        """One sensor reading. Only acted on while the needle is in a well."""
        self._record(t, channels)
        try:
            ev = self.detector.update(t, channels)
        except Exception as e:  # noqa: BLE001 — a detector bug must not kill the run
            logger.debug(f"well-empty detector error: {e}")
            return
        if ev is None or not self._in_well or self._empty_at is not None:
            return
        kind = getattr(ev, "kind", "EMPTY")
        if kind != "EMPTY":
            # SUSPECT = a suction-break dip or a lone spike. A dip is remembered:
            # followed by a stale signal it means the well is dry.
            if "dip" in str(getattr(ev, "reason", "")):
                self._last_dip = t
            logger.info(f"Low-volume: {getattr(ev, 'message', ev)}")
            return
        reason = "dip" if "dip" in str(getattr(ev, "reason", "")) else "air"
        if self.cfg.plateau_check:
            t_air = getattr(ev, "t_air", None)
            t_ref = t if t_air is None else min(t, float(t_air))
            if not self.plateau_before(t_ref):
                # No valid plateau before the air. Do not trust it, but do not
                # ignore it either: wait for a plateau, else leave flagged.
                if self._pending_noplat is None:
                    self._pending_noplat = t
                    logger.info(f"Low-volume: {getattr(ev, 'message', ev)} but NO steady "
                                f"plateau before it; waiting {self.cfg.plateau_confirm_s:g}s")
                if self._entered_t is not None and hasattr(self.detector, "rearm"):
                    self.detector.rearm(self._entered_t)
                return
        self._pending_noplat = None
        self._empty_at = t
        self._empty_reason = reason
        logger.info(f"Low-volume: {getattr(ev, 'message', ev)}")

    # ---- freshness: is fresh fluid still reaching the sensor? -------------
    def _record(self, t, channels):
        try:
            ch = [float(x) for x in list(channels)[:6]]
        except (TypeError, ValueError):
            return
        if self._prev is not None:
            for i, (a, b) in enumerate(zip(ch, self._prev)):
                self._dabs[i].append(abs(a - b))
        self._prev = ch
        self._hist.append((t, ch))
        last5 = [c for _, c in list(self._hist)[-5:]]
        self._long.append((t, [_median([c[i] for c in last5]) for i in range(6)]))
        while self._long and t - self._long[0][0] > self.cfg.baseline_window_s:
            self._long.popleft()
        keep = self.cfg.fresh_window_s + 30.0
        while self._hist and t - self._hist[0][0] > keep:
            self._hist.popleft()

    def signal_fresh(self, now: float) -> Optional[bool]:
        """True = the channels moved over the last fresh_window_s (fluid is
        arriving), False = flat (stale), None = not enough data to judge.

        Levels are 5-sample medians, so a single air spike or glitch cannot
        make a stale signal look fresh. Each channel is scaled by its own
        noise, because amplitudes differ ~50x between channels."""
        w = self.cfg.fresh_window_s
        if not self._hist or self._hist[-1][0] - self._hist[0][0] < w:
            return None
        recent = [c for _, c in self._hist][-5:]
        t_recent = self._hist[-len(recent)][0]
        past = [c for tt, c in self._hist if tt <= t_recent - w][-5:]
        if len(recent) < 2 or len(past) < 2:
            return None
        moving = 0
        for i in range(6):
            noise = _median(list(self._dabs[i])) if self._dabs[i] else 0.0
            noise = max(noise, 1e-4)
            change = abs(_median([c[i] for c in recent]) - _median([c[i] for c in past]))
            if change / noise >= self.cfg.fresh_k:
                moving += 1
        return moving >= self.cfg.fresh_min_channels

    def _noise(self, i):
        return max(_median(list(self._dabs[i])) * 1.4826 / 1.4142 if self._dabs[i] else 0.0, 1e-4)

    def plateau_before(self, t_ref: float) -> bool:
        """Was there a flat, non-baseline plateau just before t_ref?"""
        cfg = self.cfg
        lo, hi = t_ref - cfg.plateau_window_s - cfg.plateau_gap_s, t_ref - cfg.plateau_gap_s
        seg = [c for tt, c in self._hist if lo <= tt <= hi]
        base_pts = [c for tt, c in self._long if tt <= t_ref]
        if len(seg) < 4 or len(base_pts) < 10:
            return False
        good = 0
        for i in range(6):
            v = sorted(c[i] for c in seg)
            lvl = _median(v)
            rng = v[int(0.9 * (len(v) - 1))] - v[int(0.1 * (len(v) - 1))]
            b = sorted(c[i] for c in base_pts)
            base = b[int(0.1 * (len(b) - 1))]
            up, sig = lvl - base, self._noise(i)
            if up >= cfg.plateau_k * sig and up >= 0.05 and rng <= 0.5 * up + 3 * sig:
                good += 1
        return good >= 2

    def _enter(self, t: float) -> None:
        self._in_well = True
        self._empty_at = None
        self._last_dip = None
        self._entered_t = t
        self._pending_noplat = None
        try:
            self.detector.note_move(t)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"detector.note_move error: {e}")

    def _leave(self) -> None:
        self._in_well = False
        try:
            self.detector.note_lift(self.clock())
        except Exception:
            pass

    # ---- robot side -------------------------------------------------------
    async def _go_home(self, conn, timeout: float = 25.0) -> bool:
        """@T and wait for READY (the needle is home, in buffer)."""
        self._leave()
        await conn.stop_movement()
        t0 = self.clock()
        await asyncio.sleep(self.poll_s)
        while self.clock() - t0 < timeout:
            if conn.is_ready():
                return True
            await asyncio.sleep(self.poll_s)
        logger.warning("Low-volume: robot did not report READY after @T")
        return False

    async def _reenter(self, conn, method, port, on_move) -> bool:
        cmd = conn._format_method_command(port, self.cfg.max_dwell_s)
        if on_move:
            try:
                on_move(str(method.pos))
            except Exception as e:  # noqa: BLE001
                logger.error(f"on_move callback error: {e}")
        await conn.send_command(cmd, priority=_movement_priority(), expected_duration=0.2)
        if not await conn.wait_for_busy(timeout=5.0):
            await conn.send_command(cmd, priority=_movement_priority(), expected_duration=0.2)
            if not await conn.wait_for_busy(timeout=5.0):
                return False
        self._enter(self.clock())
        return True

    def _say(self, on_progress, msg, log=True):
        if log:
            logger.info(f"Low-volume: {msg}")
        if on_progress:
            try:
                on_progress(msg, 0, 0)
            except Exception:
                pass

    async def run_in_well(self, conn, method, port, stop_event,
                          on_progress=None, on_move=None) -> bool:
        """Hold the needle in the well until it runs dry. True = well done.

        Called with the move already sent and acknowledged."""
        cfg = self.cfg
        well = str(method.pos)
        self.last_error = None
        self._manual_next = False
        self._empty_reason = ""
        self._enter(self.clock())
        entered = self.clock()
        well_t0 = entered
        pump_ok = self.pump is not None and self.pump.available
        base = self.pump.base_rate() if pump_ok else 0.0
        rate = base
        last_step: Optional[float] = None
        at_max_since: Optional[float] = None
        clears = 0

        while True:
            now = self.clock()
            if stop_event.is_set():
                # STOP already halted the pump (flow_tab.pause_for_stop);
                # restarting it here would undo the operator's STOP.
                self._leave()
                await conn.stop_movement()
                return False

            # 0. The operator said this well is done.
            if self._manual_next:
                self._manual_next = False
                await self._go_home(conn)
                if pump_ok and rate != base:
                    self._set_rate(base, "well done (operator)")
                self.log.append((well, "operator", round(now - well_t0, 1), clears))
                self._say(on_progress, f"Buffer: {well} marked done by operator, moving on")
                return True

            # 1a. A call without a plateau: a plateau appearing now means it was a
            #     glitch (keep sampling); none within plateau_confirm_s means the
            #     well is empty or nothing flowed: leave, flagged.
            if self._pending_noplat is not None and self._empty_at is None:
                if self.plateau_before(now + cfg.plateau_gap_s):
                    logger.info(f"Low-volume: {well} plateau present after all; call ignored")
                    self._pending_noplat = None
                elif now - self._pending_noplat >= cfg.plateau_confirm_s:
                    self._empty_at, self._empty_reason = now, "noplateau"
                    self._pending_noplat = None

            # 1. The well ran dry -> leave.
            if self._empty_at is not None:
                await self._go_home(conn)
                if pump_ok and rate != base:
                    self._set_rate(base, "well done")
                outcome = {"air": "empty", "noplateau": "empty-NO-PLATEAU"}.get(
                    self._empty_reason, "empty-dip")
                self.log.append((well, outcome, round(now - well_t0, 1), clears))
                if self._empty_reason == "noplateau":
                    self.flagged.append(well)
                    self._say(on_progress, f"Buffer: {well} left after {now - entered:.0f}s: air "
                                           "with NO steady plateau before it (well empty or no "
                                           "flow) - its value is not valid, moving on")
                else:
                    why = {"air": "air spike", "dip": "suction dip"}.get(self._empty_reason, "dip")
                    self._say(on_progress, f"Buffer: {well} empty after {now - entered:.0f}s "
                                           f"({why}, plateau OK), moving on")
                return True

            # 2. The robot ran out its dwell cap by itself.
            if conn.is_ready() and now - entered > 15.0:
                self._leave()
                if pump_ok and rate != base:
                    self._set_rate(base, "dwell cap")
                self.log.append((well, "max_dwell", round(now - well_t0, 1), clears))
                logger.warning(f"Low-volume: {well} reached the {cfg.max_dwell_s}s dwell "
                               "cap without air")
                return True

            t_in = now - entered
            self._say(on_progress, f"Sampling: well {well} (low-volume, {t_in:.0f}s"
                                   + (f", pump {rate:g}" if pump_ok else "") + ")", log=False)

            # 3. Past the grace period, judge the signal every step_every_s.
            #    Moving = fluid is arriving, leave the pump alone. Flat = stale,
            #    push harder, up to max_rate.
            if pump_ok and t_in >= cfg.escalate_after_s \
                    and (last_step is None or now - last_step >= cfg.step_every_s):
                last_step = now
                fresh = self.signal_fresh(now)
                if fresh is False and self._last_dip is not None \
                        and now - self._last_dip <= cfg.dip_stale_s:
                    # Tip met air, then the reading froze: the well is dry.
                    logger.info(f"Low-volume: {well} dip {now - self._last_dip:.0f}s ago and "
                                "signal now stale -> well empty")
                    self._empty_at, self._empty_reason = now, "dip"
                    continue
                if fresh is True:
                    if at_max_since is not None:
                        logger.info(f"Low-volume: {well} signal moving again at "
                                    f"{rate:g}; clog countdown cancelled")
                    at_max_since = None
                elif fresh is False and rate < cfg.max_rate:
                    rate = min(cfg.max_rate, max(rate, base) + cfg.step)
                    self._set_rate(rate, f"stale signal {well}")
                    logger.info(f"Low-volume: {well} signal stale after {t_in:.0f}s, "
                                f"pump -> {rate:g}")
                    if rate >= cfg.max_rate:
                        at_max_since = now
                elif fresh is False and at_max_since is None:
                    at_max_since = now

            # 4. At max_rate for long enough and still nothing -> clog.
            if pump_ok and at_max_since is not None \
                    and now - at_max_since >= cfg.hold_at_max_s:
                if self._last_dip is not None:
                    # The tip met air in this well: it is dry, not clogged.
                    self._empty_at, self._empty_reason = now, "dip"
                    continue
                clears += 1
                if clears > cfg.max_clears:
                    await self._go_home(conn)
                    self._set_rate(base, "giving up")
                    self.last_error = (f"{well}: no air after {cfg.max_clears} clog-clearing "
                                       f"attempts at up to {cfg.max_rate:g} µL/min. Run halted "
                                       "with the needle home; check the line.")
                    self.log.append((well, "failed", round(now - well_t0, 1), clears - 1))
                    return False
                self._say(on_progress, f"Buffer: {well} no air at {rate:g} µL/min, "
                                       f"clearing clog at home ({clears}/{cfg.max_clears})")
                await self._go_home(conn)
                self._set_rate(base, "before clearing")
                if self.pump.request_clear():
                    t0 = self.clock()
                    await asyncio.sleep(self.poll_s)
                    while self.pump.clearing() and self.clock() - t0 < cfg.clear_timeout_s:
                        if stop_event.is_set():
                            self.pump.abort_clear()
                            return False
                        if self._manual_next:
                            break
                        await asyncio.sleep(self.poll_s)
                    if self.pump.clearing():
                        self.pump.abort_clear()
                else:
                    logger.warning("Low-volume: clog clearing could not start")
                if stop_event.is_set():
                    return False
                if self._manual_next:
                    # Operator: do not go back in. The needle is already home.
                    self._manual_next = False
                    if self.pump.clearing():
                        self.pump.abort_clear()
                    self._set_rate(base, "well done (operator)")
                    self.log.append((well, "operator", round(self.clock() - well_t0, 1), clears))
                    self._say(on_progress, f"Buffer: {well} marked done by operator, moving on")
                    return True
                self._set_rate(base, "after clearing")
                if not await self._reenter(conn, method, port, on_move):
                    self.last_error = (f"AMUZA did not go back into {well} after clearing.")
                    return False
                entered = self.clock()
                rate, last_step, at_max_since = base, None, None
                continue

            await asyncio.sleep(self.poll_s)


def build_detector(cfg: "LowVolumeConfig", flow_rate=None):
    """The well-empty detector for a low-volume run, configured from cfg."""
    from well_empty_detector import WellEmptyDetector
    return WellEmptyDetector(flow_rate=flow_rate, fast_dip=bool(cfg.leave_on_dip))


def _median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _movement_priority():
    from amuza_async import CommandPriority
    return CommandPriority.MOVEMENT
