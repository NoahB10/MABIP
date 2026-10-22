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
4. No air after ``escalate_after_s`` -> raise the pump rate by ``step`` every
   ``step_every_s`` up to ``max_rate``. A well that is merely slow to empty
   empties faster.
5. Still no air after ``hold_at_max_s`` at ``max_rate`` -> treat it as a CLOG.
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
from dataclasses import dataclass, asdict, fields
from typing import Callable, Optional

logger = logging.getLogger(__name__)

__all__ = ["LowVolumeConfig", "LowVolumeController", "FlowTabPump"]


@dataclass
class LowVolumeConfig:
    max_dwell_s: int = 900
    """Safety cap sent to the robot as the dwell. The run normally leaves long
    before this; reaching it means the well never showed air."""

    escalate_after_s: float = 120.0
    """Time in the well with no air before the pump rate starts rising. Must be
    longer than the dead-volume lag plus the expected empty time, or slow-but-
    healthy wells get pushed harder than needed."""

    step: float = 20.0
    """Pump rate increase per escalation step, µL/min (combined line)."""

    step_every_s: float = 30.0
    """Seconds between escalation steps."""

    max_rate: float = 300.0
    """Ceiling for the escalation, µL/min. The pump has no pressure sensor, so
    this rate is the stand-in for "maximum pressure"."""

    hold_at_max_s: float = 60.0
    """How long to run at max_rate without air before calling it a clog."""

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
        self.last_error: Optional[str] = None
        self.log: list = []          # (well, outcome, seconds_in_well, clears)

    # ---- sensor side ------------------------------------------------------
    def feed(self, t: float, channels) -> None:
        """One sensor reading. Only acted on while the needle is in a well."""
        try:
            ev = self.detector.update(t, channels)
        except Exception as e:  # noqa: BLE001 — a detector bug must not kill the run
            logger.debug(f"well-empty detector error: {e}")
            return
        if ev is None or not self._in_well or self._empty_at is not None:
            return
        kind = getattr(ev, "kind", "EMPTY")
        if kind != "EMPTY":
            # SUSPECT = a suction-break dip or a lone spike: early warning only.
            logger.info(f"Low-volume: {getattr(ev, 'message', ev)}")
            return
        self._empty_at = t
        logger.info(f"Low-volume: {getattr(ev, 'message', ev)}")

    def _enter(self, t: float) -> None:
        self._in_well = True
        self._empty_at = None
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
                self._leave()
                await conn.stop_movement()
                if pump_ok and rate != base:
                    self.pump.set_rate(base, "stopped")
                return False

            # 1. The well ran dry -> leave.
            if self._empty_at is not None:
                await self._go_home(conn)
                if pump_ok and rate != base:
                    self.pump.set_rate(base, "well done")
                self.log.append((well, "empty", round(now - well_t0, 1), clears))
                self._say(on_progress, f"Buffer: {well} empty after {now - entered:.0f}s, "
                                       "moving on")
                return True

            # 2. The robot ran out its dwell cap by itself.
            if conn.is_ready() and now - entered > 15.0:
                self._leave()
                if pump_ok and rate != base:
                    self.pump.set_rate(base, "dwell cap")
                self.log.append((well, "max_dwell", round(now - well_t0, 1), clears))
                logger.warning(f"Low-volume: {well} reached the {cfg.max_dwell_s}s dwell "
                               "cap without air")
                return True

            t_in = now - entered
            self._say(on_progress, f"Sampling: well {well} (low-volume, {t_in:.0f}s"
                                   + (f", pump {rate:g}" if pump_ok else "") + ")", log=False)

            # 3. No air yet -> push harder, up to max_rate.
            if pump_ok and t_in >= cfg.escalate_after_s and rate < cfg.max_rate \
                    and (last_step is None or now - last_step >= cfg.step_every_s):
                rate = min(cfg.max_rate, max(rate, base) + cfg.step)
                self.pump.set_rate(rate, f"escalate {well}")
                logger.info(f"Low-volume: {well} no air after {t_in:.0f}s, pump -> {rate:g}")
                last_step = now
                if rate >= cfg.max_rate:
                    at_max_since = now

            # 4. At max_rate for long enough and still nothing -> clog.
            if pump_ok and at_max_since is not None \
                    and now - at_max_since >= cfg.hold_at_max_s:
                clears += 1
                if clears > cfg.max_clears:
                    await self._go_home(conn)
                    self.pump.set_rate(base, "giving up")
                    self.last_error = (f"{well}: no air after {cfg.max_clears} clog-clearing "
                                       f"attempts at up to {cfg.max_rate:g} µL/min. Run halted "
                                       "with the needle home; check the line.")
                    self.log.append((well, "failed", round(now - well_t0, 1), clears - 1))
                    return False
                self._say(on_progress, f"Buffer: {well} no air at {rate:g} µL/min, "
                                       f"clearing clog at home ({clears}/{cfg.max_clears})")
                await self._go_home(conn)
                self.pump.set_rate(base, "before clearing")
                if self.pump.request_clear():
                    t0 = self.clock()
                    await asyncio.sleep(self.poll_s)
                    while self.pump.clearing() and self.clock() - t0 < cfg.clear_timeout_s:
                        if stop_event.is_set():
                            self.pump.abort_clear()
                            return False
                        await asyncio.sleep(self.poll_s)
                    if self.pump.clearing():
                        self.pump.abort_clear()
                else:
                    logger.warning("Low-volume: clog clearing could not start")
                if stop_event.is_set():
                    return False
                self.pump.set_rate(base, "after clearing")
                if not await self._reenter(conn, method, port, on_move):
                    self.last_error = (f"AMUZA did not go back into {well} after clearing.")
                    return False
                entered = self.clock()
                rate, last_step, at_max_since = base, None, None
                continue

            await asyncio.sleep(self.poll_s)


def _movement_priority():
    from amuza_async import CommandPriority
    return CommandPriority.MOVEMENT
