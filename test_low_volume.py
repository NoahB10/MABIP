"""LowVolumeController: leave on air, escalate, clear clogs at home, give up."""

import asyncio
import time

import pytest

from amuza_async import Method
from low_volume import LowVolumeConfig, LowVolumeController


class FakeConn:
    """Robot: READY after @T (with a short move), busy after @P."""

    def __init__(self):
        self.sent = []
        self._ready = False

    async def stop_movement(self):
        self.sent.append("@T")
        self._ready = True
        return True

    def is_ready(self):
        return self._ready

    def _format_method_command(self, port, t):
        return f"@P,M1,{int(t):04d},{port:02d},\n"

    async def send_command(self, cmd, **kw):
        self.sent.append(cmd.strip())
        if cmd.startswith("@P"):
            self._ready = False
        return True

    async def wait_for_busy(self, timeout=5.0):
        return True


class FakePump:
    def __init__(self, base=100.0, clear_s=0.05):
        self.available = True
        self._base = base
        self.rates = []
        self.clears = 0
        self._clear_until = 0.0
        self.clear_s = clear_s

    def base_rate(self):
        return self._base

    def set_rate(self, rate, label=""):
        self.rates.append(rate)

    def request_clear(self):
        self.clears += 1
        self._clear_until = time.monotonic() + self.clear_s
        return True

    def clearing(self):
        return time.monotonic() < self._clear_until

    def abort_clear(self):
        self._clear_until = 0.0


class FakeDetector:
    def __init__(self):
        self.moves = []
        self.fire = False

    def note_move(self, t):
        self.moves.append(t)

    def update(self, t, ch):
        if self.fire:
            self.fire = False
            return "air"
        return None


def _cfg(**kw):
    base = dict(max_dwell_s=900, escalate_after_s=0.1, step=20, step_every_s=0.05,
                max_rate=160, hold_at_max_s=0.1, max_clears=2, clear_timeout_s=2.0,
                fresh_window_s=0.1)
    base.update(kw)
    return LowVolumeConfig(**base)


def _ctl(cfg=None, pump=None):
    det = FakeDetector()
    ctl = LowVolumeController(cfg or _cfg(), det, pump=pump if pump is not None else FakePump(),
                              poll_s=0.01)
    return ctl, det


def _signal(kind):
    """Six channels at time t: 'flat' (stale, tiny noise), 'rising', 'falling'."""
    import math
    def f(t, i):
        noise = 0.001 * math.sin(i * 12.9898)
        if kind == "flat":
            return [1.0 + noise] * 6
        slope = 5.0 if kind == "rising" else -5.0
        return [1.0 + slope * t + noise] * 6
    return f


async def _stream(ctl, sig, until_s=10.0, dt=0.01, switch=None):
    """Feed readings every dt. `switch=(after_s, other_sig)` changes the signal."""
    t0 = time.monotonic()
    i = 0
    while time.monotonic() - t0 < until_s:
        el = time.monotonic() - t0
        s = switch[1] if switch and el >= switch[0] else sig
        ctl.feed(time.monotonic(), s(el, i))
        i += 1
        await asyncio.sleep(dt)


async def _feed_after(ctl, det, delay):
    await asyncio.sleep(delay)
    det.fire = True
    ctl.feed(time.monotonic(), [0] * 6)


def test_leaves_as_soon_as_air_is_seen():
    ctl, det = _ctl(_cfg(escalate_after_s=10))
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_feed_after(ctl, det, 0.05))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert conn.sent == ["@T"]
    assert ctl.log[0][:2] == ("A1", "empty")
    assert ctl.pump.rates == []                    # never escalated


def test_air_outside_a_well_is_ignored():
    ctl, det = _ctl()
    det.fire = True
    ctl.feed(time.monotonic(), [0] * 6)            # needle not in a well
    assert ctl._empty_at is None


def test_escalates_in_steps_then_air_resets_to_base():
    ctl, det = _ctl(_cfg(hold_at_max_s=10))
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat")))
        asyncio.ensure_future(_feed_after(ctl, det, 0.45))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert ctl.pump.rates[:3] == [120, 140, 160]   # +20 steps, capped at max
    assert max(ctl.pump.rates) == 160
    assert ctl.pump.rates[-1] == 100               # back to base after the well


def test_clog_goes_home_clears_and_returns_to_same_well():
    pump = FakePump()
    ctl, det = _ctl(pump=pump)
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat")))
        # No air until well after the first clog trip.
        asyncio.ensure_future(_feed_after(ctl, det, 0.6))
        return await ctl.run_in_well(conn, Method("B3", wait=60), 15, asyncio.Event(),
                                     on_move=lambda w: moves.append(w))
    moves = []
    assert asyncio.run(go()) is True
    assert pump.clears >= 1
    i = conn.sent.index("@T")
    assert conn.sent[i + 1] == "@P,M1,0900,15,"    # back into the SAME well, full cap
    assert moves and moves[0] == "B3"              # re-entry fires the boost callback
    assert det.moves and len(det.moves) >= 2       # detector re-armed on re-entry


def test_gives_up_after_max_clears():
    pump = FakePump()
    ctl, det = _ctl(_cfg(max_clears=1), pump=pump)
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat")))
        return await ctl.run_in_well(conn, Method("C1", wait=60), 3, asyncio.Event())
    ok = asyncio.run(go())
    assert ok is False
    assert pump.clears == 1
    assert "C1" in ctl.last_error and "halted" in ctl.last_error
    assert conn.sent[-1] == "@T"                   # left with the needle home
    assert pump.rates[-1] == 100


def test_robot_dwell_cap_counts_as_done():
    ctl, det = _ctl(_cfg(escalate_after_s=100), pump=FakePump())
    conn = FakeConn()

    async def go():
        async def finish():
            await asyncio.sleep(0.05)
            conn._ready = True
        asyncio.ensure_future(finish())
        ctl_clock_skip = time.monotonic
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    # READY within the first 15 s is ignored (that is the robot still arriving),
    # so shrink that guard by advancing the clock.
    t0 = time.monotonic()
    ctl.clock = lambda: time.monotonic() - t0 + (100 if time.monotonic() - t0 > 0.04 else 0)
    assert asyncio.run(go()) is True
    assert ctl.log[0][1] == "max_dwell"


def test_stop_sends_T_and_restores_base():
    ctl, det = _ctl(_cfg(hold_at_max_s=10))
    conn = FakeConn()
    stop = asyncio.Event()

    async def go():
        async def press():
            await asyncio.sleep(0.4)
            stop.set()
        asyncio.ensure_future(_stream(ctl, _signal("flat")))
        asyncio.ensure_future(press())
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, stop)
    assert asyncio.run(go()) is False
    assert conn.sent[-1] == "@T"
    # STOP owns the pump: the controller must not restart it at base rate.
    assert 100 not in ctl.pump.rates


def test_no_pump_means_only_waiting_for_air():
    class NoPump(FakePump):
        available = False
    p = NoPump()
    p.available = False
    ctl, det = _ctl(pump=p)
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_feed_after(ctl, det, 0.4))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert p.rates == [] and p.clears == 0


def test_execute_method_hands_the_dwell_to_the_controller():
    import amuza_async
    from amuza_async import AsyncAmuzaConnection
    c = AsyncAmuzaConnection(use_mock=False)
    sent = []

    async def send_command(cmd, **kw):
        sent.append(cmd.strip())
        if cmd.startswith("@P"):
            c.status.state = amuza_async.DeviceState.RUNNING
        return True

    async def stop_movement():
        sent.append("@T")
        c.status.state = amuza_async.DeviceState.READY
        c.status.is_moving = False
        return True

    async def wait_for_busy(timeout=5.0):
        return True

    c.send_command, c.stop_movement, c.wait_for_busy = send_command, stop_movement, wait_for_busy
    c.calculate_move_time = lambda pos: 0.0
    ctl, det = _ctl(_cfg(max_dwell_s=1200, escalate_after_s=10))

    async def go():
        asyncio.ensure_future(_feed_after(ctl, det, 0.05))
        return await c.execute_method(Method("A2", wait=150, buffer_time=0),
                                      asyncio.Event(), dwell_controller=ctl)
    assert asyncio.run(go()) is True
    assert sent[0].startswith("@P,M1,1200,")    # the cap, not the 150 s sampling time
    assert sent[-1] == "@T"


def test_detector_on_real_run_flags_only_the_dry_well():
    """22 Oct 2026 14:21 run: while the needle was in a well, only H3 (the
    candidate dry well) may be called EMPTY."""
    import os
    from datetime import datetime
    from well_empty_detector import WellEmptyDetector
    path = "Sensor_Readings/MABIP_Run_22_10_26_14_21.txt"
    if not os.path.exists(path):
        pytest.skip("run file not on this machine")
    t0 = datetime(2026, 10, 22, 14, 21, 11)
    moves = ["14:22:32", "14:26:12", "14:29:53", "14:33:34", "14:37:15", "14:40:56", "14:44:37"]
    names = ["G1", "H1", "H2", "H3", "H4", "H5", "H6"]
    mv = [(datetime.strptime("2026-10-22 " + m, "%Y-%m-%d %H:%M:%S") - t0).total_seconds()
          for m in moves]
    det = WellEmptyDetector(flow_rate=120)
    in_well_empty, k = [], 0
    for line in open(path):
        p = line.split("\t")
        if not (p[0].isdigit() and len(p) > 8):
            continue
        try:
            t, ch = float(p[1]) * 60, [float(x) for x in p[2:8]]
        except ValueError:
            continue
        while k < len(mv) and t >= mv[k]:
            det.note_move(mv[k]); k += 1
        ev = det.update(t, ch)
        if ev and ev.kind == "EMPTY" and k and t - mv[k - 1] <= 150:
            in_well_empty.append(names[k - 1])
    assert in_well_empty == ["H3"]



def test_moving_signal_never_raises_the_pump():
    """Fresh fluid arriving (signal rising) = draining fine: no escalation, no
    clog trip, however long it takes."""
    pump = FakePump()
    ctl, det = _ctl(pump=pump)
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("rising")))
        asyncio.ensure_future(_feed_after(ctl, det, 0.8))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert pump.rates == [] and pump.clears == 0
    assert ctl.log[0][1] == "empty"


def test_falling_signal_also_counts_as_fresh():
    pump = FakePump()
    ctl, det = _ctl(pump=pump)
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("falling")))
        asyncio.ensure_future(_feed_after(ctl, det, 0.5))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert pump.rates == []


def test_signal_going_stale_midway_starts_escalation():
    pump = FakePump()
    ctl, det = _ctl(_cfg(hold_at_max_s=10), pump=pump)
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("rising"),
                                      switch=(0.4, _signal("flat"))))
        asyncio.ensure_future(_feed_after(ctl, det, 1.0))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert pump.rates and pump.rates[0] == 120
    assert pump.clears == 0


def test_freshness_judgement_needs_a_full_window():
    ctl, det = _ctl(_cfg(fresh_window_s=30))
    now = time.monotonic()
    ctl.feed(now, [1.0] * 6)
    assert ctl.signal_fresh(now) is None


class DipDetector(FakeDetector):
    """Emits a suction-break dip once when told to."""
    def __init__(self):
        super().__init__()
        self.dip = False

    def update(self, t, ch):
        if self.dip:
            self.dip = False
            class E:
                kind, reason, message = "SUSPECT", "suction-break dip", "dip"
            return E()
        return super().update(t, ch)


def test_dip_then_stale_is_empty_not_a_clog():
    """22 Oct A2: the tip met air (dip), then the reading froze at its plateau.
    That is a dry well; the pump must not be raised and no clog trip made."""
    pump = FakePump()
    det = DipDetector()
    ctl = LowVolumeController(_cfg(), det, pump=pump, poll_s=0.01)
    conn = FakeConn()

    async def go():
        async def dip_soon():
            await asyncio.sleep(0.05)
            det.dip = True
        asyncio.ensure_future(_stream(ctl, _signal("flat")))
        asyncio.ensure_future(dip_soon())
        return await ctl.run_in_well(conn, Method("A2", wait=60), 9, asyncio.Event())
    assert asyncio.run(go()) is True
    assert ctl.log[0][1] == "empty-dip"
    assert pump.clears == 0 and pump.rates == []
    assert conn.sent == ["@T"]


def test_operator_next_leaves_the_well():
    ctl, det = _ctl(_cfg(escalate_after_s=100))
    conn = FakeConn()

    async def go():
        async def press():
            await asyncio.sleep(0.05)
            ctl.request_next()
        asyncio.ensure_future(press())
        return await ctl.run_in_well(conn, Method("A3", wait=60), 17, asyncio.Event())
    assert asyncio.run(go()) is True
    assert ctl.log[0][1] == "operator"
    assert conn.sent == ["@T"]


def test_operator_next_during_clog_trip_does_not_go_back_in():
    pump = FakePump(clear_s=5.0)                       # a long clearing
    ctl, det = _ctl(pump=pump)
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat")))
        async def press_when_clearing():
            while not pump.clearing():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            ctl.request_next()
        asyncio.ensure_future(press_when_clearing())
        return await ctl.run_in_well(conn, Method("A2", wait=60), 9, asyncio.Event())
    t0 = time.monotonic()
    assert asyncio.run(go()) is True
    assert time.monotonic() - t0 < 3.0                 # did not wait out the clear
    assert ctl.log[0][1] == "operator"
    assert not [c for c in conn.sent[1:] if c.startswith("@P")]   # never re-entered
    assert not pump.clearing()                         # clearing aborted
    assert pump.rates[-1] == 100


def test_detector_lag_follows_pump_rate():
    from well_empty_detector import WellEmptyDetector
    d = WellEmptyDetector(flow_rate=130)
    lag130 = d.lag_s
    d.set_flow_rate(260)
    assert abs(d.lag_s - lag130 / 2) < 0.01
