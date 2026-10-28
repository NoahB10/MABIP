"""Monitor-only mode: never touch the pump; a clog pauses the run until it is fixed."""

import asyncio
import time
from types import SimpleNamespace

from amuza_async import Method
from low_volume import LowVolumeController
from test_low_volume import FakeConn, FakePump, FakeDetector, _cfg, _signal, _stream


def _mon(**kw):
    base = dict(monitor_only=True, escalate_after_s=0.05, step_every_s=0.02,
                fresh_window_s=0.1, clog_after_s=0.2)
    base.update(kw)
    return _cfg(**base)


def _ctl(cfg=None):
    det = FakeDetector()
    pump = FakePump()
    ctl = LowVolumeController(cfg or _mon(), det, pump=pump, poll_s=0.01)
    alerts = []
    ctl.alert = lambda title, msg: alerts.append(title)
    return ctl, det, pump, alerts


async def _air_later(ctl, det, delay):
    await asyncio.sleep(delay)
    det.fire = True
    ctl.feed(time.monotonic(), [0] * 6)


def test_stale_signal_pauses_and_never_touches_the_pump():
    ctl, det, pump, alerts = _ctl()
    conn = FakeConn()
    states = []

    async def watch():
        while True:
            states.append(ctl.state)
            await asyncio.sleep(0.01)

    async def go():
        # flat (clog) for 0.8 s, then moving again, then the well runs dry
        asyncio.ensure_future(_stream(ctl, _signal("flat"), until_s=3.0,
                                      switch=(0.8, _signal("rising"))))
        asyncio.ensure_future(_air_later(ctl, det, 1.6))
        w = asyncio.ensure_future(watch())
        r = await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
        w.cancel()
        return r
    assert asyncio.run(go()) is True
    assert "clog" in states                       # it paused
    assert pump.rates == [] and pump.clears == 0  # pump never changed, never cleared
    assert conn.sent == ["@T"]                    # stayed in the well through the clog
    assert ctl.clogged == ["A1"]
    assert "MABIP: clog - run paused" in alerts
    assert ctl.log[0][:2] == ("A1", "empty")


def test_a_valid_plateau_is_not_a_clog():
    ctl, det, pump, alerts = _ctl()
    det.steady = True                             # detector: plateau is STEADY
    det.dry_flag = None
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat"), until_s=1.5))
        asyncio.ensure_future(_air_later(ctl, det, 1.0))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert ctl.clogged == [] and alerts == []


def test_go_to_buffer_waits_for_baseline_then_back_into_same_well():
    ctl, det, pump, alerts = _ctl()
    conn = FakeConn()
    washed = []

    async def wash(ev, prog, label, settle=True):
        washed.append(label)
        await asyncio.sleep(0.1)
        return "baseline"
    ctl.wash_fn = wash

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat"), until_s=3.0))
        asyncio.get_event_loop().call_later(0.5, ctl.request_buffer)
        asyncio.get_event_loop().call_later(1.2, ctl.request_next)
        return await ctl.run_in_well(conn, Method("A1", wait=60), 7, asyncio.Event())
    assert asyncio.run(go()) is True
    assert conn.sent[0] == "@T"                   # out to the buffer
    assert conn.sent[1].startswith("@P") and conn.sent[1].endswith(",07,")  # same well
    assert washed == ["A1 hold"]
    assert pump.rates == [] and pump.clears == 0


def test_buffer_hold_waits_for_continue_when_wash_caps():
    ctl, det, pump, alerts = _ctl()
    conn = FakeConn()

    async def wash(ev, prog, label, settle=True):
        await asyncio.sleep(0.05)
        return "cap"
    ctl.wash_fn = wash

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("rising"), until_s=2.0))
        asyncio.get_event_loop().call_later(0.1, ctl.request_buffer)
        asyncio.get_event_loop().call_later(0.6, ctl.request_resume)
        asyncio.ensure_future(_air_later(ctl, det, 1.0))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 3, asyncio.Event())
    assert asyncio.run(go()) is True
    assert "MABIP: still not at baseline" in alerts
    assert [c[:2] for c in conn.sent] == ["@T", "@P", "@T"]   # out, back in after Continue, out dry


def test_operator_next_during_clog_pause_moves_on():
    ctl, det, pump, alerts = _ctl()
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat"), until_s=2.0))
        asyncio.get_event_loop().call_later(0.7, ctl.request_next)
        return await ctl.run_in_well(conn, Method("B2", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert ctl.log[0][:2] == ("B2", "operator")
    assert pump.rates == []


def test_continue_during_clog_keeps_reading_the_well():
    ctl, det, pump, alerts = _ctl()
    conn = FakeConn()

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat"), until_s=2.0))
        asyncio.get_event_loop().call_later(0.5, ctl.request_resume)
        asyncio.ensure_future(_air_later(ctl, det, 0.6))
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, asyncio.Event())
    assert asyncio.run(go()) is True
    assert conn.sent == ["@T"]                    # never left before it ran dry
    assert ctl.log[0][:2] == ("A1", "empty")


def test_stop_during_clog_pause():
    ctl, det, pump, alerts = _ctl()
    conn = FakeConn()

    async def go():
        ev = asyncio.Event()
        asyncio.ensure_future(_stream(ctl, _signal("flat"), until_s=2.0))
        asyncio.get_event_loop().call_later(0.6, ev.set)
        return await ctl.run_in_well(conn, Method("A1", wait=60), 1, ev)
    assert asyncio.run(go()) is False
    assert pump.rates == []


def test_set_rate_is_refused_in_monitor_mode():
    ctl, det, pump, alerts = _ctl()
    ctl._set_rate(300.0, "anything")
    assert pump.rates == []


def test_normal_low_volume_still_escalates():
    det = FakeDetector()
    pump = FakePump()
    ctl = LowVolumeController(_cfg(), det, pump=pump, poll_s=0.01)

    async def go():
        asyncio.ensure_future(_stream(ctl, _signal("flat"), until_s=0.6))
        asyncio.ensure_future(_air_later(ctl, det, 0.5))
        return await ctl.run_in_well(FakeConn(), Method("A1", wait=60), 1, asyncio.Event())
    asyncio.run(go())
    assert pump.rates                             # unchanged behaviour outside monitor mode


# ---- Flow Control tab guards (no Qt: the methods only touch plain attributes)
def _tab(**kw):
    msgs = []
    t = SimpleNamespace(
        _monitor_only=True, line=object(), _busy=False, _bursting=False, _clearing=False,
        _phase="buffer", _n_wells=3, _auto_burst=True, _burst_done=False, _exp_follow=True,
        _ff_enabled=True, _boost_until=0.0, _ff_gen=0, _phase_flow_gen=0,
        cfg={"ff_boost_add": 50.0, "ff_boost_s": 20.0}, lbl_phase=SimpleNamespace(setText=lambda s: None),
        status_msg=SimpleNamespace(emit=msgs.append),
        _cfgf=lambda k, d: {"ff_boost_add": 50.0, "ff_boost_s": 20.0}.get(k, d),
        _num=lambda f, d: 100.0, f_rate=None)
    t.__dict__.update(kw)
    t.msgs = msgs
    return t


def test_flow_tab_does_nothing_by_itself_in_monitor_mode():
    from flow_control_tab import FlowControlTab as T
    t = _tab()
    touched = []
    t._trigger_burst = lambda auto=False: touched.append("burst")
    t._apply_phase_flow = lambda *a: touched.append("ramp")
    T.on_move_command(t, "A1")                    # no boost, no feed-forward pause
    assert t._boost_until == 0.0
    T.set_experiment_phase(t, "buffer", 3)        # no auto-burst, no follow ramp
    T.set_experiment_phase(t, "well", 3)          # no per-well rate hold
    assert T.try_clear_burst(t) is False
    assert T.request_clog_clear(t, "metabolite") is False
    assert touched == []
