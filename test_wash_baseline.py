"""BaselineWash: end the buffer wash when 3 of the 4 outputs are back at baseline."""

import asyncio
import time

from wash_baseline import BaselineWash, WashConfig, outputs_of


def _ch(glu=0.0, gln=0.0, gluc=0.0, lac=0.0, a=0.2, b=0.2, wiggle=0.0):
    """Six channels whose four outputs are (glu, gln, gluc, lac) (+ a tiny wiggle)."""
    return [a + wiggle, a - glu + wiggle, a + gln + 2 * wiggle,
            b, b + gluc + wiggle, b + lac - wiggle]


def _wash(**kw):
    base = dict(enabled=True, dead_volume_ul=1.0, hold_s=0.05, base_window_s=5.0, max_s=3.0)
    base.update(kw)
    # 600 µL/min through 1 µL -> lag 0.1 s
    return BaselineWash(WashConfig(**base), rate_fn=lambda: 600.0, poll_s=0.01)


def _fill_baseline(w, n=20):
    t = time.monotonic() - n * 0.01
    for i in range(n):
        w.feed(t + i * 0.01, _ch(wiggle=0.001 * ((-1) ** i)))
    assert w.set_baseline()


def _plateau(w, levels, n=20):
    w.note_move(time.monotonic() - 1.0)
    t = time.monotonic() - 0.5
    for i in range(n):
        w.feed(t + i * 0.01, _ch(*levels))


async def _stream(w, levels_fn, until_s):
    t0 = time.monotonic()
    i = 0
    while time.monotonic() - t0 < until_s:
        w.feed(time.monotonic(), _ch(*levels_fn(time.monotonic() - t0), wiggle=0.001 * ((-1) ** i)))
        i += 1
        await asyncio.sleep(0.005)


def test_outputs_are_the_plot_differences():
    ch = [1.0, 0.25, 1.5, 0.5, 0.75, 2.0]
    assert outputs_of(ch) == [0.75, 0.5, 0.25, 1.5]


def test_no_baseline_means_caller_falls_back():
    w = _wash()
    assert asyncio.run(w.wait(asyncio.Event())) == "nobaseline"


def test_baseline_needs_enough_readings():
    w = _wash()
    w.feed(time.monotonic(), _ch())
    assert not w.set_baseline()


def test_wash_ends_when_three_of_four_are_back():
    w = _wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    # glutamate never comes back, the other three do after 0.3 s
    lv = lambda el: (1.0, 0.0, 0.0, 0.0) if el > 0.3 else (1.0, 1.0, 1.0, 1.0)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 2.5))
        t0 = time.monotonic()
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r, time.monotonic() - t0
    r, el = asyncio.run(go())
    assert r == "baseline"
    assert 0.3 <= el < 1.5


def test_two_of_four_back_is_not_enough_and_caps():
    w = _wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    lv = lambda el: (1.0, 1.0, 0.0, 0.0)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 4.0))
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "cap"
    assert w.last["back"] == 2


def test_within_five_percent_of_the_plateau_counts_as_back():
    w = _wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    # 4 % of the plateau left on every output: back. (6 % would not be.)
    lv = lambda el: (0.04, 0.04, 0.04, 0.04) if el > 0.2 else (1.0, 1.0, 1.0, 1.0)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 2.5))
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "baseline"


def test_six_percent_left_is_not_back():
    w = _wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    lv = lambda el: (0.06, 0.06, 0.06, 0.06) if el > 0.2 else (1.0, 1.0, 1.0, 1.0)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 4.0))
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "cap"


def test_nothing_is_judged_before_one_lag():
    # lag 1.0 s: the signal is at baseline from the start (well fluid not yet
    # arrived), but the wash must not end before the lag has passed.
    w = BaselineWash(WashConfig(enabled=True, dead_volume_ul=10.0, hold_s=0.05,
                                base_window_s=5.0, max_s=5.0),
                     rate_fn=lambda: 600.0, poll_s=0.01)
    _fill_baseline(w)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lambda el: (0, 0, 0, 0), 3.0))
        t0 = time.monotonic()
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r, time.monotonic() - t0
    r, el = asyncio.run(go())
    assert r == "baseline"
    assert el >= 1.0


def test_one_air_spike_does_not_end_or_break_the_wash():
    w = _wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    # Still on the plateau, with one 2-sample dip to baseline at 0.3 s: must not end.
    def lv(el):
        return (0.0, 0.0, 0.0, 0.0) if 0.3 < el < 0.31 else (1.0, 1.0, 1.0, 1.0)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 4.0))
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "cap"


def test_operator_skip_and_stop():
    w = _wash(max_s=30.0)
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))

    async def skip():
        asyncio.get_event_loop().call_later(0.1, w.request_skip)
        return await w.wait(asyncio.Event())
    assert asyncio.run(skip()) == "skip"

    async def stop():
        ev = asyncio.Event()
        asyncio.get_event_loop().call_later(0.1, ev.set)
        return await w.wait(ev)
    assert asyncio.run(stop()) == "stop"


def test_execute_method_uses_the_buffer_wait_instead_of_the_timer():
    from amuza_async import AsyncAmuzaConnection, Method

    calls = []

    async def buffer_wait(method, ev, prog):
        calls.append(method.pos)
        return False                       # abandon: nothing else should run

    async def go():
        conn = AsyncAmuzaConnection(use_mock=True)
        t0 = time.monotonic()
        ok = await conn.execute_method(Method("A1", wait=5, buffer_time=60), asyncio.Event(),
                                       buffer_wait=buffer_wait)
        return ok, time.monotonic() - t0
    ok, el = asyncio.run(go())
    assert ok is False and calls == ["A1"]
    assert el < 5                          # the 60 s fixed buffer was not waited


def _settle_wash():
    return _wash(settle_s=0.3, settle_tol=0.03, max_s=5.0)


def test_flat_most_of_the_way_back_is_settled_drift():
    w = _settle_wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    # falls to 20 % of the plateau and stays there: the baseline drifted
    lv = lambda el: (0.2, 0.2, 0.2, 0.2) if el > 0.15 else (1.0, 1.0, 1.0, 1.0)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 4.0))
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "settled"
    assert w.last["recovered"] == 0.8


def test_flat_on_the_plateau_is_stuck_not_settled():
    w = _settle_wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lambda el: (1.0, 1.0, 1.0, 1.0), 4.0))
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "stuck"


def test_still_falling_is_not_settled():
    w = _settle_wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    # a slow steady fall: never flat, reaches baseline at ~1.2 s
    lv = lambda el: tuple([max(0.0, 1.0 - el / 1.2)] * 4)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 4.0))
        r = await w.wait(asyncio.Event())
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "baseline"


def test_settle_off_waits_for_the_real_baseline():
    w = _settle_wash()
    _fill_baseline(w)
    _plateau(w, (1.0, 1.0, 1.0, 1.0))
    lv = lambda el: (0.2, 0.2, 0.2, 0.2) if el < 1.0 else (0.0, 0.0, 0.0, 0.0)

    async def go():
        feeder = asyncio.ensure_future(_stream(w, lv, 4.0))
        r = await w.wait(asyncio.Event(), settle=False)
        feeder.cancel()
        return r
    assert asyncio.run(go()) == "baseline"


def test_rebase_takes_the_current_level():
    w = _wash()
    _fill_baseline(w)
    t = time.monotonic() - 0.45          # mostly the new level in the last window
    for i in range(45):
        w.feed(t + i * 0.01, _ch(0.3, 0.3, 0.3, 0.3))
    assert w.rebase()
    assert abs(w.baseline[0] - 0.3) < 1e-6
