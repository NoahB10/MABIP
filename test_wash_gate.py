"""The GUI's buffer phase with the baseline wash (AsyncAMUZAGUI._wash_buffer_wait).

First well: Buffer Time, then the baseline. Later wells: wash to baseline; a
"stuck" wash moves on flagged in a timed run, but HOLDS in monitor-only until
the line washes or the operator presses Continue (which re-bases)."""

import asyncio

from amuza_async import Method
from gui_async import AsyncAMUZAGUI


class FakeWash:
    def __init__(self, results, baseline=True):
        self.baseline = [0.0] * 4 if baseline else None
        self.results = list(results)
        self.calls = []
        self.rebased = 0
        self.last = {"seconds": 70.0, "back": 1, "recovered": 0.2}
        self.cfg = type("C", (), {"max_s": 600.0, "base_window_s": 30.0})()

    async def wait(self, ev, prog, label="", settle=True):
        self.calls.append(settle)
        await asyncio.sleep(0)
        return self.results.pop(0)

    def set_baseline(self):
        self.baseline = [0.0] * 4
        return True

    def rebase(self):
        self.rebased += 1
        return True


class FakeGUI:
    _wash_buffer_wait = AsyncAMUZAGUI._wash_buffer_wait

    def __init__(self):
        self.shown = []
        self._wash_flagged = []
        self._wash_drift = 0
        self.flow_tab = None
        self.connection = type("Conn", (), {"last_error": None})()

    def add_to_display(self, msg):
        self.shown.append(msg)


def _run(gui, wash, monitor, method=None):
    method = method or Method("B3", wait=60, buffer_time=0)
    return asyncio.run(gui._wash_buffer_wait(wash, monitor, method, asyncio.Event(),
                                             lambda *a: None))


def test_first_well_sets_the_baseline_after_the_buffer_time():
    gui, wash = FakeGUI(), FakeWash([], baseline=False)
    assert _run(gui, wash, monitor=True) is True
    assert wash.baseline is not None and wash.calls == []
    assert any("Baseline set" in m for m in gui.shown)


def test_baseline_ends_the_wash():
    gui, wash = FakeGUI(), FakeWash(["baseline"])
    assert _run(gui, wash, monitor=True) is True
    assert gui._wash_flagged == []


def test_settled_moves_on_and_counts_drift():
    gui, wash = FakeGUI(), FakeWash(["settled"])
    assert _run(gui, wash, monitor=True) is True
    assert gui._wash_drift == 1


def test_stuck_in_a_timed_run_moves_on_flagged():
    gui, wash = FakeGUI(), FakeWash(["stuck"])
    assert _run(gui, wash, monitor=False) is True
    assert gui._wash_flagged == ["B3"]


def test_stuck_in_monitor_mode_holds_until_the_line_washes():
    gui, wash = FakeGUI(), FakeWash(["stuck", "baseline"])
    assert _run(gui, wash, monitor=True) is True
    assert wash.calls == [True, False]        # second wait: no settle exit, real baseline only
    assert wash.rebased == 0
    assert any("Holding in the buffer" in m for m in gui.shown)


def test_continue_while_held_rebases():
    gui, wash = FakeGUI(), FakeWash(["cap", "skip"])
    assert _run(gui, wash, monitor=True) is True
    assert wash.rebased == 1


def test_continue_in_a_normal_wash_does_not_rebase():
    gui, wash = FakeGUI(), FakeWash(["skip"])
    assert _run(gui, wash, monitor=True) is True
    assert wash.rebased == 0


def test_stop_during_the_wash_abandons_the_sequence():
    gui, wash = FakeGUI(), FakeWash(["stop"])
    assert _run(gui, wash, monitor=True) is False
