"""Tests for holding the plate in the buffer while the line is blocked.

Covers the gate's decisions (AsyncAMUZAGUI._blockage_pause_gate) and the seam it
hangs off in the AMUZA layer (execute_method awaiting it after the buffer wait
and before the move). The popup itself is stubbed — what matters is that the gate
waits for an answer and honours it.
"""

import asyncio

import pytest

from amuza_async import Method
from blockage_detector import BlockageDetector
from gui_async import AsyncAMUZAGUI


class FakeGUI:
    _blockage_pause_gate = AsyncAMUZAGUI._blockage_pause_gate
    _wait_for_clearance = AsyncAMUZAGUI._wait_for_clearance
    _attempt_clearance = AsyncAMUZAGUI._attempt_clearance

    def __init__(self, blocked=False, answer=True, clears_after=None,
                 decidable=False):
        self.blockage_detector = BlockageDetector(cycle_s=60.0)
        self.blockage_detector.blocked = blocked
        self.displayed = []
        self.asked = 0
        self._answer = answer
        self._closed = 0
        self.attempts = 0

        # Stub the detector's hold mode: `decidable` says whether the stuck level
        # leaves room to fall; `clears_after` is how many polls until it does.
        det = self.blockage_detector
        det.begin_hold = lambda: None
        det.end_hold = lambda: None
        type(det).hold_is_decidable = property(lambda _s: decidable)
        self._polls = 0

        def hold_cleared():
            if not decidable:
                return None
            self._polls += 1
            if clears_after is None:
                return False
            return self._polls >= clears_after

        det.hold_cleared = hold_cleared

    def add_to_display(self, msg):
        self.displayed.append(msg)

    def _close_blockage_dialog(self):
        self._closed += 1

    async def _ask_blockage_cleared(self):
        self.asked += 1
        if self._answer is None:
            # Nobody is at the bench: the signal must decide. A bare future,
            # not a sleep -- the nosleep fixture would collapse a sleep and this
            # stub would answer after all.
            await asyncio.get_event_loop().create_future()
        return self._answer


def _gate(g, stop_event=None):
    return asyncio.run(g._blockage_pause_gate(stop_event or asyncio.Event()))


@pytest.fixture
def nosleep(monkeypatch):
    """Make the poll loop spin instantly. Bind the real sleep first -- patching
    asyncio.sleep with a lambda that calls asyncio.sleep recurses forever."""
    real = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *a, **k: real(0))


def test_unblocked_line_passes_straight_through():
    g = FakeGUI(blocked=False)
    assert _gate(g) is True
    assert g.asked == 0, "must not nag when nothing is wrong"


def test_blocked_line_asks_and_continues_when_cleared():
    g = FakeGUI(blocked=True, answer=True)
    assert _gate(g) is True
    assert g.asked == 1
    assert any("holding in buffer" in m for m in g.displayed)


def test_blocked_line_stops_run_when_user_declines():
    g = FakeGUI(blocked=True, answer=False)
    stop = asyncio.Event()
    assert asyncio.run(g._blockage_pause_gate(stop)) is False
    assert stop.is_set(), "declining must stop the run, not silently carry on"


def test_gate_stops_judging_while_parked_then_rearms():
    """Parked in buffer nothing is stepping, so the signal is *supposed* to go
    flat; judging through the pause would re-alarm on our own idleness."""
    g = FakeGUI(blocked=True, answer=True)
    seen = []
    orig = g._ask_blockage_cleared

    async def spy():
        seen.append(g.blockage_detector._cycling)
        return await orig()

    g._ask_blockage_cleared = spy
    _gate(g)
    assert seen == [False], "detector must be idled while waiting"
    assert g.blockage_detector._cycling is True, "must re-arm after resuming"


# ---- auto-resume: the signal proves the line came back ----------------------

def test_resumes_on_its_own_when_the_signal_starts_moving(nosleep):
    """The goal: stuck high, burst clears it, signal falls, run resumes -- with
    nobody at the bench and no flow sensor in line."""
    g = FakeGUI(blocked=True, answer=None, decidable=True, clears_after=3)
    assert _gate(g) is True
    assert any("line cleared" in m for m in g.displayed)
    assert g._closed == 1, "the dialog must be taken down once the signal decides"


def test_stuck_at_baseline_falls_back_to_the_dialog():
    """Nothing to fall to, so the signal can never prove clearance. Ask."""
    g = FakeGUI(blocked=True, answer=True, decidable=False)
    assert _gate(g) is True
    assert g.asked == 1
    assert any("confirm by hand" in m for m in g.displayed)


def test_user_can_still_answer_first_while_watching(nosleep):
    """Auto-watching must not lock the user out of deciding."""
    g = FakeGUI(blocked=True, answer=False, decidable=True, clears_after=None)
    assert _gate(g) is False


def test_clearance_hook_is_called_while_waiting(nosleep):
    """The seam the burst will occupy must actually be driven during the hold."""
    g = FakeGUI(blocked=True, answer=None, decidable=True, clears_after=3)

    async def counting():
        g.attempts += 1

    g._attempt_clearance = counting
    _gate(g)
    assert g.attempts >= 1, "burst hook never ran"


def test_stop_event_breaks_the_wait(nosleep):
    """A stop must not leave the run parked forever waiting on the signal."""
    stop = asyncio.Event()
    g = FakeGUI(blocked=True, answer=None, decidable=True, clears_after=None)

    async def stop_soon():
        g.attempts += 1
        if g.attempts >= 3:
            stop.set()

    g._attempt_clearance = stop_soon
    assert asyncio.run(g._blockage_pause_gate(stop)) is False


def test_gate_rearms_even_if_the_popup_raises():
    g = FakeGUI(blocked=True)

    async def boom():
        raise RuntimeError("dialog exploded")

    g._ask_blockage_cleared = boom
    with pytest.raises(RuntimeError):
        _gate(g)
    assert g.blockage_detector._cycling is True, "a crash must not leave it idled"


def test_already_stopping_does_not_prompt():
    g = FakeGUI(blocked=True)
    stop = asyncio.Event()
    stop.set()
    assert asyncio.run(g._blockage_pause_gate(stop)) is True
    assert g.asked == 0


def test_no_detector_is_a_noop():
    g = FakeGUI()
    g.blockage_detector = None
    assert _gate(g) is True


# ---- the seam in the AMUZA layer -------------------------------------------

def _stub_conn(order):
    """A bare AsyncAmuzaConnection with the move path stubbed to record order."""
    from amuza_async import AsyncAmuzaConnection

    conn = AsyncAmuzaConnection.__new__(AsyncAmuzaConnection)
    conn.well_mapping = lambda wells: [1]
    conn._format_method_command = lambda port, wait: "CMD"
    conn.calculate_move_time = lambda pos: 0.0
    conn.status = type("S", (), {"state": None, "current_well": 1})()

    async def fake_send(*a, **k):
        order.append("move")
        return True

    conn._send_command = fake_send
    return conn


def test_gate_is_awaited_before_the_move_command():
    """The whole point of the buffer seam: the gate runs *before* the move goes
    out, so a blocked line never costs a well its sampling time."""
    order = []
    conn = _stub_conn(order)

    async def gate():
        order.append("gate")
        return True

    method = Method(pos="A1", wait=0, buffer_time=0, eject=False, insert=False)
    try:
        asyncio.run(conn.execute_method(method, asyncio.Event(), pause_gate=gate))
    except Exception:
        pass  # stubbed move path may not finish; ordering is what is under test

    assert order[:1] == ["gate"], \
        f"gate must be awaited before the move command, got {order}"


def test_declined_gate_abandons_before_moving():
    """Saying no must stop the needle leaving the buffer entirely."""
    order = []
    conn = _stub_conn(order)

    async def gate():
        order.append("gate")
        return False

    method = Method(pos="A1", wait=0, buffer_time=0, eject=False, insert=False)
    ok = asyncio.run(conn.execute_method(method, asyncio.Event(), pause_gate=gate))

    assert ok is False
    assert "move" not in order, "must not move after the gate declined"


def test_no_gate_leaves_behaviour_unchanged():
    order = []
    conn = _stub_conn(order)
    method = Method(pos="A1", wait=0, buffer_time=0, eject=False, insert=False)
    try:
        asyncio.run(conn.execute_method(method, asyncio.Event()))
    except Exception:
        pass
    assert "gate" not in order
