"""End-to-end tests for the blockage retry pass.

Drives the real AsyncAMUZAGUI._rerun_blocked_wells against a fake connection
that records the sequences it is asked to run, so the loop's control flow — what
gets re-queued, how often, and when it refuses — is tested rather than described.
"""

import asyncio
import time

import pytest

from blockage_detector import BlockageDetector
from gui_async import AsyncAMUZAGUI


class FakeConn:
    """Records executed sequences; optionally marks re-runs as clean."""

    def __init__(self, gui, clean_on_rerun=True, completed=True):
        self.gui = gui
        self.clean_on_rerun = clean_on_rerun
        self.completed = completed
        self.runs = []          # list of [well_id, ...] per execute_sequence call

    async def execute_sequence(self, sequence, stop_event, well_completed_callback,
                               progress_callback, timing_provider, move_callback,
                               pause_gate=None):
        wells = [m.pos for m in sequence.methods]
        self.runs.append(wells)
        # Wells are sampled after the pass marker was stamped, as on the rig.
        t = max(self.gui._clock, time.monotonic()) + 1
        for w in wells:
            t += 100
            # A re-run is clean unless the fixture says the line is still bad.
            self.gui._well_spans.append((w, t - 100, t, float(t)))
        self.gui._clock = t
        return self.completed


class FakeState:
    async def get_timing_params(self):
        return (60, 90)


class FakeGUI:
    _rerun_blocked_wells = AsyncAMUZAGUI._rerun_blocked_wells
    _blocked_wells = AsyncAMUZAGUI._blocked_wells
    _span_blocked = AsyncAMUZAGUI._span_blocked
    MAX_BLOCKAGE_RETRY_PASSES = AsyncAMUZAGUI.MAX_BLOCKAGE_RETRY_PASSES

    def __init__(self, wells):
        self.current_sequence_wells = list(wells)
        self._well_spans = []
        self._blockage_spans = []
        self._well_started = {}
        self._active_seq_name = "Sampling Sequence"
        self.remaining_wells = []
        self.well_labels = {}
        self.displayed = []
        self.app_state = FakeState()
        self.blockage_detector = BlockageDetector(cycle_s=60.0)
        # The retry loop stamps its pass marker with time.monotonic(), and in
        # production well spans come from the same clock — so the fake must sit
        # on that timeline too, or `since` filters every span out.
        self._clock = time.monotonic()

    def add_to_display(self, msg):
        self.displayed.append(msg)


WELLS = ["A1", "A2", "A3"]


def _gui_with_spoiled_a2(**conn_kw):
    """First pass sampled A1/A2/A3; a blockage spoiled A2."""
    g = FakeGUI(WELLS)
    g._well_spans = [("A1", 0, 100, 1.0), ("A2", 100, 200, 2.0), ("A3", 200, 300, 3.0)]
    g._blockage_spans = [[110, 190]]
    conn = FakeConn(g, **conn_kw)
    g.connection = conn
    return g, conn


def _run(g, conn):
    noop = lambda *a, **k: None
    return asyncio.run(g._rerun_blocked_wells(
        asyncio.Event(), noop, noop, lambda: (60, 90), noop))


def test_spoiled_well_is_requeued_and_run():
    g, conn = _gui_with_spoiled_a2()
    assert _run(g, conn) is True
    assert conn.runs == [["A2"]], "only the spoiled well should be re-run"


def test_clean_run_triggers_no_rerun():
    g = FakeGUI(WELLS)
    g._well_spans = [("A1", 0, 100, 1.0), ("A2", 100, 200, 2.0)]
    g._blockage_spans = []
    conn = FakeConn(g)
    g.connection = conn
    assert _run(g, conn) is True
    assert conn.runs == []


def test_rerun_stops_after_one_clean_pass():
    """The re-run must not re-queue itself once it comes back clean."""
    g, conn = _gui_with_spoiled_a2()
    _run(g, conn)
    assert len(conn.runs) == 1


def test_still_blocked_line_refuses_to_burn_the_plate():
    """An unclosed blockage means the line is still bad; re-running would spoil
    the retry exactly as it spoiled the original."""
    g, conn = _gui_with_spoiled_a2()
    g._blockage_spans = [[110, None]]        # never cleared
    g.blockage_detector.blocked = True
    assert _run(g, conn) is True
    assert conn.runs == [], "must not re-run into a still-blocked line"
    assert any("still blocked" in m for m in g.displayed)


def test_user_stop_during_rerun_reports_not_completed():
    g, conn = _gui_with_spoiled_a2(completed=False)
    assert _run(g, conn) is False


def test_retry_passes_are_capped():
    """A line that keeps blocking must not loop forever."""
    g = FakeGUI(WELLS)
    g._well_spans = [("A2", 100, 200, 2.0)]
    g._blockage_spans = [[0, 1e9]]           # everything overlaps, always spoiled
    conn = FakeConn(g)
    g.connection = conn
    assert _run(g, conn) is True
    assert len(conn.runs) == g.MAX_BLOCKAGE_RETRY_PASSES
    assert any("still hit a blockage after" in m for m in g.displayed)


def test_rerun_preserves_plate_order():
    g = FakeGUI(WELLS)
    g._well_spans = [("A1", 0, 100, 1.0), ("A2", 100, 200, 2.0), ("A3", 200, 300, 3.0)]
    g._blockage_spans = [[0, 1000]]          # spoils all three
    g.blockage_detector.blocked = False
    conn = FakeConn(g)
    g.connection = conn
    _run(g, conn)
    assert conn.runs[0] == ["A1", "A2", "A3"]


def test_history_is_kept_for_the_well_log():
    """Spans must accumulate across passes — the log needs every attempt."""
    g, conn = _gui_with_spoiled_a2()
    before = len(g._well_spans)
    _run(g, conn)
    assert len(g._well_spans) == before + 1, "the re-run attempt must be recorded too"
    assert [s[0] for s in g._well_spans].count("A2") == 2
