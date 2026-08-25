"""Tests for mapping blockage spans onto the wells they spoiled.

Exercises AsyncAMUZAGUI._blocked_wells / _track_blockage without building a GUI:
both are plain logic over _well_spans / _blockage_spans, so we bind them to a
stand-in object. That keeps the test fast and headless while still testing the
real functions rather than a copy of them.
"""

import pytest

from blockage_detector import BlockageDetector, BlockageEvent
from gui_async import AsyncAMUZAGUI
from run_file import RunFile, WELL_COLUMNS


class FakeGUI:
    """Minimal stand-in carrying only what the methods under test touch."""

    _blocked_wells = AsyncAMUZAGUI._blocked_wells
    _span_blocked = AsyncAMUZAGUI._span_blocked
    _track_blockage = AsyncAMUZAGUI._track_blockage
    _finalize_well_log = AsyncAMUZAGUI._finalize_well_log

    def __init__(self, wells=()):
        self.current_sequence_wells = list(wells)
        self._well_spans = []
        self._blockage_spans = []
        self.blockage_detector = None
        self.displayed = []
        self.run_file = None

    def add_to_display(self, msg):
        self.displayed.append(msg)

    class _Sig:
        def emit(self, *_):
            pass

    blockage_changed = _Sig()


def _gui(wells, spans, blockages):
    g = FakeGUI(wells)
    g._well_spans = list(spans)
    g._blockage_spans = [list(b) for b in blockages]
    return g


# Three wells sampled back-to-back, 100 s each.
# (well_id, start_mono, end_mono, sensor_elapsed_min)
SPANS = [("A1", 0, 100, 1.0), ("A2", 100, 200, 2.0), ("A3", 200, 300, 3.0)]
WELLS = ["A1", "A2", "A3"]


def test_no_blockage_means_no_reruns():
    assert _gui(WELLS, SPANS, [])._blocked_wells() == []


def test_blockage_maps_to_the_overlapping_well():
    assert _gui(WELLS, SPANS, [[110, 190]])._blocked_wells() == ["A2"]


def test_blockage_spanning_two_wells_returns_both():
    assert _gui(WELLS, SPANS, [[150, 250]])._blocked_wells() == ["A2", "A3"]


def test_partial_overlap_counts():
    """Touching a well at all spoils it — we re-run rather than keep stale data."""
    assert _gui(WELLS, SPANS, [[95, 105]])._blocked_wells() == ["A1", "A2"]


def test_unclosed_blockage_extends_to_end_of_run():
    """A blockage still open at the end spoils every well after its onset."""
    assert _gui(WELLS, SPANS, [[150, None]])._blocked_wells() == ["A2", "A3"]


def test_results_are_in_plate_order_not_detection_order():
    g = _gui(WELLS, SPANS, [[210, 220], [10, 20]])
    assert g._blocked_wells() == ["A1", "A3"]


def test_well_spoiled_by_two_blockages_listed_once():
    g = _gui(WELLS, SPANS, [[110, 120], [130, 140]])
    assert g._blocked_wells() == ["A2"]


def test_blockage_between_wells_touches_neither():
    g = _gui(WELLS, [("A1", 0, 100, 1.0), ("A2", 150, 250, 2.0)], [[110, 140]])
    assert g._blocked_wells() == []


def test_since_restricts_to_the_current_retry_pass():
    """A retry pass must be judged on its own blockages, or the wells that
    triggered it re-queue forever."""
    g = _gui(WELLS, SPANS, [[110, 190]])          # spoiled A2 in the first pass
    assert g._blocked_wells() == ["A2"]
    # A2 re-run cleanly at t=300-400, after the retry pass began at t=300.
    g._well_spans.append(("A2", 300, 400, 4.0))
    assert g._blocked_wells(since=300) == []


def test_onset_is_backdated_by_detector_latency():
    """The alarm fires ~a cycle late, so the spoiled well is the earlier one.
    Without back-dating we would blame A3 and keep A2's stale reading."""
    g = FakeGUI(WELLS)
    g._well_spans = list(SPANS)
    det = BlockageDetector(cycle_s=60.0)          # latency = 60*1.5 + 45 = 135 s
    g.blockage_detector = det

    class R:
        channels = [0.0] * 6

    # Alarm raised at t=250 (while A3 is being sampled).
    det.update = lambda *_a, **_k: BlockageEvent(True, 250.0, ["ch6"], {}, "blocked")
    g._track_blockage(R())

    assert g._blockage_spans[0][0] == pytest.approx(115.0)   # 250 - 135
    assert g._blocked_wells() == ["A2", "A3"], "must blame the well that was live"


HEADER = ("Created: 07/14/2026\t03:00:28 PM\n"
          "counter\tt[min]\t#1ch1\t#1ch2\t#1ch3\t#1ch4\t#1ch5\t#1ch6\t#1ch7\tflow_uL_min\n"
          "Start: 07/14/2026\t03:00:28 PM\n")
SENSOR_ROW = "1\t0.0201\t-0.002\t0.000\t-0.003\t0.699\t0.719\t0.735\t20.062\t1.000\n"


def _row(well, elapsed, seq="Sampling Sequence"):
    """A well completion as the run file journals it while the run is live."""
    return f"#WELL\t{well}\t2026-07-14 15:00:00\t{elapsed:.4f}\t{seq}\tpending\tpending\n"


def _wells(run_file):
    out = {}
    for p in run_file.read().wells:
        out.setdefault(p[0], []).append((float(p[2]), p[4], p[5]))
    return out


def _finalized(tmp_path, spans, blockages, rows):
    g = FakeGUI(sorted({s[0] for s in spans}))
    g._well_spans = list(spans)
    g._blockage_spans = [list(b) for b in blockages]
    g.run_file = RunFile(tmp_path / "MABIP_Run_test.txt")
    g.run_file.write_header(HEADER)
    g.run_file.append(SENSOR_ROW + "".join(rows))
    g._finalize_well_log()
    return g, _wells(g.run_file)


def test_finalize_marks_clean_run_all_usable(tmp_path):
    _, out = _finalized(tmp_path, SPANS, [],
                        [_row("A1", 1.0), _row("A2", 2.0), _row("A3", 3.0)])
    for well, rows in out.items():
        assert rows == [(pytest.approx(float(rows[0][0])), "0", "1")], well


def test_finalize_flags_spoiled_and_points_use_at_the_rerun(tmp_path):
    """The whole point: A2 was sampled through a clog, then re-run. The log must
    say which of the two rows is the real reading."""
    spans = list(SPANS) + [("A2", 300, 400, 4.0)]      # the re-run
    rows = [_row("A1", 1.0), _row("A2", 2.0), _row("A3", 3.0),
            _row("A2", 4.0, "Blockage Retry 1")]
    _, out = _finalized(tmp_path, spans, [[110, 190]], rows)

    assert out["A2"] == [(2.0, "1", "0"), (4.0, "0", "1")], \
        "spoiled attempt must be blocked=1/use=0; the re-run use=1"
    assert out["A1"] == [(1.0, "0", "1")]
    assert out["A3"] == [(3.0, "0", "1")]


def test_finalize_no_clean_reading_gets_no_use_row(tmp_path):
    """If every attempt was blocked there is no good reading to point at, and
    nominating a stale one would be worse than admitting it."""
    spans = [("A1", 0, 100, 1.0), ("A1", 200, 300, 3.0)]
    rows = [_row("A1", 1.0), _row("A1", 3.0, "Blockage Retry 1")]
    g, out = _finalized(tmp_path, spans, [[0, 400]], rows)
    assert out["A1"] == [(1.0, "1", "0"), (3.0, "1", "0")]
    assert any("No clean reading" in m for m in g.displayed)


def test_finalize_keeps_the_sampling_rows_and_header(tmp_path):
    """Marking wells up is append-only: the sampling table above is untouched."""
    g, _ = _finalized(tmp_path, SPANS, [], [_row("A1", 1.0)])
    text = g.run_file.path.read_text()
    assert text.startswith(HEADER + SENSOR_ROW)
    run = g.run_file.read()
    assert run.header == HEADER.splitlines()
    assert run.sensor == [SENSOR_ROW.rstrip("\n")]


def test_session_end_writes_clean_sections(tmp_path):
    g, before = _finalized(tmp_path, SPANS, [[110, 190]],
                           [_row("A1", 1.0), _row("A2", 2.0), _row("A3", 3.0)])
    assert g.run_file.finalize()
    text = g.run_file.path.read_text()
    assert "=== WELL LOG ===" in text and "#WELL\t" not in text
    assert "\t".join(WELL_COLUMNS) in text
    assert text.rstrip().endswith("=== END OF RUN ===")
    assert _wells(g.run_file) == before


def test_finalize_is_idempotent(tmp_path):
    """Finalizing twice (e.g. stop then complete) must not corrupt the rows."""
    g, out1 = _finalized(tmp_path, SPANS, [[110, 190]],
                         [_row("A1", 1.0), _row("A2", 2.0), _row("A3", 3.0)])
    g._finalize_well_log()
    assert _wells(g.run_file) == out1
    g.run_file.finalize()
    g._finalize_well_log()
    assert _wells(g.run_file) == out1


def test_finalize_never_destroys_the_log_if_the_write_fails(tmp_path, monkeypatch):
    """The run file is hours of work and cannot be recreated. A failed rewrite
    must leave the original intact rather than a truncated stub."""
    import pathlib

    g, _ = _finalized(tmp_path, SPANS, [[110, 190]],
                      [_row("A1", 1.0), _row("A2", 2.0), _row("A3", 3.0)])
    original = g.run_file.path.read_text()

    real_write = pathlib.Path.write_text

    def boom(self, *a, **k):
        if self.suffix == ".tmp":
            raise OSError("disk full")
        return real_write(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "write_text", boom)
    assert g.run_file.finalize() is False          # must not raise

    assert g.run_file.path.read_text() == original, "original log was clobbered"


def test_finalize_leaves_no_temp_file_behind(tmp_path):
    g, _ = _finalized(tmp_path, SPANS, [[110, 190]],
                      [_row("A1", 1.0), _row("A2", 2.0), _row("A3", 3.0)])
    g.run_file.finalize()
    assert list(tmp_path.glob("*.tmp")) == []


def test_finalize_without_run_file_is_a_noop():
    g = FakeGUI(WELLS)
    g._finalize_well_log()          # must not raise


def test_clear_closes_the_open_span():
    g = FakeGUI(WELLS)
    g.blockage_detector = BlockageDetector(cycle_s=60.0)
    g._blockage_spans = [[100.0, None]]

    class R:
        channels = [0.0] * 6

    g.blockage_detector.update = lambda *_a, **_k: BlockageEvent(
        False, 300.0, [], {}, "cleared")
    g._track_blockage(R())
    assert g._blockage_spans[0][1] == 300.0
