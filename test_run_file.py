"""Tests for run_file: ONE file per sensor session (sampling + well log + flow log).

The file is append-only while live (sampling rows with #WELL/#FLOW journal lines
in between), rewritten into sections by finalize(), and readable in either form
— and in any mix of the two. The BioMon export must be exactly the old layout.
"""

from datetime import datetime
from pathlib import Path

import pytest

import run_file as rf
from run_file import RunFile

HEADER = ("Created: 08/25/2026\t04:27:59 PM\n"
          "counter\tt[min]\t#1ch1\t#1ch2\t#1ch3\t#1ch4\t#1ch5\t#1ch6\t#1ch7\tflow_uL_min\n"
          "Start: 08/25/2026\t04:27:59 PM\n")
LEGACY_HEADER = HEADER.replace("\tflow_uL_min", "")
T0 = datetime(2026, 8, 25, 16, 27, 59)


def srow(i, t, flow="1.000"):
    return f"{i}\t{t:.4f}\t-0.002\t0.000\t-0.003\t0.699\t0.719\t0.735\t20.062\t{flow}\n"


def make_live(tmp_path):
    """A session mid-run: header, rows, and well/flow events in arrival order."""
    f = RunFile(tmp_path / "MABIP_Run_test.txt")
    f.write_header(HEADER)
    f.append(srow(1, 0.02) + srow(2, 0.05))
    f.log_flow(1.11, T0, 0.074, False, False, "")
    f.log_well("A1", T0, 1.0, "Sampling Sequence")
    f.append(srow(3, 0.08))
    f.log_flow(2.16, T0, 0.056, True, False, "run_x")
    f.log_well("A2", T0, 2.0, "Sampling Sequence")
    return f


# ------------------------------------------------------------------ live form
def test_live_file_is_an_append_only_journal(tmp_path):
    f = make_live(tmp_path)
    text = f.path.read_text()
    assert text.startswith("Created: ")
    assert "#WELL\tA1\t2026-08-25 16:27:59\t1.0000\tSampling Sequence\tpending\tpending\n" in text
    assert "#FLOW\t2.16\t2026-08-25 16:27:59\t0.056\t1\t0\trun_x\n" in text
    # Chronological: the third sample was appended after the first well.
    assert text.index("#WELL\tA1") < text.index("\n3\t0.0800")


def test_parse_journal(tmp_path):
    run = make_live(tmp_path).read()
    assert run.header == HEADER.splitlines()
    assert [ln.split("\t")[0] for ln in run.sensor] == ["1", "2", "3"]
    assert run.wells == [
        ["A1", "2026-08-25 16:27:59", "1.0000", "Sampling Sequence", "pending", "pending"],
        ["A2", "2026-08-25 16:27:59", "2.0000", "Sampling Sequence", "pending", "pending"],
    ]
    assert run.flows == [
        ["1.11", "2026-08-25 16:27:59", "0.074", "0", "0", ""],
        ["2.16", "2026-08-25 16:27:59", "0.056", "1", "0", "run_x"],
    ]


def test_write_header_keeps_a_flow_row_that_beat_it(tmp_path):
    f = RunFile(tmp_path / "r.txt")
    f.log_flow(0.5, T0, 0.1, False, False)
    f.write_header(HEADER)
    text = f.path.read_text()
    assert text.startswith("Created: ")
    assert "#FLOW\t0.50" in text


def test_sensor_values(tmp_path):
    values = make_live(tmp_path).read().sensor_values(7)
    assert len(values) == 3 and all(len(v) == 8 for v in values)
    assert values[0][:3] == [0.02, -0.002, 0.0]


def test_for_session_name(tmp_path):
    f = RunFile.for_session(tmp_path, T0)
    assert f.name == "MABIP_Run_25_08_26_16_27.txt"
    assert f.path.parent == tmp_path


# ------------------------------------------------------------------- finalize
def test_finalize_writes_sections_and_reads_back_the_same(tmp_path):
    f = make_live(tmp_path)
    before = f.read()
    assert rf.needs_finalize(f.path)
    assert f.finalize() is True
    assert not rf.needs_finalize(f.path)

    text = f.path.read_text()
    assert "#WELL\t" not in text and "#FLOW\t" not in text
    assert text.rstrip().endswith(rf.END_MARK)
    i_sensor, i_well, i_flow = (text.index("\n3\t0.0800"), text.index(rf.WELL_SECTION),
                                text.index(rf.FLOW_SECTION))
    assert i_sensor < i_well < i_flow, "sampling, then wells, then flow"
    assert "\t".join(rf.WELL_COLUMNS) in text and "\t".join(rf.FLOW_COLUMNS) in text
    assert "A1\t2026-08-25 16:27:59\t1.0000\tSampling Sequence\tpending\tpending\n" in text

    assert f.read() == before


def test_rows_appended_after_finalize_are_still_read(tmp_path):
    """The reader may flush a last buffer after the shutdown finalize; the
    next start-up must still see those rows, and tidy them in."""
    f = make_live(tmp_path)
    f.finalize()
    f.append(srow(4, 0.11))
    f.log_well("A3", T0, 3.0, "Sampling Sequence")
    run = f.read()
    assert [ln.split("\t")[0] for ln in run.sensor] == ["1", "2", "3", "4"]
    assert [w[0] for w in run.wells] == ["A1", "A2", "A3"]
    assert rf.needs_finalize(f.path)
    assert f.finalize()
    assert f.read() == run
    assert not rf.needs_finalize(f.path)


def test_finalize_failure_leaves_the_file_intact(tmp_path, monkeypatch):
    f = make_live(tmp_path)
    original = f.path.read_text()
    real = Path.write_text

    def boom(self, *a, **k):
        if self.suffix == ".tmp":
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", boom)
    assert f.finalize() is False
    assert f.path.read_text() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_finalize_leaves_no_temp_file(tmp_path):
    f = make_live(tmp_path)
    f.finalize()
    assert list(tmp_path.glob("*.tmp")) == []


def test_tidy_folder_finalizes_only_journal_form_run_files(tmp_path):
    live = make_live(tmp_path)
    done = RunFile(tmp_path / "MABIP_Run_done.txt")
    done.write_header(HEADER); done.append(srow(1, 0.02)); done.finalize()
    legacy = tmp_path / "Sensor_readings_old.txt"
    legacy.write_text(LEGACY_HEADER + srow(1, 0.02)[:-7] + "\n")
    legacy_before = legacy.read_text()

    assert rf.tidy_folder(tmp_path) == [live.name]
    assert not rf.needs_finalize(live.path)
    assert legacy.read_text() == legacy_before, "old files are left alone"
    assert rf.tidy_folder(tmp_path) == []


# ------------------------------------------------------------------- mark-up
def test_mark_wells_supersedes_pending_rows_without_rewriting(tmp_path):
    f = make_live(tmp_path)
    size_before = f.path.stat().st_size
    spoiled, unusable = f.mark_wells({("A1", 1.0): False, ("A2", 2.0): True})
    assert (spoiled, unusable) == (1, ["A2"])
    assert f.path.stat().st_size > size_before, "append-only: verdicts are added, not rewritten"
    assert f.path.read_text().startswith("Created: ")
    wells = f.read().wells
    assert [(w[0], w[4], w[5]) for w in wells] == [("A1", "0", "1"), ("A2", "1", "0")]


def test_mark_wells_keeps_earlier_verdicts(tmp_path):
    """A second sequence in the same session must not reset the first one's flags."""
    f = make_live(tmp_path)
    f.mark_wells({("A1", 1.0): False, ("A2", 2.0): True})
    f.log_well("A3", T0, 3.0, "Sampling Sequence")
    f.mark_wells({("A3", 3.0): False})
    wells = f.read().wells
    assert [(w[0], w[4], w[5]) for w in wells] == [
        ("A1", "0", "1"), ("A2", "1", "0"), ("A3", "0", "1")]


def test_resolve_markup_points_use_at_the_rerun():
    rows = [["A1", "", "1.0000", "s", "pending", "pending"],
            ["A2", "", "2.0000", "s", "pending", "pending"],
            ["A2", "", "4.0000", "retry", "pending", "pending"]]
    out, spoiled, unusable = rf.resolve_markup(
        rows, {("A1", 1.0): False, ("A2", 2.0): True, ("A2", 4.0): False})
    assert [(r[0], r[4], r[5]) for r in out] == [("A1", "0", "1"), ("A2", "1", "0"), ("A2", "0", "1")]
    assert (spoiled, unusable) == (1, [])


# -------------------------------------------------------------------- exports
def test_export_biomon_drops_sections_and_the_flow_column(tmp_path):
    f = make_live(tmp_path)
    f.mark_wells({("A1", 1.0): False})
    out = rf.export_biomon(f.path, tmp_path / "for_biomon")   # no extension given
    assert out.name == "for_biomon.txt"
    lines = out.read_text().splitlines()
    assert lines[:3] == LEGACY_HEADER.splitlines()
    assert len(lines) == 3 + 3
    assert all(len(ln.split("\t")) == 9 for ln in lines[3:]), "flow column stripped"
    text = out.read_text()
    assert "WELL" not in text and "FLOW" not in text and "===" not in text


def test_biomon_export_of_a_legacy_file_is_byte_identical():
    legacy = LEGACY_HEADER + "1\t0.0201\t-0.002\t0.000\t-0.003\t0.699\t0.719\t0.735\t20.062\n"
    assert rf.biomon_text(rf.parse_text(legacy)) == legacy


def test_parse_legacy_file_with_trailing_comments():
    legacy = (LEGACY_HEADER
              + "1\t0.0201\t-0.002\t0.000\t-0.003\t0.699\t0.719\t0.735\t20.062\n"
              + "2\t0.0479\t-0.002\t-0.002\t-0.006\t0.697\t0.717\t0.734\t20.062\n"
              + "Some closing remark from BioMon\n")
    run = rf.parse_text(legacy)
    assert len(run.sensor) == 2 and run.wells == [] and run.flows == []
    assert run.columns[-1] == "#1ch7"


def test_snapshot_is_sectioned_and_leaves_the_live_file_alone(tmp_path):
    f = make_live(tmp_path)
    snap = f.snapshot_to(tmp_path / "copy")
    assert snap.name == "copy.txt"
    assert snap.read_text().rstrip().endswith(rf.END_MARK)
    assert "#WELL\t" in f.path.read_text(), "live file untouched"
    assert rf.parse(snap) == f.read()


def test_biomon_name_for():
    assert rf.biomon_name_for("/x/MABIP_Run_1.txt") == "/x/MABIP_Run_1_biomon.txt"


@pytest.mark.parametrize("given,expected", [
    ("run", "run.txt"), ("run.txt", "run.txt"), ("run.csv", "run.csv"),
    ("/a/b/run", "/a/b/run.txt"), ("", ""),
])
def test_ensure_txt(given, expected):
    assert rf.ensure_txt(given) == expected
