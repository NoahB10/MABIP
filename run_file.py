"""One file per metabolite-sensor session, holding everything the run produced.

Why one file: a run used to leave three files behind (Sensor_readings_*.txt,
Well_Log_*.csv, Flow_Log_*.csv) that had to be moved together and matched up by
timestamp. Now a single .txt carries the sampling rows, the well completion log
and the flow-rate log — one file IS the run.

Layout of a finished file (tab-separated throughout, one delimiter everywhere):

    Created: 08/25/2026<TAB>04:27:59 PM          <- BioMon-style 3-line header
    counter<TAB>t[min]<TAB>#1ch1 ... <TAB>flow_uL_min
    Start: 08/25/2026<TAB>04:27:59 PM
    1<TAB>0.0201<TAB>-0.002 ...                    <- sampling rows, unchanged

    === WELL LOG ===
    # blocked=1: sampled through a blockage ...
    well_id<TAB>completed_at<TAB>sensor_elapsed_min<TAB>sequence_name<TAB>blocked<TAB>use
    A1<TAB>2026-08-25 16:30:01<TAB>2.0333<TAB>Sampling Sequence<TAB>0<TAB>1

    === FLOW LOG ===
    # Fluigent flow-rate log ...
    elapsed_s<TAB>timestamp<TAB>flow_uL_min<TAB>clog<TAB>air_bubble<TAB>segment
    1.11<TAB>2026-08-25 16:27:50<TAB>0.074<TAB>0<TAB>0<TAB>
    === END OF RUN ===

While the session is live the file is APPEND-ONLY. The sensor reader appends
sampling rows; well completions and flow samples are journaled the instant they
happen as "#WELL<TAB>..." / "#FLOW<TAB>..." lines in between. Nothing is ever
rewritten mid-run, so a crash or power cut cannot lose what was on disk.
finalize() then rewrites the file into the sectioned layout above — atomically,
the old file standing until the new one is complete. Every reader in this module
accepts both layouts, and any mix of the two, so a file that was finalized and
then appended to again (or never finalized at all) is still whole.

BioMon compatibility: the sampling section IS the legacy BioMon layout plus one
trailing flow_uL_min column. export_biomon() drops everything else and that
column, producing exactly the file BioMon has always read.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

WELL_TAG = "#WELL"
FLOW_TAG = "#FLOW"
WELL_SECTION = "=== WELL LOG ==="
FLOW_SECTION = "=== FLOW LOG ==="
END_MARK = "=== END OF RUN ==="

WELL_COLUMNS = ["well_id", "completed_at", "sensor_elapsed_min",
                "sequence_name", "blocked", "use"]
FLOW_COLUMNS = ["elapsed_s", "timestamp", "flow_uL_min", "clog", "air_bubble", "segment"]
WELL_NOTES = [
    "# Well completion log. sensor_elapsed_min is on the same clock as t[min] above.",
    "# blocked=1: sampled through a blockage, reading is stale - discard.",
    "# use=1: the reading to take for this well (its last un-blocked attempt).",
    "# A well may appear twice: the spoiled attempt, then its re-run.",
]
FLOW_NOTES = [
    "# Fluigent flow-rate log (~1 Hz). elapsed_s counts from the flow sensor connecting.",
]
FLOW_COLUMN = "flow_uL_min"
DEFAULT_NAME_FORMAT = "MABIP_Run_{timestamp}.txt"
DEFAULT_TS_FORMAT = "%d_%m_%y_%H_%M"

# The least fields any sampling row has: counter, t[min], seven channels. The
# real files carry 387; the old hand-written export carried exactly 9.
MIN_SENSOR_FIELDS = 9


# --------------------------------------------------------------- small helpers
def ensure_txt(path: str) -> str:
    """Add ".txt" when a name was typed without any extension, so nobody has to
    remember to type it in the save dialog."""
    if not path:
        return path
    root, ext = os.path.splitext(path)
    return path if ext else path + ".txt"


def is_sensor_row(line: str) -> bool:
    """A sampling row starts with an integer counter and a float time and has
    at least MIN_SENSOR_FIELDS tab-separated fields. Nothing else in the file
    looks like that: well rows start with a well id, flow rows with a float and
    have six fields, header/comment lines with words."""
    parts = line.split("\t")
    if len(parts) < MIN_SENSOR_FIELDS:
        return False
    try:
        int(parts[0])
        float(parts[1])
    except ValueError:
        return False
    return True


def _pad(fields: List[str], columns: List[str]) -> List[str]:
    fields = list(fields[:len(columns)])
    return fields + [""] * (len(columns) - len(fields))


def _well_key(row: List[str]) -> Optional[Tuple[str, float]]:
    """(well_id, elapsed) identifies one completion uniquely — a well re-run
    after a blockage appears twice, at different times."""
    try:
        return (row[0], round(float(row[2]), 4))
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------- parsed form
@dataclass
class ParsedRun:
    header: List[str] = field(default_factory=list)   # up to the 3 legacy header lines
    sensor: List[str] = field(default_factory=list)   # sampling rows, raw text
    wells: List[List[str]] = field(default_factory=list)   # WELL_COLUMNS fields each
    flows: List[List[str]] = field(default_factory=list)   # FLOW_COLUMNS fields each

    @property
    def columns(self) -> List[str]:
        """Column names of the sampling table, from the header ([] if absent)."""
        for ln in self.header:
            if ln.startswith("counter\t"):
                return ln.split("\t")
        return []

    def sensor_values(self, n_channels: int = 7) -> List[List[float]]:
        """Sampling rows as [t_min, ch1..chN] floats; unparseable rows dropped."""
        out = []
        for ln in self.sensor:
            p = ln.split("\t")
            if len(p) < 2 + n_channels:
                continue
            try:
                out.append([float(p[1])] + [float(x) for x in p[2:2 + n_channels]])
            except ValueError:
                continue
        return out


def parse_text(text: str) -> ParsedRun:
    """Read a run file in ANY of its forms — live journal, finalized sections,
    a finalized file with rows appended afterwards, a legacy BioMon/Sensor_readings
    file — into one structure.

    Journal rows for the same well (re-issued when its blocked/use flags were
    resolved) collapse onto the first occurrence, last value winning, so the
    well order stays chronological."""
    run = ParsedRun()
    mode: Optional[str] = None
    well_index: Dict[Tuple[str, float], int] = {}

    def add_well(fields: List[str]):
        row = _pad(fields, WELL_COLUMNS)
        key = _well_key(row)
        if key is not None and key in well_index:
            run.wells[well_index[key]] = row
            return
        if key is not None:
            well_index[key] = len(run.wells)
        run.wells.append(row)

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith(WELL_TAG + "\t"):
            add_well(line.split("\t")[1:])
            continue
        if line.startswith(FLOW_TAG + "\t"):
            run.flows.append(_pad(line.split("\t")[1:], FLOW_COLUMNS))
            continue
        if line == WELL_SECTION:
            mode = "well"
            continue
        if line == FLOW_SECTION:
            mode = "flow"
            continue
        if line.startswith("#") or line.startswith("==="):
            continue
        if is_sensor_row(line):
            run.sensor.append(line)
            continue
        first = line.split("\t", 1)[0]
        if mode is None and len(run.header) < 3 and (
                first.startswith("Created:") or first == "counter" or first.startswith("Start:")):
            run.header.append(line)
            continue
        if mode == "well":
            if first != WELL_COLUMNS[0]:
                add_well(line.split("\t"))
        elif mode == "flow":
            if first != FLOW_COLUMNS[0]:
                run.flows.append(_pad(line.split("\t"), FLOW_COLUMNS))
        # Anything else is unknown junk (e.g. trailing comments in old BioMon
        # files) and is dropped.
    return run


def parse(path) -> ParsedRun:
    return parse_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def sectioned_text(run: ParsedRun) -> str:
    """The finished layout described at the top of this module."""
    out: List[str] = list(run.header) + list(run.sensor)
    out += ["", WELL_SECTION, *WELL_NOTES, "\t".join(WELL_COLUMNS)]
    out += ["\t".join(w) for w in run.wells]
    out += ["", FLOW_SECTION, *FLOW_NOTES, "\t".join(FLOW_COLUMNS)]
    out += ["\t".join(f) for f in run.flows]
    out.append(END_MARK)
    return "\n".join(out) + "\n"


def biomon_text(run: ParsedRun) -> str:
    """Only the sampling table, without the trailing flow column: the legacy
    BioMon file, byte-for-byte the layout it has always read."""
    header = list(run.header)
    n_with_flow = None
    for i, ln in enumerate(header):
        if ln.startswith("counter\t"):
            cols = ln.split("\t")
            if cols and cols[-1] == FLOW_COLUMN:
                n_with_flow = len(cols)
                header[i] = "\t".join(cols[:-1])
    rows = []
    for ln in run.sensor:
        if n_with_flow is not None:
            p = ln.split("\t")
            if len(p) == n_with_flow:
                ln = "\t".join(p[:-1])
        rows.append(ln)
    return "\n".join(header + rows) + "\n"


def resolve_markup(wells: List[List[str]],
                   blocked_by_row: Dict[Tuple[str, float], bool]
                   ) -> Tuple[List[List[str]], int, List[str]]:
    """Fill in the `blocked`/`use` columns.

    `use` lands on each well's last un-blocked attempt — the re-run, when there
    was one. A well whose every attempt was blocked gets no use=1 row at all;
    there is no good reading to point at, and saying so is better than
    nominating a stale one. Rows the caller has no verdict for keep whatever
    they had (an earlier sequence's verdict, or `pending`).

    Returns (rows, spoiled_count, wells_with_no_clean_reading)."""
    rows = [list(w) for w in wells]
    for p in rows:
        key = _well_key(p)
        if key is not None and key in blocked_by_row:
            p[4] = "1" if blocked_by_row[key] else "0"
    good: Dict[str, int] = {}
    for i, p in enumerate(rows):
        if p[4] == "0":
            good[p[0]] = i
    for i, p in enumerate(rows):
        p[5] = "1" if good.get(p[0]) == i else "0"
    spoiled = sum(1 for p in rows if p[4] == "1")
    unusable = sorted({p[0] for p in rows} - set(good))
    return rows, spoiled, unusable


def needs_finalize(path) -> bool:
    """True unless the file ends with the END marker — i.e. it is still in
    journal form, or rows were appended after its last finalize."""
    try:
        p = Path(path)
        size = p.stat().st_size
        if size == 0:
            return False
        with open(p, "rb") as f:
            f.seek(max(0, size - 64))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return not tail.rstrip().endswith(END_MARK)


# ------------------------------------------------------------------ the file
class RunFile:
    """Owns one session's file. Every write goes through one lock so lines
    from the sensor reader (an aiofiles thread) and the Qt thread (well/flow
    events) can never interleave mid-line."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @classmethod
    def for_session(cls, folder, start_time: datetime,
                    name_format: str = DEFAULT_NAME_FORMAT,
                    ts_format: str = DEFAULT_TS_FORMAT) -> "RunFile":
        name = name_format.format(timestamp=start_time.strftime(ts_format))
        return cls(Path(folder) / name)

    @property
    def name(self) -> str:
        return self.path.name

    def __str__(self):
        return str(self.path)

    # ---- live writes (append-only)
    def write_header(self, text: str):
        """Put the legacy 3-line header at the top. Anything already journaled
        (a flow sample that beat the sensor's first packet) is kept below it."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing = ""
            try:
                if self.path.exists():
                    existing = self.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                existing = ""
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(text)
                f.write(existing)

    def append(self, text: str):
        if not text:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)

    def log_well(self, well_id: str, completed_at: datetime, elapsed_min: float,
                 sequence_name: str, blocked: str = "pending", use: str = "pending"):
        self.append(f"{WELL_TAG}\t{well_id}\t{completed_at:%Y-%m-%d %H:%M:%S}\t"
                    f"{elapsed_min:.4f}\t{sequence_name}\t{blocked}\t{use}\n")

    def log_flow(self, elapsed_s: float, timestamp: datetime, flow_uL_min: float,
                 clog: bool, air_bubble: bool, segment: str = ""):
        self.append(f"{FLOW_TAG}\t{elapsed_s:.2f}\t{timestamp:%Y-%m-%d %H:%M:%S}\t"
                    f"{flow_uL_min:.3f}\t{1 if clog else 0}\t{1 if air_bubble else 0}\t"
                    f"{segment}\n")

    # ---- reading
    def read(self) -> ParsedRun:
        if not self.path.exists():
            return ParsedRun()
        with self._lock:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        return parse_text(text)

    # ---- end of run / end of session
    def mark_wells(self, blocked_by_row: Dict[Tuple[str, float], bool]) -> Tuple[int, List[str]]:
        """Resolve blocked/use for the wells and journal the verdicts. Still
        append-only: the resolved rows are re-issued and supersede the pending
        ones when the file is read. Returns (spoiled_count, unusable_wells)."""
        run = self.read()
        if not run.wells:
            return 0, []
        rows, spoiled, unusable = resolve_markup(run.wells, blocked_by_row)
        self.append("".join(f"{WELL_TAG}\t" + "\t".join(r) + "\n" for r in rows))
        return spoiled, unusable

    def finalize(self) -> bool:
        """Rewrite into the sectioned layout. Via a temp file and an atomic
        rename: a plain write truncates first, so a crash or power cut mid-write
        would destroy the run's only record — hours of work that cannot be
        recreated. With os.replace the old file stands until the new one is
        complete. Returns False (and leaves the file untouched) on failure."""
        if not self.path.exists():
            return False
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            try:
                run = parse_text(self.path.read_text(encoding="utf-8", errors="replace"))
                tmp.write_text(sectioned_text(run), encoding="utf-8")
                os.replace(tmp, self.path)
            except Exception as e:
                logger.error(f"Failed to finalize run file {self.path}: {e}")
                try:
                    tmp.unlink()
                except Exception:
                    pass
                return False
        return True

    def snapshot_to(self, dest) -> Path:
        """A sectioned copy of the file as it stands right now (the live file
        itself is left alone, so this is safe mid-run)."""
        dest = Path(ensure_txt(str(dest)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(sectioned_text(self.read()), encoding="utf-8")
        return dest

    def export_biomon(self, dest) -> Path:
        return export_biomon(self.path, dest)


def export_biomon(src, dest) -> Path:
    """Write the BioMon-format file for `src` (any layout) to `dest`."""
    dest = Path(ensure_txt(str(dest)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(biomon_text(parse(src)), encoding="utf-8")
    return dest


def biomon_name_for(src) -> str:
    """Default export name: <run>_biomon.txt next to the source."""
    p = Path(src)
    return str(p.with_name(p.stem + "_biomon.txt"))


def tidy_folder(folder, pattern: str = "MABIP_Run_*.txt") -> List[str]:
    """Finalize any run file left in journal form (the app was killed, or the
    power went). Called at start-up. Returns the names it tidied."""
    done = []
    try:
        candidates = sorted(Path(folder).glob(pattern))
    except OSError:
        return done
    for p in candidates:
        if p.suffix == ".tmp" or not needs_finalize(p):
            continue
        if RunFile(p).finalize():
            done.append(p.name)
    return done
