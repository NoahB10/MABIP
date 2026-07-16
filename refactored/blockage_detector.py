"""Detect flow blockages from the metabolite signal alone — no flow sensor.

Why this exists
---------------
`AsyncAMUZAGUI.metabolite_stuck()` answers a *different* question: "the flow sensor
already says clog — is the metabolite signal flat too?". It uses a 12 s window,
which is far shorter than one well cycle (~177 s). Because a healthy signal sits
flat at baseline or at plateau for 60-90 s of every cycle, a 12 s window cannot
tell "flat because blocked" from "flat because we're mid-plateau". Measured on
Sensor_readings_14_07_26_15_00.txt it fires on 4% of a known-healthy stretch
while catching only 17% of the real blockage. Fine as a corroborator, useless
standalone.

The idea
--------
During a well-plate run the metabolite signal is a *periodic square wave* driven
by the AMUZA stepping between wells: rise to plateau, fall to baseline, once per
cycle. A blockage does not make the signal flat at some special level — it makes
the **oscillation stop**. So we do not ask "is the signal flat right now?", we
ask "has the signal moved at all over a window longer than one full cycle?".
Any healthy window of 1.5 cycles necessarily contains a rise *and* a fall, so its
range is large. A blocked window's range collapses to the noise floor.

Four things make that robust:

1. **Window > cycle.** window_s = 1.5 x cycle_s. This is what the 12 s detector
   gets wrong, and it is the whole trick.
2. **Robust range (p95-p5), not max-min.** The raw channels carry single-sample
   spikes that would keep max-min wide through a real blockage.
3. **Adaptive per-channel baseline.** Channel amplitudes differ by ~50x (ch6
   range ~18, ch1 ~0.4) and drift as electrodes age, so a fixed threshold cannot
   work. We compare each channel's range against the median of its own recent
   healthy ranges. The baseline is *frozen* while flat, otherwise a long blockage
   would drag the baseline down to meet the signal and silently self-clear.
4. **Multi-channel vote.** A blockage is fluidic: every channel sees stale fluid
   and freezes together. One frozen channel is an electrode fault, not a clog.
   Only channels with enough SNR get a vote (ch1/ch2/ch4 measured SNR ~16 never
   fall below ~29% of baseline even when genuinely stuck, so they cannot vote).

Latency — the unavoidable cost
------------------------------
A blockage cannot be called faster than roughly one well cycle, because one cycle
is exactly the timescale on which a *healthy* signal is allowed to be flat. The
alarm therefore lands ~window_s + confirm_s (~5 min at default settings) after
the fluid actually stops — about two wells' worth of data. Measured on the 14 Jul
run, trading that latency away costs false positives:

    window_factor  confirm_s   false-positive rate    latency
        1.2          30 s          0.95%              ~1.9 min
        1.3          45 s          0.14%              ~2.5 min
        1.5          45 s          0.00%              ~3.0 min   <- default
        1.8          45 s          0.00%              misses real events

The defaults buy a clean zero-false-positive run at ~3 min. If you would rather
hear about a clog sooner and tolerate the odd spurious alarm, drop window_factor
to 1.3. Do not go below ~1.2: at window_factor <= 1.0 the window no longer spans
a full cycle and the detector degenerates into the 12 s detector's failure mode.

Usage (streaming, one call per sensor reading):

    det = BlockageDetector(cycle_s=177.0)
    det.set_cycling(True)              # only meaningful while wells are stepping
    ev = det.update(t_seconds, [ch1, ch2, ch3, ch4, ch5, ch6])
    if ev is not None:
        print(ev.message)              # BLOCKED / CLEARED edge

Everything is pure-Python and O(window) per sample — at 0.6 Hz with 6 channels
that is nothing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = ["BlockageDetector", "BlockageEvent", "DetectorConfig"]


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _median(vals):
    if not vals:
        return 0.0
    return _quantile(sorted(vals), 0.5)


@dataclass
class DetectorConfig:
    """Tuned against Sensor_readings_14_07_26_15_00.txt (338 min, 12178 samples,
    0.60 Hz, 2.95 min well cycle). See module docstring for the reasoning."""

    cycle_s: float = 177.0
    """One AMUZA well cycle. Measured 2.95 min from Well_Log completions and
    confirmed by autocorrelation of ch6 (peak 0.77 at 2.96 min)."""

    window_factor: float = 1.5
    """Analysis window as a multiple of cycle_s. Must exceed 1.0 so every healthy
    window contains a full rise+fall. 1.5 gave clean separation: healthy p5 range
    13.0 vs blocked <3.0 on ch6, at 0% false positives. Lower it to ~1.3 to alarm
    ~0.5 min sooner at ~0.14% false positives; see the latency table in the module
    docstring."""

    flat_frac: float = 0.25
    """Flat if range < this fraction of the channel's healthy baseline. Healthy
    ranges sit at 0.98-1.00 of baseline (p50) and only the real episodes reach
    0.06-0.17, so 0.25 sits in a wide empty gap."""

    clear_frac: float = 0.40
    """Hysteresis: recovery needs range back above this fraction. Above flat_frac
    so a signal hovering at the threshold cannot chatter."""

    min_channels: int = 2
    """How many usable channels must agree. A fluidic blockage freezes all of
    them; a single frozen channel is an electrode fault."""

    min_snr: float = 25.0
    """A channel votes only if baseline_range / sample_noise exceeds this.
    Measured SNR: ch1 15.9, ch2 15.3, ch3 40.9, ch4 16.3, ch5 81.2, ch6 178.4 —
    so this admits ch3/ch5/ch6 and rejects the three that cannot discriminate."""

    confirm_s: float = 45.0
    """Sustain time before alarming. Deliberately suppresses brief flat spells:
    the 14 Jul run has two ~0.5 min ones that are momentary, not clogs worth
    interrupting a run for. Anything shorter than this is not reported."""

    baseline_s: float = 3600.0
    """Trailing span for the healthy-amplitude baseline. Long enough to average
    over many cycles, short enough to track electrode drift."""

    min_baseline_s: float = 600.0
    """Warm-up before the detector will call anything. Below this there are too
    few cycles to know what healthy amplitude looks like."""

    n_channels: int = 6


@dataclass
class BlockageEvent:
    """Emitted on a state *edge* only (BLOCKED or CLEARED), never per-sample."""

    blocked: bool
    t: float
    channels: list = field(default_factory=list)
    ratios: dict = field(default_factory=dict)
    message: str = ""


class BlockageDetector:
    """Streaming blockage detector driven only by the metabolite channels."""

    def __init__(self, cycle_s: float = 177.0, config: Optional[DetectorConfig] = None, **kw):
        self.cfg = config or DetectorConfig(cycle_s=cycle_s, **kw)
        n = self.cfg.n_channels
        self._buf = [deque() for _ in range(n)]      # (t, value) inside window
        self._noise = [deque(maxlen=400) for _ in range(n)]
        self._prev = [None] * n
        self._base = [deque() for _ in range(n)]     # (t, range) healthy history
        self._t0: Optional[float] = None
        self._t: Optional[float] = None
        self._flat_since: Optional[float] = None
        self._wells = deque(maxlen=12)               # recent well-completion times
        self._cycling = True
        self.blocked = False
        self.last_ratios: dict = {}
        self.last_usable: list = []

    # ---- gating -----------------------------------------------------------
    def set_cycling(self, cycling: bool) -> None:
        """Tell the detector whether wells are actively stepping.

        A flat signal is only evidence of a blockage if the fluidics are supposed
        to be producing a cycle. Parked on one well, or between sequences, flat is
        correct behaviour — so we stop judging and reset the sustain timer rather
        than accumulate a false alarm."""
        if cycling != self._cycling:
            self._cycling = cycling
            self._flat_since = None
            if not cycling and self.blocked:
                self.blocked = False

    def set_cycle_s(self, cycle_s: float) -> None:
        """Update the expected well period (e.g. user changed timing mid-run)."""
        if cycle_s and cycle_s > 0:
            self.cfg.cycle_s = float(cycle_s)

    def note_well_completed(self, t: float) -> None:
        """Call from the well-completion hook on every finished well.

        Two jobs. It re-arms the cycling gate (wells are stepping, so flat is now
        meaningful), and it *learns* the true cycle period from the median gap
        between completions, so timing changes mid-run need no reconfiguration.
        Median (not mean) because a paused or retried well leaves one huge gap
        that would drag a mean far off."""
        self._wells.append(t)
        if len(self._wells) >= 3:
            gaps = [b - a for a, b in zip(list(self._wells)[:-1], list(self._wells)[1:])]
            gaps = [g for g in gaps if 0 < g < 3600]
            if gaps:
                self.set_cycle_s(_median(gaps))
        self.set_cycling(True)

    @property
    def window_s(self) -> float:
        return self.cfg.cycle_s * self.cfg.window_factor

    # ---- main -------------------------------------------------------------
    def update(self, t: float, channels: Sequence[float]) -> Optional[BlockageEvent]:
        """Feed one reading. `t` is seconds (monotonic or elapsed).

        Returns a BlockageEvent on a state change, else None."""
        cfg = self.cfg
        if self._t0 is None:
            self._t0 = t
        self._t = t
        win = self.window_s

        for i in range(min(cfg.n_channels, len(channels))):
            v = channels[i]
            if v is None:
                continue
            v = float(v)
            if self._prev[i] is not None:
                self._noise[i].append(abs(v - self._prev[i]))
            self._prev[i] = v
            b = self._buf[i]
            b.append((t, v))
            while b and t - b[0][0] > win:
                b.popleft()

        # If well completions are being reported and they have stopped arriving,
        # the plate is not stepping and a flat signal proves nothing. Safe during
        # a real blockage: the AMUZA keeps stepping on schedule while blocked —
        # it is the fluid that is stuck, not the robot — so completions continue.
        if self._wells and (t - self._wells[-1]) > 2.5 * cfg.cycle_s:
            self.set_cycling(False)

        # Need a full window and enough history to know healthy amplitude.
        if t - self._t0 < win:
            return None

        ratios, usable, flat_votes = {}, [], 0
        for i in range(cfg.n_channels):
            b = self._buf[i]
            if len(b) < 10:
                continue
            vals = sorted(v for _, v in b)
            rng = _quantile(vals, 0.95) - _quantile(vals, 0.05)

            base_hist = [r for _, r in self._base[i]]
            base = _median(base_hist) if base_hist else 0.0

            # Grow the baseline only from healthy samples. Frozen while blocked
            # so a long blockage cannot redefine "normal" and self-clear.
            if not self.blocked and (base == 0.0 or rng >= base * cfg.clear_frac):
                self._base[i].append((t, rng))
                while self._base[i] and t - self._base[i][0][0] > cfg.baseline_s:
                    self._base[i].popleft()
                base_hist = [r for _, r in self._base[i]]
                base = _median(base_hist) if base_hist else 0.0

            if base <= 0.0 or t - self._t0 < cfg.min_baseline_s:
                continue

            noise = _median(list(self._noise[i])) if self._noise[i] else 0.0
            if noise > 0 and (base / noise) < cfg.min_snr:
                continue  # channel too noisy to discriminate

            usable.append(i)
            ratio = rng / base
            ratios[i] = ratio
            thr = cfg.flat_frac if not self.blocked else cfg.clear_frac
            if ratio < thr:
                flat_votes += 1

        self.last_ratios, self.last_usable = ratios, usable
        if not self._cycling or len(usable) < cfg.min_channels:
            self._flat_since = None
            return None

        is_flat = flat_votes >= cfg.min_channels

        if is_flat:
            if self._flat_since is None:
                self._flat_since = t
            if not self.blocked and (t - self._flat_since) >= cfg.confirm_s:
                self.blocked = True
                names = [f"ch{i+1}" for i in sorted(ratios) if ratios[i] < cfg.flat_frac]
                return BlockageEvent(
                    True, t, names, dict(ratios),
                    "BLOCKAGE: metabolite signal stopped changing on "
                    f"{', '.join(names)} for {t - self._flat_since:.0f}s "
                    f"(>{cfg.confirm_s:.0f}s, window {win:.0f}s). No fresh fluid "
                    "is reaching the sensor.",
                )
        else:
            self._flat_since = None
            if self.blocked:
                self.blocked = False
                return BlockageEvent(
                    False, t, [], dict(ratios),
                    "Blockage cleared: metabolite signal is cycling again.",
                )
        return None

    def status(self) -> str:
        if not self._cycling:
            return "idle (not cycling)"
        if len(self.last_usable) < self.cfg.min_channels:
            return "warming up"
        return "BLOCKED" if self.blocked else "flowing"
