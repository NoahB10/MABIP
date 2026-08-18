"""Detect flow blockages from the metabolite signal alone — no flow sensor.

Why this exists
---------------
The rig used to infer a clog from the Fluigent flow sensor, corroborated by a
`metabolite_stuck()` helper that asked whether the raw channel sum had moved over
a 12 s window. Both were unreliable. On the 14 Jul run the flow sensor itself
dropped out for the last third of the experiment (flat ~0.1 uL/min while the
channels kept cycling normally), and it reads ~0 between bursts anyway, so
"no flow" never meant "blocked". The 12 s window was worse: one well cycle is
~177 s and a healthy signal sits flat at baseline or plateau for 60-90 s of every
cycle, so it could not tell "flat because blocked" from "flat because mid-plateau"
— it called 4% of a known-healthy stretch stuck while catching only 17% of the
real blockage. Both are gone; this replaced them.

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

    hold_move_frac: float = 0.15
    """While parked, how far the signal must move off its stuck level (as a
    fraction of the channel's healthy swing) to count as fresh fluid arriving.
    Small, because the plate is not stepping: nothing but flow can move it."""

    hold_confirm_s: float = 15.0
    """How long that movement must persist before calling the line clear. Short —
    unlike blockage detection there is no cycle to wait out, so clearance is seen
    in seconds rather than minutes."""

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
        self._armed_since: Optional[float] = None   # window refilling since
        self._t: Optional[float] = None
        self._flat_since: Optional[float] = None
        self._wells = deque(maxlen=12)               # recent well-completion times
        self._hold_level: dict = {}                  # i -> (stuck level, threshold)
        self._hold_since: Optional[float] = None
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
        if cycling == self._cycling:
            return
        self._cycling = cycling
        self._flat_since = None
        if not cycling and self.blocked:
            self.blocked = False
        if cycling:
            # While idle the signal was flat *because* nothing was stepping. That
            # data is still sitting in the window, and judging against it would
            # re-alarm the instant we resume. Drop it and re-arm once a fresh
            # window has filled. Baselines survive — the healthy amplitude did not
            # change just because we paused.
            for b in self._buf:
                b.clear()
            self._armed_since = self._t

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

    # ---- baseline helpers -------------------------------------------------
    def _baseline_range(self, i) -> float:
        return _median([r for _, r, _ in self._base[i]]) if self._base[i] else 0.0

    def _baseline_low(self, i) -> float:
        """Where the channel sits when fresh buffer is reaching the sensor."""
        return _median([lo for _, _, lo in self._base[i]]) if self._base[i] else 0.0

    def _departure_threshold(self, i) -> float:
        """How far the signal must move to count as "fluid arrived".

        Whichever is larger: comfortably above this channel's own sample noise,
        or a slice of its healthy swing. The noise term stops a quiet channel
        firing on jitter; the amplitude term stops a noisy one needing an
        implausibly large move."""
        noise = _median(list(self._noise[i])) if self._noise[i] else 0.0
        return max(8.0 * noise, self.cfg.hold_move_frac * self._baseline_range(i))

    # ---- hold mode --------------------------------------------------------
    def begin_hold(self) -> None:
        """Enter clearance-watching mode — call when the plate parks in buffer.

        Judging flips around here. While the plate is cycling, a blockage is the
        *absence* of movement. While parked, nothing is stepping, so the signal
        has no reason to move at all — which means any movement is proof that
        fresh fluid arrived, i.e. the line cleared. That works with no flow
        sensor and no well cycle, which is the point: during a burst the flow
        is not in line to be measured.

        Caveat, and it is a real one: this only works if the signal is stuck
        *above* its buffer baseline, so it has somewhere to fall to. Stuck at
        baseline already, clearing brings more buffer and nothing moves — see
        hold_cleared(), which reports that case as unknowable rather than
        guessing."""
        self._hold_level = {}
        for i in range(self.cfg.n_channels):
            b = self._buf[i]
            if len(b) < 5 or not self._base[i]:
                continue
            level = _median([v for _, v in b])
            thr = self._departure_threshold(i)
            # Only channels with somewhere to fall can prove clearance.
            if thr > 0 and (level - self._baseline_low(i)) > thr:
                self._hold_level[i] = (level, thr)
        self._hold_since = self._t

    def end_hold(self) -> None:
        self._hold_level = {}
        self._hold_since = None

    @property
    def hold_is_decidable(self) -> bool:
        """Can clearance be seen at all from where the signal is stuck?"""
        return bool(self._hold_level)

    def hold_cleared(self) -> Optional[bool]:
        """During a hold: has fresh fluid reached the sensor?

        True  -> the signal moved off its stuck level; the line is flowing again.
        False -> still sitting where it stopped.
        None  -> cannot be known from the signal: it is stuck at buffer baseline,
                 so clearing it would look exactly like staying blocked. The
                 caller must ask a human or probe a well."""
        if not self._hold_level:
            return None
        t = self._t
        for i, (level, thr) in self._hold_level.items():
            b = self._buf[i]
            recent = [v for (tt, v) in b if t - tt <= self.cfg.hold_confirm_s]
            if len(recent) < 5:
                continue
            # The sustain requirement is carried by the data, not by a timer:
            # `recent` spans hold_confirm_s, so its *median* can only have moved
            # if the signal spent most of that window away from the stuck level.
            # One spike cannot shift it, and the answer does not depend on how
            # often the caller happens to poll.
            if abs(_median(recent) - level) > thr:
                return True
        return False

    @property
    def window_s(self) -> float:
        return self.cfg.cycle_s * self.cfg.window_factor

    @property
    def latency_s(self) -> float:
        """How long after fluid actually stops that `update()` can raise the alarm.

        The range only collapses once the window sits wholly inside the flat
        stretch (window_s), and then it must persist (confirm_s). Callers that
        need to know *which* well was affected must back-date the alarm by this
        much — otherwise they blame a well that was fine, and miss the one that
        was not."""
        return self.window_s + self.cfg.confirm_s

    # ---- main -------------------------------------------------------------
    def update(self, t: float, channels: Sequence[float]) -> Optional[BlockageEvent]:
        """Feed one reading. `t` is seconds (monotonic or elapsed).

        Returns a BlockageEvent on a state change, else None."""
        cfg = self.cfg
        if self._t0 is None:
            self._t0 = t
            self._armed_since = t
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
        #
        # The 2.5 must stay above window_factor + confirm_s/cycle_s (i.e. above
        # latency_s/cycle_s, ~1.75 by default). Callers wait latency_s after the
        # last well for a verdict on it; idling sooner than that would discard the
        # verdict they are waiting for.
        if self._wells and (t - self._wells[-1]) > 2.5 * cfg.cycle_s:
            self.set_cycling(False)

        # Need a full window of *judgeable* data. After a pause this restarts,
        # so idle-flat samples can never be mistaken for a blockage.
        if self._armed_since is None or t - self._armed_since < win:
            return None

        ratios, usable, flat_votes = {}, [], 0
        for i in range(cfg.n_channels):
            b = self._buf[i]
            if len(b) < 10:
                continue
            vals = sorted(v for _, v in b)
            lo = _quantile(vals, 0.05)
            rng = _quantile(vals, 0.95) - lo

            base = self._baseline_range(i)

            # Grow the baseline only from healthy samples. Frozen while blocked
            # so a long blockage cannot redefine "normal" and self-clear. The low
            # is kept alongside the range: clearance-watching needs to know where
            # the signal sits when fresh buffer is arriving.
            if not self.blocked and (base == 0.0 or rng >= base * cfg.clear_frac):
                self._base[i].append((t, rng, lo))
                while self._base[i] and t - self._base[i][0][0] > cfg.baseline_s:
                    self._base[i].popleft()
                base = self._baseline_range(i)

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
