"""End the buffer wash when the signal is back at the run's starting baseline.

Why this exists
---------------
A fixed buffer time either wastes time (the line washed out long ago) or moves
into the next well while the previous well's fluid is still at the sensor. The
four metabolite outputs say directly when the wash is done: they are back where
they sat before the first well.

The outputs are channel differences (the same ones the plot shows, without the
gains, which only scale them):

    Glutamate = ch1 - ch2    Glutamine = ch3 - ch1
    Glucose   = ch5 - ch4    Lactate   = ch6 - ch4

Before the first well these sit close to 0, so "within 5 % of the baseline"
cannot mean 5 % of the baseline value: noise alone is bigger than that. It
means each output has come back 95 % of the way from the last well's plateau
to the baseline (Noah, 28 Oct 2026). If a well barely rose above the baseline,
being inside the baseline noise counts as back.

The wash is done when a majority (3 of 4) of the outputs are back and have
stayed back for ``hold_s``. Two guards around that:

* nothing is judged before one dead-volume lag after the lift-out, because
  until then the well's own fluid (and the lift-out air) is still on its way
  to the sensor;
* the baseline can drift. Replayed on 12 recorded plates (1086 washes, 28 Oct
  2026): at the end of the old 64 s buffer the outputs sat a median 5-20 % of
  the plateau height from the FIRST baseline (up to 60-90 % on plates with a
  start-up offset or drift), but only 1-6 % from the buffer level just before
  each well. A wash that insists on the first baseline would hang on most wells.
  So once the outputs have STOPPED moving (flat for ``settle_s`` after the lag)
  the wash ends anyway: ``"settled"`` if they recovered at least
  ``settle_min_recovery`` of the way back (drifted baseline, fluid is flowing),
  ``"stuck"`` if they did not (the line is probably not washing: a clog);
* after ``max_s`` the wash stops waiting and says so (``"cap"``).

The caller decides what "settled", "stuck" and "cap" mean (move on flagged, or
hold for the operator).

Levels are 5-sample medians, so one air spike cannot end (or extend) a wash.
Pure Python, no Qt: the GUI feeds it readings and awaits ``wait()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, asdict, fields
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["WashConfig", "BaselineWash", "OUTPUT_NAMES", "outputs_of"]

OUTPUT_NAMES = ("Glutamate", "Glutamine", "Glucose", "Lactate")


def outputs_of(ch) -> List[float]:
    """The four metabolite outputs from channels ch1..ch6 (0-based list)."""
    return [ch[0] - ch[1], ch[2] - ch[0], ch[4] - ch[3], ch[5] - ch[3]]


@dataclass
class WashConfig:
    enabled: bool = False
    """End the buffer wash at baseline instead of after the fixed Buffer Time."""

    tol_frac: float = 0.05
    """Back = within this fraction of the well's plateau height from baseline."""

    noise_k: float = 4.0
    """...or within this many baseline-noise sigmas (a well that barely rose)."""

    majority: int = 3
    """Outputs (of the 4) that must be back."""

    hold_s: float = 5.0
    """They must stay back this long (a single in-band sample is noise)."""

    base_window_s: float = 30.0
    """Baseline = median of the last this-many seconds before the first move."""

    min_lag: float = 1.0
    """Nothing is judged before this many dead-volume lags after the lift-out."""

    dead_volume_ul: float = 130.0
    """Tip-to-sensor dead volume, µL (65 s at 120 µL/min)."""

    settle_s: float = 60.0
    """Flat this long (after the lag) without reaching baseline = settled/stuck. 0 = off."""

    settle_tol: float = 0.03
    """Flat = each output moved less than this fraction of its plateau height
    over settle_s (or less than its baseline noise)."""

    settle_min_recovery: float = 0.7
    """Settled (drift) needs this share of the way back to baseline; less = stuck."""

    max_s: float = 600.0
    """Longest wash before giving up waiting (logged, flagged)."""

    @classmethod
    def from_dict(cls, d: dict) -> "WashConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


class BaselineWash:
    """Fed every sensor reading; ``wait()`` holds the buffer phase until baseline."""

    MIN_BASE_SAMPLES = 10

    def __init__(self, config: WashConfig, rate_fn: Optional[Callable[[], Optional[float]]] = None,
                 clock: Callable[[], float] = time.monotonic, poll_s: float = 0.5):
        self.cfg = config
        self.rate_fn = rate_fn
        self.clock = clock
        self.poll_s = poll_s
        self._hist = deque()          # (t, [4 outputs]) raw, for the baseline window
        self._med = deque()           # (t, [4 outputs]) 5-sample medians
        self.baseline: Optional[List[float]] = None
        self.sigma: Optional[List[float]] = None
        self._move_t: Optional[float] = None
        self._skip = False
        self.last: Optional[dict] = None   # what the last wash did (for the log)

    # ---- sensor side ------------------------------------------------------
    def feed(self, t: float, channels) -> None:
        try:
            ch = [float(x) for x in list(channels)[:6]]
        except (TypeError, ValueError):
            return
        if len(ch) < 6:
            return
        out = outputs_of(ch)
        self._hist.append((t, out))
        last5 = [o for _, o in list(self._hist)[-5:]]
        self._med.append((t, [_median([o[i] for o in last5]) for i in range(4)]))
        keep = max(self.cfg.max_s, 600.0) + 600.0     # a whole well + its wash
        while self._hist and t - self._hist[0][0] > keep:
            self._hist.popleft()
        while self._med and t - self._med[0][0] > keep:
            self._med.popleft()

    # ---- baseline ---------------------------------------------------------
    def set_baseline(self, t_end: Optional[float] = None) -> bool:
        """Freeze the baseline from the window ending at t_end (default: now).
        False if there are not enough readings yet."""
        t_end = self.clock() if t_end is None else t_end
        lo = t_end - self.cfg.base_window_s
        pts = [o for tt, o in self._hist if lo <= tt <= t_end]
        if len(pts) < self.MIN_BASE_SAMPLES:
            return False
        self.baseline = [_median([p[i] for p in pts]) for i in range(4)]
        self.sigma = [max(1.4826 * _median([abs(p[i] - self.baseline[i]) for p in pts]), 1e-3)
                      for i in range(4)]
        logger.info("Wash baseline: " + ", ".join(
            f"{n} {b:+.4f} (±{s:.4f})" for n, b, s in zip(OUTPUT_NAMES, self.baseline, self.sigma)))
        return True

    def samples_since(self, t0: float) -> int:
        return sum(1 for tt, _ in self._hist if tt >= t0)

    def note_move(self, t: float) -> None:
        """A move into a well was sent: its plateau is measured from here."""
        self._move_t = t

    def request_skip(self) -> None:
        """Operator: the wash is done, move on now."""
        self._skip = True

    def rebase(self) -> bool:
        """Operator: the buffer level NOW is the new baseline (after a drift
        that would otherwise hold every wash). Logged."""
        old = self.baseline
        ok = self.set_baseline()
        if ok:
            logger.info(f"Wash baseline RESET by operator (was {old})")
        return ok

    # ---- judging ----------------------------------------------------------
    def lag_s(self) -> float:
        r = None
        if self.rate_fn is not None:
            try:
                r = self.rate_fn()
            except Exception:
                r = None
        r = r if r and r > 0 else 120.0
        return 60.0 * self.cfg.dead_volume_ul / r

    def excursion(self, since: float, until: Optional[float] = None) -> List[float]:
        """Plateau height of each output over [since, until]: the 90th
        percentile of its distance from baseline (robust to the odd spike)."""
        seg = [o for tt, o in self._med if tt >= since and (until is None or tt <= until)]
        if not seg or self.baseline is None:
            return [0.0] * 4
        exc = []
        for i in range(4):
            d = sorted(abs(o[i] - self.baseline[i]) for o in seg)
            exc.append(d[int(0.9 * (len(d) - 1))])
        return exc

    def back(self, exc: List[float]) -> List[bool]:
        """Which outputs are back at baseline right now."""
        if not self._med or self.baseline is None:
            return [False] * 4
        now = self._med[-1][1]
        res = []
        for i in range(4):
            band = max(self.cfg.tol_frac * exc[i], self.cfg.noise_k * self.sigma[i])
            res.append(abs(now[i] - self.baseline[i]) <= band)
        return res

    def recovery(self, exc: List[float]) -> List[Optional[float]]:
        """Share of the way back from the plateau to baseline, per output
        (None for an output that never rose above its noise)."""
        if not self._med or self.baseline is None:
            return [None] * 4
        now = self._med[-1][1]
        out = []
        for i in range(4):
            if exc[i] <= self.cfg.noise_k * self.sigma[i]:
                out.append(None)
            else:
                out.append(1.0 - abs(now[i] - self.baseline[i]) / exc[i])
        return out

    def flat_for(self, exc: List[float], span: float) -> bool:
        """Majority of outputs moved less than settle_tol of their plateau
        height (or their noise) over the last `span` seconds."""
        if not self._med or span <= 0:
            return False
        t_now = self._med[-1][0]
        seg = [o for tt, o in self._med if tt >= t_now - span]
        if self._med[0][0] > t_now - span or len(seg) < 4:
            return False
        n = 0
        for i in range(4):
            v = [o[i] for o in seg]
            band = max(self.cfg.settle_tol * exc[i], self.cfg.noise_k * self.sigma[i])
            if max(v) - min(v) <= band:
                n += 1
        return n >= self.cfg.majority

    async def wait(self, stop_event, on_progress=None, label: str = "",
                   lift_t: Optional[float] = None, settle: bool = True) -> str:
        """Hold until the outputs are back at baseline.

        Returns "baseline", "skip" (operator), "settled" (flat, most of the
        way back: drifted baseline), "stuck" (flat, NOT most of the way back),
        "cap" (max_s reached), "stop", or "nobaseline" (no baseline was ever
        set: the caller should fall back to its fixed buffer time)."""
        if self.baseline is None:
            return "nobaseline"
        cfg = self.cfg
        t0 = self.clock() if lift_t is None else lift_t
        since = self._move_t if self._move_t is not None else t0 - 600.0
        self._skip = False
        in_band_since: Optional[float] = None
        result = "cap"
        while True:
            now = self.clock()
            el = now - t0
            if stop_event.is_set():
                result = "stop"
                break
            if self._skip:
                self._skip = False
                result = "skip"
                break
            lag = self.lag_s()
            # The well's fluid keeps arriving for one lag after the lift-out;
            # after that the wash itself must not shrink the plateau height.
            exc = self.excursion(since, t0 + lag)
            back = self.back(exc)
            n = sum(back)
            if el >= cfg.min_lag * lag and n >= cfg.majority:
                in_band_since = in_band_since if in_band_since is not None else now
                if now - in_band_since >= cfg.hold_s:
                    result = "baseline"
                    break
            else:
                in_band_since = None
            if settle and cfg.settle_s > 0 and el >= cfg.min_lag * lag + cfg.settle_s \
                    and self.flat_for(exc, cfg.settle_s):
                rec = [r for r in self.recovery(exc) if r is not None]
                result = "settled" if (not rec or _median(rec) >= cfg.settle_min_recovery) else "stuck"
                break
            if el >= cfg.max_s:
                result = "cap"
                break
            if on_progress:
                try:
                    wait_lag = max(0, int(round(cfg.min_lag * lag - el)))
                    msg = (f"Buffer: wash {el:.0f}s, {n}/4 at baseline"
                           + (f" (judging in {wait_lag}s)" if wait_lag else ""))
                    on_progress(msg, int(el), int(cfg.max_s))
                except Exception:
                    pass
            await asyncio.sleep(self.poll_s)
        el = self.clock() - t0
        exc = self.excursion(since, t0 + self.lag_s())
        rec = [r for r in self.recovery(exc) if r is not None]
        self.last = {"label": label, "result": result, "seconds": round(el, 1),
                     "back": sum(self.back(exc)),
                     "recovered": round(_median(rec), 2) if rec else None}
        logger.info(f"Wash {label}: {result} after {el:.0f}s "
                    f"({self.last['back']}/4 outputs at baseline"
                    + (f", {100 * self.last['recovered']:.0f} % of the way back)" if rec else ")"))
        return result


def _median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
