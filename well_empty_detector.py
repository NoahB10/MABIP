"""Streaming "well ran dry" detector for the AMUZA / Jobst SIX rig (pure Python, no Qt).

Physics (measured on ~1000 scheduled lift-outs in 14 Jan-Jul 2026 runs, see report):
when the needle tip goes from liquid to air -- a scheduled lift-out, or a well that
has been sucked dry -- the sensor shows two things:

  1. SUCTION BREAK, no lag: within one sample of the tip leaving liquid, ch1/ch2/ch4
     dip sharply together (median -15..-30 sigma, ~-0.2 abs units) and ch5/ch6
     surge +20..35 % for ~20 s.  Seen at 88-100 % of lift-outs in clean runs, but
     only 0-15 % of move-ins (which happen while the sensor sees buffer).
  2. AIR SPIKE, one dead-volume lag later: a common-mode positive spike on ch1-ch4
     of roughly the SAME ABSOLUTE size on every channel (~0.4 units, i.e. 2-3x the
     plateau on ch1/2/4 but only ~0.4x on ch3, 0.1x on ch5, ~0 on ch6), 1-sample
     rise (<1.7 s), decays to 1/e in ~2 samples (~3.4 s).

Because the spike is common-mode and NOT proportional to analyte amplitude, the
detector scales each channel by its own sample noise (sigma of first differences),
not by its plateau, and votes across ch1-ch4.  ch5/ch6 are excluded from the spike
vote (their analyte noise swamps the spike: SNR 5-7 vs ~40 on ch1/2/4).

Rule
----
* note_move(t): robot was sent into a well.  Everything until t + lag + guard_s is
  the move-in transient (buffer->well air gap) and is ignored, except that the first
  air spike found there is used to (re)learn the lag and each channel's typical
  spike amplitude.
* Armed: a DIP (>= min_votes of ch1-4 below -k_dip sigma, common-mode) marks a
  suspected suction break at t_d -> event "SUSPECT" (early warning, lag-free).
* An AIR SPIKE (>= min_votes of ch1-4 above +k_spike sigma, common-mode, and at
  least amp_frac of the learned transit-spike amplitude) CONFIRMS if it lands at
  t_d + lag +- pair_tol_s, or if a second air spike follows within train_s.
  -> event "EMPTY" and the detector disarms until the next note_move().

Spikes are judged against a centred running median, so each verdict is delayed by
half_win samples (~6.7 s at 0.6 Hz).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = ["WellEmptyDetector", "WellEmptyConfig", "WellEmptyEvent"]


def _median(v):
    s = sorted(v); n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


@dataclass
class WellEmptyConfig:
    vote_channels: tuple = (0, 1, 2, 3)   # ch1-ch4 (0-based)
    min_votes: int = 3                     # of the usable vote channels
    half_win: int = 4                      # centred median of 2*4+1 = 9 samples (~15 s)
    k_spike: float = 8.0                   # residual / sigma for an air spike
    k_dip: float = 8.0                     # -residual / sigma for a suction-break dip
    common_mode_ratio: float = 4.0         # max/min abs amplitude among voters
    amp_frac: float = 0.3                  # spike must be >= this x learned transit amplitude
    min_snr: float = 8.0                   # channel votes only if learned spike amp / sigma >= this
    lag_s: float = 25.0                    # prior dead-volume lag; relearned from move-in spikes
    lag_min_s: float = 5.0
    lag_lo_frac: float = 0.6               # move-in air searched in [0.6, 1.6] x lag
    lag_hi_frac: float = 1.6
    lag_max_s: float = 150.0
    guard_s: float = 20.0                  # extra ignore after move + lag
    pair_tol_s: float = 10.0               # dip -> spike pairing tolerance (scaled up with lag)
    pair_tol_frac: float = 0.15            # tol = max(pair_tol_s, frac*lag)
    train_s: float = 30.0                  # 2 spikes within this confirm without a dip
    noise_len: int = 400                   # samples for sigma estimate
    sigma_floor: float = 1e-4
    n_channels: int = 6


@dataclass
class WellEmptyEvent:
    kind: str            # "SUSPECT" or "EMPTY"
    t: float             # time of the evidence (dip or spike sample), seconds
    t_since_move: float
    t_air_entered: float  # best estimate of when the tip met air
    reason: str
    channels: list = field(default_factory=list)
    message: str = ""


DEAD_VOLUME_UL = 130.0
"""Line dead volume, needle tip to sensor. Measured 22 Oct 2026: fluid took
80 s to reach the sensor at 100 µL/min. Gives the prior lag for a given pump
rate (relearned from the move-in air on every well)."""


def lag_for_rate(flow_rate: float, dead_volume_ul: float = DEAD_VOLUME_UL) -> float:
    """Seconds for air entering the tip to reach the sensor."""
    return 60.0 * dead_volume_ul / flow_rate


class WellEmptyDetector:
    def __init__(self, config: Optional[WellEmptyConfig] = None,
                 flow_rate: Optional[float] = None,
                 dead_volume_ul: float = DEAD_VOLUME_UL, **kw):
        self.cfg = config or WellEmptyConfig(**kw)
        # The Jun/Jul replays showed a wrong prior lag is the main way this
        # misses: seed it from the actual pump rate when we know it.
        if flow_rate and flow_rate > 0:
            self.cfg.lag_s = min(self.cfg.lag_max_s,
                                 max(self.cfg.lag_min_s, lag_for_rate(flow_rate, dead_volume_ul)))
        n = self.cfg.n_channels
        L = 2 * self.cfg.half_win + 1
        self._win = [deque(maxlen=L) for _ in range(n)]
        self._tw = deque(maxlen=L)
        self._dabs = [deque(maxlen=self.cfg.noise_len) for _ in range(n)]
        self._prev = [None] * n
        self._lags = deque(maxlen=9)
        self._amp = [deque(maxlen=21) for _ in range(n)]
        self._move_t = None
        self._lag_learned_this_move = False
        self._armed = False
        self._dips = deque(maxlen=16)
        self._spikes = deque(maxlen=8)
        self._best = None              # (residual sum, since, residuals) of largest move-in spike
        self._last_evt_i = -99
        self._i = 0
        self.lag_s = self.cfg.lag_s

    # ------------------------------------------------------------------ API
    def note_move(self, t_s: float) -> None:
        """Call when the needle is commanded into a well (the @P send time)."""
        self._move_t = float(t_s)
        self._lag_learned_this_move = False
        self._armed = False
        self._dips.clear()
        self._spikes.clear()
        self._best = None

    def note_lift(self, t_s: float) -> None:
        """Call if the robot is pulled out deliberately (stops judging this well)."""
        self._armed = False
        self._move_t = None

    @property
    def arm_time(self):
        if self._move_t is None:
            return None
        return self._move_t + self.lag_s + self.cfg.guard_s

    def update(self, t_s: float, channels: Sequence[float]) -> Optional[WellEmptyEvent]:
        cfg = self.cfg
        self._i += 1
        for c in range(cfg.n_channels):
            v = float(channels[c])
            if self._prev[c] is not None:
                self._dabs[c].append(abs(v - self._prev[c]))
            self._prev[c] = v
            self._win[c].append(v)
        self._tw.append(float(t_s))
        if len(self._tw) < self._tw.maxlen:
            return None
        h = cfg.half_win
        tc = self._tw[h]                         # the sample being judged (delayed)
        res, sig = {}, {}
        for c in cfg.vote_channels:
            w = list(self._win[c])
            s = 1.4826 * _median(self._dabs[c]) / 1.4142 if self._dabs[c] else 0.0
            sig[c] = max(s, cfg.sigma_floor)
            res[c] = w[h] - _median(w)
        usable = self._usable(sig)
        if len(usable) < cfg.min_votes:
            return None
        z = {c: res[c] / sig[c] for c in usable}
        is_spike = self._common(z, res, +1, cfg.k_spike)
        is_dip = self._common(z, res, -1, cfg.k_dip)
        # de-duplicate: one event per 3 samples
        if (is_spike or is_dip) and self._i - self._last_evt_i <= 3:
            return None
        if is_spike or is_dip:
            self._last_evt_i = self._i
        if self._move_t is None:
            return None
        since = tc - self._move_t
        # ----- move-in transient: learn lag + spike amplitude, never alarm
        if not self._armed:
            # the buffer->well air gap: take the LARGEST common-mode spike inside
            # [lag_lo_frac, lag_hi_frac] x current lag (early hydraulic blips in some
            # runs come at +10 s and must not be mistaken for the air).
            lo, hi = cfg.lag_lo_frac * self.lag_s, cfg.lag_hi_frac * self.lag_s
            if is_spike and max(lo, cfg.lag_min_s) <= since <= min(hi, cfg.lag_max_s):
                score = sum(res[c] for c in usable)
                if self._best is None or score > self._best[0]:
                    self._best = (score, since, {c: res[c] for c in usable})
            if self.arm_time is not None and tc >= self.arm_time:
                if self._best is not None:
                    self._lags.append(self._best[1])
                    self.lag_s = _median(self._lags)
                    for c, r in self._best[2].items():
                        self._amp[c].append(r)
                self._armed = True
            else:
                return None
        # ----- armed
        if is_spike and self._amp_ok(res, usable):
            tol = max(cfg.pair_tol_s, cfg.pair_tol_frac * self.lag_s)
            self._spikes.append(tc)
            chans = [f"ch{c+1}" for c in usable if z[c] > cfg.k_spike]
            match = [d for d in self._dips if abs(tc - (d + self.lag_s)) <= tol]
            if match:
                return self._fire(tc, since, match[0], "suction-break dip + air spike one lag later", chans)
            if len(self._spikes) >= 2 and self._spikes[-1] - self._spikes[-2] <= cfg.train_s:
                return self._fire(tc, since, self._spikes[-2] - self.lag_s, "train of air spikes", chans)
            return WellEmptyEvent("SUSPECT", tc, since, tc - self.lag_s, "isolated air spike", chans,
                                  f"Air spike {since:.0f}s into the well; waiting for confirmation.")
        if is_dip:
            self._dips.append(tc)
            return WellEmptyEvent("SUSPECT", tc, since, tc, "suction-break dip",
                                  [f"ch{c+1}" for c in usable if z[c] < -cfg.k_dip],
                                  f"Suction-break dip {since:.0f}s into the well; expect air at sensor in ~{self.lag_s:.0f}s.")
        return None

    # ------------------------------------------------------------ helpers
    def _usable(self, sig):
        out = []
        for c in self.cfg.vote_channels:
            if len(self._amp[c]) >= 3:
                if _median(self._amp[c]) / sig[c] < self.cfg.min_snr:
                    continue        # channel too noisy to see an air spike
            out.append(c)
        return out

    def _common(self, z, res, sign, k):
        voters = [c for c in z if sign * z[c] > k]
        if len(voters) < self.cfg.min_votes:
            return False
        a = [abs(res[c]) for c in voters]
        return max(a) / max(min(a), 1e-12) < self.cfg.common_mode_ratio

    def _amp_ok(self, res, usable):
        ok = 0
        for c in usable:
            if len(self._amp[c]) < 3:
                ok += 1
            elif res[c] >= self.cfg.amp_frac * _median(self._amp[c]):
                ok += 1
        return ok >= self.cfg.min_votes

    def _fire(self, t, since, t_air, reason, chans):
        self._armed = False
        self._move_t = None
        air_in = since - (t - t_air)      # seconds into the well the tip met air
        return WellEmptyEvent("EMPTY", t, since, t_air, reason, chans,
                              f"WELL EMPTY: {reason} on {', '.join(chans)}; tip met air "
                              f"~{air_in:.0f}s into the well (lag {self.lag_s:.0f}s).")
