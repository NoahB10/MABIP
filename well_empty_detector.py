"""Streaming "well ran dry -> leave" detector for the AMUZA / Jobst SIX rig, v4 (pure Python, no numpy, no Qt).

Drop-in replacement for well_empty_detector.py v3: same API (WellEmptyDetector(flow_rate=..., **cfg),
WellEmptyConfig, update(t_s, channels) -> event with .kind 'EMPTY'/'SUSPECT', .reason, .message, .t_air;
note_move, note_lift, set_flow_rate, rearm(move_t)).  v3 (STEADY, P1, P2, optional P3) is kept unchanged in
_update3(); v4 adds two things.  Study: MABIP_datasets/drywell_study (25 Oct 2026).

  P4  DRY-WELL AIR FRONT.  All 13 real dry wells (22 + 25 Oct) end with the same event at the sensor: within ~3
      samples ALL SIX channels fall sharply below the plateau (ch1 often one sample ahead of the others = a front
      moving through the flow cell; the same shape as the air of a lift-out), with no upward spike, and the signal
      stays low for several samples before it partly recovers.  A glitch or bubble on a healthy plateau snaps back
      within 1-2 samples.  Rule: every channel drops >= dd_k sigma and the median >= dd_kmed sigma below the median
      of [t-15, t-4] s, median rise < dd_up sigma (a pure drop); dd_hold_s later the median channel has recovered
      <= dd_hold_rec of the drop.  After STEADY -> EMPTY (reason 'dry-well air front ...', t_air = drop - lag).
      Before STEADY -> SUSPECT 'dry-signature confirmed (no steady plateau)' and .dry_flag (for the controller),
      ignoring drops within 1 lag of a big all-channel rise (wash-in overshoot decay).
  RE-STEADY.  A big all-channel rise after STEADY that is still standing 0.4 lag later means the sample is only now
      arriving (25 Oct B1: partial clog, sample at ~152 s): STEADY is withdrawn for 0.5 lag and candidates made
      since the rise are dropped.  (Without this, v3 itself leaves B1 50 s early once plateau_check is off.)

Results (replay through the low_volume controller decision loop, live lv_config; drywell_study/results):
  * 13 real dry wells: v4 leaves 11 at +8..+25 s (median +8.5 s) after the air front reaches the sensor, none early;
    v3 as installed: 5 (median +45 s, A2/B3 only via the 199 s / 296 s backstop) and leaves 25 Oct B1 50 s early.
    Leave-one-RUN-out (tuned on one day, tested on the other): 5/7 (22 Oct) and 5/6 (25 Oct), 0 early.
    Not left: 22 Oct A2 re-entry (already empty, no sample ever) and 22 Oct A3 (dry before the plateau was steady).
  * 1376 gold wells: 1 early leave, same well as v3 (v3's P1, batch6 k73) = 0.7 / 1000 (95 % UB 4.0);
    leave-one-plate-out with the grid choice re-made per fold: 3 early = 2.2 / 1000 (UB 6.4).
    With the controller's volume gate (well_volume_ul entered, +-20 %): 0 early (UB 2.7 / 1000).
    The P4 rule WITHOUT the hold check ("none") adds 5 early leaves on gold -> keep "hold".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

__all__ = ["WellEmptyDetector", "WellEmptyConfig", "WellEmptyEvent"]

STRONG = (2, 4, 5)          # ch3, ch5, ch6
AIRCH = (0, 1, 2, 3)        # ch1-ch4
DIPCH = (0, 1, 3)           # ch1, ch2, ch4


def _median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _extrap(vals):
    """OLS line through vals (abscissa 0..M-1) extrapolated to M."""
    M = len(vals)
    xb = (M - 1) / 2.0
    yb = sum(vals) / M
    sxx = M * (M * M - 1) / 12.0
    sxy = sum((i - xb) * (v - yb) for i, v in enumerate(vals))
    return yb + (sxy / sxx) * (M - xb)


@dataclass
class WellEmptyConfig:
    lag_s: float = 25.0              # tip -> sensor dead-volume lag (seeded from flow_rate when given)
    lag_min_s: float = 5.0
    lag_max_s: float = 150.0
    guard_s: float = 20.0            # nothing judged before move + lag + guard
    noise_len: int = 400
    sigma_floor: float = 7e-4        # data quantised to 0.001
    # event
    k_dip: float = 20.0
    f_dip: float = 0.6
    k_air: float = 8.0
    f_air: float = 0.5
    amp_prior: float = 0.3           # air size (raw units) before 3 move-ins have been learned
    amp_keep: int = 8
    amp_learn_z: float = 8.0
    refractory_s: float = 5.0
    # valid measurement (STEADY)
    require_steady: bool = True
    ss_q: float = 0.1
    ss_eps: float = 0.05
    ss_h: float = 30.0
    ss_nconf: int = 2
    ss_arm_sigma: float = 8.0
    ss_vote: int = 2
    blank_after_s: float = 45.0
    # confirmation P1 (air + wash-out)
    air_lo_frac: float = 0.5
    air_hi_frac: float = 1.6
    wash_frac: float = 0.7
    wash_window_s: float = 25.0
    min_excursion: float = 24.0      # noise units (sum over ch3/5/6) for a wash-out to be judged
    # confirmation P2 (dry-well suction break)
    dry_path: bool = True
    dry_big: float = 1.5
    dry_jit: float = 3.0
    dry_frac: float = 0.7
    dry_from_s: float = 8.0
    dry_hold_s: float = 35.0
    # P3 (25 Oct 2026 low-volume test, OFF): a dry well's signal drops and then SETTLES at a clearly
    # lower level (no full wash-out, no sustained jitter), while a glitch usually returns to the same
    # plateau. Confirm when >= settle_vote of ch3/5/6 have lost >= settle_drop of their excursion and
    # stayed flat (range <= settle_flat of the excursion) for settle_hold_s, between settle_from_s and
    # settle_to_s after the event.  Tested and REJECTED for now: gains 2 of 13 real dry wells (25 Oct B3,
    # 22 Oct A4) but adds 15-16 early leaves on the 1376 gold wells (1 -> 16/17, for drop 0.25-0.6) and
    # left 22 Oct A6 86 s before it ran dry - healthy plateaus sometimes step down and stay there.
    settle_path: bool = False
    settle_drop: float = 0.4
    settle_flat: float = 0.15
    settle_vote: int = 2
    settle_from_s: float = 20.0
    settle_to_s: float = 90.0
    settle_hold_s: float = 10.0
    suspect_before_steady: bool = False  # True: also report tip dips before STEADY as SUSPECT (feeds low_volume dip+stale backstop; +4 early on gold)
    # P4 (v4): DRY-WELL AIR FRONT.  The 13 real dry wells (22 + 25 Oct 2026) all end with the same event: within
    # ~3 samples ALL SIX channels fall sharply below their plateau (the air front reaching the sensor; the same
    # signature as the air of a lift-out), with no upward spike, then partially recover.  On the healthy part of
    # 1376 gold wells such a pure all-channel drop is rare (bubbles / bad draws).
    dd_path: bool = True
    dd_pre_s: tuple = (15.0, 4.0)    # plateau level = median of x over [t-15, t-4] s
    dd_win: int = 3                  # drop measured as the min over the last dd_win samples
    dd_k: float = 10.0               # every one of dd_nch channels must fall by >= dd_k sigma ...
    dd_nch: int = 6
    dd_kmed: float = 10.0            # ... and the median channel by >= dd_kmed sigma
    dd_up: float = 5.0               # median upward excursion in the same window < dd_up sigma (a pure drop)
    dd_fair: float = 0.0             # median over ch1-4 of drop / learned air size >= dd_fair (0 = off)
    dd_confirm: str = "hold"         # "hold" (see dd_hold_s) or "none" (EMPTY at the drop: +5 early on gold)
    dd_hold_s: float = 6.0           # "hold": dd_hold_s after the drop the signal is still low (slow recovery):
    dd_hold_rec: float = 0.6         #   median recovered share of the drop <= dd_hold_rec (a glitch snaps back)
    dd_flat: float = 0.0             # plateau before the drop must be flat: median range / drop <= dd_flat (0 = off)
    resteady_up: float = 30.0        # >0: an all-channel RISE (>= 5 channels, median >= resteady_up sigma) after
                                     # STEADY means fresh sample is only now arriving (late wash-in, e.g. after a
                                     # partial clog, 25 Oct B1): the plateau must become steady again.
    resteady_check_lag: float = 0.4  # the rise must still stand (>= resteady_frac of it, >= 4 channels) this many lags later
    resteady_frac: float = 0.15
    resteady_hold_lag: float = 0.5   # after such a reset STEADY cannot return for this many lags
    dd_emit: str = "empty"           # "empty": EMPTY (reason 'dry-well air front'); "suspect": SUSPECT only
    dd_suspect_pre_steady: bool = True   # also confirm drops before STEADY -> SUSPECT 'dry-signature confirmed (no
                                         # steady plateau)' + dry_flag (the controller's dry backstop leaves flagged)
    dd_after_rise_lag: float = 1.0   # before STEADY, ignore drops within this many lags of a big all-channel rise
    n_channels: int = 6


@dataclass
class WellEmptyEvent:
    kind: str            # "SUSPECT" or "EMPTY"
    t: float
    t_since_move: float
    t_air_entered: float
    reason: str
    channels: list = field(default_factory=list)
    message: str = ""

    @property
    def t_air(self) -> float:
        return self.t_air_entered


DEAD_VOLUME_UL = 130.0


def lag_for_rate(flow_rate: float, dead_volume_ul: float = DEAD_VOLUME_UL) -> float:
    return 60.0 * dead_volume_ul / flow_rate


class _Cand:
    __slots__ = ("te", "isdip", "size", "pre", "b0", "epre", "s3", "jpre", "t_airseen", "jr_n", "jr_hi", "end", "done")


class WellEmptyDetector:
    def __init__(self, config: Optional[WellEmptyConfig] = None, flow_rate: Optional[float] = None,
                 dead_volume_ul: float = DEAD_VOLUME_UL, **kw):
        self.cfg = config or WellEmptyConfig(**kw)
        c = self.cfg
        self._rate = flow_rate if flow_rate and flow_rate > 0 else None
        if self._rate:
            c.lag_s = min(c.lag_max_s, max(c.lag_min_s, lag_for_rate(flow_rate, dead_volume_ul)))
        self.lag_s = c.lag_s
        n = c.n_channels
        self._hist = deque(maxlen=400)                 # (t, x[6]) raw
        self._recent = deque(maxlen=40)                # last 40 samples (v4 drop / rise checks)
        self._m5 = deque(maxlen=400)                   # (t, 5-sample median[6])
        self._dabs = [deque(maxlen=c.noise_len) for _ in range(n)]
        self._dx = deque(maxlen=400)                   # (t, |dx|[6])
        self._amps = deque(maxlen=c.amp_keep)
        self._amp = [c.amp_prior] * 4
        self._movein = None                            # [t_end, best_z, best_absR6]
        self._move_t: Optional[float] = None
        self._armed = False                            # False after an EMPTY until rearm / next move
        self._b0 = None
        self._kf = None
        self._steady = False
        self._cands: List[_Cand] = []
        self._last_cand_t = -1e18
        self._last_pre_t = -1e18
        self._dd_last = -1e18          # time of the last dry-signature drop (any)
        self._dd_pend = None           # [t_event, pre, mn] awaiting confirmation
        self._dd_n = 0                 # dry-signature drops seen in this well
        self.last_dd = None            # (t, median drop sigma, steady) of the most recent drop, for the controller
        self._resteady_t = None        # last late-wash-in reset of STEADY
        self._rise_pend = None         # (t, pre, peak, steady_before) of a big all-channel rise being checked
        self._last_rise_t = -1e18      # last big all-channel rise in this well (wash-in / late sample arrival)
        self.dry_flag = None           # (t_drop, steady) of the last CONFIRMED dry-signature drop (controller backstop)
        self._dd_ref = None            # (t, pre, trough) of the last dry-signature drop

    # ------------------------------------------------------------------ API
    def set_flow_rate(self, flow_rate: float) -> None:
        if not flow_rate or flow_rate <= 0:
            return
        prior = self._rate
        if prior and prior > 0:
            self.lag_s = min(self.cfg.lag_max_s, max(self.cfg.lag_min_s, self.lag_s * prior / flow_rate))
        self._rate = flow_rate

    def note_move(self, t_s: float) -> None:
        """The needle was commanded into a well (@P send time)."""
        self._finish_movein()
        self._move_t = float(t_s)
        self._armed = True
        self._b0 = None
        self._kf = None
        self._steady = False
        self._cands = []
        self._dd_pend = None
        self._dd_n = 0
        self._dd_last = -1e18
        self.last_dd = None
        self._resteady_t = None
        self._rise_pend = None
        self._last_rise_t = -1e18
        self.dry_flag = None
        self._dd_ref = None
        self._movein = [self._move_t + self.cfg.air_hi_frac * self.lag_s, 0.0, None, self._move_t + 0.5 * self.lag_s]

    def note_lift(self, t_s: float) -> None:
        """The robot was pulled out deliberately: stop judging this well."""
        self._armed = False
        self._move_t = None
        self._cands = []
        self._rise_pend = None
        self._dd_pend = None

    def rearm(self, move_t: float) -> None:
        """Keep judging the current well after a call was rejected upstream (no plateau)."""
        if self._move_t is None:
            self._move_t = float(move_t)
        self._armed = True

    @property
    def arm_time(self):
        return None if self._move_t is None else self._move_t + self.lag_s + self.cfg.guard_s

    @property
    def steady(self) -> bool:
        return self._steady

    # ------------------------------------------------------------------ helpers
    def _sig(self, c):
        d = self._dabs[c]
        return max(1.4826 * _median(d) / 1.41421356 if d else 0.0, self.cfg.sigma_floor)

    def _finish_movein(self):
        mi = self._movein
        if mi is not None and mi[2] is not None:
            self._amps.append(mi[2])
            if len(self._amps) >= 3:
                self._amp = [_median([a[i] for a in self._amps]) for i in range(4)]
        self._movein = None

    def _window(self, t0, t1, src):
        return [x for tt, x in src if t0 <= tt <= t1]

    # ------------------------------------------------------------------ main
    def update(self, t_s: float, channels: Sequence[float]) -> Optional[WellEmptyEvent]:
        c = self.cfg
        t = float(t_s)
        pend = self._rise_pend
        if pend is not None:
            self._steady = False          # no new judgement while a possible late wash-in is being checked
        ev = self._update3(t_s, channels)
        mt = self._move_t
        if pend is not None and self._rise_pend is pend and t >= pend[0] + c.resteady_check_lag * self.lag_s:
            self._rise_pend = None
            last = [v for _, v in list(self._recent)[-3:]]
            n_up = sum(1 for i in range(c.n_channels)
                       if _median([v[i] for v in last]) - pend[1][i] >= c.resteady_frac * (pend[2][i] - pend[1][i]) > 0)
            if n_up >= 4:                 # still up: fresh sample is only now arriving -> re-validate the plateau
                self._kf = None
                self._steady = False
                self._resteady_t = t
                self._cands = [cd for cd in self._cands if cd.te < pend[0] - 2.0]
                self._dd_pend = None
            else:
                self._steady = pend[3]
        if self._rise_pend is not None:
            self._steady = False
        if ev is not None and ev.kind == "EMPTY":
            return ev
        in_well = mt is not None and t >= mt + self.lag_s + c.guard_s
        if c.resteady_up > 0 and mt is not None and t >= mt + 0.3 * self.lag_s and self._rise_pend is None:
            sig = [self._sig(i) for i in range(c.n_channels)]
            br = self._big_rise(t, sig)
            dp = self._dd_ref
            if br is not None and dp is not None and t > dp[0]:
                # after a dry-signature drop the signal recovers from its trough: only a rise clearly ABOVE the
                # plateau before the drop means fresh sample is arriving
                if sum(1 for i in range(c.n_channels) if br[1][i] > dp[1][i] + 0.5 * (dp[1][i] - dp[2][i])) < 4:
                    br = None
            if br is not None:
                self._last_rise_t = t
                if self._steady and in_well:
                    self._rise_pend = (t, br[0], br[1], self._steady)
                    self._steady = False
        if c.dd_path and in_well and self._armed and self._rise_pend is None:
            sig = [self._sig(i) for i in range(c.n_channels)]
            dev = self._dd_step(t, None, sig, mt)
            if dev is not None and (ev is None or dev.kind == "EMPTY"):
                return dev
        return ev

    def _update3(self, t_s: float, channels: Sequence[float]) -> Optional[WellEmptyEvent]:
        """v3 logic, unchanged (STEADY, P1, P2, optional P3)."""
        c = self.cfg
        t = float(t_s)
        x = [float(v) for v in list(channels)[:c.n_channels]]
        prev = list(self._hist)
        if prev:
            px = prev[-1][1]
            ad = [abs(x[i] - px[i]) for i in range(c.n_channels)]
        else:
            ad = [0.0] * c.n_channels
        for i in range(c.n_channels):
            self._dabs[i].append(ad[i])
        self._dx.append((t, ad))
        self._hist.append((t, x))
        self._recent.append((t, x))
        last5 = [h[1] for h in list(self._recent)[-5:]]
        m5 = [_median([v[i] for v in last5]) for i in range(c.n_channels)]
        self._m5.append((t, m5))
        sig = [self._sig(i) for i in range(c.n_channels)]
        lag = self.lag_s

        # residuals vs causal linear extrapolation
        r6 = r4 = None
        if len(prev) >= 6:
            r6 = [x[ch] - _extrap([p[1][ch] for p in prev[-6:]]) for ch in AIRCH]
        if len(prev) >= 4:
            r4 = [x[ch] - _extrap([p[1][ch] for p in prev[-4:]]) for ch in DIPCH]

        # move-in air -> learn the air size (window [move + 0.5 lag, move + 1.6 lag])
        mi = self._movein
        if mi is not None:
            if t > mi[0]:
                self._finish_movein()
            elif t >= mi[3] and r6 is not None:
                z = _median([abs(r6[j]) / sig[ch] for j, ch in enumerate(AIRCH)])
                if z > c.amp_learn_z and z > mi[1]:
                    mi[1] = z
                    mi[2] = [abs(v) for v in r6]
        amp = self._amp

        mt = self._move_t
        # baseline at the move: median of [move - 10 s, move + 1 s]
        if mt is not None and self._b0 is None and t >= mt + 1.0:
            w = self._window(mt - 10.0, mt + 1.0, self._hist) or [x]
            self._b0 = [_median([v[i] for v in w]) for i in range(c.n_channels)]

        if mt is not None and self._b0 is not None:
            self._steady_step(t, x, sig)

        in_well = mt is not None and t >= mt + lag + c.guard_s
        # ---- flags
        dip = air = False
        size = 0.0
        if r4 is not None and r6 is not None:
            dip = all(r4[j] < -min(c.k_dip * sig[ch], c.f_dip * amp[ch if ch < 3 else 3]) for j, ch in enumerate(DIPCH))
            thr = [min(c.k_air * sig[ch], c.f_air * amp[ch]) for ch in AIRCH]
            air = all(r6[j] > thr[j] for j in range(4)) or all(r6[j] < -thr[j] for j in range(4))
            if all(v < 0 for v in r4):
                size = max(size, min(-v for v in r4) / (sum(amp[a] for a in (0, 1, 3)) / 3.0))
            if all(v > 0 for v in r6) or all(v < 0 for v in r6):
                size = max(size, min(abs(r6[j]) / amp[j] for j in range(4)))
        air_ok = air and in_well

        out = None
        # ---- update / confirm live candidates
        keep = []
        for cd in self._cands:
            if t > cd.end:
                continue
            tt = t - cd.te
            hit = None
            # P1: air at the sensor, then wash-out
            if cd.t_airseen is None:
                if not cd.isdip:
                    cd.t_airseen = 0.0
                elif air_ok and c.air_lo_frac * lag <= tt <= c.air_hi_frac * lag:
                    cd.t_airseen = tt
            if cd.t_airseen is not None and cd.epre is not None and cd.t_airseen <= tt <= cd.t_airseen + c.wash_window_s:
                enow = sum(abs(m5[ch] - cd.b0[ch]) / cd.s3[j] for j, ch in enumerate(STRONG))
                if 1.0 - enow / cd.epre >= c.wash_frac:
                    hit = "air at sensor + wash-out"
            # P2: dry-well suction break with sustained turbulence
            if c.dry_path and cd.size >= c.dry_big and not cd.done:
                if c.dry_from_s <= tt <= c.dry_hold_s:
                    jr = _median([ad[ch] / cd.jpre[j] for j, ch in enumerate(AIRCH)])
                    cd.jr_n += 1
                    cd.jr_hi += 1 if jr >= c.dry_jit else 0
                elif tt > c.dry_hold_s:
                    cd.done = True
                    if hit is None and cd.jr_n and cd.jr_hi / cd.jr_n >= c.dry_frac:
                        hit = "suction break + sustained air in line (dry well)"
            # P3: settled at a clearly lower level (dry well), not back on the plateau (glitch)
            if hit is None and c.settle_path and cd.epre is not None and c.settle_from_s <= tt <= c.settle_to_s:
                hw = self._window(t - c.settle_hold_s, t, self._m5)
                if len(hw) >= 4:
                    votes = 0
                    for j, ch in enumerate(STRONG):
                        exc = cd.pre[ch] - cd.b0[ch]
                        if abs(exc) < 8 * cd.s3[j]:
                            continue
                        vals = [v[ch] for v in hw]
                        lost = [(cd.pre[ch] - v) / exc for v in vals]      # sign-agnostic share of excursion lost
                        flat = (max(vals) - min(vals)) / abs(exc)
                        if min(lost) >= c.settle_drop and flat <= c.settle_flat:
                            votes += 1
                    if votes >= c.settle_vote:
                        hit = "signal settled clearly lower after the event (dry well)"
            if hit is not None:
                # each candidate confirms once (consumed even if it cannot be emitted)
                if out is None and self._armed and self._move_t is not None:
                    t_air = cd.te if cd.isdip else cd.te - lag
                    out = self._fire(t, t_air, hit)
                continue
            keep.append(cd)
        self._cands = keep
        if out is not None:
            return out

        # ---- new candidate (valid measurement first)
        if mt is None or not in_well or not (dip or air):
            return None
        if self._rise_pend is not None:
            return None
        if c.require_steady and not self._steady:
            if dip and c.suspect_before_steady and t - self._last_pre_t >= c.refractory_s:
                self._last_pre_t = t
                return WellEmptyEvent("SUSPECT", t, t - mt, t, "suction-break dip (no steady plateau yet)",
                                      ["ch1", "ch2", "ch4"], f"Suction-break dip {t - mt:.0f}s into the well before "
                                      "the plateau was steady: not a valid end of sample (watching).")
            return None
        if t - self._last_cand_t < c.refractory_s:
            return None
        self._last_cand_t = t
        pre_w = self._window(t - 20.0, t - 3.0, self._hist)
        if len(pre_w) < 3:
            return None
        cd = _Cand()
        cd.te = t
        cd.isdip = bool(dip)
        cd.size = size
        cd.pre = [_median([v[i] for v in pre_w]) for i in range(c.n_channels)]
        cd.b0 = list(self._b0)
        cd.s3 = [sig[ch] for ch in STRONG]
        e = sum(abs(cd.pre[ch] - cd.b0[ch]) / cd.s3[j] for j, ch in enumerate(STRONG))
        cd.epre = e if e >= c.min_excursion else None
        jw = self._window(t - 30.0, t - 3.0, self._dx)
        cd.jpre = [max(_median([v[ch] for v in jw]), 7e-4) if jw else 7e-4 for ch in AIRCH]
        cd.t_airseen = None
        cd.jr_n = 0
        cd.jr_hi = 0
        cd.done = False
        cd.end = t + 3.5 * lag + 40.0
        self._cands.append(cd)
        # the event sample itself can confirm P1 (air event with an immediate wash-out is impossible in one
        # sample, but keep the order identical to the offline study: the check runs from the next sample on)
        why = "suction-break dip" if dip else "common-mode air"
        return WellEmptyEvent("SUSPECT", t, t - mt, t if dip else t - lag, why, ["ch1", "ch2", "ch3", "ch4"],
                              f"{why} {t - mt:.0f}s into the well (size {size:.1f}x air); confirming "
                              f"(air at sensor + wash-out, or sustained suction)")

    def _big_rise(self, t, sig):
        c = self.cfg
        h = self._recent
        lo, hi = t - c.dd_pre_s[0], t - c.dd_pre_s[1]
        pw = [v for tt, v in h if lo <= tt <= hi]
        if len(pw) < 4 or len(h) < c.dd_win:
            return None
        last = [v for _, v in list(h)[-c.dd_win:]]
        pre = [_median([v[i] for v in pw]) for i in range(c.n_channels)]
        mx = [max(v[i] for v in last) for i in range(c.n_channels)]
        up = [(mx[i] - pre[i]) / sig[i] for i in range(c.n_channels)]
        if sum(1 for v in up if v >= 0.5 * c.resteady_up) >= 5 and _median(up) >= c.resteady_up:
            return pre, mx
        return None

    def _dd_drop(self, t, sig):
        """(median drop sigma, pre, mn) if the last dd_win samples are a pure all-channel drop, else None."""
        c = self.cfg
        h = self._recent
        if len(h) < c.dd_win + 6:
            return None
        lo, hi = t - c.dd_pre_s[0], t - c.dd_pre_s[1]
        pw = [v for tt, v in h if lo <= tt <= hi]
        if len(pw) < 4:
            return None
        last = [v for _, v in list(h)[-c.dd_win:]]
        n = c.n_channels
        pre = [_median([v[i] for v in pw]) for i in range(n)]
        mn = [min(v[i] for v in last) for i in range(n)]
        mx = [max(v[i] for v in last) for i in range(n)]
        dn = [(pre[i] - mn[i]) / sig[i] for i in range(n)]
        up = [(mx[i] - pre[i]) / sig[i] for i in range(n)]
        if sum(1 for v in dn if v >= c.dd_k) < c.dd_nch:
            return None
        dmed = _median(dn)
        if c.dd_flat > 0:
            fl = _median([(max(v[i] for v in pw) - min(v[i] for v in pw)) / max(pre[i] - mn[i], 1e-9) for i in range(n)])
            if fl > c.dd_flat:
                return None
        if dmed < c.dd_kmed or _median(up) >= c.dd_up:
            return None
        if c.dd_fair > 0:
            fa = _median([(pre[ch] - mn[ch]) / max(self._amp[ch], 1e-6) for ch in AIRCH])
            if fa < c.dd_fair:
                return None
        return dmed, pre, mn

    def _dd_step(self, t, x, sig, mt):
        c = self.cfg
        pd = self._dd_pend
        if pd is not None and t >= pd[0] + c.dd_hold_s:
            # HOLD: a real air front leaves the signal low for several samples; a glitch snaps back
            self._dd_pend = None
            te, pre, mn, dmed, was_steady = pd
            last = [v for _, v in list(self._recent)[-2:]]
            lv = [_median([v[i] for v in last]) for i in range(c.n_channels)]
            rec = _median([(lv[i] - mn[i]) / (pre[i] - mn[i]) for i in range(c.n_channels) if pre[i] - mn[i] > 0] or [9.0])
            if rec <= c.dd_hold_rec:
                self.dry_flag = (te, was_steady and self._steady)
                if was_steady and self._steady:
                    return self._dd_emit(t, te, f"dry-well air front (all six channels dropped, slow recovery {rec:.2f})")
                return WellEmptyEvent("SUSPECT", t, t - mt, te, "dry-signature confirmed (no steady plateau)",
                                      ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"],
                                      f"Dry-well air front {te - mt:.0f}s into the well but the plateau was never "
                                      "steady: the sample is probably not valid.")
        r = self._dd_drop(t, sig)
        if r is None or t - self._dd_last < c.refractory_s:
            if r is not None:
                self._dd_last = t
            return None
        self._dd_last = t
        dmed, pre, mn = r
        steady = self._steady or not c.require_steady
        if not steady:
            # before STEADY: only a drop from a settled level counts (not the decay after a wash-in overshoot)
            if not c.dd_suspect_pre_steady or t - self._last_rise_t < c.dd_after_rise_lag * self.lag_s:
                return None
        self._dd_n += 1
        self.last_dd = (t, dmed, steady)
        if c.dd_confirm == "none" and steady:
            self.dry_flag = (t, True)
            return self._dd_emit(t, t, "dry-well air front (all six channels dropped)")
        if self._dd_pend is None:
            self._dd_pend = [t, pre, mn, dmed, steady]
            self._dd_ref = (t, pre, mn)
        return WellEmptyEvent("SUSPECT", t, t - mt, t, "dry-signature drop" + ("" if steady else " (no steady plateau yet)"),
                              ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"],
                              f"All six channels dropped ({dmed:.0f} sigma) {t - mt:.0f}s into the well; confirming.")

    def _dd_emit(self, t, te, reason):
        if self.cfg.dd_emit != "empty":
            mt = self._move_t
            return WellEmptyEvent("SUSPECT", t, t - mt, te, "dry-signature " + reason,
                                  ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"], f"Dry-well signature: {reason}.")
        # the air front reaches the sensor ~1 dead volume after the tip met air
        return self._fire(t, te - self.lag_s, reason)

    def _fire(self, t, t_air, reason):
        mt = self._move_t
        self._armed = False
        return WellEmptyEvent("EMPTY", t, t - mt, t_air, reason, ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"],
                              f"WELL EMPTY: {reason}; tip met air ~{t_air - mt:.0f}s into the well "
                              f"(confirmed {t - t_air:.0f}s later, after a steady plateau).")

    # ------------------------------------------------------------------ STEADY (valid measurement)
    def _steady_step(self, t, x, sig):
        c = self.cfg
        mt = self._move_t
        if t < mt + max(10.0, self.lag_s - 2.0):
            return
        b0 = self._b0
        if self._kf is None:
            s0 = list(sig)
            self._kf = dict(s0=s0, f={}, ss=set())
        K = self._kf
        s0 = K["s0"]
        armed_n = 0
        for ch in STRONG:
            f = K["f"].get(ch)
            if f is None:
                K["f"][ch] = [x[ch], 0.0, [[20 * s0[ch] ** 2, 0.0], [0.0, 20 * s0[ch] ** 2]], t, 0]
                continue
            lv, sl, P, tp, cnt = f
            dt = t - tp
            R_ = 2 * s0[ch] ** 2
            Q_ = c.ss_q * s0[ch] ** 2
            lvp = lv + sl * dt
            p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1]
            p01 = P[0][1] + dt * P[1][1]
            p10 = P[1][0] + dt * P[1][1]
            p11 = P[1][1] + Q_ * dt
            S = p00 + R_
            k0, k1 = p00 / S, p10 / S
            e = x[ch] - lvp
            lv = lvp + k0 * e
            sl = sl + k1 * e
            P = [[p00 - k0 * p00, p01 - k0 * p01], [p10 - k1 * p00, p11 - k1 * p01]]
            if abs(x[ch] - b0[ch]) >= c.ss_arm_sigma * s0[ch]:
                armed_n += 1
                cnt = cnt + 1 if abs(sl) * c.ss_h <= c.ss_eps * abs(lv - b0[ch]) + 2 * s0[ch] else 0
                if cnt >= c.ss_nconf:
                    K["ss"].add(ch)
            else:
                cnt = 0
            K["f"][ch] = [lv, sl, P, t, cnt]
        if self._resteady_t is not None and t < self._resteady_t + c.resteady_hold_lag * self.lag_s:
            K["ss"].clear()
            return
        if not self._steady:
            if armed_n and len(K["ss"]) >= min(c.ss_vote, armed_n):
                self._steady = True
            elif armed_n == 0 and t >= mt + self.lag_s + c.blank_after_s:
                self._steady = True
