"""Streaming "well ran dry -> leave" detector for the AMUZA / Jobst SIX rig, v3 (pure Python, no numpy, no Qt).

Drop-in replacement for well_empty_detector.py (same API: WellEmptyDetector(flow_rate=..., **cfg), WellEmptyConfig,
update(t_s, channels) -> event with .kind 'EMPTY'/'SUSPECT', .reason, .message, .t_air; note_move, note_lift,
set_flow_rate, rearm(move_t)).

Design: EVENT -> CONFIRM, and only after a valid measurement.  (Study: scratchpad/leave/, 25 Oct 2026.)

  0. VALID MEASUREMENT first.  Nothing is judged until the well's sample has reached a STEADY plateau: local-linear-
     trend Kalman filter on ch3/ch5/ch6, steady when the trend projected over 30 s is <= 5 % of the excursion
     (+2 sigma) on 2 consecutive samples, on 2 of the channels that have left the baseline (atlas definition).  Most
     early exits of v2 were wash-in transients that happen before this point.
  1. EVENT (sign-agnostic, scaled by the LEARNED air size, not only by noise -> survives the Jun-Aug noise rise and
     Batch14's inverted polarity):
       tip dip  : ch1, ch2, ch4 all fall below a 4-sample linear extrapolation by > min(k_dip*sigma, f_dip*air size)
       air      : ch1-ch4 all deviate with the SAME sign from a 6-sample extrapolation by > min(k_air*sigma, f_air*air size)
     The air size is learned on every well from the move-in air (buffer->well air gap, ~1 lag after the move).
     An event is only a SUSPECT: on its own it leaves early (bad buffer draws, mid-plateau glitches).
  2. CONFIRM (any of):
       P1 air at the sensor + wash-out: the air is seen at the sensor (the event itself, or for a tip dip an air flag
          0.5-1.6 lag later) and within wash_window_s the noise-weighted excursion of ch3/5/6 has fallen by >= wash_frac
          of its pre-event value (the plateau has really ended).  A mid-plateau glitch is followed by the same plateau.
       P2 dry-well suction break (no air peak needed, 22 Oct pattern): event >= dry_big x the learned air size, and
          ch1-4 sample-to-sample jitter stays >= dry_jit x its pre-event level in >= dry_frac of the samples from
          dry_from_s to dry_hold_s after it (the line keeps sucking air; a glitch settles in a few samples).
     -> EMPTY with t_air = the dip (or air - lag).  Candidates expire 3.5 lag + 40 s after the event.

Delay: P1 decides ~1 lag + 5..25 s after the tip meets air (it waits for the air and wash-out to reach the sensor);
P2 decides dry_hold_s (35 s) after the suction break.  Causal: every decision uses samples <= t only.

Results (scratchpad/leave/results, real low_volume controller logic, plateau rule as installed):
  * 20/80 plate splits (3 train / 12 test, 8 stratified splits): left early 0.22 +/- 0.11 % (installed 1.53 %),
    right time 77.1 +/- 3.8 % (installed 61.0 %), median latency 36 s (installed 21 s); every leave after STEADY.
  * leave-one-plate-out: 2 early in 1376 wells (1.5 per 1000, 95 % upper bound 5.2 per 1000).
  * 22 Oct real dry wells: A5, A6 left +35 s; A2 +199 s; A1, A3, A4 NOT left by the detector (A2/A3 never reached
    STEADY before the air; A1/A4 not confirmed) -> they depend on the controller backstops / dwell cap.
Deployed parameters = the values tuned on ALL 15 gold plates (identical to the choice in most splits).
Controller note: low_volume keeps only 60 s of history (fresh_window_s + 30) but v3 confirms ~40 s after the air,
so plateau_before(t_air) sometimes lacks its window -> 3.8 % of right-time calls turn into 'never'.  Either keep
>= 120 s of history in LowVolumeController._record or set plateau_check=False (v3 already requires STEADY).
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
        self._movein = [self._move_t + self.cfg.air_hi_frac * self.lag_s, 0.0, None, self._move_t + 0.5 * self.lag_s]

    def note_lift(self, t_s: float) -> None:
        """The robot was pulled out deliberately: stop judging this well."""
        self._armed = False
        self._move_t = None
        self._cands = []

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
        last5 = [h[1] for h in list(self._hist)[-5:]]
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
        if not self._steady:
            if armed_n and len(K["ss"]) >= min(c.ss_vote, armed_n):
                self._steady = True
            elif armed_n == 0 and t >= mt + self.lag_s + c.blank_after_s:
                self._steady = True
