"""Tests for BlockageDetector.

Synthetic signals mimic the real rig: a square wave per well cycle on the
high-amplitude channels (ch3/ch5/ch6 in the real data), plus low-amplitude noisy
channels that must never get a vote.
"""

import math

import pytest

from blockage_detector import BlockageDetector, DetectorConfig

CYCLE = 177.0
DT = 1.662          # real sample interval, 0.602 Hz


def _noise(i, scale):
    """Deterministic pseudo-noise — no RNG seed juggling."""
    return scale * math.sin(i * 12.9898) * 0.5


def _square(t, amp, cycle=CYCLE):
    """Plateau for the first half of the cycle, baseline for the second."""
    return amp if (t % cycle) < cycle / 2 else 0.0


def _channels(t, i, cycling=True, stuck_at=None):
    """Six channels shaped like the real rig.

    ch3/ch5/ch6 carry real amplitude (1.4 / 6.3 / 18.2 in the data);
    ch1/ch2/ch4 are low-amplitude and noisy, as measured.
    """
    if stuck_at is not None:
        big = [stuck_at * k for k in (1.4, 6.3, 18.2)]
    elif cycling:
        big = [_square(t, a) for a in (1.4, 6.3, 18.2)]
    else:
        big = [0.0, 0.0, 0.0]
    return [
        0.3 + _noise(i, 0.026),          # ch1  low amp, noisy
        0.2 + _noise(i + 1, 0.024),      # ch2
        big[0] + _noise(i, 0.035),       # ch3
        0.2 + _noise(i + 2, 0.021),      # ch4
        big[1] + _noise(i, 0.078),       # ch5
        big[2] + _noise(i, 0.102),       # ch6
    ]


def _run(det, duration_s, start=0.0, **kw):
    """Feed synthetic samples; return (events, final_t)."""
    events, t, i = [], start, int(start / DT)
    while t < start + duration_s:
        ev = det.update(t, _channels(t, i, **kw))
        if ev:
            events.append((t, ev))
        t += DT
        i += 1
    return events, t


def _fresh(**kw):
    kw.setdefault("cycle_s", CYCLE)
    return BlockageDetector(config=DetectorConfig(**kw))


def test_healthy_cycling_never_alarms():
    """The core false-positive guard: a healthy square wave sits flat at plateau
    and at baseline for ~90 s per cycle, which is what defeats a short window."""
    det = _fresh()
    events, _ = _run(det, 40 * 60)
    assert [e for _, e in events if e.blocked] == []
    assert not det.blocked


def test_blockage_is_detected():
    det = _fresh()
    _run(det, 30 * 60)                       # establish healthy baseline
    events, _ = _run(det, 15 * 60, start=30 * 60, stuck_at=0.42)
    assert any(e.blocked for _, e in events), "should alarm on a stuck signal"
    assert det.blocked


def test_recovery_clears():
    det = _fresh()
    _run(det, 30 * 60)
    _run(det, 15 * 60, start=30 * 60, stuck_at=0.42)
    assert det.blocked
    events, _ = _run(det, 15 * 60, start=45 * 60)
    assert any(not e.blocked for _, e in events), "should clear once cycling"
    assert not det.blocked


def test_stuck_level_does_not_matter():
    """A blockage freezes the signal wherever it happened to be — at baseline,
    mid-rise, or at plateau. All must alarm."""
    for level in (0.0, 0.42, 1.0):
        det = _fresh()
        _run(det, 30 * 60)
        _run(det, 15 * 60, start=30 * 60, stuck_at=level)
        assert det.blocked, f"missed a blockage stuck at level {level}"


def test_not_cycling_suppresses_alarm():
    """Parked on one well, flat is correct behaviour, not a blockage."""
    det = _fresh()
    _run(det, 30 * 60)
    det.set_cycling(False)
    events, _ = _run(det, 15 * 60, start=30 * 60, stuck_at=0.42)
    assert [e for _, e in events if e.blocked] == []
    assert not det.blocked


def test_short_dip_does_not_alarm():
    """A flat stretch shorter than confirm_s must not fire."""
    det = _fresh(confirm_s=45.0)
    _run(det, 30 * 60)
    events, _ = _run(det, 20, start=30 * 60, stuck_at=0.42)
    assert [e for _, e in events if e.blocked] == []


def test_single_frozen_channel_is_not_a_blockage():
    """One dead electrode is not a fluidic blockage — min_channels must hold."""
    det = _fresh()
    _run(det, 30 * 60)
    t, i = 30 * 60, int(30 * 60 / DT)
    events = []
    while t < 45 * 60:
        ch = _channels(t, i)
        ch[5] = 7.6                       # freeze ch6 only; ch3/ch5 keep cycling
        ev = det.update(t, ch)
        if ev:
            events.append(ev)
        t += DT
        i += 1
    assert [e for e in events if e.blocked] == []
    assert not det.blocked


def test_warmup_does_not_alarm():
    """Before a baseline exists the detector must stay silent, not guess."""
    det = _fresh()
    events, _ = _run(det, 8 * 60, stuck_at=0.42)
    assert [e for _, e in events if e.blocked] == []


def test_baseline_frozen_during_blockage():
    """A long blockage must not redefine 'normal' and silently self-clear."""
    det = _fresh()
    _run(det, 30 * 60)
    _run(det, 60 * 60, start=30 * 60, stuck_at=0.42)   # a full hour stuck
    assert det.blocked, "long blockage self-cleared — baseline was not frozen"


def test_resume_after_pause_does_not_realarm_on_idle_data():
    """While paused in the buffer the signal is flat because nothing is stepping.
    That data sits in the window; judging against it would re-alarm the instant we
    resume, popping the dialog again on a line the user just cleared."""
    det = _fresh()
    _run(det, 30 * 60)                                  # healthy baseline

    det.set_cycling(False)
    _run(det, 10 * 60, start=30 * 60, stuck_at=0.42)    # parked: flat, ignored
    assert not det.blocked

    det.set_cycling(True)
    # Resume healthy. The window still holds the flat pause data at first, so the
    # detector must stay silent until a fresh window has filled.
    events, _ = _run(det, 20 * 60, start=40 * 60)
    assert [e for _, e in events if e.blocked] == [], \
        "re-alarmed on its own idle data after resuming"


def test_resume_still_detects_a_genuine_blockage():
    """Re-arming must not blind it — a real clog after resuming still fires."""
    det = _fresh()
    _run(det, 30 * 60)
    det.set_cycling(False)
    _run(det, 5 * 60, start=30 * 60, stuck_at=0.42)
    det.set_cycling(True)
    _run(det, 15 * 60, start=35 * 60)                   # healthy, re-arms
    _run(det, 15 * 60, start=50 * 60, stuck_at=0.42)    # genuinely stuck again
    assert det.blocked


def test_note_well_completed_learns_cycle():
    det = _fresh(cycle_s=60.0)            # deliberately wrong
    for k in range(6):
        det.note_well_completed(k * 177.0)
    assert det.cfg.cycle_s == pytest.approx(177.0, abs=1.0)


def test_well_completions_stopping_idles_detector():
    det = _fresh()
    _run(det, 30 * 60)
    det.note_well_completed(30 * 60)
    # No further completions: >2.5 cycles later the detector should idle rather
    # than blame the fluidics for a plate that is not stepping.
    _run(det, 15 * 60, start=30 * 60, stuck_at=0.42)
    assert not det.blocked


@pytest.mark.parametrize("window_factor", [1.2, 1.3, 1.5, 1.8])
def test_auto_idle_never_beats_the_verdict_it_is_waiting_for(window_factor):
    """Callers wait latency_s after the last well for a verdict on it. If the
    auto-idle grace (2.5 cycles) were shorter than that, the detector would idle
    and drop the verdict mid-wait, and the last wells of every run would be
    silently marked clean."""
    det = _fresh(window_factor=window_factor)
    assert det.latency_s < 2.5 * det.cfg.cycle_s


def test_idle_grace_outlasts_latency_in_practice():
    """The last well must still be judgeable while the plate sits idle."""
    det = _fresh()
    _run(det, 30 * 60)
    det.note_well_completed(30 * 60)
    # Stuck from the moment the last well ends; the alarm must land within
    # latency_s, i.e. before the auto-idle grace expires.
    events, _ = _run(det, int(det.latency_s) + 30, start=30 * 60, stuck_at=0.42)
    assert any(e.blocked for _, e in events), "verdict lost to auto-idle"


def test_emits_edges_only():
    """Events fire on state change, not once per sample."""
    det = _fresh()
    _run(det, 30 * 60)
    events, _ = _run(det, 15 * 60, start=30 * 60, stuck_at=0.42)
    assert len([e for _, e in events if e.blocked]) == 1
