"""Tests for notifier.py (ntfy push) and the detector gating it depends on.

The network is faked at the urlopen seam so these run anywhere. The gating
tests pin the behaviour that makes a push notification trustworthy: a parked
plate must never raise a blockage, and idle data must not teach the detector
that "flat" is healthy.
"""

import json
import threading
import time

import pytest

from blockage_detector import BlockageDetector
from notifier import Notifier, DEFAULT_SERVER


class FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Capture:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.evt = threading.Event()

    def __call__(self, req, timeout=None):
        self.calls.append((req, timeout))
        self.evt.set()
        if self.fail:
            raise self.fail
        return FakeResp()


def _notifier(**kw):
    n = Notifier(enabled=True, topic="mabip-test", desktop=False, **kw)
    cap = Capture()
    n._urlopen = cap
    return n, cap


def test_send_now_posts_json_to_server_root():
    n, cap = _notifier()
    ok, detail = n.send_now("Title", "Body – with dash ✓", priority="urgent",
                            tags=["rotating_light"])
    assert ok, detail
    req, timeout = cap.calls[0]
    assert req.full_url == DEFAULT_SERVER + "/"
    assert req.get_method() == "POST"
    body = json.loads(req.data.decode("utf-8"))
    assert body == {"topic": "mabip-test", "title": "Title",
                    "message": "Body – with dash ✓", "priority": 5,
                    "tags": ["rotating_light"]}
    assert timeout == n.timeout_s


def test_send_is_async_and_never_raises():
    n, cap = _notifier()
    cap.fail = OSError("no network")
    assert n.send("t", "m") is True          # queued
    assert cap.evt.wait(2.0)
    # Give the thread a moment to record the failure.
    for _ in range(50):
        if n.failed_count:
            break
        time.sleep(0.02)
    assert n.failed_count == 1
    assert "no network" in n.last_error


def test_disabled_sends_nothing():
    n, cap = _notifier()
    n.enabled = False
    assert n.send("t", "m") is False
    assert not cap.calls


def test_rate_limit_per_key():
    n, cap = _notifier(min_gap_s=60.0)
    assert n.send("t", "m", key="blocked") is True
    assert n.send("t", "m again", key="blocked") is False   # suppressed
    assert n.send("t", "m", key="cleared") is True          # different key
    assert n.send("t", "m", key=None) is True               # no key, no limit


def test_send_now_reports_http_error():
    import urllib.error
    n, cap = _notifier()
    cap.fail = urllib.error.HTTPError("u", 404, "not found", {}, None)
    ok, detail = n.send_now("t", "m")
    assert not ok and "404" in detail


def test_send_now_requires_topic():
    n = Notifier(enabled=True, topic="", desktop=False)
    ok, detail = n.send_now("t", "m")
    assert not ok and "topic" in detail.lower()


def test_new_topic_is_random_and_prefixed():
    a, b = Notifier.new_topic(), Notifier.new_topic()
    assert a != b and a.startswith("mabip-") and len(a) == len("mabip-") + 8


def test_configure_normalises_server():
    n = Notifier()
    n.configure(server="https://example.org/", topic=" abc ", enabled=True)
    assert n.server == "https://example.org"
    assert n.topic == "abc"
    assert n.topic_url == "https://example.org/abc"


# ---- detector gating -------------------------------------------------------

CYCLE = 60.0
DT = 1.0


def _feed(det, t0, seconds, fn):
    t = t0
    end = t0 + seconds
    events = []
    while t < end:
        ev = det.update(t, fn(t))
        if ev is not None:
            events.append(ev)
        t += DT
    return t, events


def _square(t, amp):
    return amp if (t % CYCLE) < CYCLE / 2 else 0.0


def _healthy(t):
    return [0.0, 0.0, _square(t, 5.0), 0.0, _square(t, 20.0), _square(t, 60.0)]


def _flat(_t):
    return [0.0] * 6


def test_idle_plate_never_alarms_and_does_not_learn_flat_baseline():
    det = BlockageDetector(cycle_s=CYCLE, min_baseline_s=0.0)
    det.set_cycling(False)
    # Sensor connected, plate parked for a long time: flat is correct.
    t, events = _feed(det, 0.0, 20 * CYCLE, _flat)
    assert events == []
    assert det.status() == "idle (not cycling)"
    assert all(len(b) == 0 for b in det._base), "idle data must not become baseline"

    # Run starts: healthy cycling must be learned as the baseline...
    det.set_cycling(True)
    t, events = _feed(det, t, 6 * CYCLE, _healthy)
    assert events == []
    assert det.status() == "flowing"
    # ...so a real blockage right after is caught.
    t, events = _feed(det, t, 4 * CYCLE, _flat)
    assert [e.blocked for e in events] == [True]


def test_baseline_learned_from_idle_would_mask_blockage_without_gating():
    """Documents the failure the gating prevents: with learning allowed while
    idle, the flat idle period becomes "healthy" and a blockage is missed."""
    det = BlockageDetector(cycle_s=CYCLE, min_baseline_s=0.0)
    det.set_cycling(False)
    # Simulate the old behaviour by forcing the baseline from the idle stretch.
    det._cycling = True
    t, _ = _feed(det, 0.0, 20 * CYCLE, _flat)
    det._cycling = False
    det.set_cycling(True)
    t, _ = _feed(det, t, 2 * CYCLE, _healthy)
    t, events = _feed(det, t, 4 * CYCLE, _flat)
    assert events == [], "sanity: idle-learned baseline hides the blockage"


def test_well_completions_set_cycle_and_idle_when_they_stop():
    det = BlockageDetector(cycle_s=10.0, min_baseline_s=0.0)
    det.set_cycling(False)
    for k in range(4):
        det.note_well_completed(k * CYCLE)
    assert det.cfg.cycle_s == pytest.approx(CYCLE)
    assert det._cycling
    # No completion for > 2.5 cycles → detector idles itself.
    t, events = _feed(det, 3 * CYCLE, 3 * CYCLE, _healthy)
    assert not det._cycling
    assert events == []


def test_second_run_arms_despite_stale_completion():
    """Review finding: a previous run's last completion must not idle the
    detector the moment the next run starts."""
    det = BlockageDetector(cycle_s=CYCLE, min_baseline_s=0.0)
    det.set_cycling(False)
    det.set_cycling(True)
    t, _ = _feed(det, 0.0, 6 * CYCLE, _healthy)
    for k in range(4):
        det.note_well_completed(k * CYCLE)
    det.set_cycling(False)                      # run 1 ends
    t, _ = _feed(det, t, 10 * CYCLE, _flat)     # long idle
    det.set_cycling(True)                       # run 2 starts, blocked at once
    t, events = _feed(det, t, 1.0, _flat)       # first sample after Start
    assert det._cycling, "stale completion un-armed the detector"
    # The robot keeps stepping on schedule while the fluid is stuck, so
    # completions keep arriving; the detector must stay armed and alarm.
    for _ in range(4):
        t, ev = _feed(det, t, CYCLE, _flat)
        events += ev
        det.note_well_completed(t)
    assert [e.blocked for e in events] == [True]


def test_warmup_counts_from_first_baseline_entry():
    det = BlockageDetector(cycle_s=CYCLE, min_baseline_s=5 * CYCLE)
    det.set_cycling(False)
    t, _ = _feed(det, 0.0, 30 * CYCLE, _flat)   # connected long before Start
    det.set_cycling(True)
    t, _ = _feed(det, t, 2 * CYCLE, _healthy)
    assert det.status() == "warming up"
    t, _ = _feed(det, t, 6 * CYCLE, _healthy)
    assert det.status() == "flowing"


def test_judging_only_once_a_baseline_exists_and_resets_on_rearm():
    det = BlockageDetector(cycle_s=CYCLE, min_baseline_s=3 * CYCLE)
    det.set_cycling(False)
    assert not det.judging
    det.set_cycling(True)
    t, _ = _feed(det, 0.0, 2 * CYCLE, _healthy)
    assert not det.judging                      # warming up
    t, _ = _feed(det, t, 6 * CYCLE, _healthy)
    assert det.judging
    det.set_cycling(False)
    assert not det.judging
    det.set_cycling(True)                       # new run: stale verdict dropped
    assert not det.judging
