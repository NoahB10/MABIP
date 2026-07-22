"""Tests for firing a clog-clearing burst while parked in the buffer.

Two layers: FlowControlTab.try_clear_burst (the buffer-only / no-stacking guard,
which is the safety-critical part) and AsyncAMUZAGUI._attempt_clearance (the
cooldown and the no-op-without-a-pump behaviour). Neither needs real hardware —
the burst trigger is stubbed and we assert on what got called.
"""

import time

import pytest

from gui_async import AsyncAMUZAGUI


# ---- FlowControlTab.try_clear_burst -----------------------------------------

class FakeTab:
    """Stand-in exposing exactly what try_clear_burst reads."""

    try_clear_burst = None      # bound below from the real class

    def __init__(self, phase="buffer", rate=5.0, busy=False, bursting=False,
                 line=object()):
        self.line = line
        self._busy = busy
        self._bursting = bursting
        self._phase = phase
        self.f_rate = rate
        self.triggered = 0

    def _num(self, w, default=None):
        return float(w) if w is not None else default

    def _trigger_burst(self, auto=False):
        self.triggered += 1
        self._bursting = True       # as the real one does, synchronously


def _import_real_method():
    from flow_control_tab import FlowControlTab
    FakeTab.try_clear_burst = FlowControlTab.try_clear_burst


_import_real_method()


def test_bursts_when_parked_in_buffer():
    tab = FakeTab(phase="buffer")
    assert tab.try_clear_burst() is True
    assert tab.triggered == 1


def test_never_bursts_mid_well():
    """The safety guarantee the user asked for: buffer only, never in a well."""
    tab = FakeTab(phase="well")
    assert tab.try_clear_burst() is False
    assert tab.triggered == 0


def test_never_bursts_when_idle():
    tab = FakeTab(phase="idle")
    assert tab.try_clear_burst() is False
    assert tab.triggered == 0


def test_does_not_stack_bursts():
    """While a burst is running the loop must not fire another."""
    tab = FakeTab(phase="buffer", bursting=True)
    assert tab.try_clear_burst() is False
    assert tab.triggered == 0


def test_declines_while_busy():
    tab = FakeTab(phase="buffer", busy=True)
    assert tab.try_clear_burst() is False


def test_declines_without_a_rate_to_boost_from():
    tab = FakeTab(phase="buffer", rate=0.0)
    assert tab.try_clear_burst() is False
    assert tab.triggered == 0


def test_declines_without_a_pump():
    tab = FakeTab(phase="buffer", line=None)
    assert tab.try_clear_burst() is False


# ---- AsyncAMUZAGUI._attempt_clearance ---------------------------------------

class FakeGUI:
    _attempt_clearance = AsyncAMUZAGUI._attempt_clearance
    BLOCKAGE_BURST_COOLDOWN_S = AsyncAMUZAGUI.BLOCKAGE_BURST_COOLDOWN_S

    def __init__(self, tab):
        self.flow_tab = tab
        self._last_burst_attempt = 0.0
        self.displayed = []

    def add_to_display(self, msg):
        self.displayed.append(msg)


class BurstSpy:
    def __init__(self, fires=True):
        self.calls = 0
        self._fires = fires

    def try_clear_burst(self):
        self.calls += 1
        return self._fires


async def _call(gui):
    await gui._attempt_clearance()


def test_attempt_clearance_fires_the_burst():
    import asyncio
    spy = BurstSpy(fires=True)
    gui = FakeGUI(spy)
    asyncio.run(_call(gui))
    assert spy.calls == 1
    assert any("Bursting in the buffer" in m for m in gui.displayed)


def test_attempt_clearance_is_a_noop_without_a_pump():
    import asyncio
    gui = FakeGUI(None)
    asyncio.run(_call(gui))          # must not raise
    assert gui.displayed == []


def test_cooldown_prevents_spinning():
    """Back-to-back polls must not fire a burst every second."""
    import asyncio
    spy = BurstSpy(fires=True)
    gui = FakeGUI(spy)
    asyncio.run(_call(gui))
    asyncio.run(_call(gui))          # immediately again — inside the cooldown
    assert spy.calls == 1, "second call within the cooldown should be suppressed"


def test_cooldown_only_advances_when_a_burst_actually_fired():
    """A declined burst must not start the cooldown, or a rig that can't burst
    would wait pointlessly between attempts."""
    import asyncio
    spy = BurstSpy(fires=False)      # e.g. wrong phase / no rate
    gui = FakeGUI(spy)
    asyncio.run(_call(gui))
    asyncio.run(_call(gui))
    assert spy.calls == 2, "declined attempts should keep polling, not cool down"
