"""Regressions from 22 Oct 2026.

1. A modal dialog opened inside a coroutine killed the AMUZA command loops, so
   moves were never sent and wells were marked done anyway.
2. A move the robot never acknowledged fell through to "completed".
3. STOP / RESUME must stop and restart the pump.
"""

import ast
import asyncio
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import amuza_async
from amuza_async import AsyncAmuzaConnection, Method


# ---- 1. no nested event loops inside coroutines ---------------------------

MODAL = ("QMessageBox.warning", "QMessageBox.critical", "QMessageBox.information",
         "QMessageBox.question", "QMessageBox.about")


@pytest.mark.parametrize("path", ["gui_async.py", "flow_control_tab.py"])
def test_no_modal_dialogs_inside_async_functions(path):
    tree = ast.parse(open(path).read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    name = ast.unparse(sub.func)
                    if name in MODAL or name.endswith((".exec_", ".exec")) \
                            or name.startswith(("QFileDialog.get", "QInputDialog.get")):
                        bad.append(f"{path}:{sub.lineno} {node.name}: {name}")
    assert not bad, "nested Qt loop inside a coroutine (use _popup):\n" + "\n".join(bad)


# ---- 2. unacknowledged move must fail, not complete -----------------------

def _conn(alive=True, goes_busy_on_attempt=None):
    c = AsyncAmuzaConnection(use_mock=False)
    sent = []
    attempts = {"n": 0}

    async def send_command(cmd, **kw):
        sent.append(cmd)
        return True

    async def wait_for_busy(timeout=5.0):
        attempts["n"] += 1
        return goes_busy_on_attempt is not None and attempts["n"] >= goes_busy_on_attempt

    c.send_command = send_command
    c.wait_for_busy = wait_for_busy
    c.calculate_move_time = lambda pos: 0.0
    c.last_status_t = time.monotonic() if alive else time.monotonic() - 60
    # Ready from before the command: this is what used to fake a completion.
    c.status.state = amuza_async.DeviceState.READY
    c.status.is_moving = False
    return c, sent


def _run(c, method):
    return asyncio.run(c.execute_method(method, asyncio.Event()))


def test_move_never_acknowledged_fails_after_one_resend():
    c, sent = _conn(alive=True, goes_busy_on_attempt=None)
    ok = _run(c, Method(pos="G1", wait=5, buffer_time=0))
    assert ok is False
    assert len([s for s in sent if s.startswith("@P")]) == 2     # original + one resend
    assert "G1" in c.last_error and "not marked done" in c.last_error


def test_dead_link_fails_without_resend():
    c, sent = _conn(alive=False)
    ok = _run(c, Method(pos="G1", wait=5, buffer_time=0))
    assert ok is False
    assert len([s for s in sent if s.startswith("@P")]) == 1
    assert "Restart MABIP" in c.last_error


def test_resend_that_works_continues_normally():
    c, sent = _conn(alive=True, goes_busy_on_attempt=2)
    ok = _run(c, Method(pos="G1", wait=5, buffer_time=0))
    assert ok is True                       # robot moved on the resend; READY = done
    assert c.last_error is None


def test_heartbeat_judges_liveness():
    c = AsyncAmuzaConnection(use_mock=False)
    assert c._sender_alive() is False        # never heard from the device
    c._parse_status_response("@q,01,0,000,0000,23.0,23.0,0")
    assert c._sender_alive() is True
    c.last_status_t -= 30
    assert c._sender_alive() is False


# ---- 3. STOP / RESUME drive the pump --------------------------------------

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication
import flow_control_tab as fct


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeLine:
    def __init__(self):
        self.calls = []

    def start_flow_single(self, machine_rate, direction=None, **kw):
        self.calls.append(("flow", round(machine_rate, 3)))

    def stop(self):
        self.calls.append(("stop",))


def _spin(ms):
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def _tab(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tab = fct.FlowControlTab(main_gui=None)
    tab.line = FakeLine()
    tab.f_rate.setText("120")
    tab.cfg.update({"direction": "withdraw", "n": 1, "ff_resume_ramp": 0.0})
    return tab


def test_stop_halts_pump_and_cancels_pending_boost(app, tmp_path, monkeypatch):
    tab = _tab(tmp_path, monkeypatch)
    tab.cfg.update({"ff_boost_add": 90.0, "ff_boost_s": 0.1})
    tab.on_move_command("G1")                 # boost 210 now, back to 120 in 0.1 s
    assert tab.line.calls == [("flow", 210.0)]
    tab.pause_for_stop("STOP pressed")
    _spin(250)                                # the boost-end timer must NOT restart it
    assert ("flow", 120.0) not in tab.line.calls
    assert tab.line.calls[-1] == ("stop",)


def test_resume_restarts_at_flow_rate(app, tmp_path, monkeypatch):
    tab = _tab(tmp_path, monkeypatch)
    tab.pause_for_stop("STOP pressed")
    tab.resume_after_stop()
    assert tab.line.calls[-1] == ("flow", 120.0)
    tab.resume_after_stop()                   # second RESUME is a no-op
    assert tab.line.calls.count(("flow", 120.0)) == 1


def test_resume_without_stop_does_nothing(app, tmp_path, monkeypatch):
    tab = _tab(tmp_path, monkeypatch)
    tab.resume_after_stop()
    assert tab.line.calls == []
