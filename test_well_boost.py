"""Well-entry boost: after a move, the pump runs at the boost rate for boost_s,
then drops to the Flow rate. Timers are real Qt timers, scaled down to ms."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
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
        self.calls.append(("flow", round(machine_rate, 3), direction))

    def stop(self):
        self.calls.append(("stop",))


def _spin(ms):
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def _tab(app, tmp_path, monkeypatch, **cfg):
    monkeypatch.setenv("HOME", str(tmp_path))      # keep the real settings file out
    tab = fct.FlowControlTab(main_gui=None)
    tab.line = FakeLine()
    tab.f_rate.setText("100")
    tab.cfg.update({"ff_pause_s": 0.01, "ff_resume_s": 0.03, "ff_resume_ramp": 0.0,
                    "direction": "withdraw", "n": 1})
    tab.cfg.update(cfg)
    return tab


def test_boost_then_base_rate_with_pause(app, tmp_path, monkeypatch):
    tab = _tab(app, tmp_path, monkeypatch, ff_boost_rate=160.0, ff_boost_s=0.05)
    tab._ff_enabled = True
    tab.on_move_command("A1")
    _spin(200)
    assert tab.line.calls == [("stop",),
                              ("flow", 160.0, "withdraw"),
                              ("flow", 100.0, "withdraw")]
    assert tab._boost_until == 0.0


def test_boost_works_without_pause_on_moves(app, tmp_path, monkeypatch):
    tab = _tab(app, tmp_path, monkeypatch, ff_boost_rate=160.0, ff_boost_s=0.05)
    tab._ff_enabled = False
    tab.on_move_command("A2")
    _spin(200)
    assert tab.line.calls == [("flow", 160.0, "withdraw"), ("flow", 100.0, "withdraw")]


def test_no_boost_is_the_old_behaviour(app, tmp_path, monkeypatch):
    tab = _tab(app, tmp_path, monkeypatch, ff_boost_rate=0.0, ff_boost_s=30.0)
    tab._ff_enabled = True
    tab.on_move_command("A3")
    _spin(150)
    assert tab.line.calls == [("stop",), ("flow", 100.0, "withdraw")]


def test_nothing_scheduled_when_both_off(app, tmp_path, monkeypatch):
    tab = _tab(app, tmp_path, monkeypatch)
    tab._ff_enabled = False
    tab.on_move_command("A4")
    _spin(100)
    assert tab.line.calls == []


def test_well_hold_does_not_cut_boost_short(app, tmp_path, monkeypatch):
    tab = _tab(app, tmp_path, monkeypatch, ff_boost_rate=160.0, ff_boost_s=0.3)
    tab._ff_enabled = False
    tab._exp_follow = False
    tab.on_move_command("A5")
    _spin(80)                                   # boost running
    tab.set_experiment_phase("well", 1)         # MABIP says the well started
    assert ("flow", 100.0, "withdraw") not in tab.line.calls
    _spin(350)
    assert tab.line.calls[-1] == ("flow", 100.0, "withdraw")


def test_experiment_file_sets_boost(app, tmp_path, monkeypatch):
    exp = fct.parse_experiment("flow_rate: 100\nboost_rate: 160\nboost_s: 30\n")
    assert exp["boost_rate"] == "160" and exp["boost_s"] == "30"


def test_apply_file_is_merged_once(app, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".mabip"; d.mkdir()
    (d / "flow_settings.apply.json").write_text(
        '{"cfg": {"ff_boost_rate": 160, "ff_boost_s": 30, "bogus": 1}, "fields": {"rate": "100"}}')
    tab = fct.FlowControlTab(main_gui=None)
    assert tab.cfg["ff_boost_rate"] == 160 and tab.cfg["ff_boost_s"] == 30
    assert "bogus" not in tab.cfg
    assert tab.f_rate.text() == "100"
    assert not (d / "flow_settings.apply.json").exists()
    assert (d / "flow_settings.json").exists()


def test_apply_file_persists_even_when_settings_are_current(app, tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("HOME", str(tmp_path))
    d = tmp_path / ".mabip"; d.mkdir()
    (d / "flow_settings.json").write_text(json.dumps(
        {"cfg_version": fct.FlowControlTab.CFG_VERSION, "cfg": {"ff_boost_rate": 0.0},
         "fields": {"rate": "90", "vol": "1", "target": "5"}}))
    (d / "flow_settings.apply.json").write_text(
        '{"cfg": {"ff_boost_rate": 160, "ff_boost_s": 30}, "fields": {"rate": "100"}}')
    fct.FlowControlTab(main_gui=None)
    saved = json.loads((d / "flow_settings.json").read_text())
    assert saved["cfg"]["ff_boost_rate"] == 160 and saved["cfg"]["ff_boost_s"] == 30
    assert saved["fields"]["rate"] == "100"
    assert not (d / "flow_settings.apply.json").exists()
