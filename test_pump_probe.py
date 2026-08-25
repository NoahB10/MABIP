"""Connecting the Chemyx pump must fail when nothing answers on the line.

The pump's FTDI adapter is USB-powered, so the serial port opens fine with
the pump switched off; "connected" has to mean the pump actually replied.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardware"))

import pytest  # noqa: E402

import chemyx_pump  # noqa: E402
from chemyx_pump import ChemyxPump, PumpError  # noqa: E402


class FakeSerial:
    """Stand-in for serial.Serial. `reply` is what the pump sends back after
    any write; None models a powered-off pump (port opens, nothing answers)."""

    def __init__(self, reply):
        self._reply = reply
        self._buf = b""
        self.is_open = True
        self.writes = []

    def write(self, data):
        self.writes.append(data)
        if self._reply is not None:
            self._buf += data.strip() + b"\r\n" + self._reply + b"\r\n>"
        return len(data)

    @property
    def in_waiting(self):
        return len(self._buf)

    def read(self, n):
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._buf = b""

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False


def _fake_pump_line(monkeypatch, reply):
    made = []

    def fake_serial(*args, **kwargs):
        s = FakeSerial(reply)
        made.append(s)
        return s

    monkeypatch.setattr(chemyx_pump.serial, "Serial", fake_serial)
    monkeypatch.setattr(chemyx_pump.time, "sleep", lambda s: None)
    return made


def test_open_fails_when_pump_is_silent(monkeypatch):
    made = _fake_pump_line(monkeypatch, reply=None)
    with pytest.raises(PumpError, match="powered on"):
        ChemyxPump(port="/dev/fake", timeout=0.05)
    assert made and not made[0].is_open, "port must be released, not left half-open"


def test_open_succeeds_when_pump_answers(monkeypatch):
    made = _fake_pump_line(monkeypatch, reply=b"10.0 0.001 50.0")
    pump = ChemyxPump(port="/dev/fake", timeout=0.05)
    assert pump.is_open
    assert made[0].writes[0].startswith(b"read limit parameter")


def test_open_fails_on_line_noise(monkeypatch):
    _fake_pump_line(monkeypatch, reply=b"\xff\xfe??")
    with pytest.raises(PumpError, match="Garbled"):
        ChemyxPump(port="/dev/fake", timeout=0.05)


def test_command_raises_instead_of_returning_empty(monkeypatch):
    made = _fake_pump_line(monkeypatch, reply=b"1 2 3")
    pump = ChemyxPump(port="/dev/fake", timeout=0.05)
    made[0]._reply = None          # pump switched off mid-session
    with pytest.raises(PumpError, match="No reply"):
        pump.command("elapsed time")
