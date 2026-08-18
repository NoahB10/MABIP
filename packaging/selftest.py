"""Frozen-bundle self test, built as the `mabip-selftest` console binary.

Exercises the exact import chain that gui_async.py wraps in try/except at the
FlowControlTab import — the failure mode where a broken bundle still launches
but the Flow Control tab silently disappears. Exits 0 iff everything is
healthy. Runs headless (offscreen Qt), safe with no hardware attached.
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

failures = []


def check(name, fn):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL  {name}: {e!r}")


check("PyQt5 imports", lambda: __import__("PyQt5.QtWidgets"))


def _qapp():
    from PyQt5.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv)


check("QApplication constructs (offscreen)", _qapp)
check("matplotlib qt5agg backend", lambda: __import__("matplotlib.backends.backend_qt5agg"))
check("flow_control_tab imports (the silent-tab trap)", lambda: __import__("flow_control_tab"))
check("dual_syringe chain", lambda: __import__("dual_syringe"))


def _fgt():
    import Fluigent.SDK  # noqa: F401
    from Fluigent.SDK import low_level
    libdir = getattr(low_level, "_libdir", None)
    assert libdir and os.path.isdir(str(libdir)), f"fgt lib dir not on disk: {libdir!r}"


check("Fluigent SDK native lib resolves on disk", _fgt)
check("bluetooth (PyBluez C ext + libbluetooth)", lambda: __import__("bluetooth"))
check("serial + list_ports", lambda: __import__("serial.tools.list_ports"))
check("serial_asyncio", lambda: __import__("serial_asyncio"))
check("qasync", lambda: __import__("qasync"))
check("aiofiles", lambda: __import__("aiofiles"))
check("pandas", lambda: __import__("pandas"))


def _cfg():
    from config import FILES
    assert "MABIP_Data" in FILES.SENSOR_READINGS_FOLDER, FILES.SENSOR_READINGS_FOLDER


check("frozen data-dir redirect (~/MABIP_Data)", _cfg)


def _calib():
    p = os.path.join(sys._MEIPASS, "burst_calibration.json")
    assert os.path.isfile(p), p


check("burst_calibration.json shipped", _calib)

print("SELFTEST OK" if not failures else f"SELFTEST FAILED: {failures}")
sys.exit(1 if failures else 0)
