# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for MABIP: onedir bundle with two entry points sharing one
# COLLECT — the GUI (MABIP) and a headless health check (mabip-selftest).
import os

REPO = "/home/pi/Documents/MABIP-testing"
APP = REPO
PKG = os.path.join(REPO, "packaging")

# The whole vendored hardware tree ships as DATA (real files, layout intact) —
# it must stay out of pathex/PYZ so Fluigent's pkg_resources loader can resolve
# shared/linux/arm64/libfgt_SDK.so as an on-disk file. The runtime hook puts
# it on sys.path.
hardware_tree = Tree(os.path.join(REPO, "hardware"), prefix="hardware",
                     excludes=["__pycache__", "*.pyc", "*.md"])

hiddenimports = [
    "flow_control_tab",            # gui_async imports it in a try/except that only logs
    "pkg_resources",               # used only by Fluigent code shipped as data
    "serial", "serial.tools.list_ports",
    "bluetooth", "serial_asyncio", "aiofiles",
    "qasync",
    "matplotlib.backends.backend_qt5agg",
]
excludes = [
    "tkinter", "_tkinter",
    "pytest", "IPython", "jupyter",
    "PyQt6", "PySide2", "PySide6",
    "PyQt5.QtWebEngineWidgets", "PyQt5.QtQml", "PyQt5.QtQuick",
    "PyQt5.QtMultimedia", "PyQt5.QtBluetooth", "PyQt5.QtSql",
    "PyQt5.QtTest", "PyQt5.QtDesigner", "PyQt5.QtLocation",
]

a = Analysis(
    [os.path.join(APP, "gui_async.py")],
    pathex=[APP],
    datas=[(os.path.join(APP, "burst_calibration.json"), ".")],
    hiddenimports=hiddenimports,
    excludes=excludes,
    runtime_hooks=[os.path.join(PKG, "pyi_rth_mabip.py")],
)
st = Analysis(
    [os.path.join(PKG, "selftest.py")],
    pathex=[APP],
    hiddenimports=hiddenimports,
    excludes=excludes,
    runtime_hooks=[os.path.join(PKG, "pyi_rth_mabip.py")],
)

pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [],
          exclude_binaries=True, name="MABIP", console=False, upx=False)

pyz_st = PYZ(st.pure)
exe_st = EXE(pyz_st, st.scripts, [],
             exclude_binaries=True, name="mabip-selftest", console=True, upx=False)

coll = COLLECT(exe, a.binaries, a.datas, hardware_tree,
               exe_st, st.binaries, st.datas,
               name="MABIP", upx=False)
