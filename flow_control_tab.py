"""
flow_control_tab — Fluigent + Chemyx flow control/plotting embedded in MABIP.

A self-contained PyQt5 QWidget (matplotlib plot, matching MABIP's stack) that
reuses the toolkit-agnostic `dual_syringe.DualSyringeLine` backend. Drop it into
the main window as a tab. It:

* connects the Fluigent flow sensor + Chemyx pump and plots the combined line
  flow live,
* drives the pump in machine terms (Start/apply flow, Run volume, Ramp),
* detects a CLOG (flow far below expected while pumping) and flags it,
* exposes `latest_flow` and `is_clogged` so the main window can record them in
  the well log.

Hardware libs (Fluigent SDK, chemyx_pump, fluigent_sensor, dual_syringe) are
vendored in the repo's hardware/ folder; on the Pi rig the live copies in
~/pumpcontrol-project take precedence when present.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from collections import deque

# --- make the pumpcontrol backend + Fluigent SDK importable from MABIP's venv.
# Vendored copies live in <repo>/hardware; the Pi's ~/pumpcontrol-project (the
# live dev copies) wins over them when it exists. Insert order = reverse
# priority: each insert(0) lands in front of the previous one.
_HW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardware")
_PC = "/home/rpi/pumpcontrol-project"
for p in (_HW, os.path.join(_PC, "fgt-SDK", "Python"), _PC):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox,
    QPushButton, QLineEdit, QLabel, QComboBox, QFrame, QCheckBox,
    QDialog, QDialogButtonBox, QButtonGroup, QMessageBox, QFileDialog, QPlainTextEdit,
    QStackedWidget, QScrollArea, QApplication)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ACCENT = "#2f81f7"; RED = "#e5484d"; GREEN = "#2ea043"; AMBER = "#d9a406"; MUTED = "#6b7785"


from experiment_parse import parse_experiment, num_list as _num_list


def _sensor_readings_dir():
    """Writable Sensor_Readings folder. Frozen (PyInstaller) builds must not
    write inside the bundle directory, so when config is unavailable fall back
    to ~/MABIP_Data instead of the module's own folder."""
    try:
        from config import FILES
        return FILES.SENSOR_READINGS_FOLDER
    except Exception:
        base = (os.path.join(os.path.expanduser("~"), "MABIP_Data")
                if getattr(sys, "frozen", False)
                else os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "Sensor_Readings")


class _SignedSensor:
    """Wraps the Fluigent sensor so the positive direction can be flipped.

    Which way the sensor calls "positive" depends on how it is plumbed into the
    line, so on one rig a push reads +70 and on another it reads -70. Correct it
    once, here at the source, rather than sprinkling sign handling through every
    consumer — the plot, the readout, the flow log, the burst triggers and
    `dual_syringe.read_flow()` all then agree that forward flow is positive.

    Doing it anywhere later would not be enough: `calibrate()` trims cal_factor by
    measured/target, so an inverted sensor silently drives cal_factor negative.

    The sign is read through a callable on every sample, not captured, so changing
    it in Definitions takes effect immediately on an already-connected sensor.
    Everything else (unit, air_bubble, close) passes straight through."""

    def __init__(self, inner, sign_fn):
        self._inner = inner
        self._sign_fn = sign_fn

    def read(self, *a, **kw):
        return self._sign_fn() * float(self._inner.read(*a, **kw))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class FlowDefinitionsDialog(QDialog):
    """Pop-up for the rarely-changed 'definitions' so they don't crowd the tab."""

    _FIELDS = [
        ("settle", "Settle (s)"), ("measure", "Measure (s)"), ("window", "Plot window (s)"),
        ("r_start", "Ramp start"), ("r_max", "Ramp max"), ("r_step", "Ramp step"),
        ("r_dwell", "Ramp dwell (s)"), ("r_tol", "Ramp tol (%)"),
        ("b_mult", "Burst multiplier (×)"),
        ("b_high_s", "Burst max/timed (s)"), ("b_stop_s", "Burst stop (s)"),
        ("b_backflow_s", "Burst backflow (s)"), ("b_backflow_rate", "Burst backflow rate (µL/min, 0=baseline)"),
        ("b_settle_base_s", "Burst settle base (s)"), ("b_settle_k", "Burst settle k (s·µL/min)"),
        ("b_settle_tol", "Burst steady tol (µL/min)"), ("b_settle_hold", "Burst steady hold (s)"),
        # flow_sign is not here: it is the "flip" checkbox below the columns,
        # since +1/-1 in a text box reads worse than a tick box.
        ("clog_frac", "Clog: flow-below frac (0-1)"), ("clog_seconds", "Clog: sustained (s)"),
        ("clog_arm_frac", "Clog: arm at frac of setpoint (0-1)"),
        ("clog_arm_timeout_s", "Clog: startup budget before warning (s)"),
        ("clear_fwd_bursts", "Clear: # forward bursts"), ("clear_wait_s", "Clear: wait before reverse (s)"),
        ("rev_rate", "Clear: reverse rate (µL/min, 0=baseline)"), ("rev_time_s", "Clear: reverse time (s)"),
        ("rev_attempts", "Clear: max reverse attempts"), ("clear_settle_s", "Clear: recheck settle (s)"),
        ("prime_pull_vol", "Prime pull vol (µL)"), ("prime_pull_rate", "Prime pull rate (µL/min)"),
        ("prime_push_vol", "Prime push vol (µL)"), ("prime_push_rate", "Prime push rate (µL/min)"),
        ("exp_buffer_rate", "Well-flow: buffer rate"), ("exp_well_rate", "Well-flow: well rate"),
        ("exp_approach_ramp", "Well-flow: ramp→well (s)"), ("exp_recover_ramp", "Well-flow: ramp→buffer (s)"),
        ("ff_pause_s", "Feed-fwd: pause after move (s)"), ("ff_resume_s", "Feed-fwd: resume after move (s)"),
        ("ff_resume_ramp", "Feed-fwd: resume ramp (s)"),
        ("exp_settle_max_var", "Max flow variation (µL/min)"), ("exp_settle_hold", "Settle hold (s)"),
        ("exp_settle_timeout", "Settle timeout (s)"), ("exp_settle_bump", "Settle raise step (%)"),
    ]

    # What an operator needs to set up and run the rig. Everything else in
    # _FIELDS is a bench-tuning constant — burst shapes, clog thresholds,
    # feed-forward timings, the ramp/prime numbers — and those stay behind
    # developer mode so this dialog is readable at a glance.
    _BASIC_FIELDS = ("window",)

    # Value boxes are sized to their contents — 3-6 characters for the tuning
    # numbers, a bit more for a port name. Left to stretch they take the whole
    # column width, which reads as a form full of empty boxes.
    _VALUE_W = 96
    _BASIC_VALUE_W = 130

    # Nobody reads a bore off a syringe barrel — they read "20 mL" off the
    # wrapper. The pump still needs the inner diameter, so pick the syringe and
    # convert here. Values are the BD Plastipak / Luer-Lok bores, the same table
    # syringe-pump firmware ships (NE-1000, Chemyx); other brands differ by a
    # few tenths, which is what Custom is for.
    _SYRINGES = [
        ("10 mL", 10.0, 14.50),
        ("20 mL", 20.0, 19.13),
        ("30 mL", 30.0, 21.70),
        ("50 / 60 mL", 60.0, 26.70),
    ]

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flow — Definitions / Settings")
        self.cfg = dict(cfg); self.w = {}
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 16, 16, 16); lay.setSpacing(8)

        self.w["port"] = QLineEdit(str(cfg["port"]))

        # Syringe size -> bore. A diameter already in cfg that matches no known
        # syringe is kept as a Custom entry rather than being rounded onto the
        # nearest standard one: it may have been measured against this rig.
        self._size = QComboBox()
        saved_d = float(cfg.get("diameter", 19.13))
        for label, ml, bore in self._SYRINGES:
            self._size.addItem(label, bore)
        match = next((i for i, (_l, _ml, bore) in enumerate(self._SYRINGES)
                      if abs(bore - saved_d) < 0.005), None)
        if match is None:
            self._size.addItem(f"Custom — {saved_d:g} mm", saved_d)
            match = self._size.count() - 1
        self._size.setCurrentIndex(match)
        self._size.setFixedWidth(self._BASIC_VALUE_W)

        # Show the bore the pump will actually be given. The conversion is the
        # whole point of the dropdown, so it should not be invisible.
        self._bore_lbl = QLabel()
        self._bore_lbl.setStyleSheet(f"color:{MUTED};")
        self._size.currentIndexChanged.connect(self._show_bore)
        self._show_bore()

        size_row = QWidget()
        srh = QHBoxLayout(size_row); srh.setContentsMargins(0, 0, 0, 0); srh.setSpacing(8)
        srh.addWidget(self._size); srh.addWidget(self._bore_lbl); srh.addStretch(1)

        seg = QWidget(); sh = QHBoxLayout(seg); sh.setContentsMargins(0, 0, 0, 0); sh.setSpacing(6)
        self._syr = {}
        self._syr_group = QButtonGroup(seg); self._syr_group.setExclusive(True)
        for n in (1, 2):
            b = QPushButton(str(n)); b.setCheckable(True); b.setFixedWidth(46)
            b.setStyleSheet("QPushButton:checked{background:#2f81f7;color:white;font-weight:700;}")
            b.setChecked(int(cfg["n"]) == n)
            self._syr_group.addButton(b); sh.addWidget(b); self._syr[n] = b
        sh.addStretch(1)

        # Basics: the hardware the pump is plumbed with, plus how much history
        # the plot shows. One column — there are few enough to read at a glance.
        form_basic = QFormLayout(); form_basic.setVerticalSpacing(7)
        form_basic.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        self.w["port"].setFixedWidth(self._BASIC_VALUE_W)
        form_basic.addRow("Pump port", self.w["port"])
        form_basic.addRow("Syringe size", size_row)
        form_basic.addRow("# syringes", seg)
        # Every field is built either way, so hiding one never drops its saved
        # value: values() still reads the whole of self.w.
        for key, label in self._FIELDS:
            self.w[key] = QLineEdit(str(cfg[key]))
            if key in self._BASIC_FIELDS:
                self.w[key].setFixedWidth(self._BASIC_VALUE_W)
                form_basic.addRow(label, self.w[key])
        lay.addLayout(form_basic)

        # Sensor sign: a tick box, not a +1/-1 field, so which way is "positive"
        # is a yes/no question rather than a number to get wrong.
        self.chk_flip = QCheckBox("Flip flow sensor direction — invert the +/- of every reading")
        self.chk_flip.setToolTip(
            "Off: the sensor already reads POSITIVE when the line flows forward.\n"
            "On: it is plumbed the other way round, so readings are negated at the "
            "source — plot, readout, flow log, burst triggers and calibration all "
            "then agree that forward flow is positive.\n"
            "Takes effect on the next sample, even while connected.")
        self.chk_flip.setChecked(float(cfg.get("flow_sign", 1.0)) < 0)
        self.chk_flip.setStyleSheet("QCheckBox{font-weight:600;padding-top:6px;}")
        lay.addWidget(self.chk_flip)

        # Dev mode: the one switch on this dialog that changes what the TAB shows,
        # so it sits apart from the numeric fields rather than lost among them.
        # Kept short on purpose: this label sets the dialog's width, and the
        # basics-only view is otherwise narrow.
        self.chk_dev = QCheckBox("Developer mode — bench controls + tuning settings")
        self.chk_dev.setToolTip("Off: only the controls and settings used for a normal run. "
                                "On: adds the experiment-authoring and bench-test buttons, "
                                "and reveals the tuning constants on this dialog.")
        self.chk_dev.setChecked(bool(cfg.get("dev_mode", False)))
        self.chk_dev.setStyleSheet("QCheckBox{font-weight:600;padding-top:6px;}")
        lay.addWidget(self.chk_dev)

        # Advanced block, revealed by the checkbox directly above it. Two
        # columns so the ~40 rows stay short enough to leave OK/Cancel onscreen.
        self._adv = QGroupBox("Advanced — bench tuning")
        adv_cols = QHBoxLayout(self._adv); adv_cols.setSpacing(24)
        form_l = QFormLayout(); form_r = QFormLayout()
        for _f in (form_l, form_r):
            _f.setVerticalSpacing(7)
            _f.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        adv_rows = [(label, self.w[key]) for key, label in self._FIELDS
                    if key not in self._BASIC_FIELDS]
        half = (len(adv_rows) + 1) // 2
        for i, (label, widget) in enumerate(adv_rows):
            widget.setFixedWidth(self._VALUE_W)
            # Without a floor, a dialog taller than the screen is resolved by
            # shrinking every row below the font's height, which clips the
            # digits' descenders — the squashed boxes.
            widget.setMinimumHeight(widget.sizeHint().height())
            (form_l if i < half else form_r).addRow(label, widget)
        adv_cols.addLayout(form_l); adv_cols.addLayout(form_r)

        # 40-odd rows will outgrow a short screen. Scroll them rather than let
        # the layout compress the rows to fit.
        self._adv_scroll = QScrollArea()
        self._adv_scroll.setWidgetResizable(True)
        self._adv_scroll.setFrameShape(QFrame.NoFrame)
        self._adv_scroll.setWidget(self._adv)
        lay.addWidget(self._adv_scroll, 1)

        # Never open taller than the screen; the scroll area absorbs the rest.
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setMaximumHeight(max(320, screen.availableGeometry().height() - 80))

        self.chk_dev.toggled.connect(self._toggle_advanced)
        self._toggle_advanced(self.chk_dev.isChecked())

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _toggle_advanced(self, on):
        """Show or hide the tuning block as developer mode is switched, without
        closing the dialog. Qt keeps the larger height once the block has been
        shown, so the minimum has to be relaxed before adjustSize() can shrink
        the window back down to the basics."""
        self._adv_scroll.setVisible(bool(on))
        self.setMinimumWidth(920 if on else 380)
        self.setMinimumHeight(0)

        if on:
            # A QScrollArea's own sizeHint is small, so adjustSize() alone would
            # open the dialog a couple of rows tall with everything behind a
            # scrollbar. Ask for the full block, capped by what the screen has.
            screen = QApplication.primaryScreen()
            avail = screen.availableGeometry().height() if screen else 900
            chrome = 280        # basics + the two checkboxes + OK/Cancel + title
            self._adv_scroll.setMinimumHeight(
                min(self._adv.sizeHint().height() + 8, max(240, avail - chrome)))

        self.resize(self.minimumWidth(), self.sizeHint().height())
        self.adjustSize()

    def _show_bore(self, *_):
        self._bore_lbl.setText(f"→ {float(self._size.currentData()):g} mm bore")

    def values(self):
        out = dict(self.cfg)
        for k, w in self.w.items():
            if k == "port":
                out[k] = w.text()
            else:
                try:
                    out[k] = float(w.text())
                except ValueError:
                    pass
        # The pump is still driven by bore; the dropdown only chooses it.
        out["diameter"] = float(self._size.currentData())
        out["syringe_ml"] = next((ml for _l, ml, bore in self._SYRINGES
                                  if abs(bore - out["diameter"]) < 0.005), None)
        out["flow_sign"] = -1.0 if self.chk_flip.isChecked() else 1.0
        out["dev_mode"] = bool(self.chk_dev.isChecked())
        out["n"] = 2 if self._syr[2].isChecked() else 1
        return out


class FlowControlTab(QWidget):
    """Flow control + live plot + clog detection, as a MABIP tab."""

    clog_changed = pyqtSignal(bool)
    clog_ui = pyqtSignal(bool, str)  # banner/button paint; queued when a worker raises it
    exp_running = pyqtSignal(bool)    # True while an experiment is running (toggles Stop button)
    status_msg = pyqtSignal(str)
    cal_msg = pyqtSignal(float)
    pump_result = pyqtSignal(bool)   # pump connect worker -> UI
    sensor_result = pyqtSignal(bool) # sensor connect worker -> UI
    prime_confirm = pyqtSignal()     # part-1 done -> prompt user to reconnect hose

    def __init__(self, parent=None, main_gui=None):
        super().__init__(parent)
        self.main_gui = main_gui         # for mutual-exclusive plotting
        self.line = None
        self.latest_flow = None          # read by the well-log writer
        self.flow_unit = "uL/min"
        self.is_clogged = False          # read by the well-log writer
        self._clog_flagged = False       # clog raised but NOT auto-cleared (user must act)
        self._steady = False             # pump in steady flow (clog check active)
        self._expected = 0.0
        self._clog_since = None
        self._flow_seen = False          # sensor has reached the setpoint since this command
        self._steady_since = 0.0         # when the current flow command was issued
        self._arm_warned = False         # one-shot "never got going" warning
        self._busy = False
        self._abort = False
        self._shutdown_done = False      # safe_shutdown latch; cleared on reconnect
        # burst / experiment-phase gating
        self._phase = "idle"             # 'idle' | 'buffer' | 'well'
        self._n_wells = 0
        self._auto_burst = False
        self._burst_done = False         # one auto-burst per buffer entry
        self._bursting = False
        self._burst_settled = True        # last burst confirmed back at baseline
        self._exp_follow = False         # pump follows well/buffer phases
        self._phase_flow_rate = None     # last commanded phase-flow line rate
        self._phase_flow_gen = 0         # cancels a running phase ramp
        self._ff_enabled = False         # feed-forward pause/resume across moves
        self._ff_gen = 0                 # cancels stale pause/resume timers
        self._last_air = False           # latest Fluigent air-bubble flag
        self._clearing = False           # True while the clog-clearing escalation runs
        self._abort_clear = False        # STOP / resume flag that unwinds the escalation
        self._last_clear_method = None   # "forward burst" | "reverse push" | None(failed)
        self._clear_baseline = 0.0       # line rate to restore after clearing
        self._flow_log_path = None       # own continuous flow-rate log file
        self._flow_log_last = 0.0
        self._seg_label = ""             # tags flow-log rows during an experiment
        self._exp_sweep = None           # loaded flow-sweep config
        self._exp_runs = None            # loaded well-plate multi-run list
        self._exp_calib = None           # loaded burst-calibration config
        self._calibrating = False        # True while the fast sampler owns the sensor
        self._t0 = time.monotonic()
        self._tick = 0
        self.t = deque(maxlen=36000)
        self.v = deque(maxlen=36000)

        self.cfg = {
            # diameter is the bore the pump is driven with; syringe_ml records
            # which syringe it came from, for the status line and the dialog.
            "port": "auto", "diameter": 19.13, "syringe_ml": 20.0,
            "n": 1, "direction": "withdraw",
            "settle": 30.0, "measure": 15.0, "window": 120.0,
            "r_start": 5.0, "r_max": 60.0, "r_step": 5.0, "r_dwell": 20.0, "r_tol": 5.0,
            # +1: the sensor already reads positive when flowing forward. -1: it is
            # plumbed the other way round, so flip it at the source.
            "flow_sign": 1.0,
            "clog_frac": 0.4, "clog_seconds": 6.0,
            # Arming: the watch judges nothing until the sensor has reached
            # clog_arm_frac of setpoint at least once since the flow command. Above
            # clog_frac so establishing flow and losing it cannot chatter.
            #
            # 0.5, not 0.8: air leaking into the line means this rig routinely
            # settles well short of the commanded rate, so an 80% bar was never
            # cleared on a healthy run and the "FLOW NEVER STARTED" banner fired
            # on rigs that were working as well as they ever do.
            "clog_arm_frac": 0.5, "clog_arm_timeout_s": 120.0,
            "metab_window": 12.0, "metab_thresh": 0.15,
            # clog-clearing escalation (Strategy 1 forward burst -> Strategy 2 reverse push).
            # Sequence: N forward bursts to completion, wait, then reverse-push attempts.
            "clear_fwd_bursts": 2.0,        # forward bursts to run before escalating
            "clear_wait_s": 60.0,           # wait after the forward bursts before reversing
            "auto_clear": False,            # OFF: a clog only warns; the user presses Burst now
            "clear_settle_s": 8.0,          # settle+measure window when checking if flow returned
            "rev_rate": 0.0,                # reverse-push line rate µL/min (0 => use baseline)
            "rev_time_s": 10.0,             # reverse-push duration per attempt (fixed time)
            "rev_attempts": 3.0,            # max reverse-push attempts before giving up
            "b_mult": 1.7, "b_high_s": 10.0, "b_stop_s": 8.0,
            # burst resettle: backflow pulse bleeds the overshoot, then confirm
            # the sensor is back at baseline before the well starts. The settle
            # budget is rate-scaled (base + k/rate): slow settling at low flow.
            "b_backflow_s": 5.0, "b_backflow_rate": 0.0,          # 0 => use baseline rate
            "b_settle_base_s": 8.0, "b_settle_k": 300.0,          # budget = base + k/line_rate
            "b_settle_overshoot_s": 10.0,                         # extra budget per (mult-1) overshoot
            "b_backflow_relief": 0.8,                             # fraction of overshoot a full backflow cancels
            "b_settle_tol": 5.0, "b_settle_hold": 4.0,            # steady = within ±tol for hold s
            "b_fit_margin_s": 3.0,                                # safety gap left in the buffer
            "prime_pull_vol": 600.0, "prime_pull_rate": 1000.0,   # 1 mL/min
            "prime_push_vol": 600.0, "prime_push_rate": 200.0,
            # flow-follows-wells (combined line rates + ramp seconds)
            "exp_buffer_rate": 20.0, "exp_well_rate": 50.0,
            "exp_approach_ramp": 5.0, "exp_recover_ramp": 5.0,
            # feed-forward pause/resume across a move (calibrated 2026-07-15)
            "ff_pause_s": 0.45, "ff_resume_s": 10.0, "ff_resume_ramp": 0.0,
            # closed-loop settle: reach the specified flow (sensor) before a run starts
            "exp_settle": False, "exp_settle_max_var": 5.0, "exp_settle_hold": 5.0,
            "exp_settle_timeout": 120.0, "exp_settle_bump": 5.0,
            # Dev mode reveals the bench/experiment-authoring controls (Run
            # volume, Ramp, and the load/run/stop experiment buttons). Day-to-day
            # plate runs are driven from the Sampling tab, so they are hidden by
            # default to keep this panel to the controls actually used.
            "dev_mode": False,
            # One burst per buffer entry during a run, on by default. Bursts are
            # phase-gated regardless of this switch — never mid-well.
            "auto_burst": False,
        }
        self._load_cfg()   # restore saved definitions/fields over the defaults

        self.status_msg.connect(lambda s: self.lbl_status.setText(s))
        self.status_msg.connect(self._console_log)
        self.clog_ui.connect(self._apply_clog_ui)
        self.cal_msg.connect(lambda c: self.lbl_status.setText(f"cal_factor = {c:.4f}"))
        self.pump_result.connect(self._after_pump)
        self.sensor_result.connect(self._after_sensor)
        self.prime_confirm.connect(self._prime_confirm)
        self._build()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start(150)

    # ------------------------------------------------------------------ UI
    def _build(self):
        outer = QHBoxLayout(self); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(12)

        # ---- left controls
        left = QVBoxLayout(); left.setSpacing(8)
        # Status pill ABOVE its button, both full width: side by side, a 268 px
        # column left the buttons too narrow and clipped "Disconnect sensor".
        conn1 = QVBoxLayout(); conn1.setSpacing(3)
        self.pill_pump = QLabel()
        self.btn_pump = QPushButton("Connect pump")
        self.btn_pump.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;}}")
        self.btn_pump.clicked.connect(self._connect_pump)
        self.btn_pump.setToolTip("Connect / disconnect the Chemyx syringe pump. Works with NO flow sensor.")
        conn1.addWidget(self.pill_pump); conn1.addWidget(self.btn_pump)
        left.addLayout(conn1)
        conn2 = QVBoxLayout(); conn2.setSpacing(3)
        self.pill_sensor = QLabel()
        self.btn_sensor = QPushButton("Connect sensor")
        self.btn_sensor.clicked.connect(self._connect_sensor)
        self.btn_sensor.setToolTip("Connect / disconnect the Fluigent flow sensor (OPTIONAL). "
                                   "Enables live plotting, flow logging, clog detection, and Ramp/Calibrate/Verify.")
        conn2.addWidget(self.pill_sensor); conn2.addWidget(self.btn_sensor)
        left.addLayout(conn2)
        self._set_pill(self.pill_pump, False, "pump")
        self._set_pill(self.pill_sensor, False, "sensor")

        params = QGroupBox("Run parameters")
        pf = QFormLayout(params); pf.setVerticalSpacing(6)
        # Label above field, not beside it. Side by side, label + entry could not
        # both fit the 268 px column and Qt resolved it by squeezing the label
        # column to zero width, leaving unlabelled boxes.
        pf.setRowWrapPolicy(QFormLayout.WrapAllRows)
        sf = getattr(self, "_saved_fields", {})
        self.f_rate = QLineEdit(sf.get("rate", "5"))
        self.f_vol = QLineEdit(sf.get("vol", "1000"))
        self.f_target = QLineEdit(sf.get("target", "60"))
        self.f_rate.setToolTip("Desired flow in the COMBINED line. The pump runs each of N syringes at rate ÷ N.")
        self.f_vol.setToolTip("Volume the 'Run volume' button dispenses (one shot, then stops). "
                              "Total into the line — each of N syringes delivers volume ÷ N. "
                              "Not used by Start (continuous) or Prime.")
        pf.addRow("Flow rate (µL/min, line)", self.f_rate)
        pf.addRow("Run volume (µL, line)", self.f_vol)
        pf.addRow("Target sensor (µL/min)", self.f_target)
        # The Run-volume field only feeds the dev-only 'Run volume' button, so it
        # hides with it (label included).
        self._vol_label = pf.labelForField(self.f_vol)
        self.f_target.textChanged.connect(self._sync_target)
        for _f in (self.f_rate, self.f_vol, self.f_target):
            _f.editingFinished.connect(self._save_cfg)   # remember the run params
        left.addWidget(params)

        # push/pull direction — a big two-state button on the main panel, so the
        # current direction is readable at a glance and one click flips it
        self.chk_pull = QPushButton()
        self.chk_pull.setCheckable(True)
        self.chk_pull.setChecked(self.cfg["direction"] == "withdraw")
        # Two lines of text: below ~52 px the layout squeezes it and clips both.
        self.chk_pull.setMinimumHeight(52)
        self.chk_pull.setToolTip("Flow direction for Start / Run volume / Burst / Ramp. "
                                 "Click to switch between PUSH (infuse) and PULL (withdraw).")
        self.chk_pull.toggled.connect(self._on_pull_toggled)
        self._style_direction_btn(self.chk_pull.isChecked())
        left.addWidget(self.chk_pull)

        grid = QGridLayout(); grid.setSpacing(6); self.btn = {}
        self._btn_grid = grid          # _apply_dev_mode re-spans Burst on this
        def mk(key, text, r, c, slot, style=""):
            b = QPushButton(text); b.clicked.connect(slot)
            if style:
                b.setStyleSheet(style)
            grid.addWidget(b, r, c); self.btn[key] = b
        acc = f"QPushButton{{background:{ACCENT};color:white;}}"
        dng = f"QPushButton{{background:{RED};color:white;font-weight:700;}}"
        mk("startflow", "Start", 0, 0, self._start_flow, acc)
        mk("stop", "STOP", 0, 1, self._stop, dng)
        mk("run", "Run volume", 1, 0, self._run)
        mk("ramp", "Ramp ▶", 1, 1, self._ramp)
        mk("burst", "Burst now", 2, 0, self._burst)
        mk("prime", "Prime", 2, 1, self._prime)
        _tips = {
            "startflow": "Start / update CONTINUOUS flow at the Flow rate. Change the rate and press again to apply it live.",
            "stop": "Immediately STOP the pump and abort any burst / ramp / prime.",
            "run": "Deliver the Run volume ONCE at the Flow rate, then stop.",
            "ramp": "Step the rate up (start→max by step, from Definitions) while watching the sensor; stops when it reaches Target.",
            "burst": "Clog-clear pulse: boost to (multiplier × current rate) until flow recovers to the original rate, then resume. Buffer only.",
            "prime": "2-step wet prime: PULL liquid in with empty syringes, wait for you to reconnect the hose, then PUSH it out to wet the line.",
        }
        for _k, _t in _tips.items():
            self.btn[_k].setToolTip(_t)
        left.addLayout(grid)
        self.chk_ff = QCheckBox("Pause pump on moves")
        self.chk_ff.setToolTip("On each well move: PAUSE the pump ~0.45 s after the move command "
                               "(tip lifts out of liquid) and RESUME ~10 s later (tip back in liquid) — "
                               "kills the air surge. Runs at the Flow rate; calibrated, needs NO sensor.")
        self.chk_ff.toggled.connect(lambda v: setattr(self, "_ff_enabled", bool(v)))
        left.addWidget(self.chk_ff)
        self.chk_settle = QCheckBox("Settle to target rate")
        self.chk_settle.setToolTip("Before an experiment run starts, treat the run's flow rate as a TARGET "
                                   "at the sensor and wait for the sensor to actually read it — if the "
                                   "measured flow is low, raise the pump until it holds the target, then "
                                   "start the wells. The pump rate that got there is written back into "
                                   "Flow rate, since it is usually not the target itself. "
                                   "Needs the flow sensor connected. (Definitions: tolerance/hold/timeout.)")
        self.chk_settle.setChecked(bool(self.cfg.get("exp_settle", False)))
        self.chk_settle.toggled.connect(lambda v: self.cfg.__setitem__("exp_settle", bool(v)))
        left.addWidget(self.chk_settle)
        self.chk_follow = QCheckBox("Flow follows wells")
        self.chk_follow.setToolTip("During a well-plate run, drive the pump to the buffer-rate and "
                                   "well-rate (with ramps) from Definitions, instead of a single Start rate. "
                                   "Ramps happen on each buffer→well and well→buffer transition.")
        self.chk_follow.toggled.connect(lambda v: setattr(self, "_exp_follow", bool(v)))
        left.addWidget(self.chk_follow)
        self.chk_auto = QCheckBox("Auto-burst in buffer")
        self.chk_auto.setToolTip("Automatically fire one Burst each time the run enters the buffer "
                                 "(needs ≥1 well). Bursts NEVER fire mid-well — the buffer is the "
                                 "only window where one can't spoil a reading.")
        self.chk_auto.setChecked(bool(self.cfg.get("auto_burst", False)))
        self._auto_burst = self.chk_auto.isChecked()
        self.chk_auto.toggled.connect(self._on_auto_burst_toggled)
        left.addWidget(self.chk_auto)
        self.chk_autoclear = QCheckBox("Auto-clear on clog")
        self.chk_autoclear.setToolTip("OFF (default): a clog only RAISES A WARNING — the banner turns red and "
                                      "'Burst now' lights up, and you decide whether to clear. ON: the pump runs "
                                      "the forward-burst → reverse-push escalation by itself. Leave it off unless "
                                      "you trust the flow sensor: a dead or unprimed sensor reads ~0 and looks "
                                      "exactly like a clog.")
        self.chk_autoclear.setChecked(bool(self.cfg.get("auto_clear", False)))
        self.chk_autoclear.toggled.connect(lambda v: self.cfg.__setitem__("auto_clear", bool(v)))
        left.addWidget(self.chk_autoclear)
        self.lbl_phase = QLabel("phase: idle")
        self.lbl_phase.setStyleSheet(f"color:{MUTED};")
        left.addWidget(self.lbl_phase)

        self.btn_savelog = QPushButton("💾  Save flow log…")
        self.btn_savelog.setToolTip("Save a copy of this run's flow-rate log (elapsed, flow, air) "
                                    "to a file you choose. Also auto-logs to Sensor_Readings/, and "
                                    "appends to the metabolite file when the sensor is recording.")
        self.btn_savelog.clicked.connect(self._save_log)
        left.addWidget(self.btn_savelog)

        self.btn_defs = QPushButton("⚙  Settings…")
        self.btn_defs.setToolTip("Rarely-changed settings: port, syringe Ø, # syringes, direction, "
                                 "ramp / clog / burst / prime parameters, and the developer-mode "
                                 "switch. Saved automatically.")
        self.btn_defs.clicked.connect(self._open_defs)
        left.addWidget(self.btn_defs)
        self.btn_loadexp = QPushButton("📂  Load experiment…")
        self.btn_loadexp.setToolTip("Load a text experiment file (wells, sample/buffer times, flow rate, "
                                    "direction, pause-on-move, resume ramp). Applies the pump settings here "
                                    "and sets the wells + timing on the Sampling tab — then press Start Sampling.")
        self.btn_loadexp.clicked.connect(self._load_experiment)
        left.addWidget(self.btn_loadexp)
        self.btn_runexp = QPushButton("▶  Run experiment")
        self.btn_runexp.setToolTip("Run the loaded experiment automatically, saving all segments to one "
                                   "flow log tagged per rate/run so you can analyze them together.")
        self.btn_runexp.setEnabled(False)
        self.btn_runexp.clicked.connect(self._run_experiment)
        left.addWidget(self.btn_runexp)
        self.btn_stopexp = QPushButton("⏹  Stop experiment")
        self.btn_stopexp.setToolTip("Halt the running experiment: finishes the current well, then stops "
                                    "(no further runs/wells). Also stops the pump and aborts any burst.")
        self.btn_stopexp.setStyleSheet("QPushButton { background:#e53935; color:white; font-weight:700; } "
                                       "QPushButton:hover{ background:#d32f2f; }")
        self.btn_stopexp.setEnabled(False)
        self.btn_stopexp.clicked.connect(self._stop_experiment)
        left.addWidget(self.btn_stopexp)
        # enable Stop only while an experiment runs; disable Run to avoid re-entry
        self.exp_running.connect(self._on_exp_running)
        left.addStretch(1)

        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(268)
        outer.addWidget(lw)

        # ---- right: readout + clog banner + plot
        right = QVBoxLayout(); right.setSpacing(6)
        rr = QHBoxLayout()
        self.lbl_flow = QLabel("—"); self.lbl_flow.setStyleSheet(f"font-size:34px;font-weight:700;color:{ACCENT};")
        u = QLabel("µL/min (combined)"); u.setStyleSheet(f"color:{MUTED};")
        rr.addWidget(self.lbl_flow); rr.addWidget(u, alignment=Qt.AlignBottom); rr.addStretch(1)
        right.addLayout(rr)

        # Clog banner — hidden until the watcher flags one. This is the whole
        # point of not auto-clearing: the operator sees the call and decides.
        self.lbl_clog = QLabel("")
        self.lbl_clog.setWordWrap(True)
        self.lbl_clog.setVisible(False)
        right.addWidget(self.lbl_clog)

        self.fig = Figure(figsize=(5, 3)); self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("time (s)"); self.ax.set_ylabel("flow (µL/min)")
        self.ax.grid(True, alpha=0.25)
        (self.trace,) = self.ax.plot([], [], color=ACCENT, lw=1.6)
        self.hline = self.ax.axhline(0.0, color=RED, ls="--", lw=1.0)

        # While the metabolite sensor runs, the flow trace is drawn on the
        # Plotting tab's right axis and this canvas is frozen (see _poll). A dead
        # plot that still looks live is worse than no plot, so cover it with a
        # panel that says where the trace went and keeps the live number visible.
        cover = QWidget()
        cover.setStyleSheet("background:#11151c;border:1px solid #2a3340;border-radius:6px;")
        cl = QVBoxLayout(cover); cl.setContentsMargins(24, 24, 24, 24); cl.setSpacing(4)
        cl.addStretch(1)
        cap = QLabel("CURRENT FLOW"); cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet(f"color:{MUTED};font-size:12px;font-weight:700;"
                          "letter-spacing:2px;border:none;")
        self.lbl_cover_flow = QLabel("—"); self.lbl_cover_flow.setAlignment(Qt.AlignCenter)
        self.lbl_cover_flow.setStyleSheet(
            f"color:{ACCENT};font-size:56px;font-weight:700;border:none;")
        cunit = QLabel("µL/min (combined)"); cunit.setAlignment(Qt.AlignCenter)
        cunit.setStyleSheet(f"color:{MUTED};font-size:13px;border:none;")
        msg = QLabel("Live flow plotting is on the <b>Plotting</b> tab —\n"
                     "the trace is drawn there on the right-hand axis "
                     "while the metabolite sensor is recording.")
        msg.setAlignment(Qt.AlignCenter); msg.setWordWrap(True)
        msg.setStyleSheet("color:#cfe3ff;font-size:14px;padding-top:18px;border:none;")
        for wdg in (cap, self.lbl_cover_flow, cunit, msg):
            cl.addWidget(wdg)
        cl.addStretch(1)

        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.canvas)   # 0 = live plot
        self.plot_stack.addWidget(cover)         # 1 = "see the Plotting tab"
        right.addWidget(self.plot_stack, 1)

        self.lbl_status = QLabel("Not connected. Press Connect to plot the Fluigent sensor.")
        self.lbl_status.setStyleSheet(f"color:{MUTED};")
        right.addWidget(self.lbl_status)

        # scrolling console: every status line is timestamped and kept here so a
        # running experiment can be tracked (the single label only shows the latest)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(500)
        self.console.setFixedHeight(150)
        self.console.setStyleSheet(
            "QPlainTextEdit{background:#11151c;color:#cfe3ff;border:1px solid #2a3340;"
            # Name real families first: Cocoa has no "monospace" alias, and Qt
            # walks the whole font list to discover that on every launch.
            "font-family:Menlo,'DejaVu Sans Mono',Consolas,monospace;"
            "font-size:11px;}")
        self.console.setPlaceholderText("Activity log — connect, load an experiment, press Run…")
        right.addWidget(self.console)
        outer.addLayout(right, 1)

        self._sync_target(); self._refresh_actions(); self._apply_dev_mode()

        # Say out loud when a saved setting was carried onto a new default —
        # a threshold that changes itself silently is worse than the old value.
        for note in getattr(self, "_migration_notes", []):
            self.status_msg.emit(f"[settings] updated: {note}")

    # ------------------------------------------------------------- helpers
    def _set_pill(self, lbl, ok, name):
        lbl.setText(f"● {name} on" if ok else f"● {name} off")
        lbl.setStyleSheet(f"color:{GREEN if ok else MUTED};font-weight:600;")

    def _sync_target(self):
        try:
            self.hline.set_ydata([float(self.f_target.text())] * 2)
        except ValueError:
            pass

    def _num(self, w, default=None):
        try:
            return float(w.text())
        except (ValueError, AttributeError):
            if default is None:
                raise
            return default

    def _style_direction_btn(self, pulling: bool):
        """Label + colour the direction button for its current state."""
        # Keep both lines short enough to fit the 268 px control column — the
        # longer "Direction: ▲ PULL (withdraw)" wording clipped at both ends.
        if pulling:
            self.chk_pull.setText("▲  PULL  (withdraw)\nclick to switch to PUSH")
            bg, hover = "#6a1b9a", "#7b27ab"
        else:
            self.chk_pull.setText("▼  PUSH  (infuse)\nclick to switch to PULL")
            bg, hover = ACCENT, "#1565c0"
        self.chk_pull.setStyleSheet(
            f"QPushButton{{background:{bg};color:white;font-weight:700;text-align:center;}}"
            f"QPushButton:hover{{background:{hover};}}")

    def _on_auto_burst_toggled(self, on):
        self._auto_burst = bool(on)
        self.cfg["auto_burst"] = bool(on)
        self._save_cfg()

    def _on_pull_toggled(self, checked):
        self.cfg["direction"] = "withdraw" if checked else "infuse"
        self._style_direction_btn(bool(checked))
        if self.line is not None:
            self.line.direction = self.cfg["direction"]
        self._save_cfg()
        self.status_msg.emit(f"Flow direction: {'PULL (withdraw)' if checked else 'PUSH (infuse)'}")

    # ------------------------------------------------------- persist settings
    # Bump when a default changes in a way that must reach machines whose saved
    # settings already pin the old value (see _migrate_cfg).
    CFG_VERSION = 3

    def _load_cfg(self):
        """Restore saved definitions + run params over the defaults, so nothing
        resets between launches."""
        import json, os
        self._cfg_path = os.path.expanduser("~/.mabip/flow_settings.json")
        self._saved_fields = {}
        saved_version = 0
        try:
            with open(self._cfg_path) as f:
                saved = json.load(f)
            for k, v in (saved.get("cfg") or {}).items():
                if k in self.cfg:
                    self.cfg[k] = v
            self._saved_fields = saved.get("fields") or {}
            saved_version = int(saved.get("cfg_version") or 0)
        except Exception:
            pass
        self._migrate_cfg(saved_version)

    def _migrate_cfg(self, saved_version: int):
        """Carry changed defaults onto settings files that pin the old value.

        _load_cfg lays the saved file OVER the defaults, so simply editing a
        default here never reaches a machine that has already saved settings.
        Each step only rewrites a value that still equals the superseded
        default, so a deliberately chosen number is left alone.
        """
        if saved_version >= self.CFG_VERSION:
            return
        notes = []
        if saved_version < 2 and abs(float(self.cfg.get("clog_arm_frac", 0.5)) - 0.8) < 1e-9:
            self.cfg["clog_arm_frac"] = 0.5
            notes.append("clog arm threshold 80% -> 50% of setpoint "
                         "(air leaks keep this rig below 80%)")
        if saved_version < 3:
            # Both of these used to arrive pre-ticked, so a run could settle or
            # fire bursts on its own before anyone asked for it. They start off
            # now. Unlike a number, a bool cannot say whether True was chosen or
            # merely inherited, so this does clear a deliberate tick once — the
            # note below is there to say so rather than let it look like a bug.
            for key, label in (("exp_settle", "Settle to target rate"),
                               ("auto_burst", "Auto-burst in buffer")):
                if bool(self.cfg.get(key)):
                    self.cfg[key] = False
                    notes.append(f"'{label}' now starts unticked — re-tick it if you want it")
        self._migration_notes = notes
        # Stamp the version even with nothing to change, so this runs once.
        self._save_cfg()

    def _save_cfg(self):
        """Persist the current definitions + run params (called on every change).

        Also runs before _build(), from the migration in _load_cfg, so the run-param
        widgets may not exist yet — fall back to the values just read off disk
        rather than losing them.
        """
        import json, os
        saved = getattr(self, "_saved_fields", {}) or {}

        def _field(widget_name, key):
            w = getattr(self, widget_name, None)
            return w.text() if w is not None else saved.get(key, "")

        try:
            os.makedirs(os.path.dirname(self._cfg_path), exist_ok=True)
            with open(self._cfg_path, "w") as f:
                json.dump({"cfg_version": self.CFG_VERSION,
                           "cfg": self.cfg,
                           "fields": {"rate": _field("f_rate", "rate"),
                                      "vol": _field("f_vol", "vol"),
                                      "target": _field("f_target", "target")}}, f, indent=2)
        except Exception:
            pass

    def _set_n(self, n):
        self.cfg["n"] = n
        for k, b in getattr(self, "_syr", {}).items():
            b.setChecked(k == n)
        if self.line is not None:
            self.line.n_syringes = n

    def _apply_defs(self, *_):
        # Definitions live in the pop-up dialog, which writes self.cfg directly.
        # Just make sure the live connection matches cfg before an action runs.
        if self.line is not None:
            self.line.direction = self.cfg.get("direction", "infuse")
            self.line.n_syringes = int(self.cfg.get("n", 2))

    def _open_defs(self):
        dlg = FlowDefinitionsDialog(self.cfg, self)
        if dlg.exec_():
            new = dlg.values()
            diam_changed = new.get("diameter") != self.cfg.get("diameter")
            sign_before = self._flow_sign()
            self.cfg = new
            if self._flow_sign() != sign_before:
                # Live from the next sample, but the trace still holds old-sign
                # points — drop them so the plot isn't half-flipped.
                self.t.clear(); self.v.clear()
                self.status_msg.emit(
                    f"Flow sensor sign set to {self._flow_sign():+.0f} — plot cleared. "
                    "Re-run Calibrate if cal_factor was trimmed against the old sign.")
            if self.line is not None:
                self.line.n_syringes = int(new["n"])
                if diam_changed:
                    # The setter marks the bore stale; the pump is told on the
                    # next run command (it ignores settings sent mid-run).
                    self.line.diameter_mm = float(new["diameter"])
            dev = self._apply_dev_mode()
            self._save_cfg()
            # A running pump keeps the OLD bore until it is re-commanded, so say
            # so rather than let "saved" read as "the pump is using this now".
            pending = (" — pump keeps the old Ø until you press Apply/Start"
                       if diam_changed and self.line is not None and self._steady else "")
            ml = new.get("syringe_ml")
            syr = f"{ml:g} mL (Ø{new['diameter']:g} mm)" if ml else f"Ø{new['diameter']:g} mm"
            self.status_msg.emit(f"Definitions saved — {int(new['n'])} × {syr}, "
                                 f"{new['direction']}, "
                                 f"dev mode {'ON' if dev else 'off'}.{pending}")

    def _load_experiment(self):
        """Load a text experiment file: apply the pump/flow settings here and set
        the wells + sample/buffer times on the Sampling tab."""
        start_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "experiments"))
        if not os.path.isdir(start_dir):
            start_dir = ""      # fall back to the last/default location
        path, _ = QFileDialog.getOpenFileName(
            self, "Load experiment file", start_dir, "Experiment (*.txt *.exp *.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path) as f:
                exp = parse_experiment(f.read())
        except Exception as e:
            self.status_msg.emit(f"Load failed: {e}"); return

        # ---- burst timing calibration (pure pump + sensor, no wells) ----
        if exp.get("experiment", "").lower() in ("burst_calibration", "burst_calib", "calib"):
            def _f(k, d):
                try:
                    return float(exp.get(k, d))
                except (ValueError, TypeError):
                    return d
            try:
                highs = _num_list(exp.get("high_mults", "4,5,6,7,8"))
                revs = _num_list(exp.get("rev_mults", exp.get("rev_rates", "4,5,6,7,8")))
            except ValueError:
                self.status_msg.emit("Calibration file: bad high_mults/rev_mults."); return
            if "direction" in exp:
                d = exp["direction"].lower()
                self.cfg["direction"] = "withdraw" if d.startswith(("w", "pull")) else "infuse"
                self.chk_pull.setChecked(self.cfg["direction"] == "withdraw")
            self._exp_calib = dict(
                baseline=_f("baseline", 40.0), ceiling=_f("ceiling", 115.0),
                high_mults=highs, rev_mults=revs, repeats=int(_f("repeats", 3)),
                settle_s=_f("settle_s", 15.0), rise_timeout_s=_f("rise_timeout_s", 20.0),
                max_rate=_f("max_rate", 400.0))
            self._exp_sweep = None; self._exp_runs = None
            self._refresh_runbtn()
            msg = (f"Loaded burst calibration — base {self._exp_calib['baseline']:g} → "
                   f"ceiling {self._exp_calib['ceiling']:g} µL/min · up ×{highs} · rev ×{revs} · "
                   f"{self._exp_calib['repeats']} reps. Connect pump+SENSOR, then ▶ Run experiment.")
            self.status_msg.emit(msg)
            QMessageBox.information(self, "Burst calibration loaded", msg)
            return

        # ---- flow-rate decay sweep (pure pump, no wells) ----
        if exp.get("experiment", "").lower() in ("flow_sweep", "sweep", "decay"):
            def _flist(s):
                out = []
                for tok in s.replace(";", ",").split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    if ".." in tok:                      # range: 10..50..10  = start..stop..step
                        a = [float(x) for x in tok.split("..")]
                        start, stop = a[0], a[1]
                        step = a[2] if len(a) > 2 else 10.0
                        v = start
                        while (v <= stop + 1e-9) if step > 0 else (v >= stop - 1e-9):
                            out.append(round(v, 6)); v += step
                    else:
                        out.append(float(tok))
                return out
            try:
                rates = _flist(exp.get("rates", ""))
            except ValueError:
                rates = []
            if not rates:
                self.status_msg.emit("Sweep file has no valid 'rates:'."); return
            def _f(k, d):
                try:
                    return float(exp.get(k, d))
                except (ValueError, TypeError):
                    return d
            if "direction" in exp:
                d = exp["direction"].lower()
                self.cfg["direction"] = "withdraw" if d.startswith(("w", "pull")) else "infuse"
                self.chk_pull.setChecked(self.cfg["direction"] == "withdraw")
            self._exp_sweep = dict(rates=rates, hold_s=_f("hold_s", 30.0),
                                   decay_s=_f("decay_s", 60.0), ramp_s=_f("ramp_s", 0.0))
            self._exp_runs = None
            self._refresh_runbtn()
            msg = (f"Loaded sweep — {len(rates)} rates ({', '.join(f'{r:g}' for r in rates)}) · "
                   f"hold {self._exp_sweep['hold_s']:g}s · decay {self._exp_sweep['decay_s']:g}s. "
                   f"Press ▶ Run experiment.")
            self.status_msg.emit(msg)
            QMessageBox.information(self, "Flow sweep loaded", msg)
            return

        # ---- well-plate MULTI-RUN (A–D sweep, run back-to-back) ----
        if exp.get("_runs"):
            runs = exp["_runs"]
            self._exp_runs = runs
            self._exp_sweep = None

            def _wlist(s):
                return [w.strip().upper() for w in str(s or "").replace(";", ",").split(",") if w.strip()]
            wells = _wlist(exp.get("wells", ""))
            # Runs may name their own wells (distinct per run). For the plate preview
            # highlight the union of all wells the experiment will visit.
            per_run = [_wlist(r.get("wells", "")) for r in runs]
            all_wells = wells[:] if wells else []
            for pw in per_run:
                for w in pw:
                    if w not in all_wells:
                        all_wells.append(w)
            distinct_runs = any(pw for pw in per_run)

            def _int(x):
                try:
                    return int(float(x))
                except (ValueError, TypeError):
                    return 0
            sampling = _int(exp.get("sample_time", exp.get("sampling_time", 0)))
            buffer_t = _int(exp.get("buffer_time", 0))
            if all_wells and self.main_gui is not None and hasattr(self.main_gui, "load_experiment_setup"):
                try:
                    self.main_gui.load_experiment_setup(all_wells, sampling, buffer_t)
                except Exception as e:
                    self.status_msg.emit(f"(wells not set: {e})")
            self.apply_run(dict(runs[0], name=""), start=False)   # preview 1st run's settings
            self._refresh_runbtn()
            names = ", ".join(r.get("name", "?") for r in runs)
            msg = (f"Loaded {len(runs)} runs ({names}) · {len(all_wells)} wells · "
                   f"sample {sampling}s buffer {buffer_t}s. Connect pump + robot, then ▶ Run experiment.")
            self.status_msg.emit(msg)
            wells_note = ("Each run uses its OWN wells (distinct per run); "
                          if distinct_runs else
                          "Each run repeats the same wells with its own flow settings; ")
            QMessageBox.information(self, "Experiment loaded",
                                   msg + "\n\n" + wells_note + "all runs are tagged per-run in the flow "
                                   "log so you can analyze them together. Keep 'Flow follows wells' OFF "
                                   "for a constant rate per run.")
            return

        applied = []
        if "flow_rate" in exp:
            self.f_rate.setText(exp["flow_rate"]); applied.append(f"flow {exp['flow_rate']} µL/min")
        if "direction" in exp:
            d = exp["direction"].lower()
            self.cfg["direction"] = "withdraw" if d.startswith(("w", "pull")) else "infuse"
            self.chk_pull.setChecked(self.cfg["direction"] == "withdraw")
            applied.append(self.cfg["direction"])
        for fkey, ckey in (("pause_after", "ff_pause_s"), ("resume_after", "ff_resume_s"),
                           ("resume_ramp", "ff_resume_ramp")):
            if fkey in exp:
                try:
                    self.cfg[ckey] = float(exp[fkey])
                except ValueError:
                    pass
        if "pause_on_move" in exp:
            on = exp["pause_on_move"].lower() in ("yes", "true", "1", "on", "y")
            self._ff_enabled = on; self.chk_ff.setChecked(on)
            applied.append(f"pause-on-move {'ON' if on else 'off'}")
        if "resume_ramp" in exp:
            applied.append(f"resume ramp {exp['resume_ramp']}s")
        # AMUZA wells + timing -> Sampling tab
        wells = [w.strip().upper() for w in exp.get("wells", "").replace(";", ",").split(",") if w.strip()]

        def _int(x):
            try:
                return int(float(x))
            except (ValueError, TypeError):
                return 0
        sampling = _int(exp.get("sample_time", exp.get("sampling_time", 0)))
        buffer_t = _int(exp.get("buffer_time", 0))
        if wells and self.main_gui is not None and hasattr(self.main_gui, "load_experiment_setup"):
            try:
                self.main_gui.load_experiment_setup(wells, sampling, buffer_t)
                applied.append(f"{len(wells)} wells, sample {sampling}s, buffer {buffer_t}s")
            except Exception as e:
                applied.append(f"(wells not set: {e})")
        # single well-plate file → make it a 1-run experiment so ▶ Run experiment works too
        if wells:
            self._exp_runs = [dict(exp, name=exp.get("name", "run"))]
            self._exp_sweep = None
            self._refresh_runbtn()
        self._save_cfg()
        msg = "Loaded — " + " · ".join(applied) if applied else "Nothing recognized in the file."
        self.status_msg.emit(msg)
        QMessageBox.information(self, "Experiment loaded",
                               msg + "\n\nPump/flow settings are applied here. Press ▶ Run experiment to run "
                               "it automatically, or Start Sampling on the Sampling tab.")

    def _dev_widgets(self):
        """Controls shown only in developer mode.

        Everything here is for authoring/bench work rather than running a plate:
        a normal run is Connect -> direction -> Start -> (Burst if it clogs), and
        the plate itself is driven from the Sampling tab.
        """
        return [w for w in (self.btn.get("run"), self.btn.get("ramp"),
                            self.btn.get("prime"),
                            self.f_vol, getattr(self, "_vol_label", None),
                            self.btn_loadexp, self.btn_runexp, self.btn_stopexp)
                if w is not None]

    def _apply_dev_mode(self):
        """Show or hide the developer-only controls to match cfg['dev_mode']."""
        dev = bool(self.cfg.get("dev_mode", False))
        for w in self._dev_widgets():
            w.setVisible(dev)

        # Burst shares its row with Prime. With Prime hidden it would sit at
        # half width beside an empty cell, so give it the whole row back.
        grid = getattr(self, "_btn_grid", None)
        burst = self.btn.get("burst")
        if grid is not None and burst is not None:
            grid.removeWidget(burst)
            grid.addWidget(burst, 2, 0, 1, 1 if dev else 2)
            burst.setVisible(True)
        return dev

    def _refresh_actions(self):
        """Enable buttons per connection state: pump-only actions need the pump;
        Ramp/Calibrate/Verify additionally need the flow sensor."""
        pump = self.line is not None
        sensor = pump and getattr(self.line, "sensor", None) is not None
        busy = self._busy
        for k in ("startflow", "run", "burst", "prime"):      # work without a sensor
            self.btn[k].setEnabled(pump and not busy)
        self.btn["ramp"].setEnabled(sensor and not busy)      # needs the flow sensor
        self.btn["stop"].setEnabled(pump)

    def _set_busy(self, b):
        self._busy = b
        self._refresh_actions()

    def _guard(self):
        if self.line is None:
            self.status_msg.emit("Connect first."); return True
        if self._busy:
            self.status_msg.emit("Busy — wait or press STOP."); return True
        return False

    def _work(self, fn):
        def job():
            self._busy = True; self._set_busy(True)
            try:
                fn()
            except Exception as e:
                self.status_msg.emit(f"ERROR: {e}")
            finally:
                self._busy = False; self._set_busy(False)
        threading.Thread(target=job, daemon=True).start()

    # ------------------------------------------------------------- connect
    def _connect_pump(self):
        """Connect the Chemyx pump only (no flow sensor required)."""
        if self.line is not None:
            self._disconnect_pump(); return
        self._apply_defs()
        self.status_msg.emit("Connecting pump…"); self.btn_pump.setEnabled(False)

        def job():
            ok = False
            try:
                from dual_syringe import DualSyringeLine
                line = DualSyringeLine(diameter_mm=float(self.cfg["diameter"]),
                                       n_syringes=int(self.cfg["n"]),
                                       direction=self.cfg["direction"],
                                       pump_port=self.cfg["port"],
                                       require_sensor=False, sensor=None, verbose=False)
                self.line = line
                ok = True
                # New hardware in hand: a previous safe shutdown must not latch
                # this one out.
                self._shutdown_done = False
                self._abort = False
                self.status_msg.emit(f"Pump connected — {line.pump.port.split('/')[-1]}. "
                                     "Connect the flow sensor too if you have one.")
            except Exception as e:
                self.status_msg.emit(f"Pump connect failed: {e}")
            self.pump_result.emit(ok)
        threading.Thread(target=job, daemon=True).start()

    def _after_pump(self, ok):
        self.btn_pump.setEnabled(True)
        if ok and not self.poll_timer.isActive():
            self.poll_timer.start(150)   # a previous safe shutdown stopped it

        self._set_pill(self.pill_pump, ok, "pump")
        self.btn_pump.setText("Disconnect pump" if ok else "Connect pump")
        self.btn_sensor.setEnabled(ok)          # sensor attaches to a live pump
        if not ok:
            self.line = None
        self._refresh_actions()

    def _disconnect_pump(self):
        """Disconnect the pump (and the sensor, if attached)."""
        self._steady = False
        try:
            if self.line is not None:
                self.line.disconnect_sensor()
        except Exception:
            pass
        try:
            if self.line is not None:
                self.line.close()
        except Exception:
            pass
        self.line = None; self.latest_flow = None; self._flow_log_path = None
        self._set_pill(self.pill_pump, False, "pump")
        self._set_pill(self.pill_sensor, False, "sensor")
        self.btn_pump.setText("Connect pump")
        self.btn_sensor.setText("Connect sensor"); self.btn_sensor.setEnabled(False)
        self._refresh_actions()
        self.status_msg.emit("Pump disconnected.")

    def _flow_sign(self):
        """+1.0 or -1.0. Anything that isn't clearly negative means "don't flip"."""
        try:
            return -1.0 if float(self.cfg.get("flow_sign", 1.0)) < 0 else 1.0
        except (TypeError, ValueError):
            return 1.0

    def _connect_sensor(self):
        """Attach the Fluigent flow sensor to the live pump (optional)."""
        if self.line is None:
            self.status_msg.emit("Connect the pump first."); return
        if getattr(self.line, "sensor", None) is not None:
            self._disconnect_sensor(); return
        self.status_msg.emit("Connecting flow sensor…"); self.btn_sensor.setEnabled(False)

        def job():
            ok = False
            try:
                self.line.connect_sensor(channel=int(self.cfg.get("sensor_channel", 0)))
                # Correct the orientation at the source, before anything reads it.
                self.line.sensor = _SignedSensor(self.line.sensor, self._flow_sign)
                ok = True
                inv = " (readings INVERTED: flow_sign = -1)" if self._flow_sign() < 0 else ""
                self.status_msg.emit(f"Flow sensor connected — plotting + logging.{inv}")
            except Exception as e:
                self.status_msg.emit(f"Sensor connect failed: {e}")
            self.sensor_result.emit(ok)
        threading.Thread(target=job, daemon=True).start()

    def _after_sensor(self, ok):
        self.btn_sensor.setEnabled(self.line is not None)
        self._set_pill(self.pill_sensor, ok, "sensor")
        self.btn_sensor.setText("Disconnect sensor" if ok else "Connect sensor")
        if ok:
            self._t0 = time.monotonic(); self.t.clear(); self.v.clear()
            self._start_flow_log()   # only log flow when a sensor is present
        self._refresh_actions()

    def _disconnect_sensor(self):
        try:
            if self.line is not None:
                self.line.disconnect_sensor()
        except Exception:
            pass
        self.latest_flow = None; self._flow_log_path = None
        self._set_pill(self.pill_sensor, False, "sensor")
        self.btn_sensor.setText("Connect sensor")
        self._refresh_actions()
        self.status_msg.emit("Flow sensor disconnected (pump still connected).")

    def _start_flow_log(self):
        """Create the flow tab's OWN continuous log file (independent of the
        metabolite file, which separately gets the flow via its flow_uL_min col)."""
        import os
        from datetime import datetime
        folder = _sensor_readings_dir()
        try:
            os.makedirs(folder, exist_ok=True)
            ts = datetime.now().strftime("%d_%m_%y_%H_%M")
            self._flow_log_path = os.path.join(folder, f"Flow_Log_{ts}.csv")
            with open(self._flow_log_path, "w") as f:
                f.write(f"# Flow rate log — started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write("elapsed_s,timestamp,flow_uL_min,clog,air_bubble,segment\n")
            self._flow_log_last = 0.0
            self.status_msg.emit(f"Flow log → {os.path.basename(self._flow_log_path)}")
        except Exception as e:
            self._flow_log_path = None
            self.status_msg.emit(f"Flow log not started: {e}")

    def _console_log(self, msg):
        """Append a timestamped line to the activity console under the graph."""
        from datetime import datetime
        try:
            self.console.appendPlainText(f"{datetime.now():%H:%M:%S}  {msg}")
        except Exception:
            pass

    def log(self, msg):
        """Public: push a line into the flow-tab console (used by the experiment
        orchestrator so a running experiment is visible on THIS tab too)."""
        self.status_msg.emit(msg)

    def _save_log(self):
        """Save a copy of the current flow log wherever you choose."""
        import os, shutil
        if not self._flow_log_path or not os.path.exists(self._flow_log_path):
            self.status_msg.emit("No flow log yet — connect and let it record first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save flow log", os.path.basename(self._flow_log_path), "CSV (*.csv)")
        if path:
            if not path.lower().endswith(".csv"):   # supply the extension, don't demand it
                path += ".csv"
            try:
                shutil.copyfile(self._flow_log_path, path)
                self.status_msg.emit(f"Flow log saved → {path}")
            except Exception as e:
                self.status_msg.emit(f"Save failed: {e}")

    # ------------------------------------------------------------- actions
    def _nsyr(self):
        """Number of syringes (>=1). What you type is the COMBINED/line value;
        each syringe is driven at value / _nsyr()."""
        try:
            return max(1, int(self.cfg.get("n", 1)))
        except (ValueError, TypeError):
            return 1

    def _start_flow(self):
        if self._guard():
            return
        line_rate = self._num(self.f_rate)          # what you want in the LINE
        n = self._nsyr(); machine = line_rate / n   # per-syringe rate to the pump
        self._abort = False
        self._arm_clog_watch(self.line.expected_combined(machine))
        self._clog_flagged = False; self._set_clog_ui(False)
        self.status_msg.emit(f"Flowing {line_rate:g} µL/min in the line "
                             f"({machine:g}/syringe × {n}). Change + press again to update.")
        self._work(lambda: self.line.start_flow_single(machine, direction=self.cfg["direction"]))

    def _run(self):
        if self._guard():
            return
        line_vol, line_rate = self._num(self.f_vol), self._num(self.f_rate)
        n = self._nsyr(); mvol, mrate = line_vol / n, line_rate / n
        self._abort = False
        self._arm_clog_watch(self.line.expected_combined(mrate))
        self._clog_flagged = False; self._set_clog_ui(False)
        self.status_msg.emit(f"Running {line_vol:g} µL @ {line_rate:g} µL/min in the line "
                             f"({mvol:g} µL @ {mrate:g}/syringe × {n})…")
        def fn():
            self.line.deliver_single(mvol, mrate, direction=self.cfg["direction"])
            done = self._wait_run(mvol, mrate, "Run")
            self._steady = False
            self.status_msg.emit("Run complete." if done else "Run stopped.")
        self._work(fn)

    def _ramp(self):
        if self._guard():
            return
        self._apply_defs()
        try:
            start = float(self.cfg["r_start"]); mx = float(self.cfg["r_max"])
            step = float(self.cfg["r_step"]); dwell = float(self.cfg["r_dwell"])
            tol = float(self.cfg["r_tol"]) / 100.0
        except (ValueError, KeyError):
            self.status_msg.emit("Set ramp values in Definitions."); return
        target = self._num(self.f_target)            # sensor target (combined)
        n = self._nsyr()
        ms, mm, mstep = start / n, mx / n, step / n  # per-syringe rates for the ramp
        self._abort = False; self._steady = False    # clog check off during ramp
        self.status_msg.emit(f"Ramp {start:g}→{mx:g} µL/min (line) until sensor ≈ {target:g}…")
        def fn():
            rep = self.line.ramp_single(
                ms, mm, mstep, dwell, target, tol_frac=tol, direction=self.cfg["direction"],
                on_step=lambda r: self.status_msg.emit(
                    f"ramp: line {r['rate']*n:.1f} → sensor {r['measured']:+.2f} (target {target:g})"),
                should_stop=lambda: self._abort)
            if rep["reached"]:
                self.status_msg.emit(f"✓ Target reached at line {rep['stop_rate']*n:.1f} µL/min "
                                     f"(sensor {rep['measured']:+.2f}).")
            elif self._abort:
                self.status_msg.emit("Ramp stopped.")
            else:
                self.status_msg.emit(f"Ramp hit max {mx:g} without reaching target "
                                     f"(last {rep['measured']:+.2f}).")
        self._work(fn)

    def _isleep(self, seconds):
        """Sleep up to `seconds`, waking early if STOP is pressed. Returns True if
        it ran the full time, False if aborted."""
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if self._abort:
                return False
            time.sleep(min(0.2, max(0.0, end - time.monotonic())))
        return not self._abort

    def _run_flow_sweep(self, rates, hold_s, decay_s, ramp_s):
        """Decay-characterization sweep: for each rate, ramp/step up, HOLD, then
        STOP and watch the flow decay — all in one flow log, tagged per rate so
        you can compare settle/decay across rates. No wells / no robot."""
        if self._guard():
            return
        n = self._nsyr()
        direction = self.cfg["direction"]
        if self._flow_log_path is None:
            self.status_msg.emit("⚠ No flow log — connect the SENSOR first so the decay is recorded.")
        self._abort = False
        self.exp_running.emit(True)
        self.status_msg.emit(f"Flow sweep: {', '.join(f'{r:g}' for r in rates)} µL/min · "
                             f"hold {hold_s:g}s · decay {decay_s:g}s.")

        def fn():
            for i, rate in enumerate(rates, 1):
                if self._abort:
                    break
                self._seg_label = f"r{rate:g}_hold"
                self.status_msg.emit(f"[sweep {i}/{len(rates)}] up to {rate:g} µL/min…")
                if ramp_s > 0:                       # ramp 0→rate over ramp_s
                    steps = max(1, int(ramp_s))
                    for s in range(1, steps + 1):
                        if self._abort:
                            break
                        self.line.start_flow_single((rate * s / steps) / n, direction=direction)
                        if not self._isleep(ramp_s / steps):
                            break
                else:
                    self.line.start_flow_single(rate / n, direction=direction)
                self.status_msg.emit(f"[sweep {i}/{len(rates)}] holding {rate:g} ({hold_s:g}s)…")
                if not self._isleep(hold_s):
                    break
                # stop and watch it decay
                self._seg_label = f"r{rate:g}_decay"
                try:
                    self.line.stop()
                except Exception:
                    pass
                self.status_msg.emit(f"[sweep {i}/{len(rates)}] pump OFF, logging decay ({decay_s:g}s)…")
                self._isleep(decay_s)
            self._seg_label = ""
            try:
                self.line.stop()
            except Exception:
                pass
            self.status_msg.emit("✓ Flow sweep complete — all rates in the flow log."
                                 if not self._abort else "Flow sweep stopped.")
        def fn_wrapped():
            try:
                fn()
            finally:
                self.exp_running.emit(False)   # re-enable Run / clear Stop when done
        self._work(fn_wrapped)

    def _run_burst_calibration(self, p):
        """Characterise how fast the line jumps UP to the sensor ceiling and can be
        driven BACK to baseline — the data for a burst timing calibration curve.
        Pump + SENSOR only (no wells/robot). Writes a high-resolution trace CSV plus
        a per-trial summary CSV. Runs in the worker thread with the live poll paused
        so the fast sampler owns the sensor."""
        if self._guard():
            return
        if getattr(self.line, "sensor", None) is None:
            self.status_msg.emit("Burst calibration needs the flow SENSOR connected."); return
        import os
        from datetime import datetime
        base = float(p["baseline"]); ceil = float(p["ceiling"])
        highs = list(p["high_mults"]); revs = list(p["rev_mults"])
        reps = int(p["repeats"]); settle_s = float(p["settle_s"])
        rise_to = float(p["rise_timeout_s"]); max_rate = float(p["max_rate"])
        n = self._nsyr()
        fwd = self.cfg["direction"]
        rev = "infuse" if fwd == "withdraw" else "withdraw"
        ch = getattr(self.line, "sensor_channel", 0)
        folder = _sensor_readings_dir()
        os.makedirs(folder, exist_ok=True)
        ts = datetime.now().strftime("%d_%m_%y_%H_%M")
        trace_path = os.path.join(folder, f"Burst_Calib_{ts}.csv")
        summ_path = os.path.join(folder, f"Burst_Calib_{ts}_summary.csv")
        self._abort = False
        self.exp_running.emit(True)

        def rd():
            try:
                return float(self.line.sensor.read(ch))
            except Exception:
                return 0.0

        def fn():
            self._calibrating = True                 # pause the live poll's sensor reads
            tf = open(trace_path, "w"); tf.write("sweep,trial,mult,target_rate,phase,t_s,flow\n")
            sf = open(summ_path, "w")
            sf.write("sweep,trial,mult,target_rate,time_s,peak_or_min_flow,escalated_rate,reached\n")
            self.status_msg.emit(f"Burst calibration @ base {base:g} → ceiling {ceil:g}. "
                                 f"Trace → {os.path.basename(trace_path)}")

            def sample_until(cond, timeout_s, sweep, trial, mult, target, phase, going_up):
                """Fast-sample (~50 Hz), logging each read, until cond(flow) or timeout.
                Returns (met, t_rel_at_hit, extreme_flow_seen)."""
                t0 = time.monotonic(); extreme = None
                while not self._abort:
                    t = time.monotonic() - t0
                    f = rd()
                    tf.write(f"{sweep},{trial},{mult:g},{target:g},{phase},{t:.3f},{f:.3f}\n")
                    extreme = f if extreme is None else (max(extreme, f) if going_up else min(extreme, f))
                    if cond(f):
                        return True, t, extreme
                    if t >= timeout_s:
                        return False, t, extreme
                    time.sleep(0.02)
                return False, time.monotonic() - t0, extreme

            def settle_base(trial):
                """Command baseline and WAIT until the sensor actually LEVELS at it
                (within ±tol held ~2 s). Keep waiting as long as the flow is still
                approaching the target — slow is fine, it will get there. Only give
                up if the flow STALLS (no progress toward target for stall_s while
                still far), which means something is wrong (air/blockage) — then
                ABORT rather than measure bad data. `settle_s` is the stall window,
                NOT a start-anyway clock."""
                self.line.start_flow_single(base / n, direction=fwd)
                tol = max(3.0, 0.08 * base)
                hold_s = 2.0; stall_s = max(15.0, settle_s); prog_eps = max(1.5, 0.5 * tol)
                in_band = None; best_err = float("inf"); last_prog = time.monotonic()
                while not self._abort:
                    t = time.monotonic() - last_prog          # only for logging offset
                    f = rd()
                    tf.write(f"settle,{trial},0,{base:g},level,{time.monotonic():.3f},{f:.3f}\n")
                    err = abs(f - base)
                    if err <= tol:
                        if in_band is None:
                            in_band = time.monotonic()
                        if time.monotonic() - in_band >= hold_s:
                            return True                        # leveled — safe to measure
                    else:
                        in_band = None
                    if err < best_err - prog_eps:              # still closing on target → keep waiting
                        best_err = err; last_prog = time.monotonic()
                    elif err > 2 * tol and (time.monotonic() - last_prog) >= stall_s:
                        self.status_msg.emit(
                            f"✗ Baseline STALLED at {f:.0f} µL/min (target {base:g}) — not "
                            f"approaching for {stall_s:g}s. Likely AIR/blockage; prime the line. "
                            f"Aborting calibration (won't measure bad data).")
                        self._abort = True                     # stop the run; do NOT start anyway
                        return False
                    time.sleep(0.05)
                return False

            try:
                for trial in range(1, reps + 1):
                    if self._abort:
                        break
                    # ===== Sweep A — RISE: baseline → ceiling across up rates =====
                    for m in highs:
                        if self._abort:
                            break
                        settle_base(trial)
                        rate = m * base; esc = rate
                        self.status_msg.emit(f"[rise t{trial}] {m:g}× = {rate:g} µL/min → time to {ceil:g}…")
                        self.line.start_flow_single(rate / n, direction=fwd)
                        met, thit, peak = sample_until(lambda f: f >= ceil, rise_to,
                                                       "rise", trial, m, rate, "up", True)
                        # "just up it": escalate the rate if the ceiling wasn't reached in time
                        while not met and not self._abort and esc < max_rate:
                            esc = min(max_rate, round(esc * 1.5, 1))
                            self.status_msg.emit(f"[rise] {ceil:g} not reached in {rise_to:g}s → up to {esc:g} µL/min")
                            self.line.start_flow_single(esc / n, direction=fwd)
                            met, thit, peak = sample_until(lambda f: f >= ceil, rise_to,
                                                           "rise", trial, m, esc, "up", True)
                        sf.write(f"rise,{trial},{m:g},{rate:g},{thit:.3f},{peak:.2f},{esc:g},{int(met)}\n"); sf.flush()
                        self.status_msg.emit(f"[rise t{trial}] {m:g}×: {'reached' if met else 'MISSED'} {ceil:g} "
                                             f"in {thit:.2f}s (peak {peak:.0f}).")
                        # return safely toward baseline before the next rate
                        self.line.start_flow_single((max(revs) * base) / n, direction=rev)
                        sample_until(lambda f: f <= base, 15.0, "rise", trial, m, base, "return", False)
                        self.line.start_flow_single(base / n, direction=fwd)
                    # ===== Sweep B — FALL: ceiling → baseline across reverse rates =====
                    ref = max(highs) * base                   # fixed up rate to reach the ceiling
                    for m in revs:
                        if self._abort:
                            break
                        settle_base(trial)
                        self.line.start_flow_single(ref / n, direction=fwd)
                        up_ok, _, _ = sample_until(lambda f: f >= ceil, rise_to,
                                                   "fall", trial, m, ref, "up_ref", True)
                        if not up_ok:
                            self.status_msg.emit(f"[fall t{trial}] couldn't reach {ceil:g} — skip {m:g}×."); continue
                        revrate = m * base
                        self.status_msg.emit(f"[fall t{trial}] reverse {m:g}× = {revrate:g} → time {ceil:g}→{base:g}…")
                        self.line.start_flow_single(revrate / n, direction=rev)
                        met, thit, _ = sample_until(lambda f: f <= base, 20.0,
                                                    "fall", trial, m, revrate, "down", False)
                        # cut the reverse, level off at baseline; watch ~2 s for undershoot
                        self.line.start_flow_single(base / n, direction=fwd)
                        _, _, mn = sample_until(lambda f: False, 2.0, "fall", trial, m, base, "level", False)
                        over = "OVERSHOOT" if (mn is not None and mn < base - 3) else "ok"
                        sf.write(f"fall,{trial},{m:g},{revrate:g},{thit:.3f},{mn:.2f},,{int(met)}\n"); sf.flush()
                        self.status_msg.emit(f"[fall t{trial}] {m:g}×: {ceil:g}→{base:g} in {thit:.2f}s "
                                             f"(min {mn:.0f}, {over}).")
                    settle_base(trial)
                self.status_msg.emit("✓ Burst calibration complete — "
                                     f"trace {os.path.basename(trace_path)}, summary {os.path.basename(summ_path)}."
                                     if not self._abort else "Burst calibration stopped.")
            finally:
                self._calibrating = False
                tf.close(); sf.close()
                try:
                    self.line.stop()
                except Exception:
                    pass

        def fn_wrapped():
            try:
                fn()
            finally:
                self.exp_running.emit(False)
        self._work(fn_wrapped)

    def apply_run(self, run, start=True):
        """Apply one experiment run's flow settings (called per run by the orchestrator,
        or once at load for preview with start=False). Tags the flow log with the run
        name and, when start=True and the pump is connected, begins flowing at its rate."""
        name = str(run.get("name", "") or "")
        if name:
            self._seg_label = f"run_{name}"
        if "flow_rate" in run:
            self.f_rate.setText(str(run["flow_rate"]))
        if "direction" in run:
            d = str(run["direction"]).lower()
            self.cfg["direction"] = "withdraw" if d.startswith(("w", "pull")) else "infuse"
            self.chk_pull.setChecked(self.cfg["direction"] == "withdraw")
        for fkey, ckey in (("pause_after", "ff_pause_s"), ("resume_after", "ff_resume_s"),
                           ("resume_ramp", "ff_resume_ramp")):
            if fkey in run:
                try:
                    self.cfg[ckey] = float(run[fkey])
                except (ValueError, TypeError):
                    pass
        if "pause_on_move" in run:
            on = str(run["pause_on_move"]).lower() in ("yes", "true", "1", "on", "y")
            self._ff_enabled = on
            if hasattr(self, "chk_ff"):
                self.chk_ff.setChecked(on)
        # per-run burst settings: multiplier, timed/stop/backflow seconds, and
        # whether to auto-burst once per buffer entry during this run.
        for bkey in ("b_mult", "b_high_s", "b_stop_s", "b_backflow_s", "b_backflow_rate"):
            if bkey in run:
                try:
                    self.cfg[bkey] = float(run[bkey])
                except (ValueError, TypeError):
                    pass
        if "auto_burst" in run:
            on = str(run["auto_burst"]).lower() in ("yes", "true", "1", "on", "y")
            self._auto_burst = on
            if hasattr(self, "chk_auto"):
                self.chk_auto.setChecked(on)
        if start and self.line is not None:
            try:
                rate = self._num(self.f_rate, 0.0); n = self._nsyr()
                if rate > 0:
                    self.line.start_flow_single(rate / n, direction=self.cfg["direction"])
            except Exception as e:
                self.status_msg.emit(f"apply_run flow error: {e}")
        self.status_msg.emit(f"Run '{name or '—'}': {self._num(self.f_rate, 0):g} µL/min (line), "
                             f"pause-on-move {'ON' if self._ff_enabled else 'off'}.")

    def clear_run_tag(self):
        self._seg_label = ""

    def _refresh_runbtn(self):
        btn = getattr(self, "btn_runexp", None)
        if btn is None:
            return
        if self._exp_calib:
            btn.setText(f"▶  Run burst calibration ({self._exp_calib['repeats']} reps)")
            btn.setEnabled(True)
        elif self._exp_sweep:
            btn.setText(f"▶  Run sweep ({len(self._exp_sweep['rates'])} rates)")
            btn.setEnabled(True)
        elif self._exp_runs:
            nr = len(self._exp_runs)
            btn.setText(f"▶  Run experiment ({nr} run{'s' if nr != 1 else ''})")
            btn.setEnabled(True)
        else:
            btn.setText("▶  Run experiment")
            btn.setEnabled(False)

    def _run_experiment(self):
        """Dispatch the loaded experiment: flow-decay sweep runs here; a well-plate
        multi-run is handed to the Sampling side."""
        if self._exp_calib:
            self._run_burst_calibration(self._exp_calib)   # toggles exp_running itself
        elif self._exp_sweep:
            self._run_flow_sweep(**self._exp_sweep)    # toggles exp_running itself
        elif self._exp_runs and self.main_gui is not None \
                and hasattr(self.main_gui, "run_experiment_runs"):
            # run_experiment_runs toggles exp_running itself (True on start, False
            # in its finally) so the Stop button tracks the real task lifecycle.
            self.main_gui.run_experiment_runs(self._exp_runs)
        else:
            self.status_msg.emit("No experiment loaded — press 📂 Load experiment first.")

    def _on_exp_running(self, running):
        """Toggle the Stop-experiment button (and lock out Run) while a run is live."""
        self.btn_stopexp.setEnabled(bool(running))
        if running:
            self.btn_runexp.setEnabled(False)
        else:
            self._refresh_runbtn()      # re-enable Run per the loaded experiment

    def _stop_experiment(self):
        """Halt the running experiment: signal the run loop to stop so it finishes
        the CURRENT well (flow kept on through it) then halts with no further
        runs/wells. The pump is stopped at each path's natural end (sweep end / run
        task finally), not yanked mid-well. Mirrors STOP on the Sampling tab but
        from the Flow tab. `_abort` also breaks an in-tab flow sweep / burst worker."""
        self.status_msg.emit("⏹ Stop experiment — finishing the current well, then halting.")
        self.btn_stopexp.setEnabled(False)
        self._abort = True                       # aborts an in-tab sweep / burst worker
        if self.main_gui is not None and hasattr(self.main_gui, "stop_experiment"):
            try:
                self.main_gui.stop_experiment()  # sets the app-state stop_event
            except Exception as e:
                self.status_msg.emit(f"stop_experiment error: {e}")

    def _calibrate(self):
        if self._guard():
            return
        line_rate = self._num(self.f_rate); n = self._nsyr(); machine = line_rate / n
        self._abort = False; self._steady = False
        self.status_msg.emit(f"Calibrating at {line_rate:g} µL/min (line)…")
        self._work(lambda: self.cal_msg.emit(
            self.line.calibrate_single(machine, settle_s=float(self.cfg["settle"]),
                                       measure_s=float(self.cfg["measure"]))["cal_factor"]))

    def _verify(self):
        if self._guard():
            return
        line_rate = self._num(self.f_rate); n = self._nsyr(); machine = line_rate / n
        self._abort = False; self._steady = False
        self.status_msg.emit(f"Verifying at {line_rate:g} µL/min (line)…")
        def fn():
            r = self.line.verify_single(machine, settle_s=float(self.cfg["settle"]),
                                        measure_s=float(self.cfg["measure"]))
            self.status_msg.emit(f"Verify: sensor {r['measured_mean']:+.2f} vs expected "
                                 f"{r['expected_combined']:g} ({r['error_pct']:+.1f}%)")
        self._work(fn)

    def _stop(self):
        self._abort = True; self._abort_clear = True
        self._steady = False; self._bursting = False
        self._clog_flagged = False; self._clog_since = None
        self._set_clog_ui(False)
        if self.line is None:
            return
        try:
            self.line.stop(); self.status_msg.emit("STOP sent.")
        except Exception as e:
            self.status_msg.emit(f"Stop error: {e}")

    # ------------------------------------------------------------- clog arming
    def _arm_clog_watch(self, expected):
        """Command accepted: start watching, but do NOT start judging yet.

        `clog_seconds` is the sustained-low duration, not a startup grace — it used
        to begin the instant the button was pressed, which meant a dry or
        depressurised line had 6 s to reach 40% of setpoint or be called a clog. It
        cannot: the syringe has to take up backlash and pressurise the line first.

        So judging is gated on flow having been ESTABLISHED at least once since this
        command (`_flow_seen`). That is the correct definition anyway — a clog is the
        LOSS of flow that was working. Below setpoint before flow ever started is a
        priming problem, and it is reported as one. It also means a dead or unprimed
        sensor, which reads ~0 forever, can never arm the watch and so can never
        trigger a burst against nothing."""
        self._expected = expected
        self._steady = True
        self._flow_seen = False
        self._arm_warned = False
        self._steady_since = time.monotonic()
        self._clog_since = None

    # ------------------------------------------------------------- clog UI
    def _set_clog_ui(self, on, msg=""):
        """Show/hide the clog banner and light up 'Burst now'.

        Purely cosmetic — it never starts the pump. When auto-clear is off this is
        the ONLY thing a detected clog does, so the operator can look at the trace
        and decide whether it is a real blockage or a sensor that is reading zero.

        Goes through a signal because `_finish_clear` calls it from the clearing
        worker thread, and Qt widgets may only be touched on the GUI thread."""
        self.clog_ui.emit(bool(on), msg)

    def _apply_clog_ui(self, on, msg=""):
        try:
            self.lbl_clog.setVisible(bool(on))
            if on:
                self.lbl_clog.setText(msg)
                self.lbl_clog.setStyleSheet(
                    f"background:{RED};color:white;font-weight:700;padding:6px;"
                    "border-radius:4px;")
            b = self.btn.get("burst")
            if b is not None:
                b.setStyleSheet(
                    f"QPushButton{{background:{AMBER};color:#11151c;font-weight:700;}}"
                    if on else "")
                b.setText("Burst now  ⚠" if on else "Burst now")
        except Exception:
            pass

    # ------------------------------------------------------------- polling
    def _clog_watch(self, val):
        """Flag a clog and kick the clearing escalation when the pump is COMMANDED
        to flow but the sensor reads far below expected for a sustained time.

        Idle/standalone only — during a plate run the metabolite blockage detector
        drives clearing instead (this suppresses itself while phase != 'idle'). Does
        nothing unless a steady flow is commanded, so a stopped pump never alarms."""
        if not self._steady or self._expected <= 0 or self._clearing or self._bursting:
            if not self._steady:
                self._clog_since = None
            return
        if self._phase != "idle":              # a run is active -> metabolite path owns it
            return
        frac = self._cfgf("clog_frac", 0.4)
        secs = self._cfgf("clog_seconds", 6.0)
        now = time.monotonic()

        # ---- arming: nothing is judged until flow has actually been established.
        if not self._flow_seen:
            arm_frac = self._cfgf("clog_arm_frac", 0.5)
            # Arming AT or BELOW the clog threshold would let a run arm and trip
            # on the same reading. Definitions is free-text, so enforce the gap
            # here instead of trusting whatever was typed in.
            if arm_frac <= frac:
                arm_frac = frac * 1.25
                if not getattr(self, "_arm_frac_warned", False):
                    self._arm_frac_warned = True
                    self.status_msg.emit(
                        f"⚠ [clog] arm fraction must sit above the clog fraction "
                        f"({frac:.0%}) — using {arm_frac:.0%} for this run.")
            if abs(val) >= arm_frac * abs(self._expected):
                self._flow_seen = True
                self._clog_since = None
                self.status_msg.emit(
                    f"[clog] flow established ({abs(val):.1f} µL/min ≥ {arm_frac:.0%} "
                    f"of {abs(self._expected):.1f}) — clog watch armed.")
                return
            # Still climbing. Only complain once, and only after a generous startup
            # budget — and call it what it is (never got going), not a clog.
            arm_to = self._cfgf("clog_arm_timeout_s", 120.0)
            if not self._arm_warned and (now - self._steady_since) >= arm_to:
                self._arm_warned = True
                bar = arm_frac * abs(self._expected)
                self.status_msg.emit(
                    f"⚠ [flow] never reached {arm_frac:.0%} of "
                    f"{abs(self._expected):.1f} µL/min ({bar:.1f}) in {arm_to:g}s "
                    f"(best so far {abs(val):.1f}). NOT calling this a clog — check "
                    "priming, air in the line, and that the sensor is reading. "
                    "Clog watch stays disarmed.")
                self._set_clog_ui(True, f"⚠  FLOW NEVER STARTED — still "
                                        f"{abs(val):.1f} µL/min after {arm_to:g}s, below "
                                        f"the {arm_frac:.0%} arm bar ({bar:.1f}) for a "
                                        f"{abs(self._expected):.1f} µL/min setpoint. "
                                        "Prime the line / check for air leaks.")
            return

        low = abs(val) < frac * abs(self._expected)
        if low:
            if self._clog_since is None:
                self._clog_since = now
            elif (now - self._clog_since) >= secs and not self.is_clogged:
                self.is_clogged = True
                try:
                    self.clog_changed.emit(True)
                except Exception:
                    pass
                head = (f"flow {abs(val):.1f} < {frac:.0%} of "
                        f"{abs(self._expected):.1f} µL/min for {secs:g}s")
                # Detection and ACTION are separate. Detecting a clog never earns
                # the right to drive the pump on its own — that is opt-in, because
                # a sensor reading ~0 (unprimed, dry, dropped out) is indistinguish-
                # able from a real blockage and would burst against nothing.
                if self.cfg.get("auto_clear", False):
                    self.status_msg.emit(f"⚠ [clog] {head} — auto-clearing.")
                    self.request_clog_clear("flow-sensor")
                else:
                    self._clog_flagged = True
                    self.status_msg.emit(
                        f"⚠ [clog] {head} — NOT clearing (auto-clear is off). "
                        "Press 'Burst now' to clear it, or check that the sensor "
                        "is primed and reading.")
                    self._set_clog_ui(True, f"⚠  CLOG SUSPECTED — {head}.  "
                                            "Press 'Burst now' to clear, or STOP.")
        else:
            self._clog_since = None
            if self.is_clogged:                # recovered on its own
                self.is_clogged = False
                self._clog_flagged = False
                self._set_clog_ui(False)
                self.status_msg.emit("✓ [clog] flow recovered on its own.")
                try:
                    self.clog_changed.emit(False)
                except Exception:
                    pass

    def _show_plot_cover(self, covered: bool):
        """Swap the plot area between the live canvas and the cover panel.

        Only touched on a real change, so the stack isn't churned every 150 ms
        tick. The big readout above the plot keeps updating either way; the
        cover repeats it so the number stays large once the trace is gone."""
        idx = 1 if covered else 0
        if self.plot_stack.currentIndex() != idx:
            self.plot_stack.setCurrentIndex(idx)
            if covered:
                self.lbl_cover_flow.setText(self.lbl_flow.text())

    def _poll(self):
        # Nothing to read/plot/log without a flow sensor (pump-only is fine).
        if self.line is None or getattr(self.line, "sensor", None) is None:
            # No sensor => no trace anywhere, so never leave the cover up.
            self._show_plot_cover(False)
            return
        # During a burst calibration the worker thread owns the sensor (fast
        # sampling); don't read it here too (concurrent HID reads corrupt data).
        if self._calibrating:
            return
        ch = getattr(self.line, "sensor_channel", 0)
        try:
            val = float(self.line.sensor.read(ch))
        except Exception as e:
            self.status_msg.emit(f"sensor read error: {e}"); return
        now = time.monotonic() - self._t0
        self.latest_flow = val
        try:
            self._last_air = bool(self.line.sensor.air_bubble(ch))
        except Exception:
            self._last_air = False
        self.t.append(now); self.v.append(val)
        self.lbl_flow.setText(f"{val:+.2f}")

        # standalone clog watch — only when a steady flow is COMMANDED (pump
        # pumping). If the pump isn't pumping, no flow is expected, so nothing to do.
        self._clog_watch(val)

        # own flow-rate log file (independent of the metabolite file), ~1 Hz
        if self._flow_log_path and (now - self._flow_log_last) >= 1.0:
            self._flow_log_last = now
            try:
                from datetime import datetime
                with open(self._flow_log_path, "a") as _lf:
                    _lf.write(f"{now:.2f},{datetime.now():%Y-%m-%d %H:%M:%S},{val:.3f},"
                              f"{1 if self.is_clogged else 0},{1 if self._last_air else 0},"
                              f"{self._seg_label}\n")
            except Exception:
                pass

        # Mutual exclusivity: when the metabolite sensor is running, the flow is
        # drawn on that plot's right axis — don't also draw it here. Cover this
        # canvas so a frozen trace can't be mistaken for the live one.
        if self._metabolites_running():
            self._show_plot_cover(True)
            self.lbl_cover_flow.setText(f"{val:+.2f}")
            self._tick += 1
            if self._tick % 40 == 0:
                self.lbl_status.setText("Metabolite plot active — flow shown there (right axis).")
            return
        self._show_plot_cover(False)

        # redraw ~ every 300 ms — but only while this tab is actually on screen.
        # Re-rendering a matplotlib canvas behind another tab burned a whole CPU
        # core on the Pi and made the UI (well-plate dragging especially) stutter.
        self._tick += 1
        if self._tick % 2 or not self.isVisible():
            return
        try:
            win = max(10.0, float(self.cfg["window"]))
        except (ValueError, KeyError):
            win = 120.0
        lo = now - win
        xs = [t for t in self.t if t >= lo]
        ys = [self.v[i] for i, t in enumerate(self.t) if t >= lo]
        self.trace.set_data(xs, ys)
        try:
            tgt = float(self.f_target.text())
        except ValueError:
            tgt = 0.0
        self.ax.set_xlim(max(0.0, lo), max(now, win))
        allv = ys + [tgt, 0.0]
        ymin, ymax = min(allv), max(allv)
        pad = 0.12 * (ymax - ymin) if ymax > ymin else 1.0
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.canvas.draw_idle()

    # ------------------------------------------------------- experiment phase
    def on_move_command(self, well_id=""):
        """MABIP just sent a move command (the exact anchor). Feed-forward PAUSE
        the pump ~0.45 s later (tip lifts out) and RESUME ~10 s later (tip back in
        liquid), covering the air gap. Fires off local timers, so it's precise."""
        if not self._ff_enabled or self.line is None:
            return
        try:
            pause_ms = int(float(self.cfg.get("ff_pause_s", 0.45)) * 1000)
            resume_ms = int(float(self.cfg.get("ff_resume_s", 10.0)) * 1000)
        except (ValueError, TypeError):
            pause_ms, resume_ms = 450, 10000
        rate = self._num(self.f_rate, 0.0)
        n = self._nsyr()
        self._ff_gen += 1
        gen = self._ff_gen
        self.status_msg.emit(f"[ff] move→{well_id}: pause in {pause_ms/1000:.2f}s, "
                             f"resume {rate:g} µL/min in {resume_ms/1000:.1f}s.")

        ramp_s = float(self.cfg.get("ff_resume_ramp", 0.0) or 0.0)

        def do_pause():
            if gen != self._ff_gen or self.line is None:
                return
            try:
                self.line.stop()
            except Exception:
                pass
            self._phase_flow_rate = 0.0     # so a ramped resume starts from 0
            self.status_msg.emit(f"[ff] pump paused (moving to {well_id}).")

        def do_resume():
            if gen != self._ff_gen or self.line is None:
                return
            if ramp_s > 0:                  # gentle ramp up (not a step) to avoid overshoot
                self._apply_phase_flow(rate, ramp_s, f"resume→{well_id}")
            else:
                try:
                    self.line.start_flow_single(rate / n, direction=self.cfg["direction"])
                except Exception:
                    pass
                self._phase_flow_rate = rate
                self.status_msg.emit(f"[ff] pump resumed {rate:g} µL/min (in {well_id}).")

        QTimer.singleShot(pause_ms, do_pause)
        QTimer.singleShot(resume_ms, do_resume)

    def set_experiment_phase(self, phase, n_wells=0):
        """Called by MABIP: phase in {'idle','buffer','well'} with well count.
        Drives flow-follows-wells (ramp to per-phase rate) and auto-burst."""
        prev = self._phase
        self._phase = phase
        self._n_wells = int(n_wells)
        try:
            self.lbl_phase.setText(f"phase: {phase}  (wells={n_wells})")
        except Exception:
            pass
        # flow-follows-wells: ramp the pump to the per-phase rate on a transition
        if self._exp_follow and self.line is not None and phase != prev:
            if phase == "buffer":
                self._apply_phase_flow(float(self.cfg.get("exp_buffer_rate", 20.0)),
                                       float(self.cfg.get("exp_recover_ramp", 0.0)), "→buffer")
            elif phase == "well":
                self._apply_phase_flow(float(self.cfg.get("exp_well_rate", 50.0)),
                                       float(self.cfg.get("exp_approach_ramp", 0.0)), "→well")
        # auto-burst once per buffer entry
        if phase == "buffer" and self._n_wells >= 1 and self._auto_burst \
                and self.line is not None and not self._busy and not self._burst_done:
            self._burst_done = True
            self._trigger_burst(auto=True)
        if phase != "buffer":
            self._burst_done = False
        # safety net: the well must not start mid-burst or before the flow has
        # resettled. With the backflow + fit-to-buffer logic this shouldn't trip;
        # if it does, the buffer window is too short for the configured burst.
        if phase == "well" and prev == "buffer" and self._auto_burst:
            if self._bursting or not self._burst_settled:
                self.status_msg.emit(
                    "⚠ Well started before the burst finished resettling — flow may "
                    "not be at baseline. Increase buffer_time or shorten the burst.")
        # hold the run's baseline flow (e.g. 80 µL/min) through the whole well, so
        # recording always happens at the set rate even if a buffer burst left the
        # flow slightly off. Skipped when flow-follows-wells or feed-forward is
        # already driving the pump. Fires once per well entry (prev != "well").
        if phase == "well" and prev != "well" and self.line is not None \
                and not self._exp_follow and not self._ff_enabled:
            rate = self._num(self.f_rate, 0.0)
            if rate > 0:
                try:
                    self.line.start_flow_single(rate / self._nsyr(),
                                                direction=self.cfg["direction"])
                    self._phase_flow_rate = rate
                    self.status_msg.emit(f"Holding {rate:g} µL/min through the well.")
                except Exception as e:
                    self.status_msg.emit(f"well-hold flow error: {e}")

    def _apply_phase_flow(self, target_line_rate, ramp_s, label=""):
        """Move the COMBINED flow toward `target_line_rate` over `ramp_s` seconds
        (0 = step change). Runs in its own thread; a newer call cancels this one.
        Uses per-syringe = rate / N like every other command."""
        n = self._nsyr()
        start = self._phase_flow_rate if self._phase_flow_rate is not None else target_line_rate
        self._phase_flow_gen += 1
        gen = self._phase_flow_gen
        self.status_msg.emit(f"Flow {label}: {start:g} → {target_line_rate:g} µL/min "
                             f"({'ramp %gs' % ramp_s if ramp_s > 0 else 'step'}).")
        def worker():
            steps = max(1, min(30, int(round(ramp_s)))) if ramp_s and ramp_s > 0 else 1
            for i in range(1, steps + 1):
                if gen != self._phase_flow_gen:      # a newer transition took over
                    return
                r = start + (target_line_rate - start) * i / steps
                try:
                    self.line.start_flow_single(r / n, direction=self.cfg["direction"])
                except Exception:
                    pass
                self._phase_flow_rate = r
                if i < steps:
                    time.sleep(max(0.2, ramp_s / steps))
            self._phase_flow_rate = target_line_rate
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------- burst/prime
    def _cfgf(self, key, default):
        """Config value as float, tolerant of blanks/garbage."""
        try:
            return float(self.cfg.get(key, default))
        except (ValueError, TypeError):
            return float(default)

    def _buffer_window_s(self):
        """Seconds the run dwells in the buffer between wells (the window a burst
        must finish and re-stabilise inside). Read from the Sampling tab's state;
        falls back to 60 s if unavailable."""
        g = self.main_gui
        for attr in ("app_state",):
            st = getattr(g, attr, None) if g is not None else None
            if st is not None and getattr(st, "t_buffer", None):
                try:
                    return float(st.t_buffer)
                except (ValueError, TypeError):
                    pass
        return 60.0

    def _settle_budget(self, line_rate, mult=1.0, backflow_s=None):
        """Rate-scaled estimate of the seconds needed to return to a steady
        baseline after a burst: `base + k/rate` (compliant tubing settles slowly
        at low flow). A higher multiplier overshoots more, adding
        `overshoot_s*(mult-1)`; the backflow pulse cancels a fraction
        (`b_backflow_relief`) of that term, scaled by its duration vs the default
        5 s — so BOTH levers (lower multiplier / add backflow) genuinely shorten the
        budget. Clamped to [6, 45] s. Coefficients are defaults — trim them from a
        Tier-1 burst-characterisation run."""
        base = self._cfgf("b_settle_base_s", 8.0)
        k = self._cfgf("b_settle_k", 300.0)
        os_s = self._cfgf("b_settle_overshoot_s", 10.0)
        relief_frac = self._cfgf("b_backflow_relief", 0.8)
        if backflow_s is None:
            backflow_s = self._cfgf("b_backflow_s", 5.0)
        r = max(1e-6, abs(line_rate))
        relief = relief_frac * min(1.0, backflow_s / 5.0)        # full relief at >=5 s backflow
        overshoot = os_s * max(0.0, mult - 1.0) * (1.0 - relief)
        return max(6.0, min(45.0, base + k / r + overshoot))

    def _await_steady(self, expected, tol=None, hold_s=None, timeout_s=None):
        """Block (in the worker thread) until the measured line flow stays within
        ±tol of `expected` for `hold_s` continuous seconds, or `timeout_s` elapses.
        Returns (ok: bool, elapsed_s: float). No sensor -> (False, 0.0). Honours
        STOP via self._abort."""
        if self.latest_flow is None:
            return (False, 0.0)
        tol = self._cfgf("b_settle_tol", 5.0) if tol is None else tol
        hold_s = self._cfgf("b_settle_hold", 4.0) if hold_s is None else hold_s
        if timeout_s is None:
            timeout_s = self._settle_budget(expected)
        t0 = time.monotonic()
        in_band_since = None
        while (time.monotonic() - t0) < timeout_s:
            if self._abort:
                return (False, time.monotonic() - t0)
            f = abs(self.latest_flow or 0.0)
            if abs(f - abs(expected)) <= tol:
                if in_band_since is None:
                    in_band_since = time.monotonic()
                if (time.monotonic() - in_band_since) >= hold_s:
                    return (True, time.monotonic() - t0)
            else:
                in_band_since = None
            time.sleep(0.1)
        return (False, time.monotonic() - t0)

    def _burst(self):
        if self._guard():
            return
        if self._phase == "well":
            self.status_msg.emit("Burst blocked: recording a well. Allowed only in buffer/idle.")
            return
        if self._phase == "buffer" and self._n_wells < 1:
            self.status_msg.emit("Burst blocked: no active wells (wells<1).")
            return
        if self._clog_flagged:      # the operator answered the banner — stand it down
            self._clog_flagged = False
            self._set_clog_ui(False)
            self._clog_since = None
        self._trigger_burst(auto=False)

    def try_clear_burst(self) -> bool:
        """Fire one clog-clearing burst, but ONLY while parked in the buffer.

        Driven by the blockage hold loop, which calls this repeatedly while the
        plate is held. Refuses unless we are in the buffer phase — a burst mid-well
        would ruin that well's reading, and the whole point of holding is that the
        needle is parked in buffer, not in a well. Also refuses while a burst is
        already running (``_bursting`` is set synchronously by ``_trigger_burst``,
        so the loop cannot stack them) and if no line rate is set to boost from.

        Returns True if a burst was actually started, False if declined."""
        if self.line is None or self._busy or self._bursting:
            return False
        if self._phase != "buffer":            # buffer only — never mid-well
            return False
        if self._num(self.f_rate, 0.0) <= 0:   # nothing to boost from
            return False
        self._trigger_burst(auto=True)
        return True

    # ---------------------------------------------------- clog-clear escalation
    def request_clog_clear(self, source="auto") -> bool:
        """Start the two-strategy clog-clearing escalation once.

        Idempotent — returns False (and does nothing) if a sequence is already
        running, the pump is missing/busy, we're recording a well, or no baseline
        rate is set. Safe to call repeatedly from a watch loop.

        Escalation:
          1. Strategy 1 — run `clear_fwd_bursts` hard FORWARD bursts to completion.
          2. Wait `clear_wait_s` for the line to recover.
          3. Strategy 2 — if still blocked, REVERSE-push (infuse the opposite way)
             for a fixed time, up to `rev_attempts` times, firing one forward burst
             to re-establish flow the moment it moves.
        The needle must be parked in the buffer or idle (never mid-well) — enforced
        here. Whichever strategy works is recorded in `_last_clear_method`, then the
        baseline flow rate is restored.

        NOT YET VERIFIED ON THE RIG — the reverse-push maneuver expels a small bolus
        back into the buffer; watch the first runs by hand."""
        if self.line is None or self._busy or self._bursting or self._clearing:
            return False
        if self._phase == "well":              # never while recording a well
            return False
        base = self._num(self.f_rate, 0.0)
        if base <= 0:                          # nothing to boost from / restore to
            return False
        self._clear_baseline = base
        self._abort_clear = False
        self._clearing = True
        self._last_clear_method = None
        self.status_msg.emit(f"[clog] clearing started ({source}) — baseline {base:g} µL/min.")
        threading.Thread(target=self._clog_clear_worker, args=(source,), daemon=True).start()
        return True

    def abort_clog_clear(self):
        """Unwind the escalation (called on STOP or when a run resumes)."""
        if self._clearing:
            self._abort_clear = True

    def _clear_sleep(self, seconds) -> bool:
        """Sleep in small steps; return False if aborted mid-sleep."""
        t0 = time.monotonic()
        while (time.monotonic() - t0) < seconds:
            if self._abort_clear or self._abort:
                return False
            time.sleep(0.2)
        return True

    def _wait_burst_done(self):
        """Block until the forward burst started by _trigger_burst finishes."""
        while self._bursting and not (self._abort_clear or self._abort):
            time.sleep(0.2)

    def _flow_recovered(self) -> bool:
        """True if the flow sensor shows flow back near baseline. No sensor -> False
        (can't judge here; the metabolite hold loop decides that case instead)."""
        if self.latest_flow is None or self.line is None:
            return False
        expected = self.line.expected_combined(self._clear_baseline / self._nsyr())
        ok, _ = self._await_steady(expected, hold_s=2.0,
                                   timeout_s=self._cfgf("clear_settle_s", 8.0))
        return ok

    def _reverse_push(self):
        """Strategy 2: push the OPPOSITE way for a fixed time at a set rate, then
        stop and resume forward baseline. Expels a small bolus back into the buffer
        to dislodge the clog. Bounded volume = rev_rate * rev_time_s."""
        n = self._nsyr()
        base = self._clear_baseline
        rev_line = self._cfgf("rev_rate", 0.0) or base
        rev_s = self._cfgf("rev_time_s", 10.0)
        fwd = self.cfg["direction"]
        rev = "infuse" if fwd == "withdraw" else "withdraw"
        vol = rev_line * rev_s / 60.0
        self._busy = True; self._set_busy(True)
        try:
            self.status_msg.emit(f"[clog] reverse-push: {rev_line:g} µL/min {rev} for "
                                 f"{rev_s:g}s (~{vol:g} µL back into buffer).")
            self.line.start_flow_single(rev_line / n, direction=rev)
            self._clear_sleep(rev_s)
            self.line.stop()
            if not (self._abort_clear or self._abort):
                # resume forward baseline so _flow_recovered measures the real direction
                self.line.start_flow_single(base / n, direction=fwd)
        except Exception as e:
            self.status_msg.emit(f"[clog] reverse-push error: {e}")
        finally:
            self._busy = False; self._set_busy(False)

    def _clog_clear_worker(self, source):
        n_fwd = max(1, int(self._cfgf("clear_fwd_bursts", 2.0)))
        wait_s = self._cfgf("clear_wait_s", 60.0)
        n_rev = max(0, int(self._cfgf("rev_attempts", 3.0)))
        try:
            # ---- Strategy 1: forward bursts to completion --------------------
            for i in range(1, n_fwd + 1):
                if self._abort_clear or self._abort:
                    return
                self.status_msg.emit(f"[clog] forward burst {i}/{n_fwd}…")
                self._trigger_burst(auto=False)         # full-strength, own _work job
                self._wait_burst_done()
                if self._flow_recovered():
                    self._finish_clear("forward burst"); return

            # ---- wait, then re-check ----------------------------------------
            self.status_msg.emit(f"[clog] forward bursts done — waiting {wait_s:g}s "
                                 "before reversing…")
            if not self._clear_sleep(wait_s):
                return
            if self._flow_recovered():
                self._finish_clear("forward burst"); return

            # ---- Strategy 2: reverse-push attempts --------------------------
            for a in range(1, n_rev + 1):
                if self._abort_clear or self._abort:
                    return
                self.status_msg.emit(f"[clog] reverse-push attempt {a}/{n_rev}…")
                self._reverse_push()
                if self._abort_clear or self._abort:
                    return
                if self._flow_recovered():
                    self.status_msg.emit("[clog] flow moving — forward burst to re-establish.")
                    self._trigger_burst(auto=False)
                    self._wait_burst_done()
                    self._finish_clear("reverse push"); return

            self._finish_clear(None)     # exhausted both strategies
        finally:
            self._clearing = False

    def _finish_clear(self, method):
        """Record which strategy worked and restore the baseline forward flow."""
        self._last_clear_method = method
        n = self._nsyr()
        try:
            if self.line is not None and self._clear_baseline > 0 \
                    and not (self._abort_clear or self._abort):
                self.line.start_flow_single(self._clear_baseline / n,
                                            direction=self.cfg["direction"])
                # Re-arm from scratch: the line has just been burst and reversed, so
                # it has to re-establish flow before "low" means anything again.
                self._arm_clog_watch(self.line.expected_combined(self._clear_baseline / n))
        except Exception as e:
            self.status_msg.emit(f"[clog] restore-flow error: {e}")
        if method:
            self.is_clogged = False
            self._clog_flagged = False
            self._set_clog_ui(False)
            self.status_msg.emit(f"✓ [clog] cleared by {method}. Restored "
                                 f"{self._clear_baseline:g} µL/min.")
            try:
                self.clog_changed.emit(False)
            except Exception:
                pass
        else:
            self._set_clog_ui(True, "⚠  CLOG NOT CLEARED — forward bursts and reverse "
                                    "pushes both failed. Needs a manual prime/clear.")
            self.status_msg.emit("⚠ [clog] NOT cleared after forward bursts + reverse "
                                 "pushes — needs a manual prime/clear.")

    def _burst_constants(self):
        """Measured burst-protocol constants for the CURRENT syringe count, from
        burst_calibration.json (created by headless_rig.py calib). None if this
        syringe count hasn't been calibrated yet."""
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "burst_calibration.json")
        try:
            return json.load(open(path)).get(str(self._nsyr()))
        except Exception:
            return None

    def _trigger_burst(self, auto=False):
        if self.line is None or self._busy:
            return
        line_base = self._num(self.f_rate, 0.0)          # combined/line baseline
        if line_base <= 0:
            self.status_msg.emit("Set a flow rate before bursting."); return
        n = self._nsyr()
        cal = self._burst_constants()
        margin = self._cfgf("b_fit_margin_s", 3.0)
        window = self._buffer_window_s() if auto else None
        has_sensor = self.latest_flow is not None
        mbase = line_base / n                            # per-syringe baseline

        if cal:
            # === CALIBRATED protocol: ABSOLUTE rates, no multipliers ===========
            # Measured on this rig: only hard overdrive moves the line (knee
            # ~340-360), and the descent is discharge-dominated, so the recipe is
            # the same regardless of the baseline flow rate:
            #   up      : up_cmd until the sensor reads up_trigger (or up_max_s)
            #   reverse : rev_cmd until the sensor is back near baseline
            #             (baseline + rev_stop_offset), max rev_max_s
            #   level   : resume the baseline command
            mult = None
            hs = float(cal.get("up_max_s", 25.0)); ss = 0.0
            bf_s = float(cal.get("rev_max_s", 25.0))
            mhigh = float(cal.get("up_cmd", 400.0)) / n
            mback = float(cal.get("rev_cmd", 280.0)) / n
            trig = float(cal.get("up_trigger", 115.0))          # sensor: burst peak
            bf_trig = line_base + float(cal.get("rev_stop_offset", 8.0))
            typical = float(cal.get("expected", {}).get("typical_total_burst_s", 43.0))
            if auto and window and window < typical + margin + 10.0:
                self.status_msg.emit(
                    f"⚠ Buffer {window:g}s is tight for the calibrated burst "
                    f"(typically ~{typical:g}s + settle). Consider a longer buffer_time.")
            self.status_msg.emit(
                f"{'[auto] ' if auto else ''}Burst [calibrated, n={n}] → "
                f"{mhigh*n:g} µL/min until sensor ≥ {trig:g} (max {hs:g}s), reverse "
                f"{mback*n:g} until ≤ {bf_trig:g} (max {bf_s:g}s), resume {line_base:g}.")
        else:
            # === legacy multiplier protocol (uncalibrated syringe counts) ======
            try:
                mult = float(self.cfg.get("b_mult", 1.7))
                hs = float(self.cfg["b_high_s"]); ss = float(self.cfg["b_stop_s"])
            except (ValueError, KeyError):
                self.status_msg.emit("Set burst values in Definitions."); return
            bf_s = self._cfgf("b_backflow_s", 5.0)
            bf_rate_line = self._cfgf("b_backflow_rate", 0.0) or line_base
            if hs + ss + bf_s > 60:
                self.status_msg.emit("Burst high+stop+backflow must total ≤ 60 s."); return
            adjusted = False
            if auto and window:
                def total_at(m):
                    return hs + ss + bf_s + self._settle_budget(line_base, m, bf_s) + margin
                if total_at(mult) > window:
                    while mult > 1.05 and total_at(mult) > window:
                        mult = round(mult - 0.1, 2)
                    adjusted = True
                if total_at(mult) > window:
                    self.status_msg.emit(
                        f"⚠ Burst may not resettle within the {window:g}s buffer even at "
                        f"{mult:g}× (need ~{total_at(mult):.0f}s). Shorten high/stop or "
                        f"raise buffer_time.")
            mhigh = mult * mbase
            mback = bf_rate_line / n
            trig = self.line.expected_combined(mbase)      # sensor ≈ line_base
            bf_trig = None
            mode = (f"until flow recovers to {trig:g}" if has_sensor
                    else f"for {hs:g}s (no sensor — timed)")
            self.status_msg.emit(
                f"{'[auto] ' if auto else ''}Burst → {mult*line_base:g} µL/min line "
                f"({mult:g}× {line_base:g}){' [rate lowered to fit buffer]' if adjusted else ''} "
                f"{mode}, stop {ss:g}s, backflow {bf_s:g}s, resume {line_base:g}.")

        self._abort = False; self._bursting = True; self._steady = False
        self._burst_settled = False       # set True only once resettle is confirmed
        seg0 = self._seg_label
        def fn():
            try:
                rep = self.line.burst(
                    mbase, mhigh, trig, hs, ss,
                    read_flow=lambda: self.latest_flow or 0.0,
                    direction=self.cfg["direction"],
                    backflow_rate=mback, backflow_seconds=bf_s,
                    backflow_trigger_flow=bf_trig,
                    on_phase=lambda p: (setattr(self, "_seg_label", f"{seg0}_burst_{p}" if seg0 else f"burst_{p}"),
                                        self.status_msg.emit(f"burst: {p} phase")),
                    should_abort=lambda: self._abort)
                self.status_msg.emit(
                    f"Burst done ({'flow recovered' if rep['triggered'] else 'timed out'}, "
                    f"peak {rep['peak_flow']:.1f} µL/min, {rep['total_s']:.1f}s"
                    f"{', backflow' if rep.get('backflow') else ''}). Resumed {line_base:g}.")
                # Confirm the sensor is actually back at BASELINE before we let the
                # run leave the buffer. For an auto-burst the budget is capped by
                # whatever buffer time is left after the pulse.
                if self.line is not None and not self._abort and has_sensor:
                    resettle_target = self.line.expected_combined(mbase)  # ≈ baseline
                    if cal:
                        budget = float(cal.get("level_settle_s", 10.0)) + 5.0
                    else:
                        budget = self._settle_budget(line_base, mult or 1.0, bf_s)
                    if auto and window:
                        budget = min(budget, max(2.0, window - rep['total_s'] - margin))
                    self._seg_label = f"{seg0}_burst_settle" if seg0 else "burst_settle"
                    ok, t_steady = self._await_steady(resettle_target, timeout_s=budget)
                    self._burst_settled = ok
                    self.status_msg.emit(
                        f"Burst resettle: {'steady' if ok else '⚠ NOT steady'} at "
                        f"{resettle_target:g} µL/min in {t_steady:.1f}s (budget {budget:.0f}s).")
                else:
                    # No sensor to verify against (timed burst) -> can't confirm;
                    # assume settled so the safety net doesn't false-alarm.
                    self._burst_settled = True
            finally:
                self._seg_label = seg0
                self._bursting = False
                if self._abort:
                    self._burst_settled = True   # user STOPped; don't warn downstream
                if self.line is not None and not self._abort:
                    self._arm_clog_watch(self.line.expected_combined(mbase))
        self._work(fn)

    def _wait_run(self, volume, rate, label="Running"):
        """Wait the COMPUTED run time (volume ÷ rate) with a live countdown — an
        internal clock, so it doesn't depend on the pump's flaky elapsed-time
        reporting. Interruptible via STOP. Returns True if it completed."""
        secs = (abs(volume) / abs(rate) * 60.0) if rate else 0.0
        secs += 1.0  # small settle margin
        end = time.monotonic() + secs
        while not self._abort:
            left = end - time.monotonic()
            if left <= 0:
                return True
            self.status_msg.emit(f"{label} — {left:.0f}s left…")
            time.sleep(min(1.0, left))
        return False

    def _prime(self):
        """Wet prime. If the PULL volume is 0, skip the pull + reconnect step and
        just PUSH (fluid already loaded). Otherwise:
        1) with EMPTY syringes + thick pull tubing, PULL liquid in,
        2) wait for the user to reconnect the hose to the final config,
        3) PUSH it all out to wet the line — then it's ready to pull.
        """
        if self._guard():
            return
        v_line = float(self.cfg.get("prime_pull_vol", 600.0))     # combined/line
        r_line = float(self.cfg.get("prime_pull_rate", 1000.0))
        self._abort = False; self._steady = False
        if v_line <= 0:
            # Pull disabled -> push-only prime (no pull, no reconnect prompt).
            self.status_msg.emit("Prime: pull volume is 0 → push-only.")
            self._prime_push()
            return
        n = self._nsyr(); p1v, p1r = v_line / n, r_line / n       # per-syringe
        eta = (p1v / p1r * 60.0) if p1r else 0.0
        self.status_msg.emit(f"Prime 1/2: PULLING {v_line:g} µL @ {r_line:g} µL/min (line; "
                             f"{p1v:g} µL/syringe) ~{eta:.0f}s. Empty syringes + thick tubing.")
        def part1():
            self.line.deliver_single(p1v, p1r, direction="withdraw")
            self._wait_run(p1v, p1r, "Prime 1/2 pulling")
            if not self._abort:
                self.prime_confirm.emit()   # prompt on the GUI thread
            else:
                self.status_msg.emit("Prime stopped during part 1.")
        self._work(part1)

    def _prime_confirm(self):
        """GUI-thread: part 1 done — ask the user to reconnect, then push."""
        ok = QMessageBox.question(
            self, "Prime — reconnect the hose",
            "Part 1 complete — liquid is pulled into the syringes.\n\n"
            "Reconnect the hose to the FINAL configuration, then click OK to push it "
            "all out and finish priming.\n\n(Cancel to stop.)",
            QMessageBox.Ok | QMessageBox.Cancel) == QMessageBox.Ok
        if not ok:
            self.status_msg.emit("Prime cancelled after part 1 (nothing pushed out).")
            return
        self._prime_push()

    def _prime_push(self):
        """Push the prime_push_vol out to wet the line (used by both the 2-step
        prime and the push-only prime)."""
        v_line = float(self.cfg.get("prime_push_vol", 600.0))     # combined/line
        r_line = float(self.cfg.get("prime_push_rate", 200.0))
        n = self._nsyr(); p2v, p2r = v_line / n, r_line / n       # per-syringe
        self._abort = False
        self.status_msg.emit(f"Prime push: PUSHING {v_line:g} µL @ {r_line:g} µL/min (line; "
                             f"{p2v:g} µL/syringe)…")
        def push():
            self.line.deliver_single(p2v, p2r, direction="infuse")
            self._wait_run(p2v, p2r, "Prime pushing")
            if not self._abort:
                self.status_msg.emit("✓ Priming complete — line wet, ready to pull.")
        self._work(push)

    def _metabolites_running(self):
        """True when MABIP's SIX metabolite sensor is actively recording."""
        mg = self.main_gui
        return bool(mg and getattr(mg, "sensor_reader", None)
                    and getattr(mg.sensor_reader, "is_running", False))

    # --------------------------------------------------------------- shutdown
    def safe_shutdown(self, reason="") -> list:
        """Bring the flow hardware to a safe state. Idempotent; never raises.

        Order matters and is the whole point of this method:

        1. Raise the abort flags FIRST. Worker threads (ramp, burst, clog-clear,
           experiment phases) check them between steps; stopping the pump while
           one is still running just means it commands flow again a second later.
           The generation counters kill any in-flight phase ramp / feed-forward
           resume timer for the same reason.
        2. Stop the poll timer — no point reading a sensor we are about to close.
        3. STOP the pump, and keep stopping it: this rig's FTDI link drops
           commands (see the EMI notes), and a dropped stop leaves syringes
           driving into a closed line. Sent, given the workers a moment to
           unwind, then sent again and verified.
        4. Only then release the sensor and the serial port.

        Returns a list of human-readable lines describing what it did, for the
        caller to log — a shutdown that silently half-worked is the failure mode
        this is meant to prevent."""
        if getattr(self, "_shutdown_done", False):
            return []
        self._shutdown_done = True
        notes = []

        self._abort = True
        self._abort_clear = True
        self._steady = False
        self._bursting = False
        self._auto_burst = False
        self._exp_follow = False
        self._ff_enabled = False
        self._phase_flow_gen += 1
        self._ff_gen += 1
        self._phase = "idle"

        try:
            self.poll_timer.stop()
        except Exception:
            pass

        if self.line is None:
            notes.append("pump not connected — nothing to stop")
            return notes

        # 3. Stop, wait for any worker to notice the abort, stop again.
        stopped = self._insist_stop()
        notes.append("pump STOP sent" if stopped else
                     "pump STOP FAILED — check the pump is off at the front panel")
        deadline = time.monotonic() + 2.0
        while self._busy and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._busy:
            notes.append("a flow worker was still running — stop re-sent over it")
        if self._insist_stop():
            notes.append("pump confirmed stopped")

        # 4. Release the hardware. close() stops once more on its way out.
        try:
            self.line.disconnect_sensor()
            notes.append("flow sensor released")
        except Exception as e:
            notes.append(f"flow sensor release error: {e}")
        try:
            self.line.close()
            notes.append("pump port closed")
        except Exception as e:
            notes.append(f"pump close error: {e}")
        self.line = None
        self.latest_flow = None

        try:
            self._set_pill(self.pill_pump, False, "pump")
            self._set_pill(self.pill_sensor, False, "sensor")
            self.btn_pump.setText("Connect pump")
            self.btn_sensor.setText("Connect sensor"); self.btn_sensor.setEnabled(False)
            self._refresh_actions()
        except Exception:
            pass

        if reason:
            notes.append(f"({reason})")
        return notes

    def _insist_stop(self, tries=3):
        """Send STOP up to `tries` times; True once one gets through.

        One dropped write on this link is normal, so a single failed stop is not
        evidence the pump is still running — but it is not evidence it stopped
        either, which is why this retries rather than reporting the first error."""
        for i in range(tries):
            try:
                self.line.stop()
                return True
            except Exception:
                time.sleep(0.15)
        return False

    def shutdown(self):
        """Back-compat alias — the full sequence, same guarantees."""
        return self.safe_shutdown()
