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
_HW = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "hardware")
_PC = "/home/rpi/pumpcontrol-project"
for p in (_HW, os.path.join(_PC, "fgt-SDK", "Python"), _PC):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox,
    QPushButton, QLineEdit, QLabel, QComboBox, QFrame, QCheckBox,
    QDialog, QDialogButtonBox, QButtonGroup, QMessageBox, QFileDialog, QPlainTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ACCENT = "#2f81f7"; RED = "#e5484d"; GREEN = "#2ea043"; AMBER = "#d9a406"; MUTED = "#6b7785"


from experiment_parse import parse_experiment, num_list as _num_list


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
        ("prime_pull_vol", "Prime pull vol (µL)"), ("prime_pull_rate", "Prime pull rate (µL/min)"),
        ("prime_push_vol", "Prime push vol (µL)"), ("prime_push_rate", "Prime push rate (µL/min)"),
        ("exp_buffer_rate", "Well-flow: buffer rate"), ("exp_well_rate", "Well-flow: well rate"),
        ("exp_approach_ramp", "Well-flow: ramp→well (s)"), ("exp_recover_ramp", "Well-flow: ramp→buffer (s)"),
        ("ff_pause_s", "Feed-fwd: pause after move (s)"), ("ff_resume_s", "Feed-fwd: resume after move (s)"),
        ("ff_resume_ramp", "Feed-fwd: resume ramp (s)"),
        ("exp_settle_max_var", "Max flow variation (µL/min)"), ("exp_settle_hold", "Settle hold (s)"),
        ("exp_settle_timeout", "Settle timeout (s)"), ("exp_settle_bump", "Settle raise step (%)"),
    ]

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flow — Definitions / Settings")
        self.setMinimumWidth(330)
        self.cfg = dict(cfg); self.w = {}
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 16, 16, 16); lay.setSpacing(8)
        form = QFormLayout(); form.setVerticalSpacing(7)

        self.w["port"] = QLineEdit(str(cfg["port"])); form.addRow("Pump port", self.w["port"])
        self.w["diameter"] = QLineEdit(str(cfg["diameter"])); form.addRow("Syringe Ø (mm)", self.w["diameter"])
        seg = QWidget(); sh = QHBoxLayout(seg); sh.setContentsMargins(0, 0, 0, 0); sh.setSpacing(6)
        self._syr = {}
        self._syr_group = QButtonGroup(seg); self._syr_group.setExclusive(True)
        for n in (1, 2):
            b = QPushButton(str(n)); b.setCheckable(True); b.setFixedWidth(46)
            b.setStyleSheet("QPushButton:checked{background:#2f81f7;color:white;font-weight:700;}")
            b.setChecked(int(cfg["n"]) == n)
            self._syr_group.addButton(b); sh.addWidget(b); self._syr[n] = b
        sh.addStretch(1); form.addRow("# syringes", seg)
        for key, label in self._FIELDS:
            self.w[key] = QLineEdit(str(cfg[key])); form.addRow(label, self.w[key])
        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

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
        out["n"] = 2 if self._syr[2].isChecked() else 1
        return out


class FlowControlTab(QWidget):
    """Flow control + live plot + clog detection, as a MABIP tab."""

    clog_changed = pyqtSignal(bool)
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
        self._steady = False             # pump in steady flow (clog check active)
        self._expected = 0.0
        self._clog_since = None
        self._busy = False
        self._abort = False
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
            "port": "auto", "diameter": 19.13, "n": 1, "direction": "withdraw",
            "settle": 30.0, "measure": 15.0, "window": 120.0,
            "r_start": 5.0, "r_max": 60.0, "r_step": 5.0, "r_dwell": 20.0, "r_tol": 5.0,
            "clog_frac": 0.4, "clog_seconds": 6.0,
            "metab_window": 12.0, "metab_thresh": 0.15,
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
            "exp_settle": True, "exp_settle_max_var": 5.0, "exp_settle_hold": 5.0,
            "exp_settle_timeout": 120.0, "exp_settle_bump": 5.0,
        }
        self._load_cfg()   # restore saved definitions/fields over the defaults

        self.status_msg.connect(lambda s: self.lbl_status.setText(s))
        self.status_msg.connect(self._console_log)
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
        conn1 = QHBoxLayout()
        self.pill_pump = QLabel()
        self.btn_pump = QPushButton("Connect pump")
        self.btn_pump.setStyleSheet(f"QPushButton{{background:{ACCENT};color:white;}}")
        self.btn_pump.clicked.connect(self._connect_pump)
        self.btn_pump.setToolTip("Connect / disconnect the Chemyx syringe pump. Works with NO flow sensor.")
        conn1.addWidget(self.pill_pump); conn1.addStretch(1); conn1.addWidget(self.btn_pump)
        left.addLayout(conn1)
        conn2 = QHBoxLayout()
        self.pill_sensor = QLabel()
        self.btn_sensor = QPushButton("Connect sensor")
        self.btn_sensor.clicked.connect(self._connect_sensor)
        self.btn_sensor.setToolTip("Connect / disconnect the Fluigent flow sensor (OPTIONAL). "
                                   "Enables live plotting, flow logging, clog detection, and Ramp/Calibrate/Verify.")
        conn2.addWidget(self.pill_sensor); conn2.addStretch(1); conn2.addWidget(self.btn_sensor)
        left.addLayout(conn2)
        self._set_pill(self.pill_pump, False, "pump")
        self._set_pill(self.pill_sensor, False, "sensor")

        params = QGroupBox("Run parameters")
        pf = QFormLayout(params); pf.setVerticalSpacing(6)
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
        self.f_target.textChanged.connect(self._sync_target)
        for _f in (self.f_rate, self.f_vol, self.f_target):
            _f.editingFinished.connect(self._save_cfg)   # remember the run params
        left.addWidget(params)

        # push/pull direction, right on the main panel
        self.chk_pull = QCheckBox("Pull (withdraw)   —   unchecked = push / infuse")
        self.chk_pull.setChecked(self.cfg["direction"] == "withdraw")
        self.chk_pull.setToolTip("Direction for Start / Run / Burst: checked = PULL (withdraw), unchecked = PUSH (infuse).")
        self.chk_pull.toggled.connect(self._on_pull_toggled)
        left.addWidget(self.chk_pull)

        grid = QGridLayout(); grid.setSpacing(6); self.btn = {}
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
        self.chk_ff = QCheckBox("Pause pump during moves (feed-forward)")
        self.chk_ff.setToolTip("On each well move: PAUSE the pump ~0.45 s after the move command "
                               "(tip lifts out of liquid) and RESUME ~10 s later (tip back in liquid) — "
                               "kills the air surge. Runs at the Flow rate; calibrated, needs NO sensor.")
        self.chk_ff.toggled.connect(lambda v: setattr(self, "_ff_enabled", bool(v)))
        left.addWidget(self.chk_ff)
        self.chk_settle = QCheckBox("Settle to specified rate before each run")
        self.chk_settle.setToolTip("Before an experiment run starts, command the pump to the run's flow "
                                   "rate and wait for the SENSOR to actually reach it — if the measured "
                                   "flow is low, raise the pump until it hits the target, then start the "
                                   "wells. Needs the flow sensor connected. (Definitions: tolerance/hold/timeout.)")
        self.chk_settle.setChecked(bool(self.cfg.get("exp_settle", True)))
        self.chk_settle.toggled.connect(lambda v: self.cfg.__setitem__("exp_settle", bool(v)))
        left.addWidget(self.chk_settle)
        self.chk_follow = QCheckBox("Flow follows wells (buffer↔well)")
        self.chk_follow.setToolTip("During a well-plate run, drive the pump to the buffer-rate and "
                                   "well-rate (with ramps) from Definitions, instead of a single Start rate. "
                                   "Ramps happen on each buffer→well and well→buffer transition.")
        self.chk_follow.toggled.connect(lambda v: setattr(self, "_exp_follow", bool(v)))
        left.addWidget(self.chk_follow)
        self.chk_auto = QCheckBox("Auto-burst in buffer (during a run)")
        self.chk_auto.setToolTip("Automatically fire one Burst each time the run enters the buffer (needs ≥1 well).")
        self.chk_auto.toggled.connect(lambda v: setattr(self, "_auto_burst", bool(v)))
        left.addWidget(self.chk_auto)
        self.lbl_phase = QLabel("phase: idle")
        self.lbl_phase.setStyleSheet(f"color:{MUTED};")
        left.addWidget(self.lbl_phase)

        self.btn_defs = QPushButton("⚙  Definitions / Settings…")
        self.btn_defs.setToolTip("Rarely-changed settings: port, syringe Ø, # syringes, direction, ramp / clog / burst / prime parameters. Saved automatically.")
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

        self.fig = Figure(figsize=(5, 3)); self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("time (s)"); self.ax.set_ylabel("flow (µL/min)")
        self.ax.grid(True, alpha=0.25)
        (self.trace,) = self.ax.plot([], [], color=ACCENT, lw=1.6)
        self.hline = self.ax.axhline(0.0, color=RED, ls="--", lw=1.0)
        right.addWidget(self.canvas, 1)

        # Save-log row directly under the graph
        saverow = QHBoxLayout()
        self.btn_savelog = QPushButton("💾  Save flow log…")
        self.btn_savelog.setToolTip("Save a copy of this run's flow-rate log (elapsed, flow, air) "
                                    "to a file you choose. Also auto-logs to Sensor_Readings/, and "
                                    "appends to the metabolite file when the sensor is recording.")
        self.btn_savelog.clicked.connect(self._save_log)
        saverow.addStretch(1); saverow.addWidget(self.btn_savelog)
        right.addLayout(saverow)

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
            "font-family:monospace;font-size:11px;}")
        self.console.setPlaceholderText("Activity log — connect, load an experiment, press Run…")
        right.addWidget(self.console)
        outer.addLayout(right, 1)

        self._sync_target(); self._refresh_actions()

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

    def _on_pull_toggled(self, checked):
        self.cfg["direction"] = "withdraw" if checked else "infuse"
        if self.line is not None:
            self.line.direction = self.cfg["direction"]
        self._save_cfg()

    # ------------------------------------------------------- persist settings
    def _load_cfg(self):
        """Restore saved definitions + run params over the defaults, so nothing
        resets between launches."""
        import json, os
        self._cfg_path = os.path.expanduser("~/.mabip/flow_settings.json")
        self._saved_fields = {}
        try:
            with open(self._cfg_path) as f:
                saved = json.load(f)
            for k, v in (saved.get("cfg") or {}).items():
                if k in self.cfg:
                    self.cfg[k] = v
            self._saved_fields = saved.get("fields") or {}
        except Exception:
            pass

    def _save_cfg(self):
        """Persist the current definitions + run params (called on every change)."""
        import json, os
        try:
            os.makedirs(os.path.dirname(self._cfg_path), exist_ok=True)
            with open(self._cfg_path, "w") as f:
                json.dump({"cfg": self.cfg,
                           "fields": {"rate": self.f_rate.text(),
                                      "vol": self.f_vol.text(),
                                      "target": self.f_target.text()}}, f, indent=2)
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
            self.cfg = new
            if self.line is not None:
                self.line.n_syringes = int(new["n"])
                if diam_changed:
                    self.line.diameter_mm = float(new["diameter"])
                    self.line._configured = False   # re-push diameter on next action
            self._save_cfg()
            self.status_msg.emit(f"Definitions saved — {int(new['n'])} syringe(s), "
                                 f"Ø{new['diameter']} mm, {new['direction']}.")

    def _load_experiment(self):
        """Load a text experiment file: apply the pump/flow settings here and set
        the wells + sample/buffer times on the Sampling tab."""
        start_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
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
                self.status_msg.emit(f"Pump connected — {line.pump.port.split('/')[-1]}. "
                                     "Connect the flow sensor too if you have one.")
            except Exception as e:
                self.status_msg.emit(f"Pump connect failed: {e}")
            self.pump_result.emit(ok)
        threading.Thread(target=job, daemon=True).start()

    def _after_pump(self, ok):
        self.btn_pump.setEnabled(True)
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
                ok = True
                self.status_msg.emit("Flow sensor connected — plotting + logging.")
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
        try:
            from config import FILES
            folder = FILES.SENSOR_READINGS_FOLDER
        except Exception:
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sensor_Readings")
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
        self._expected = self.line.expected_combined(machine); self._steady = True
        self._clog_since = None
        self.status_msg.emit(f"Flowing {line_rate:g} µL/min in the line "
                             f"({machine:g}/syringe × {n}). Change + press again to update.")
        self._work(lambda: self.line.start_flow_single(machine, direction=self.cfg["direction"]))

    def _run(self):
        if self._guard():
            return
        line_vol, line_rate = self._num(self.f_vol), self._num(self.f_rate)
        n = self._nsyr(); mvol, mrate = line_vol / n, line_rate / n
        self._abort = False
        self._expected = self.line.expected_combined(mrate); self._steady = True
        self._clog_since = None
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
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sensor_Readings")
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
        self._abort = True; self._steady = False; self._bursting = False
        if self.line is None:
            return
        try:
            self.line.stop(); self.status_msg.emit("STOP sent.")
        except Exception as e:
            self.status_msg.emit(f"Stop error: {e}")

    # ------------------------------------------------------------- polling
    def _poll(self):
        # Nothing to read/plot/log without a flow sensor (pump-only is fine).
        if self.line is None or getattr(self.line, "sensor", None) is None:
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
        # drawn on that plot's right axis — don't also draw it here.
        if self._metabolites_running():
            self._tick += 1
            if self._tick % 40 == 0:
                self.lbl_status.setText("Metabolite plot active — flow shown there (right axis).")
            return

        # redraw ~ every 300 ms
        self._tick += 1
        if self._tick % 2:
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
        self._trigger_burst(auto=False)

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
                    self._expected = self.line.expected_combined(mbase)
                    self._clog_since = None; self._steady = True
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

    def shutdown(self):
        try:
            self.poll_timer.stop()
            if self.line:
                try:
                    self.line.stop()
                except Exception:
                    pass
                self.line.close()
        except Exception:
            pass
