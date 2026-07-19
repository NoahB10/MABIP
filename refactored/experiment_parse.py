"""Experiment-file parsing shared by the GUI (flow_control_tab) and the headless
runner (headless_rig). No Qt imports here — keep it importable anywhere."""


def parse_experiment(text):
    """Parse a MABIP experiment file into {lowercase key: string value}.

    Format = `key: value` lines, `#` starts a comment. Recognized keys:
      wells: A1, A2, A3 ...        # AMUZA wells to visit, in order
      sample_time: 110             # seconds at each well
      buffer_time: 60              # seconds in buffer between wells
      flow_rate: 50                # combined line µL/min
      direction: withdraw|infuse
      pause_on_move: yes|no        # feed-forward pause during transit
      pause_after: 0.45            # s after move command to pause
      resume_after: 10             # s after move command to resume
      resume_ramp: 3               # s to ramp up on resume (0 = step)

    A well-plate multi-run file adds `[run] Name` (or `[Name]`) headers; lines under
    each header are that run's settings (each run inherits the shared settings above).
    A run may set its OWN `wells:` (distinct wells per run) and burst settings —
    `auto_burst: yes`, `b_mult`, `b_high_s`, `b_stop_s`, `b_backflow_s` — to fire and
    tune a per-buffer burst during that run. A run with no `wells:` reuses the shared
    selection. Multiple runs land in exp["_runs"] = [dict, ...]; no headers => a single
    flat dict. A flow-decay sweep sets `experiment: flow_sweep` with
    rates/hold_s/decay_s/ramp_s. A burst timing calibration sets
    `experiment: burst_calibration` with baseline/ceiling/high_mults/rev_mults/....
    """
    shared = {}
    runs = []
    cur = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):        # run header: "[run] Name", "[Name]" or "[run:Name]"
            close = line.find("]")
            head = (line[1:close] if close != -1 else line[1:]).strip()
            rest = (line[close + 1:] if close != -1 else "").strip()
            if head.lower().startswith("run"):
                name = rest or head[3:].strip(": ").strip()
            else:
                name = head or rest
            cur = {"name": name or f"run{len(runs) + 1}"}
            runs.append(cur)
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        (cur if cur is not None else shared)[k.strip().lower()] = v.strip()
    exp = dict(shared)
    if runs:
        exp["_runs"] = [dict(shared, **r) for r in runs]
    return exp


def num_list(s):
    """Parse '4,5,6' or a '4..8..1' start..stop..step range into a list of floats."""
    out = []
    for tok in str(s or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ".." in tok:
            a = [float(x) for x in tok.split("..")]
            start, stop = a[0], a[1]
            step = a[2] if len(a) > 2 else 1.0
            v = start
            while (v <= stop + 1e-9) if step > 0 else (v >= stop - 1e-9):
                out.append(round(v, 6)); v += step
        else:
            out.append(float(tok))
    return out
