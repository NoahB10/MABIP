# Air-bubble analysis behind `well_empty_detector.py`

Built 22 Oct 2026 for low-volume mode (`low_volume.py`). Replays the saved runs
(Sensor_Readings/*.txt, Well_Log_*.csv, Amuza_Logs/*.log) to characterise the
air that reaches the sensor when the needle tip meets air.

- `replay.py` replays the detector over every run, using the scheduled
  lift-out as the stand-in for "well ran dry" (no labelled dry wells exist yet).
  Results in `replay_out.txt`: 77 % caught on 8 clean runs with 0 % EMPTY false
  alarms on healthy plateaus; 56 % / 0.5 % across all runs.
- `grid.py` is the threshold sweep. `load.py` loads runs and move times.
- Figures: average lift-out vs move-in signature, H2 vs H3 on 22 Oct, and a
  raster of spike times locked to moves across 17 runs.

Signature: lift-out gives a no-lag common-mode DIP on ch1/ch2/ch4 (suction
break) and, one dead-volume lag later, a common-mode SPIKE of ~0.4 raw units on
ch1–ch4 (rise < 1.7 s, decay ~3.4 s); ch5/ch6 are too noisy to vote.

Needed next: one deliberate run that pumps a well of known volume dry, with the
AMUZA log on, to get a real label. Re-run `replay.py` after any change to the
detector. Scripts may need path tweaks; they were written in a scratch folder.
