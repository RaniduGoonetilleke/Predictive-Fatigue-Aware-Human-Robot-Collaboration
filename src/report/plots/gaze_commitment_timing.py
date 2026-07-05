"""Plot the gaze-commitment timing diagram: saccade -> fixation -> prediction
confidence -> hand-motion onset, with the lead-time annotated.

CSV schema (sample rate = Tobii 50-100 Hz):
    t_sec,gaze_speed_dps,is_fixation,pred_confidence,hand_speed

where
  gaze_speed_dps : instantaneous gaze speed in deg/s
  is_fixation    : 0/1 flag from classifier (2 deg window, 400 ms)
  pred_confidence: 0.0-1.0 from camera_node/robot_controller
  hand_speed     : normalised hand speed (units/s) — used to detect motion onset

Usage
-----
  python3 gaze_commitment_timing.py --csv ../data/gaze_log.csv --out gaze.png
  python3 gaze_commitment_timing.py --simulate --out gaze.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def simulate(duration_s=3.0, fs=100):
    t = np.arange(0, duration_s, 1 / fs)
    rng = np.random.default_rng(11)
    # model: saccade burst at t=0.7 s, fixation from 0.75 to 2.0 s, another saccade at 2.05
    gaze_speed = rng.normal(15, 5, t.size).clip(0)
    # saccades
    for center in (0.7, 2.05):
        mask = (t > center - 0.02) & (t < center + 0.03)
        gaze_speed[mask] = rng.uniform(350, 500, mask.sum())
    # fixation 0.75-2.0
    fix_mask = (t > 0.75) & (t < 2.0)
    gaze_speed[fix_mask] = rng.normal(8, 3, fix_mask.sum()).clip(0)
    is_fix = fix_mask.astype(int)

    # prediction confidence ramps to 1.0 once 400 ms into fixation
    commit_time = 0.75 + 0.4
    pred_conf = np.zeros_like(t)
    pred_conf[t >= commit_time] = 1.0

    # hand motion onset at commit + 0.22 s
    hand_onset = commit_time + 0.22
    hand_speed = np.zeros_like(t)
    ramp_mask = (t >= hand_onset) & (t <= hand_onset + 0.4)
    hand_speed[ramp_mask] = np.linspace(0, 0.9, ramp_mask.sum())
    hand_speed[t > hand_onset + 0.4] = 0.9 + rng.normal(0, 0.02, (t > hand_onset + 0.4).sum())
    return t, gaze_speed, is_fix, pred_conf, hand_speed


def load_csv(path: Path):
    cols = {k: [] for k in ("t_sec", "gaze_speed_dps", "is_fixation", "pred_confidence", "hand_speed")}
    with path.open() as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]) if k != "is_fixation" else int(row[k]))
    return (np.array(cols["t_sec"]), np.array(cols["gaze_speed_dps"]),
            np.array(cols["is_fixation"]), np.array(cols["pred_confidence"]),
            np.array(cols["hand_speed"]))


def plot(t, gaze_speed, is_fix, pred_conf, hand_speed, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4]})

    # panel 1: gaze speed with saccade/fixation shading
    ax = axes[0]
    ax.plot(t, gaze_speed, color="#34495E", linewidth=1.2)
    ax.set_ylabel("Gaze speed (deg/s)")
    fix_regions = _binary_regions(is_fix)
    for s, e in fix_regions:
        ax.axvspan(t[s], t[e], color="#2ECC71", alpha=0.15,
                   label="Fixation" if s == fix_regions[0][0] else None)
    sacc_mask = gaze_speed > 100
    for s, e in _binary_regions(sacc_mask.astype(int)):
        ax.axvspan(t[s], t[e], color="#E74C3C", alpha=0.35,
                   label="Saccade" if s == _binary_regions(sacc_mask.astype(int))[0][0] else None)
    ax.axhline(100, color="gray", linewidth=0.6, linestyle=":")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Gaze-commitment timing: saccade -> fixation -> commit -> hand motion")

    # panel 2: prediction confidence
    ax = axes[1]
    ax.plot(t, pred_conf, color="#8E44AD", linewidth=2.0)
    ax.set_ylabel("Pred. confidence")
    ax.set_ylim(-0.05, 1.1)
    commit_idx = np.argmax(pred_conf >= 1.0) if (pred_conf >= 1.0).any() else None
    if commit_idx is not None and commit_idx > 0:
        ax.axvline(t[commit_idx], color="#8E44AD", linewidth=1.0, linestyle="--")
        ax.text(t[commit_idx], 1.05, "  commitment", color="#8E44AD", fontsize=9)
    ax.grid(True, alpha=0.3)

    # panel 3: hand speed
    ax = axes[2]
    ax.plot(t, hand_speed, color="#1E6091", linewidth=1.6)
    ax.set_ylabel("Hand speed (norm.)")
    ax.set_xlabel("Time (s)")
    onset_idx = np.argmax(hand_speed > 0.05) if (hand_speed > 0.05).any() else None
    if onset_idx is not None and onset_idx > 0:
        ax.axvline(t[onset_idx], color="#1E6091", linewidth=1.0, linestyle="--")
        ax.text(t[onset_idx], hand_speed.max() * 0.95, "  motion onset",
                color="#1E6091", fontsize=9)
    ax.grid(True, alpha=0.3)

    # annotate lead time
    if commit_idx is not None and onset_idx is not None:
        lead_ms = (t[onset_idx] - t[commit_idx]) * 1000
        axes[0].annotate(
            f"lead time ≈ {lead_ms:.0f} ms",
            xy=(t[commit_idx], axes[0].get_ylim()[1] * 0.6),
            xytext=(t[onset_idx] + 0.1, axes[0].get_ylim()[1] * 0.7),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


def _binary_regions(flag):
    flag = np.asarray(flag)
    if flag.size == 0:
        return []
    edges = np.diff(np.r_[0, flag, 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts, ends))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("gaze.png"))
    args = ap.parse_args()
    if args.simulate or args.csv is None:
        data = simulate()
    else:
        data = load_csv(args.csv)
    plot(*data, args.out)


if __name__ == "__main__":
    main()
