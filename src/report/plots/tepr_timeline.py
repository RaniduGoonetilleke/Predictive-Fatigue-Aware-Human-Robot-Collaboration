"""Plot pupil diameter vs time with the rolling baseline and 15 percent spike
threshold annotated. Marks the Soft E-Stop trigger.

CSV schema (Tobii sample rate, e.g. 50-100 Hz):
    t_sec,pupil_mm,baseline_mm,trigger
where `trigger` is 0/1 (1 on the sample where TEPR fired).

Note: the released tepr_log.csv contains no trigger==1 rows (all trials
used the Wizard-of-Oz channel, which bypasses the pupil comparison), so
the trigger marker in the published figure comes from --simulate mode;
the figure is labelled as an illustrative synthetic trace in the paper.

Usage
-----
  python3 tepr_timeline.py --csv ../data/tepr_log.csv --out tepr.png
  python3 tepr_timeline.py --simulate --out tepr.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASELINE_WIN_S = 3.0
SPIKE_FRAC = 0.15


def simulate(duration_s=15.0, fs=50):
    t = np.arange(0, duration_s, 1 / fs)
    rng = np.random.default_rng(7)
    # baseline drift (slow)
    base = 4.0 + 0.12 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.02, t.size)
    # sudden stress event at t=8 s: fast dilation up to +24 percent, decaying
    stress_center = 8.0
    stress = 0.95 * np.exp(-0.5 * ((t - stress_center) / 0.6) ** 2)
    pupil = base + stress + rng.normal(0, 0.03, t.size)
    # rolling baseline
    win = int(BASELINE_WIN_S * fs)
    baseline = np.array([pupil[max(0, i - win):i + 1].mean() for i in range(pupil.size)])
    trigger_idx = np.argmax(((pupil - baseline) / baseline) > SPIKE_FRAC)
    trig = np.zeros_like(t, dtype=int)
    trig[trigger_idx] = 1
    return t, pupil, baseline, trig


def load_csv(path: Path):
    t, p, b, tr = [], [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_sec"]))
            p.append(float(row["pupil_mm"]))
            b.append(float(row["baseline_mm"]))
            tr.append(int(row.get("trigger", 0)))
    return np.array(t), np.array(p), np.array(b), np.array(tr)


def plot(t, pupil, baseline, trig, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, pupil, color="#1E6091", linewidth=1.5, label="Pupil diameter (mm)")
    ax.plot(t, baseline, color="#7F8C8D", linewidth=1.2, linestyle="--",
            label=f"{BASELINE_WIN_S:.0f} s rolling baseline")
    upper = baseline * (1 + SPIKE_FRAC)
    ax.plot(t, upper, color="#E67E22", linewidth=1.0, linestyle=":",
            label=f"+{int(SPIKE_FRAC*100)}% spike threshold")

    trig_idx = np.where(trig == 1)[0]
    if trig_idx.size:
        t_fire = t[trig_idx[0]]
        ax.axvline(t_fire, color="#C0392B", linewidth=1.5, alpha=0.8)
        ax.text(t_fire, ax.get_ylim()[1] * 0.95, " Soft E-Stop",
                color="#C0392B", fontsize=9, va="top")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil diameter (mm)")
    ax.set_title("TEPR-triggered Soft E-Stop detection")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("tepr.png"))
    args = ap.parse_args()
    if args.simulate or args.csv is None:
        t, p, b, tr = simulate()
    else:
        t, p, b, tr = load_csv(args.csv)
    plot(t, p, b, tr, args.out)


if __name__ == "__main__":
    main()
