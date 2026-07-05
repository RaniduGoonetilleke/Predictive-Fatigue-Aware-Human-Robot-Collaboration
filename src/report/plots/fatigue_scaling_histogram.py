"""Fatigue scaling histogram per trial: slew_log.csv.

Counts samples at each unique fatigue_scale level. Expected discrete
values from the 4-band fatigue model: 1.000 (FRESH), 0.625 (MILD),
0.375 (MODERATE), 0.000 (SEVERE/FROZEN).
"""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CSV = Path(__file__).resolve().parent.parent / "data" / "slew_log.csv"
OUT = Path(__file__).resolve().parent / "fatigue_scaling.png"


def load():
    vals = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            vals.append(round(float(r["fatigue_scale"]), 3))
    return vals


def plot(vals):
    c = Counter(vals)
    expected = [1.000, 0.625, 0.375, 0.000]
    labels = ["FRESH\n(1.000)", "MILD\n(0.625)", "MODERATE\n(0.375)", "SEVERE\n(0.000)"]
    counts = [c.get(v, 0) for v in expected]
    colors = ["#27AE60", "#F39C12", "#E67E22", "#C0392B"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    total = sum(counts)
    for b, n in zip(bars, counts):
        pct = (n / total * 100) if total else 0.0
        ax.text(b.get_x() + b.get_width() / 2, n * 1.15,
                f"{n}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1, max(counts) * 4)
    ax.set_ylabel("Slew-log samples (log scale)")
    ax.set_title("Fatigue-scaling samples per band (torch-mode ticks, concatenated sessions)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"saved {OUT}")
    print(f"  other unique fatigue_scale values: "
          f"{sorted(set(vals) - set(expected))[:10]}")


if __name__ == "__main__":
    vals = load()
    print(f"  loaded {len(vals)} rows")
    plot(vals)
