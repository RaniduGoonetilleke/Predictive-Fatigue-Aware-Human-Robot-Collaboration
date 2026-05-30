"""Gesture detection latency histogram — latency_log.csv.

Schema: event, gesture, timestamp
Computes inter-sample latency per gesture class to give a practical
"time between classifications" distribution. Reports median + 95th
percentile.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CSV = Path.home() / "ros2_ws" / "data" / "latency_log.csv"
OUT = Path(__file__).resolve().parent / "gesture_latency.png"
CLASSES = ["POINT", "STOP", "OKAY", "GRASP", "NEUTRAL"]


def load():
    by_class = defaultdict(list)
    with CSV.open() as f:
        for r in csv.DictReader(f):
            if r["event"] != "gesture_detected":
                continue
            by_class[r["gesture"]].append(float(r["timestamp"]))
    # Convert to inter-sample deltas in ms
    deltas = {}
    for g, ts in by_class.items():
        if len(ts) < 2:
            continue
        arr = np.diff(np.array(sorted(ts))) * 1000.0
        arr = arr[(arr > 1.0) & (arr < 500.0)]  # drop stalls + bursts
        if arr.size:
            deltas[g] = arr
    return deltas


def plot(deltas):
    present = [g for g in CLASSES if g in deltas]
    if not present:
        print("  no gesture data present — skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#1E6091", "#C0392B", "#27AE60", "#8E44AD", "#F39C12"]
    for g, c in zip(present, colors):
        d = deltas[g]
        ax.hist(d, bins=25, alpha=0.55, label=f"{g} (n={d.size}, med={np.median(d):.0f} ms)",
                color=c, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Inter-sample latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Random Forest gesture-classification inter-sample latency per class")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"saved {OUT}")
    for g in present:
        d = deltas[g]
        print(f"  {g:8s}  n={d.size:5d}  median={np.median(d):6.1f} ms  p95={np.percentile(d, 95):6.1f} ms")


if __name__ == "__main__":
    d = load()
    plot(d)
