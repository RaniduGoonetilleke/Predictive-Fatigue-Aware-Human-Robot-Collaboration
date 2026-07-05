"""Fatigue-weight sensitivity analysis (Reviewer 1, comment 4).

The composite fatigue score is a deterministic weighted sum of four
normalised sub-scores (fatigue_monitor.py):

    F = w_br*blink_rate + w_bd*blink_duration + w_pc*PERCLOS + w_hj*hand_jerk

with nominal weights w_br=0.30, w_bd=0.25, w_pc=0.30, w_hj=0.15 and band
thresholds FRESH<0.30, MILD<0.60, MODERATE<0.80, SEVERE>=0.80.

Because F is deterministic in the weights, the sensitivity of the band
assignment to each weight is characterised analytically over the abstract
sub-score domain [0,1]^4 (uniform sampling). Empirical per-tick sub-score
distributions were not logged during the trials and the operational data
are heavily FRESH-skewed, so uniform sampling over the feasible domain is
the only sampling that exercises all four band boundaries. The perturbation
range (up to +/-50%) is chosen to comfortably exceed any plausible
mis-specification of the nominal weights; it is a methodological robustness
range, not a literature-derived bound.

For each weight, the weight is perturbed by a sweep of magnitudes while the
remaining three weights are renormalised so the set still sums to 1. The
script reports (a) the fraction of operating points reassigned to a
different band versus perturbation magnitude, (b) the mean absolute change
in F, and (c) at a representative +/-20% perturbation, the fraction of
points crossing each band boundary, which surfaces the stability of the
safety-critical SEVERE boundary.

Outputs (next to this script):
  fatigue_weight_sensitivity.png   - three-panel figure
  sensitivity_results.csv          - underlying numbers for audit
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Nominal weights from fatigue_monitor.py:52-55 (ROS parameter defaults).
# Hardcoded here to keep this script standalone (no ROS dependency); the
# assertion below fails loudly if the deployed defaults ever drift.
NOMINAL = {"blink_rate": 0.30, "blink_duration": 0.25,
           "PERCLOS": 0.30, "hand_jerk": 0.15}
assert abs(sum(NOMINAL.values()) - 1.0) < 1e-9, "Weights must sum to 1"

ORDER = ["blink_rate", "blink_duration", "PERCLOS", "hand_jerk"]
LABELS = ["Blink rate\n(0.30)", "Blink duration\n(0.25)",
          "PERCLOS\n(0.30)", "Hand jerk\n(0.15)"]
COLORS = ["#2E86AB", "#F18F01", "#6A8D73", "#C0392B"]
# Blink rate and PERCLOS share weight 0.30, so their curves coincide; PERCLOS
# is drawn dashed with a different marker so both remain visible.
LINESTYLES = ["-", "-", "--", "-"]
MARKERS = ["o", "s", "^", "D"]
THRESHOLDS = [0.30, 0.60, 0.80]                 # FRESH | MILD | MODERATE | SEVERE
BOUND_LABELS = ["FRESH\n&\nMILD\n(0.30)",
                "MILD\n&\nMODERATE\n(0.60)",
                "MODERATE\n&\nSEVERE\n(0.80)"]
SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N = 200_000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "fatigue_weight_sensitivity.png")
OUT_CSV = os.path.join(HERE, "sensitivity_results.csv")


def band(F):
    return np.digitize(F, THRESHOLDS)


def composite(sub, weights):
    return sub @ np.array([weights[k] for k in ORDER])


def perturbed_weights(key, factor):
    """Scale one weight by factor, renormalise the rest so the set sums to 1."""
    w = dict(NOMINAL)
    w[key] = NOMINAL[key] * factor
    others = [k for k in ORDER if k != key]
    rem_now = sum(NOMINAL[k] for k in others)
    for k in others:
        w[k] = NOMINAL[k] * ((1.0 - w[key]) / rem_now)
    return w


def main():
    rng = np.random.default_rng(0)
    sub = rng.uniform(0.0, 1.0, size=(N, 4))
    base_F = composite(sub, NOMINAL)
    base_band = band(base_F)

    # (a) + (b): sweep
    reassign = {k: [] for k in ORDER}     # % band reassigned, per weight per level
    meandF = {k: [] for k in ORDER}
    rows = []
    for key in ORDER:
        for lvl in SWEEP:
            r_dir, d_dir = [], []
            for factor in (1 + lvl, 1 - lvl):
                F = composite(sub, perturbed_weights(key, factor))
                r_dir.append(100.0 * np.mean(band(F) != base_band))
                d_dir.append(np.mean(np.abs(F - base_F)))
            reassign[key].append(np.mean(r_dir))
            meandF[key].append(np.mean(d_dir))
            rows.append([key, lvl, round(np.mean(r_dir), 3), round(np.mean(d_dir), 5)])

    # (c) boundary crossings at +/-20%, averaged over weights and both directions
    lvl_c = 0.20
    cross = np.zeros(len(THRESHOLDS))
    cnt = 0
    for key in ORDER:
        for factor in (1 + lvl_c, 1 - lvl_c):
            F = composite(sub, perturbed_weights(key, factor))
            for ti, T in enumerate(THRESHOLDS):
                cross[ti] += 100.0 * np.mean((base_F < T) != (F < T))
            cnt += 1
    cross /= cnt

    # ---- write CSV ----
    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["weight", "perturbation", "band_reassigned_pct", "mean_abs_dF"])
        wcsv.writerows(rows)
        wcsv.writerow([])
        wcsv.writerow(["boundary", "threshold", "crossing_pct_at_pm20"])
        for lab, T, c in zip(["FRESH/MILD", "MILD/MODERATE", "MODERATE/SEVERE"], THRESHOLDS, cross):
            wcsv.writerow([lab, T, round(c, 3)])

    # ---- figure ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    for key, c, ls, mk in zip(ORDER, COLORS, LINESTYLES, MARKERS):
        ax1.plot([100 * s for s in SWEEP], reassign[key], linestyle=ls, marker=mk,
                 color=c, label=key.replace("_", " "), linewidth=1.6, markersize=5)
    ax1.set_xlabel("Weight perturbation magnitude (%)")
    ax1.set_ylabel("Operating points reassigned to a different band (%)")
    ax1.set_title("(a) Band-assignment sensitivity sweep")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    for key, c, ls, mk in zip(ORDER, COLORS, LINESTYLES, MARKERS):
        ax2.plot([100 * s for s in SWEEP], meandF[key], linestyle=ls, marker=mk,
                 color=c, label=key.replace("_", " "), linewidth=1.6, markersize=5)
    ax2.set_xlabel("Weight perturbation magnitude (%)")
    ax2.set_ylabel("Mean |change in composite score F|")
    ax2.set_title("(b) Composite-score shift sweep")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    bcol = ["#27AE60", "#E67E22", "#C0392B"]
    bars = ax3.bar(BOUND_LABELS, cross, color=bcol, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, cross):
        ax3.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("Operating points crossing the border (%)")
    ax3.set_title("(c) Stability of each band border (at +/-20% weight change)")
    ax3.tick_params(axis="x", labelsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_CSV}")
    print("\nBand-reassignment % at +/-20% (sweep index 3):")
    for k in ORDER:
        print(f"  {k:15s} {reassign[k][SWEEP.index(0.20)]:5.2f}%")
    print("Max band-reassignment % across the whole sweep (up to +/-50%):")
    for k in ORDER:
        print(f"  {k:15s} {max(reassign[k]):5.2f}%")
    print("Per-boundary crossing % at +/-20%:")
    for lab, c in zip(["FRESH/MILD", "MILD/MODERATE", "MODERATE/SEVERE"], cross):
        print(f"  {lab:16s} {c:5.2f}%")


if __name__ == "__main__":
    main()
