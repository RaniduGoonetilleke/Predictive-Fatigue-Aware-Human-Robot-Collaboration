"""Feature 3 plot: a representative torch-mode soft-e-stop episode from slew_log.csv.

Isolates the torch-mode activation at rows 4619-4679 (a single
contiguous episode within one session of the concatenated log;
t_sec = 389.34 to 392.34, a 3 s latched window). Context padding
extends to rows 4589-4710 on the x-axis. Ignores the other 8 e-stop
activations that appear elsewhere in the log across earlier launch
sessions.

Two mechanisms act together while estop_scale = 0.1 is latched:
  1. The scale itself targets a fixed 10x reduction of the commanded
     velocity.
  2. The slew limiter (max_delta = 0.015 rad/step per joint) shapes
     the trajectory, so v_limited cannot snap to 10% of v_raw
     instantaneously. The resulting instantaneous v_limited/v_raw
     ratio sits between ~1.5% and ~11% across the window, correctly reflecting the interplay of both mechanisms. This is the
     "gentle decel" behaviour, not a failure to hit 10x.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CSV = Path(__file__).resolve().parent.parent / "data" / "slew_log.csv"
OUT = Path(__file__).resolve().parent / "feature3_estop.png"
JOINT = "shoulder_lift"

# CSV row numbers (1-based, including header at row 1)
CTX_LO, CTX_HI = 4589, 4710      # x-axis display window
ACTIVE_LO, ACTIVE_HI = 4619, 4679  # latched e-stop window to shade and summarise


def load_window():
    rows = []
    with CSV.open() as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=2):  # row 2 is first data row
            if r["joint_name"] != JOINT:
                continue
            if i < CTX_LO or i > CTX_HI:
                continue
            rows.append((
                i,
                float(r["t_sec"]),
                float(r["v_raw"]),
                float(r["v_limited"]),
                float(r["estop_scale"]),
            ))
    return np.array(rows)


def plot(data):
    row_no = data[:, 0]
    t = data[:, 1] - data[0, 1]
    v_raw, v_lim, estop = data[:, 2], data[:, 3], data[:, 4]

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(t, v_raw, color="#B03A2E", linewidth=1.4, linestyle="--",
             label=r"$v_{\mathrm{raw}}$ (unscaled P-loop)")
    ax1.plot(t, v_lim, color="#1E6091", linewidth=2.0,
             label=r"$v_{\mathrm{limited}}$ (after scale + slew limiter)")
    ax1.set_xlabel("Time (s, zeroed to window start)")
    ax1.set_ylabel(f"{JOINT} velocity (rad/s)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(t, estop, where="post", color="#E67E22", linewidth=1.4,
             alpha=0.95, label="estop_scale")
    ax2.set_ylabel("estop_scale", color="#E67E22")
    ax2.tick_params(axis="y", labelcolor="#E67E22")
    ax2.set_ylim(-0.05, 1.15)

    # Shade only the target activation
    active_mask = (row_no >= ACTIVE_LO) & (row_no <= ACTIVE_HI) & (estop < 0.99)
    if active_mask.any():
        idx = np.where(active_mask)[0]
        ax1.axvspan(t[idx[0]], t[idx[-1]], color="#F1C40F", alpha=0.20,
                    label="estop_scale = 0.1 (latched)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)
    ax1.set_title(
        f"Torch-mode soft e-stop: scale + slew limiter on {JOINT}"
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"saved {OUT}")

    # Report the honest statistics
    active = (row_no >= ACTIVE_LO) & (row_no <= ACTIVE_HI) & (estop < 0.99)
    if active.any():
        ratio = np.abs(v_lim[active]) / np.abs(v_raw[active]).clip(1e-6)
        ratio = ratio[np.isfinite(ratio)]
        print(f"  active window: rows {ACTIVE_LO}-{ACTIVE_HI} "
              f"(t = {data[active, 1][0]:.2f}-{data[active, 1][-1]:.2f} s)")
        print(f"  v_raw range while latched:     "
              f"{v_raw[active].min():.3f} to {v_raw[active].max():.3f} rad/s")
        print(f"  v_limited range while latched: "
              f"{v_lim[active].min():.3f} to {v_lim[active].max():.3f} rad/s")
        print(f"  instantaneous v_lim/v_raw:     "
              f"{ratio.min()*100:.1f}% to {ratio.max()*100:.1f}% "
              f"(median {np.median(ratio)*100:.1f}%)")


if __name__ == "__main__":
    data = load_window()
    print(f"  loaded {len(data)} rows for joint={JOINT!r} "
          f"(context rows {CTX_LO}-{CTX_HI})")
    plot(data)
