"""Generate the §4.1 confusion matrix PNG from the five gesture CSVs.

Leak-free split-then-augment pipeline (mirrors train_model.py after the
May-2026 restructure):
  raw -> balanced (520/class) -> stratified 80/20 split -> mirror-augment
  the training fold only -> fit Random Forest -> confusion matrix on the
  raw, unaugmented test set.
Output: confusion_matrix.png next to this script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

GESTURES = {
    "grasp.csv":   (0, "GRASP"),
    "point.csv":   (1, "POINT"),
    "stop.csv":    (2, "STOP"),
    "okay.csv":    (3, "OKAY"),
    "neutral.csv": (4, "NEUTRAL"),
}
DATA_DIR = os.path.expanduser("~/ros2_ws/data")
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.png")


def make_relative(landmarks):
    reshaped = landmarks.reshape(-1, 3)
    return reshaped - reshaped[0, :]


def load_raw():
    """Return balanced raw (unaugmented) wrist-relative landmark vectors."""
    counts = [len(pd.read_csv(os.path.join(DATA_DIR, f))) for f in GESTURES]
    n = min(counts)
    X_all, y_all = [], []
    for fname, (label, _) in GESTURES.items():
        df = pd.read_csv(os.path.join(DATA_DIR, fname)).sample(frac=1, random_state=42).head(n)
        raw = df.iloc[:, 1:].values
        rel = np.array([make_relative(r).flatten() for r in raw])
        X_all.append(rel)
        y_all.append(np.full(len(rel), label))
    return np.vstack(X_all), np.concatenate(y_all)


def mirror_x(X):
    """Flip the x-coordinate of each landmark (post wrist-relative transform)."""
    out = X.copy().reshape(-1, 21, 3)
    out[:, :, 0] *= -1
    return out.reshape(-1, 63)


def main():
    X_raw, y_raw = load_raw()
    Xtr, Xte, ytr, yte = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
    # Augment training fold only: test set stays raw.
    Xtr_aug = np.vstack([Xtr, mirror_x(Xtr)])
    ytr_aug = np.concatenate([ytr, ytr])
    Xtr, ytr = Xtr_aug, ytr_aug
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred) * 100

    labels = [name for _, name in GESTURES.values()]
    cm = confusion_matrix(yte, pred)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Gesture Classifier Confusion Matrix (accuracy {acc:.2f}%)")
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
