"""Comparative evaluation of gesture-classifier families on the same dataset
used by train_model.py. Reproduces the comparison table in the paper.

Run from this directory:
    python compare_classifiers.py

Compares Random Forest (the deployed model), MLP, and KNN on:
  - Test accuracy
  - Training wall-clock time
  - Inference time per sample (the metric that justifies the deployed choice
    given the 20 Hz real-time control budget)
  - Serialised model size in memory

Uses the same data loader, 80/20 split (random_state=42), and preprocessing
as train_model.py. Saves no models, this script is for evaluation only.

Note: train_model.load_data() shuffles each per-class CSV without a fixed
seed before truncating to the smallest class, so the exact sample membership
varies run-to-run. Within a single run all three classifiers see the same
shuffled dataset, so the comparison is internally consistent; absolute
accuracy can drift by ~0.1-0.5% between runs.
"""

import time
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from train_model import load_data, mirror_augment


TARGET_NAMES = ['Grasp', 'Point', 'Stop', 'Okay', 'Neutral']


def evaluate(name, clf, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = clf.predict(X_test)
    pred_total = time.perf_counter() - t0
    inference_us = (pred_total / len(X_test)) * 1e6  # microseconds per sample

    acc = accuracy_score(y_test, preds) * 100
    size_kb = len(pickle.dumps(clf)) / 1024

    return {
        "name": name,
        "accuracy": acc,
        "train_s": train_time,
        "inf_us": inference_us,
        "size_kb": size_kb,
        "report": classification_report(y_test, preds, target_names=TARGET_NAMES),
    }


def main():
    # Load raw balanced data, split first, then augment only the training fold
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pre_aug = len(X_train)
    X_train, y_train = mirror_augment(X_train, y_train)

    print(f"\nDataset (leak-free split-then-augment):")
    print(f"  Raw:   {len(X)} samples ({len(X) // 5} per class)")
    print(f"  Train: {len(X_train)} ({pre_aug} raw + {pre_aug} mirrored)")
    print(f"  Test:  {len(X_test)} (raw, unaugmented)\n")

    classifiers = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("MLP (64,32)",   MLPClassifier(hidden_layer_sizes=(64, 32),
                                        max_iter=500, random_state=42)),
        ("KNN (k=5)",     KNeighborsClassifier(n_neighbors=5)),
    ]

    results = []
    for name, clf in classifiers:
        print(f"--- Training {name} ---")
        r = evaluate(name, clf, X_train, y_train, X_test, y_test)
        results.append(r)
        print(r["report"])

    print("\n=== COMPARISON TABLE (paste into paper) ===\n")
    print("| Classifier      | Accuracy | Train (s) | Inference (us/sample) | Size (KB) |")
    print("|-----------------|---------:|----------:|----------------------:|----------:|")
    for r in results:
        print(f"| {r['name']:<15} "
              f"| {r['accuracy']:>7.2f}% "
              f"| {r['train_s']:>9.2f} "
              f"| {r['inf_us']:>21.1f} "
              f"| {r['size_kb']:>9.1f} |")
    print()


if __name__ == "__main__":
    main()
