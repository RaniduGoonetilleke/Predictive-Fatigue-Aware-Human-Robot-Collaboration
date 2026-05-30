"""Trains the production gesture classifier. Output: data/gesture_brain.joblib
(loaded by camera_node.py). For the paper's comparison evaluation across
classifier families, see compare_classifiers.py."""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.metrics import confusion_matrix

def make_relative(landmarks):
    # Reshape linear array into [21, 3] matrix
    reshaped = landmarks.reshape(-1, 3)
    # Get Wrist Position (First landmark)
    wrist = reshaped[0, :]
    # Subtract wrist from all points
    relative = reshaped - wrist
    return relative

def load_data():
    """Load and class-balance the raw gesture-landmark data. Returns one
    sample per CSV row (no mirror augmentation). Augmentation must be
    applied AFTER the train/test split to avoid mirror-twin leakage between
    folds, see mirror_augment()."""
    gestures = {
        'grasp.csv': 0,
        'point.csv': 1,
        'stop.csv':  2,
        'okay.csv':  3,
        'neutral.csv': 4
    }

    home = os.path.expanduser('~')
    data_path = os.path.join(home, 'ros2_ws', 'data')

    all_data = []
    all_labels = []

    # 1. Find min count for balancing
    counts = []
    for filename in gestures.keys():
        full_path = os.path.join(data_path, filename)
        if os.path.exists(full_path):
            count = len(pd.read_csv(full_path))
            counts.append(count)

    if not counts:
        print("NO DATA FOUND!")
        return None, None

    min_samples = min(counts)
    print(f"BALANCING: Limiting each class to {min_samples} raw samples.")

    # 2. Load and preprocess (NO augmentation here)
    for filename, label_id in gestures.items():
        full_path = os.path.join(data_path, filename)
        if not os.path.exists(full_path): continue

        df = pd.read_csv(full_path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.head(min_samples)

        raw_X = df.iloc[:, 1:].values
        processed = np.array([make_relative(row).flatten() for row in raw_X])
        labels = np.full(len(processed), label_id)

        print(f" -> Loaded {filename}: {len(processed)} raw samples")

        all_data.append(processed)
        all_labels.append(labels)

    X_final = np.vstack(all_data)
    y_final = np.concatenate(all_labels)

    return X_final, y_final


def mirror_augment(X, y):
    """Double the dataset by mirror-flipping the X-axis of each landmark
    (simulates left-handed operators). Apply AFTER train_test_split, only
    to the training fold — to avoid mirror-twin leakage into the test set."""
    mirrored = []
    for row in X:
        reshaped = row.reshape(-1, 3).copy()
        reshaped[:, 0] *= -1
        mirrored.append(reshaped.flatten())
    mirrored = np.array(mirrored)

    X_aug = np.vstack([X, mirrored])
    y_aug = np.concatenate([y, y.copy()])
    return X_aug, y_aug

def train():
    # Load raw, balanced data (no augmentation yet)
    X, y = load_data()
    print(f"\nRaw dataset: {len(X)} samples ({len(X) // 5} per class)")

    # SPLIT FIRST on raw data (stratified, fixed seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # AUGMENT ONLY the training fold — no mirror twin can cross folds
    pre_aug = len(X_train)
    X_train, y_train = mirror_augment(X_train, y_train)

    print(f"Split-then-augment:")
    print(f"  Train: {len(X_train)} samples ({pre_aug} raw + {pre_aug} mirrored)")
    print(f"  Test:  {len(X_test)} samples (raw, unaugmented — leak-free)")

    print("\nTRAINING AMBIDEXTROUS BRAIN...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("\nEVALUATING...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    target_names = ['Grasp', 'Point', 'Stop', 'Okay', 'Neutral']
    print(classification_report(y_test, predictions, target_names=target_names))

    cm = confusion_matrix(y_test, predictions)
    print("\nCONFUSION MATRIX:")
    print(f"{'':>12} {'GRASP':>8} {'POINT':>8} {'STOP':>8} {'OKAY':>8} {'NEUT':>8}")
    for i, name in enumerate(['GRASP', 'POINT', 'STOP', 'OKAY', 'NEUTRAL']):
        row_str = f"{name:>12}"
        for val in cm[i]:
            row_str += f"{val:>8}"
        print(row_str)
    
    save_path = os.path.join(os.path.expanduser('~'), 'ros2_ws', 'data', 'gesture_brain.joblib')
    joblib.dump(model, save_path)
    print(f"\nSUCCESS! Brain saved to: {save_path}")

    

if __name__ == "__main__":
    train()