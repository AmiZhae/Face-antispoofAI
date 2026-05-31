import argparse
import os
import csv
import pickle
import numpy as np

import config

MODEL_PATH = config.RF_MODEL_PATH
CSV_PATH   = config.RESULTS_CSV


def load_training_data():
    """Load features and labels from experiment_results.csv."""
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found.")
        print("        Run experiments first: press X in main.py.")
        return None, None

    X, y = [], []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ear = float(row["ear_score"])
                tex = float(row["texture_score"])
                chg = float(row["challenge_score"])
                lbl = 1 if row["ground_truth"] == "REAL" else 0
                X.append([ear, tex, chg])
                y.append(lbl)
            except (KeyError, ValueError):
                continue

    if len(X) < 10:
        print(f"[ERROR] Only {len(X)} valid rows found in {CSV_PATH}.")
        print("        Need at least 10 trials. Run more experiments.")
        return None, None

    print(f"[DATA] Loaded {len(X)} trials  "
          f"(REAL={sum(y)}, FAKE={len(y)-sum(y)})")
    return np.array(X), np.array(y)


def train_model(evaluate=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix

    X, y = load_training_data()
    if X is None:
        return

    # ── Train Random Forest ───────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handles imbalanced REAL/FAKE counts
        random_state=42,
    )

    if evaluate and len(X) >= 10:
        # Cross-validation for honest evaluation
        cv     = StratifiedKFold(n_splits=min(5, len(X)//2), shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"\n[CV]  Accuracy: {scores.mean()*100:.1f}% "
              f"(+/- {scores.std()*100:.1f}%)")

    # Train on full dataset
    clf.fit(X, y)

    # ── Feature importance ────────────────────────────────────────────────────
    names = ["EAR Score", "Texture Score", "Challenge Score"]
    print("\n[IMPORTANCE] Learned feature weights:")
    for name, imp in zip(names, clf.feature_importances_):
        bar = "█" * int(imp * 40)
        print(f"  {name:<20} {imp:.3f}  {bar}")

    if evaluate:
        y_pred = clf.predict(X)
        print("\n[REPORT]")
        print(classification_report(y, y_pred, target_names=["FAKE","REAL"]))
        cm = confusion_matrix(y, y_pred)
        print(f"Confusion matrix:\n{cm}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n[SAVED] Random Forest model → {MODEL_PATH}")
    print("        liveness.py will now use AI fusion instead of fixed weights.")


def main():
    parser = argparse.ArgumentParser(description="Random Forest fusion trainer")
    parser.add_argument("--eval", action="store_true",
                        help="Show cross-validation and classification report")
    args = parser.parse_args()
    train_model(evaluate=args.eval)


if __name__ == "__main__":
    main()
