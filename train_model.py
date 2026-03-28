from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA = "connect4_data.csv"
DEFAULT_MODEL = "connect4_ml_model.pkl"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Connect Four ML model.")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help=f"Input CSV (default: {DEFAULT_DATA})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Output model file (default: {DEFAULT_MODEL})")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Fraction of data reserved for testing (default: 0.15)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Maximum MLP epochs / iterations (default: 300)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate for Adam (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[Error] Data file not found: '{data_path}'")
        print("        Run data_generator.py first to create it.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"[Train] Loading '{data_path}' …")
    t0 = time.time()
    df = pd.read_csv(data_path)
    print(f"        {len(df):,} rows loaded in {time.time()-t0:.1f}s")
    print(f"        Columns: {list(df.columns[:4])} … num_p, winner")
    print(f"        Winner distribution:\n{df['winner'].value_counts().sort_index().to_string()}")
    print()

    # ------------------------------------------------------------------
    # 2. Build feature matrix and target vector
    # ------------------------------------------------------------------
    # Feature columns: cell_0 … cell_55 + num_p  (57 features)
    cell_cols = [f"cell_{i}" for i in range(56)]
    feature_cols = cell_cols + ["num_p"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[Error] Missing columns in CSV: {missing}")
        sys.exit(1)

    X: np.ndarray = df[feature_cols].values.astype(np.float32)
    y: np.ndarray = df["winner"].values.astype(int)

    print(f"[Train] Feature shape : {X.shape}  (samples × features)")
    print(f"[Train] Target classes : {sorted(set(y))}")
    print()

    # ------------------------------------------------------------------
    # 3. Train / test split (stratified so all winner classes present)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"[Train] Split: {len(X_train):,} train / {len(X_test):,} test")

    # ------------------------------------------------------------------
    # 4. Build pipeline: StandardScaler → MLPClassifier
    #
    #    Architecture: 57 → 256 → 128 → 64 → softmax(num_classes)
    #    Using Adam, early stopping on 10% of training set, ReLU activations.
    # ------------------------------------------------------------------
    print("[Train] Building MLPClassifier pipeline …")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,           # L2 regularisation
        learning_rate_init=args.lr,
        max_iter=args.epochs,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=args.seed,
        verbose=True,         # prints loss every 10 epochs
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("[Train] Fitting pipeline (this may take a few minutes) …\n")
    t_fit = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t_fit
    print(f"\n[Train] Fitting complete in {fit_time:.1f}s")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print()
    print("=" * 60)
    print(f"  Test-set Accuracy : {acc * 100:.2f}%")
    print("-" * 60)
    print("  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=[f"class_{c}" for c in sorted(set(y))],
                                zero_division=0))
    print("=" * 60)

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    model_path = Path(args.model)
    joblib.dump(pipeline, model_path)
    print(f"\n[Train] Model saved to '{model_path}'  "
          f"({model_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
