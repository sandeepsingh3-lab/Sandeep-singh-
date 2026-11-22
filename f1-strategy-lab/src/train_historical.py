"""Train and evaluate a historical pit-entry classifier.

Rows use previous-lap features to predict pit entry on the current lap.
Train/validation/test are separated chronologically by season to reduce leakage.
This remains an offline research baseline, not a real-time strategy system.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "pit_stop_decisions.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

NUMERIC = ["lap_number", "prev_TyreLife", "prev_Stint", "prev_Position",
           "prev_LapTime_seconds"]
CATEGORICAL = ["prev_Compound", "prev_TrackStatus"]


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Dataset missing. Run: python src/collect_fastf1_data.py")
    df = pd.read_csv(DATA)
    df["prev_LapTime_seconds"] = pd.to_timedelta(
        df["prev_LapTime"], errors="coerce"
    ).dt.total_seconds()
    df["target_pit_this_lap"] = pd.to_numeric(
        df["target_pit_this_lap"], errors="coerce"
    )
    df = df.dropna(subset=["season", "target_pit_this_lap"])
    df["season"] = df["season"].astype(int)
    df["target_pit_this_lap"] = df["target_pit_this_lap"].astype(int)
    df = df[df["target_pit_this_lap"].isin([0, 1])]

    # Fixed chronological split: older seasons train; later seasons test.
    train = df[df["season"] <= 2022]
    validation = df[df["season"].isin([2023])]
    test = df[df["season"].isin([2024, 2025])]
    if train.empty or validation.empty or test.empty:
        raise SystemExit("Need data from seasons <=2022, 2023, and 2024-2025.")

    X_train, y_train = train[NUMERIC + CATEGORICAL], train["target_pit_this_lap"]
    X_val, y_val = validation[NUMERIC + CATEGORICAL], validation["target_pit_this_lap"]
    X_test, y_test = test[NUMERIC + CATEGORICAL], test["target_pit_this_lap"]

    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    val_probability = model.predict_proba(X_val)[:, 1]
    # Tune threshold on validation only; keep final test season untouched.
    thresholds = np.linspace(0.10, 0.90, 81)
    threshold = max(
        thresholds,
        key=lambda t: 2 * (
            precision_score(y_val, val_probability >= t, zero_division=0)
            * recall_score(y_val, val_probability >= t, zero_division=0)
        ) / max(
            precision_score(y_val, val_probability >= t, zero_division=0)
            + recall_score(y_val, val_probability >= t, zero_division=0), 1e-12
        )
    )
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    metrics = {
        "dataset": "FastF1 historical race laps",
        "train_seasons": sorted(train["season"].unique().tolist()),
        "validation_seasons": sorted(validation["season"].unique().tolist()),
        "test_seasons": sorted(test["season"].unique().tolist()),
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "positive_rate_test": float(y_test.mean()),
        "decision_threshold_selected_on_validation": float(threshold),
        "test": {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y_test, prediction, zero_division=0)),
            "average_precision": float(average_precision_score(y_test, probability)),
            "roc_auc": float(roc_auc_score(y_test, probability)) if y_test.nunique() == 2 else None,
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
            "classification_report": classification_report(
                y_test, prediction, target_names=["NO PIT", "PIT"], output_dict=True,
                zero_division=0
            )
        },
        "majority_baseline_accuracy": float(accuracy_score(y_test, baseline_pred)),
        "warning": "Offline historical classifier; not a live pit-wall recommendation system."
    }
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": float(threshold),
                 "numeric_features": NUMERIC, "categorical_features": CATEGORICAL},
                MODEL_DIR / "historical_pit_classifier.joblib")
    (REPORT_DIR / "historical_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Saved model to models/historical_pit_classifier.joblib")


if __name__ == "__main__":
    main()
