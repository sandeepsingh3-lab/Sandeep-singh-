"""Train an educational synthetic F1 pit-stop classifier.

Synthetic labels are generated from a toy formula. Metrics measure imitation
of that formula, not real-world Formula 1 strategy performance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["tyre_age", "tyre_temperature", "tyre_degradation",
            "laps_remaining", "rain_probability", "gap_behind"]

def make_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tyre_age": rng.integers(1, 45, n),
        "tyre_temperature": rng.uniform(70, 120, n),
        "tyre_degradation": rng.uniform(0, 100, n),
        "laps_remaining": rng.integers(1, 60, n),
        "rain_probability": rng.uniform(0, 100, n),
        "gap_behind": rng.uniform(0, 30, n),
    })
    score = (.04 * df.tyre_age + .035 * (df.tyre_temperature - 70)
             + .035 * df.tyre_degradation + .02 * df.rain_probability
             - .025 * df.laps_remaining)
    df["pit_stop"] = (score > 4.0).astype(int)
    return df

def main():
    df = make_data()
    X, y = df[FEATURES], df["pit_stop"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(
        y_test, pred, target_names=["STAY OUT", "PIT STOP"], output_dict=True)
    metrics = {
        "dataset": "synthetic_rule_labelled",
        "samples": len(df),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": report,
        "features": FEATURES,
        "warning": "Synthetic labels only; not validated against real F1 outcomes."
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "f1_strategy_model.joblib")
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Confusion matrix [STAY OUT, PIT STOP]:")
    print(np.array(metrics["confusion_matrix"]))
    print(classification_report(y_test, pred, target_names=["STAY OUT", "PIT STOP"]))
    print("Saved model and reports locally.")

if __name__ == "__main__":
    main()
