"""Example prediction using the trained educational model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "models" / "f1_strategy_model.joblib"
if not path.exists():
    raise SystemExit("Run python src/train.py first.")

scenario = pd.DataFrame([{
    "tyre_age": 32, "tyre_temperature": 108, "tyre_degradation": 78,
    "laps_remaining": 20, "rain_probability": 5, "gap_behind": 8.0
}])
model = joblib.load(path)
label = int(model.predict(scenario)[0])
proba = model.predict_proba(scenario)[0]
print("Educational recommendation:", "PIT STOP" if label else "STAY OUT")
print(f"Estimated PIT STOP class probability: {proba[1]:.1%}")
print("Synthetic training labels only; not a real F1 strategy system.")
