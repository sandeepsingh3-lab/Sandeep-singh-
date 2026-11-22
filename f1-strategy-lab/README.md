# F1 Strategy Lab — AI Race Strategy Predictor

An educational motorsport analytics project exploring how machine learning and data analysis can support Formula 1 race-strategy research.

**Author:** Sandeep Singh  
**Status:** Learning prototype (v0.1)

> **Important limitation:** The initial model uses synthetic, rule-labelled data. Its accuracy measures agreement with artificial labels, not real-world Formula 1 strategy performance. The browser dashboard demonstrates a transparent heuristic, not a deployed Random Forest model. No live team telemetry is used.

## Goals
- Explore a pit-stop / stay-out classification problem.
- Build a reproducible Python baseline with scikit-learn.
- Learn feature engineering, train/test splits and classification metrics.
- Visualize a hypothetical race scenario in an interactive dashboard.
- Establish a foundation for later work with licensed historical race data.

## Structure
- `app/index.html` — interactive browser-based strategy heuristic.
- `src/train.py` — synthetic baseline training pipeline.
- `src/predict.py` — example prediction using the saved model.
- `data/` — dataset provenance and licensing notes.
- `models/` — model artifact instructions.
- `notebooks/` — experimentation guide.

## Quick start
1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Train: `python src/train.py`.
4. Predict: `python src/predict.py`.
5. Open `app/index.html` in a browser.

## Features
`tyre_age`, `tyre_temperature`, `tyre_degradation`, `laps_remaining`, `rain_probability`, and `gap_behind`.

Target: `pit_stop` (1 = PIT STOP, 0 = STAY OUT).

## Evaluation
The training script reports accuracy, precision, recall, F1-score and support. Accuracy alone is insufficient; inspect class balance, confusion matrix and calibration. Since labels are generated from a hand-written formula, high scores can simply mean the model learned that formula.

The previously observed 92.75% accuracy and 83.33% class probability came from the initial local synthetic experiment; they are not independently reproduced or validated here.

## Roadmap
- [ ] Reproduce the initial experiment from a fixed random seed.
- [ ] Add confusion matrix and calibration analysis.
- [ ] Source historical race data with a documented licence.
- [ ] Engineer race-aware features and prevent data leakage.
- [ ] Use chronological validation across race weekends.
- [ ] Compare baselines and calibrated models.
- [ ] Connect a model-backed API only after validation.

## Data integrity
Before adding public data, document its source, licence, access date, units, coverage, missing-value handling and limitations. Do not commit proprietary telemetry or data that cannot legally be redistributed.

## Deployment
The `app/` directory is a static HTML/CSS/JavaScript app suitable for Vercel. Set Vercel's Root Directory to `app`. The Python pipeline runs separately and is not executed by this static dashboard.

## License
MIT License. See `LICENSE`.


## Historical-data experiment (new)

FastF1 is used to collect public historical session timing data. The collector defaults to completed seasons 2018–2025 and writes a local CSV; detailed timing data is generally available from 2018 onward. See the [FastF1 data reference](https://github.com/theOehrly/Fast-F1/blob/main/docs/data_reference/index.rst) and [timing data reference](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst).

### Run the historical pipeline

**Google Colab notebook:** [Open the historical-data experiment](https://colab.research.google.com/github/sandeepsingh3-lab/Sandeep-singh-/blob/main/f1-strategy-lab/notebooks/F1_Historical_Data_Pipeline.ipynb). It clones this public repository, installs dependencies, collects data and runs the chronological evaluation.

```bash
pip install -r requirements.txt
python src/collect_fastf1_data.py
python src/train_historical.py
```

The collector labels whether a driver enters the pits on a lap and shifts predictors to the previous completed lap. The model trains on seasons through 2022, selects a decision threshold on 2023, and evaluates once on 2024–2025. It reports accuracy, precision, recall, F1, average precision, ROC AUC, confusion matrix and a majority-class baseline. The downloaded CSV, cache, trained model and metrics are generated locally and are not committed by default.

**Interpretation caution:** pit-entry prediction is not the same as optimal pit strategy. A model can learn historical correlations but cannot establish the best action without modelling alternatives, pit-lane loss, tyre pace, traffic, weather, safety-car probabilities and race position. Do not use this offline baseline for live race decisions.
