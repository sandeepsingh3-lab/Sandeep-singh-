"""Download historical race lap data with FastF1 and build a pit-stop dataset.

Uses completed seasons by default. FastF1 timing data availability is generally
2018 onward. The target is whether a driver enters the pits on the current lap;
features are shifted from the previous completed lap to avoid using in-lap data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".fastf1_cache"
OUT = ROOT / "data" / "processed" / "pit_stop_decisions.csv"


def build_session_rows(year: int, event_name: str) -> pd.DataFrame:
    session = fastf1.get_session(year, event_name, "R")
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps.copy()
    required = {"Driver", "LapNumber", "PitInTime", "TyreLife", "Compound",
                "Stint", "Position", "LapTime", "TrackStatus"}
    missing = required.difference(laps.columns)
    if missing:
        print(f"Skip {year} {event_name}: missing columns {sorted(missing)}")
        return pd.DataFrame()

    rows = []
    for driver, group in laps.groupby("Driver", sort=False):
        group = group.sort_values("LapNumber").copy()
        # Current-lap target; predictors are values from the previous completed lap.
        group["target_pit_this_lap"] = group["PitInTime"].notna().astype(int)
        for col in ["TyreLife", "Compound", "Stint", "Position", "LapTime", "TrackStatus"]:
            group[f"prev_{col}"] = group[col].shift(1)
        group["lap_number"] = group["LapNumber"]
        group["season"] = year
        group["event"] = event_name
        group["driver"] = driver
        rows.append(group[[
            "season", "event", "driver", "lap_number",
            "prev_TyreLife", "prev_Compound", "prev_Stint",
            "prev_Position", "prev_LapTime", "prev_TrackStatus",
            "target_pit_this_lap"
        ]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025,
                        help="Use completed seasons for a clean historical baseline.")
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    all_rows = []
    for year in range(args.start_year, args.end_year + 1):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            event_name = str(event["EventName"])
            try:
                frame = build_session_rows(year, event_name)
                if not frame.empty:
                    all_rows.append(frame)
                    print(f"Collected {year} {event_name}: {len(frame)} driver-laps")
            except Exception as exc:
                print(f"Skipped {year} {event_name}: {type(exc).__name__}: {exc}")
    if not all_rows:
        raise SystemExit("No race-lap data collected. Check network access and FastF1 availability.")
    data = pd.concat(all_rows, ignore_index=True)
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data):,} rows to {OUT}")
    print("Target counts:")
    print(data["target_pit_this_lap"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
