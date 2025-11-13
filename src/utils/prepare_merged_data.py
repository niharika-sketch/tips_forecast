"""
prepare_merged_data.py
------------------------------------------------------------
Merge daily sales/tips with weather data for training.
Adds a coverage summary highlighting low-coverage cities.
------------------------------------------------------------
Outputs:
  1. data/daily_sales_tips_weather.csv
  2. data/weather_merge_summary.csv
------------------------------------------------------------
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
WEATHER_DIR = os.path.join(DATA_DIR, "weather")
OUT_MERGED = os.path.join(DATA_DIR, "daily_sales_tips_weather.csv")
OUT_SUMMARY = os.path.join(DATA_DIR, "weather_merge_summary.csv")


def load_sales_tips():
    path = os.path.join(DATA_DIR, "daily_tips_sales_train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    expected = {"date", "location_id", "city_id", "daily_sales", "daily_tips"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Missing columns in daily_sales_tips.csv: {expected}")
    return df


def load_weather_data():
    if not os.path.isdir(WEATHER_DIR):
        raise FileNotFoundError(f"Missing weather directory: {WEATHER_DIR}")
    frames = []
    for f in os.listdir(WEATHER_DIR):
        if not (f.startswith("weather_") and f.endswith(".csv")):
            continue
        path = os.path.join(WEATHER_DIR, f)
        city = f.replace("weather_", "").replace(".csv", "")
        try:
            w = pd.read_csv(path, parse_dates=["date"])
            w.columns = [c.strip().lower().replace(" ", "_") for c in w.columns]
            w["city"] = city
            keep = ["date", "city_id", "temp_max", "temp_min", "temp_mean", "precip_mm"]
            w = w[[c for c in keep if c in w.columns] + ["city"]]
            frames.append(w)
        except Exception as e:
            print(f"[warn] Skipped {city}: {e}")
    if not frames:
        raise ValueError("No valid weather CSVs found.")
    df_weather = pd.concat(frames, ignore_index=True)
    return df_weather


def merge_sales_weather():
    df_sales = load_sales_tips()
    print(f"Loaded {len(df_sales)} sales/tips rows across {df_sales['city_id'].nunique()} cities")

    df_weather = load_weather_data()
    print(f"Loaded {len(df_weather)} weather rows across {df_weather['city'].nunique()} cities")

    # Coerce numeric values safely
    for col in ["temp_max", "temp_min", "temp_mean", "precip_mm"]:
        if col in df_weather.columns:
            df_weather[col] = pd.to_numeric(df_weather[col], errors="coerce")

    merged = df_sales.merge(df_weather, on=["city_id", "date"], how="left", suffixes=("", "_w"))
    merged = merged.sort_values(["location_id", "date"]).reset_index(drop=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    merged.to_csv(OUT_MERGED, index=False)
    print(f"Saved merged dataset → {OUT_MERGED}")


    # --- Weather coverage summary ---
    summary = (
        merged.groupby("city_id")
        .agg(
            n_total=("date", "count"),
            n_with_weather=("temp_mean", lambda x: x.notna().sum()),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    summary["weather_coverage_%"] = 100.0 * summary["n_with_weather"] / summary["n_total"]

    # Flag low coverage
    low_cov = summary[summary["weather_coverage_%"] < 90.0]
    if not low_cov.empty:
        print("\n Low weather coverage (<90%) detected for:")
        for _, row in low_cov.iterrows():
            print(f"   city_id={row['city_id']} — coverage={row['weather_coverage_%']:.1f}% "
                  f"({int(row['n_with_weather'])}/{int(row['n_total'])} days)")

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved weather coverage summary → {OUT_SUMMARY}")
    print(summary.head(10))


if __name__ == "__main__":
    merge_sales_weather()
