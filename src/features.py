

# python
import pandas as pd
import numpy as np
from math import pi

def add_calendar_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # Cyclical encodings
    df["dow_sin"] = np.sin(2 * pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * pi * df["day_of_week"] / 7)
    df["woy_sin"] = np.sin(2 * pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * pi * df["week_of_year"] / 52)
    df["mon_sin"] = np.sin(2 * pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * pi * df["month"] / 12)
    return df

def add_rolling_features(df, id_col="location_id", target_col="daily_tips"):
    df = df.sort_values([id_col, "date"]).copy()

    # ensure an explicit weekday column for grouped rolling
    df["dow"] = df["date"].dt.dayofweek

    # Rolling over previous 4 occurrences of the same weekday (shifted so current day not included)
    df["ma_dow_4"] = (
        df.groupby([id_col, "dow"])[target_col]
          .apply(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
          .reset_index(level=[id_col, "dow"], drop=True)
    )

    # Rolling means and counts for windows W
    for W in [7, 28]:
        df[f"ma_{W}"] = (
            df.groupby(id_col)[target_col]
              .apply(lambda x: x.shift(1).rolling(W, min_periods=1).mean())
              .reset_index(level=[id_col], drop=True)
        )
        df[f"n_obs_{W}"] = (
            df.groupby(id_col)[target_col]
              .apply(lambda x: x.shift(1).rolling(W, min_periods=1).count())
              .reset_index(level=[id_col], drop=True)
        )

    # Exponentially weighted moving average (alpha=0.3), shifted so current day not included
    df["ewma_03"] = (
        df.groupby(id_col)[target_col]
          .apply(lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean())
          .reset_index(level=[id_col], drop=True)
    )

    return df

def add_fallback_city_ma(df, id_col="location_id", city_col="city_id", target_col="daily_tips"):
    df = df.sort_values(["date"]).copy()

    # City-level mean using same-day open observations (no shift yet)
    df["city_ma_7"] = df.groupby([city_col, "date"])[target_col].transform("mean")
    # shift city mean so we only use past info
    df["city_ma_7"] = df.groupby(city_col)["city_ma_7"].shift(1)

    # Ensure n_obs_7 exists (count of location observations over past 7 days expected from rolling step)
    if "n_obs_7" not in df.columns:
        df["n_obs_7"] = df.groupby(id_col)[target_col].transform("count")

    # Fallback to city mean when location has few observations, otherwise use location ma_7
    df["ma_7_fallback"] = np.where(df["n_obs_7"] < 7, df["city_ma_7"], df.get("ma_7"))

    # Global mean fallback (shifted)
    df["global_mean"] = df.groupby("date")[target_col].transform("mean").shift(1)
    df["ma_7_fallback"] = df["ma_7_fallback"].fillna(df["global_mean"])

    return df

def assemble_feature_frame(df):
    # baseline: prefer ma_dow_4, else ma_7_fallback
    baseline = df.get("ma_dow_4").fillna(df.get("ma_7_fallback"))
    df["baseline"] = baseline

    num_cols = [
        "ma_7_fallback", "ma_28", "ewma_03", "baseline",
        "temperature", "precip_mm", "rain_flag", "storm_flag",
        "dow_sin", "dow_cos", "woy_sin", "woy_cos", "mon_sin", "mon_cos"
    ]

    # Guarantee numeric columns exist
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0

    return df.copy(), num_cols


"""
With is_open

import pandas as pd
import numpy as np
from math import pi

# ------------------------------------
# 1️⃣ Calendar features
# ------------------------------------
def add_calendar_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # Cyclical encodings
    df["dow_sin"] = np.sin(2 * pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * pi * df["day_of_week"] / 7)
    df["woy_sin"] = np.sin(2 * pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * pi * df["week_of_year"] / 52)
    df["mon_sin"] = np.sin(2 * pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * pi * df["month"] / 12)
    return df


# ------------------------------------
# 2️⃣ Rolling features (open-day aware)
# ------------------------------------
def add_rolling_features(df, id_col="location_id", target_col="daily_tips"):
    df = df.sort_values([id_col, "date"]).copy()
    if "is_open" not in df.columns:
        # Default to True if not provided
        df["is_open"] = (~df[target_col].isna()) | (~df["daily_sales"].isna())

    df["dow"] = df["date"].dt.dayofweek

    # Rolling over previous 4 occurrences of the same weekday (only when open)
    df["ma_dow_4"] = (
        df[df["is_open"]]
        .groupby([id_col, "dow"])[target_col]
        .apply(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        .reset_index(level=[id_col, "dow"], drop=True)
    )
    for W in [7, 28]:
        df[f"ma_{W}"] = (
            df[df["is_open"]]
            .groupby(id_col)[target_col]
            .apply(lambda x: x.shift(1).rolling(W, min_periods=1).mean())
            .reset_index(level=[id_col], drop=True)
        )
        df[f"n_obs_{W}"] = (
            df[df["is_open"]]
            .groupby(id_col)[target_col]
            .apply(lambda x: x.shift(1).rolling(W, min_periods=1).count())
            .reset_index(level=[id_col], drop=True)
        )

    # Exponentially weighted moving average (open days only)
    df["ewma_03"] = (
        df[df["is_open"]]
        .groupby(id_col)[target_col]
        .apply(lambda x: x.shift(1).ewm(alpha=0.3, adjust=False).mean())
        .reset_index(level=[id_col], drop=True)
    )

    # Forward fill so last known stats persist to next open day
    for col in ["ma_dow_4", "ma_7", "ma_28", "ewma_03"]:
        if col in df.columns:
            df[col] = df.groupby(id_col)[col].ffill()

    return df


# ------------------------------------
# 3️⃣ Fallback city mean (open-day aware)
# ------------------------------------
def add_fallback_city_ma(df, id_col="location_id", city_col="city_id", target_col="daily_tips"):
    df = df.sort_values(["date"]).copy()
    if "is_open" not in df.columns:
        df["is_open"] = (~df[target_col].isna()) | (~df["daily_sales"].isna())

    # Compute city-level mean only using open days
    city_ma = (
        df[df["is_open"]]
        .groupby([city_col, "date"])[target_col]
        .mean()
        .rename("city_ma_7")
        .reset_index()
    )
    df = df.merge(city_ma, on=[city_col, "date"], how="left")

    # Use city-level average when location has few observations
    if "n_obs_7" not in df.columns:
        df["n_obs_7"] = df.groupby(id_col)[target_col].transform("count")

    df["city_ma_7"] = df.groupby(city_col)["city_ma_7"].shift(1)
    df["ma_7_fallback"] = np.where(df["n_obs_7"] < 7, df["city_ma_7"], df["ma_7"])

    # Fallback to global mean if still missing
    df["global_mean"] = df.groupby("date")[target_col].transform("mean").shift(1)
    df["ma_7_fallback"] = df["ma_7_fallback"].fillna(df["global_mean"])
    return df


# ------------------------------------
# 4️⃣ Assemble full feature frame
# ------------------------------------
def assemble_feature_frame(df):
    # Compute baseline using fallback averages
    baseline = df["ma_dow_4"].fillna(df["ma_7_fallback"])
    df["baseline"] = baseline

    # Ensure open-day context columns exist
    if "days_open_this_week" not in df.columns:
        df["days_open_this_week"] = 0
    if "pct_open_week" not in df.columns:
        df["pct_open_week"] = 0.0

    num_cols = [
        "ma_7_fallback", "ma_28", "ewma_03", "baseline",
        "temperature", "precip_mm", "rain_flag", "storm_flag",
        "dow_sin", "dow_cos", "woy_sin", "woy_cos", "mon_sin", "mon_cos",
        "days_open_this_week", "pct_open_week"
    ]

    # Guarantee numeric types
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0

    return df.copy(), num_cols
"""