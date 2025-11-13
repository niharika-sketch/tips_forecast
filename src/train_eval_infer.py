# python
import os, io, json, argparse, numbers
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model import ResidualNetWithEmbeddings
from features import (
    add_calendar_features,
    add_rolling_features,
    add_fallback_city_ma,
    assemble_feature_frame,
)

# -----------------------------
# Losses / metrics
# -----------------------------
def asymmetric_smooth_l1(pred, target, beta=5.0, underweight=1.5, overweight=1.0):
    diff = pred - target
    abs_diff = diff.abs()
    weight = torch.where(diff < 0, underweight, overweight)
    loss = torch.where(
        abs_diff < beta,
        0.5 * (abs_diff ** 2) / beta,
        abs_diff - 0.5 * beta
    )
    return (loss * weight).mean()

def asymmetric_mae(y_true, y_pred, underweight=1.5, overweight=1.0):
    diff = y_pred - y_true
    w = np.where(diff < 0, underweight, overweight)
    return np.mean(np.abs(diff) * w)

def json_safe(o):
    import numpy as np, pandas as pd
    if isinstance(o, numbers.Integral): return int(o)
    if isinstance(o, numbers.Real):    return float(o)
    if isinstance(o, (pd.Timestamp,)): return o.isoformat()
    if isinstance(o, (np.bool_,)):     return bool(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# -----------------------------
# Data utils
# -----------------------------
BAD_STRINGS = {"missingvalues", "missing", "na", "n/a", "null", ""}

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].where(~df[c].str.lower().isin(BAD_STRINGS), np.nan)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def impute_missing_sales_and_tips(df, out_dir=None):
    df = df.copy()
    if "daily_sales" not in df.columns:
        df["daily_sales"] = np.nan
    if "daily_tips" not in df.columns:
        df["daily_tips"] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        df["tip_rate"] = df["daily_tips"] / df["daily_sales"]
    df.loc[~np.isfinite(df["tip_rate"]), "tip_rate"] = np.nan

    tip_rate_summary = (
        df.groupby("location_id")["tip_rate"]
          .agg(["median", "mean", "std", "count"])
          .rename(columns={"median":"tip_rate_median","mean":"tip_rate_mean","std":"tip_rate_std"})
          .reset_index()
    )
    tip_rate_map = tip_rate_summary.set_index("location_id")["tip_rate_median"]
    global_rate = float(np.nanmedian(df["tip_rate"])) if np.isfinite(np.nanmedian(df["tip_rate"])) else 0.15

    df["imputed_sales"] = False
    df["imputed_tips"]  = False

    # sales missing, tips present
    m_sales = df["daily_sales"].isna() & df["daily_tips"].notna()
    loc_rate = df.loc[m_sales, "location_id"].map(tip_rate_map)
    df.loc[m_sales, "daily_sales"] = df.loc[m_sales, "daily_tips"] / loc_rate
    m_sales2 = df["daily_sales"].isna() & df["daily_tips"].notna()
    df.loc[m_sales2, "daily_sales"] = df.loc[m_sales2, "daily_tips"] / global_rate
    df.loc[m_sales | m_sales2, "imputed_sales"] = True

    # tips missing, sales present
    m_tips = df["daily_tips"].isna() & df["daily_sales"].notna()
    loc_rate2 = df.loc[m_tips, "location_id"].map(tip_rate_map)
    df.loc[m_tips, "daily_tips"] = df.loc[m_tips, "daily_sales"] * loc_rate2
    m_tips2 = df["daily_tips"].isna() & df["daily_sales"].notna()
    df.loc[m_tips2, "daily_tips"] = df.loc[m_tips2, "daily_sales"] * global_rate
    df.loc[m_tips | m_tips2, "imputed_tips"] = True

    df["daily_sales"] = pd.to_numeric(df["daily_sales"], errors="coerce").fillna(0.0)
    df["daily_tips"]  = pd.to_numeric(df["daily_tips"],  errors="coerce").fillna(0.0)

    if out_dir:
        out = tip_rate_summary.copy()
        imp = df.groupby("location_id")[["imputed_sales", "imputed_tips"]].sum().reset_index()
        out = out.merge(imp, on="location_id", how="left")
        out_path = os.path.join(out_dir, "tip_rate_summary.csv")
        out.to_csv(out_path, index=False)
        print(f"Saved tip-rate diagnostics → {out_path}")

    return df

def evaluate_weekly(df, preds_daily):
    tmp = df[["location_id","date","daily_tips"]].copy()
    tmp["pred_daily"] = preds_daily
    tmp["week"] = tmp["date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly = (
        tmp.groupby(["location_id","week"], as_index=False)
           .agg(
               true_weekly=("daily_tips","sum"),
               pred_weekly=("pred_daily","sum"),
               n_days=("daily_tips","count"),
               true_avg_per_day=("daily_tips","mean"),
               pred_avg_per_day=("pred_daily","mean"),
           )
    )
    weekly["is_partial_week"] = weekly["n_days"] < 5
    return weekly

# -----------------------------
# Dataset
# -----------------------------
class TipDataset(Dataset):
    def __init__(self, df, num_cols, fit_scaler=False, scaler=None, target_mode="tips"):
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        elif not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")

        df = df.reset_index(drop=True).copy()
        self.df = df
        self.num_cols = list(num_cols)
        self.scaler = scaler or StandardScaler()
        self.target_mode = target_mode

        X = self.df[self.num_cols].to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)

        for col in ("loc_code", "city_code"):
            if col not in self.df.columns:
                raise KeyError(f"Missing '{col}' in frame: {self.df.columns.tolist()}")

        self.loc  = torch.tensor(self.df["loc_code"].to_numpy(dtype=np.int64))
        self.city = torch.tensor(self.df["city_code"].to_numpy(dtype=np.int64))

        if "daily_tips" not in self.df.columns:
            raise KeyError("Missing 'daily_tips'")

        if target_mode == "residual" and "ma_7_fallback" in self.df.columns:
            y = self.df["daily_tips"].to_numpy() - self.df["ma_7_fallback"].fillna(0).to_numpy()
        else:
            y = self.df["daily_tips"].to_numpy()
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        if "ma_7_fallback" in self.df.columns:
            base = self.df["ma_7_fallback"].fillna(0).to_numpy()
            self.baseline = torch.tensor(base, dtype=torch.float32).unsqueeze(1)
        else:
            self.baseline = torch.zeros_like(self.y)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        return self.X[idx], self.loc[idx], self.city[idx], self.y[idx], self.baseline[idx]

# -----------------------------
# Splits
# -----------------------------
def time_split(df, test_ratio=0.15, val_ratio=0.15, random_state=42):
    np.random.seed(random_state)
    df = df.sort_values(["location_id", "date"]).copy()
    train_idx, val_idx, test_idx = [], [], []

    for _, sub in df.groupby("location_id"):
        n = len(sub)
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_test = int(test_ratio * n)
        n_val  = int(val_ratio  * n)
        test_idx_loc = sub.iloc[idx[:n_test]].index
        val_idx_loc  = sub.iloc[idx[n_test:n_test+n_val]].index
        train_idx_loc = sub.iloc[idx[n_test+n_val:]].index
        train_idx.extend(train_idx_loc); val_idx.extend(val_idx_loc); test_idx.extend(test_idx_loc)

    train = df.loc[train_idx].sort_values("date")
    val   = df.loc[val_idx].sort_values("date")
    test  = df.loc[test_idx].sort_values("date")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# -----------------------------
def load_and_merge_weather(df, weather_dir, outputs_dir=None):
    import os
    import pandas as pd

    weather_frames = []
    if weather_dir and os.path.isdir(weather_dir):
        for f in os.listdir(weather_dir):
            if f.startswith("weather_") and f.endswith(".csv"):
                p = os.path.join(weather_dir, f)
                try:
                    wf = pd.read_csv(p, parse_dates=["date"])
                    wf.columns = [c.strip().lower().replace(" ", "_") for c in wf.columns]
                    wf["city"] = wf["city"].str.strip().str.lower()
                    weather_frames.append(wf)
                except Exception as e:
                    print(f"[warn] Skipped {p}: {e}")
    if not weather_frames:
        raise FileNotFoundError("No weather CSVs found")

    df_weather = pd.concat(weather_frames, ignore_index=True)
    print(f"[info] Loaded {len(df_weather)} total weather rows across {df_weather['city'].nunique()} cities")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df_weather.columns = [c.strip().lower().replace(" ", "_") for c in df_weather.columns]

    weather_cols = ["date", "city_id", "temp_max", "temp_min", "temp_mean", "precip_mm"]
    df_weather = df_weather[weather_cols].drop_duplicates(subset=["city_id", "date"])
    valid_cities = df["city_id"].unique()
    df_weather = df_weather[df_weather["city_id"].isin(valid_cities)].copy()

    merged = df.merge(df_weather, on=["date", "city_id"], how="left")
    merged = merged.sort_values(["location_id", "date"]).drop_duplicates(subset=["location_id", "date"], keep="first")

    for c in ["temp_max", "temp_min", "temp_mean", "precip_mm"]:
        if c in merged.columns:
            merged[c] = merged.groupby("city_id")[c].ffill().bfill()

    merged["temperature"] = merged["temp_mean"].fillna(merged["temp_max"])
    merged["rain_flag"] = (merged["precip_mm"] > 1).astype(int)
    merged["storm_flag"] = (merged["precip_mm"] > 10).astype(int)

    numeric_cols = ["daily_sales", "daily_tips", "temp_max", "temp_min", "temp_mean", "precip_mm", "temperature"]
    merged = _coerce_numeric(merged, numeric_cols)

    merged = impute_missing_sales_and_tips(merged, out_dir=outputs_dir)

    return merged

# -----------------------------
# Utilities (centralize repeated behavior)
# -----------------------------
def safe_torch_load(path, map_location):
    """
    Use context manager when available (PyTorch >=2.6) or fallback to add_safe_globals.
    """
    try:
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
            return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        # older versions expose add_safe_globals
        try:
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        except Exception:
            pass
        return torch.load(path, map_location=map_location, weights_only=False)

def _factorize_with_mappings(df, id_col, known_uniqs, new_token="<UNK>"):
    """
    Map raw IDs to integer codes using known uniqs from training.
    Unknowns are appended after known space; mapping keys are compared as strings to support non-int ids.
    """
    known = pd.Index([str(v) for v in known_uniqs])
    mapper = {k: i for i, k in enumerate(known)}
    # map using stringified values so types don't break matches
    raw_str = df[id_col].astype(str)
    codes = raw_str.map(mapper)
    unseen_mask = codes.isna()
    if unseen_mask.any():
        unseen_vals = pd.Index(raw_str.loc[unseen_mask].unique())
        start = len(known)
        mapping_new = {v: i for i, v in enumerate(unseen_vals, start=start)}
        codes.loc[unseen_mask] = raw_str.loc[unseen_mask].map(mapping_new)
    return codes.astype(int)

# -----------------------------
# Training pipeline
# -----------------------------
def train_pipeline(args):
    os.makedirs(args.out, exist_ok=True)
    df_raw = pd.read_csv(args.data, parse_dates=["date"])

    merged = load_and_merge_weather(df_raw, args.weather_dir, outputs_dir=args.out)
    merged = merged.dropna(subset=["daily_tips"])
    merged.to_csv(os.path.join(args.out, "actual_restaurants.csv"), index=False)
    print(f" Saved merged dataset: {os.path.join(args.out, 'actual_restaurants.csv')}")

    df = add_calendar_features(merged)
    df = add_rolling_features(df, id_col="location_id", target_col="daily_tips")
    df = add_fallback_city_ma(df, id_col="location_id", city_col="city_id", target_col="daily_tips")
    feats, num_cols = assemble_feature_frame(df.dropna(subset=["daily_tips"]))

    # Feature sanity check (unchanged)
    def feature_sanity_check(feats, num_cols):
        print("\n Feature Sanity Check:")
        stats = []
        for col in num_cols:
            if col not in feats.columns:
                print(f"  [missing] {col}")
                continue
            mean_val = feats[col].mean()
            std_val = feats[col].std()
            zero_frac = (feats[col] == 0).mean() * 100
            stats.append((col, mean_val, std_val, zero_frac))
            if std_val == 0:
                print(f"  {col} has zero variance — constant feature")
            elif zero_frac > 90:
                print(f"  {col}: {zero_frac:.1f}% zeros → likely missing signal")
            else:
                print(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}, zeros={zero_frac:.1f}%")
        return pd.DataFrame(stats, columns=["feature", "mean", "std", "pct_zero"])

    feat_report = feature_sanity_check(feats, num_cols)
    feat_report.to_csv(os.path.join(args.out, "feature_sanity_report.csv"))
    print(f" Saved feature_sanity_report.csv with {len(feat_report)} features\n")

    feats["loc_code"], loc_uniqs = pd.factorize(feats["location_id"], sort=True)
    feats["city_code"], city_uniqs = pd.factorize(feats["city_id"], sort=True)
    n_locations = int(feats["loc_code"].max() + 1)
    n_cities    = int(feats["city_code"].max() + 1)

    train_df, val_df, test_df = time_split(feats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    train_ds = TipDataset(train_df, num_cols, fit_scaler=True, scaler=scaler, target_mode=args.target_mode)
    val_ds   = TipDataset(val_df,   num_cols, fit_scaler=False, scaler=scaler, target_mode=args.target_mode)
    test_ds  = TipDataset(test_df,  num_cols, fit_scaler=False, scaler=scaler, target_mode=args.target_mode)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

    model = ResidualNetWithEmbeddings(
        num_numeric_features=len(num_cols),
        n_locations=n_locations,
        n_cities=n_cities,
        hidden=128, emb_dim_loc=8, emb_dim_city=4, dropout=0.1
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val = 1e9
    ckpt_path = os.path.join(args.out, "best_model.pt")

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, loc, city, yb, base in train_loader:
            xb, loc, city, yb, base = xb.to(device), loc.to(device), city.to(device), yb.to(device), base.to(device)
            city = city.clamp(min=0, max=model.city_emb.num_embeddings - 1)
            loc = loc.clamp(min=0, max=model.loc_emb.num_embeddings - 1)
            resid_pred = model(xb, loc, city)
            final_pred = resid_pred + base
            loss = asymmetric_smooth_l1(final_pred, yb, beta=5.0, underweight=args.underweight, overweight=1.0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * xb.size(0)
        train_loss = tot / len(train_loader.dataset)

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, loc, city, yb, base in val_loader:
                xb, loc, city, yb, base = xb.to(device), loc.to(device), city.to(device), yb.to(device), base.to(device)
                resid_pred = model(xb, loc, city)
                final_pred = resid_pred + base
                vloss = asymmetric_smooth_l1(final_pred, yb, beta=5.0, underweight=args.underweight, overweight=1.0)
                vtot += vloss.item() * xb.size(0)
        val_loss = vtot / len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "num_cols": num_cols,
                "loc_uniqs": loc_uniqs.tolist(),
                "city_uniqs": city_uniqs.tolist(),
            }, ckpt_path)
        print(f"Epoch {epoch+1}: train_MAE={train_loss:.3f} val_MAE={val_loss:.3f}")

    # reload best
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    def predict_loader(loader):
        preds, trues = [], []
        with torch.no_grad():
            for xb, loc, city, yb, base in loader:
                xb, loc, city, yb, base = xb.to(device), loc.to(device), city.to(device), yb.to(device), base.to(device)
                resid = model(xb, loc, city)
                pred = (base + resid).cpu().numpy().ravel()
                preds.append(pred); trues.append(yb.cpu().numpy().ravel())
        return np.concatenate(preds), np.concatenate(trues)

    pred_val, true_val = predict_loader(val_loader)
    pred_test, true_test = predict_loader(test_loader)

    def pack_metrics(name, y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        amae = asymmetric_mae(y_true, y_pred, underweight=args.underweight)
        bias = float(np.mean(y_pred - y_true))
        under = float(np.mean(y_pred < y_true) * 100.0)
        print(f"[{name}] MAE={mae:.2f}, aMAE={amae:.2f}, RMSE={rmse:.2f}, Bias={bias:+.2f}, Under%={under:.1f}")
        return {"split": name, "MAE_daily": mae, "RMSE_daily": rmse, "aMAE_daily": amae, "bias": bias, "underpred_pct": under}

    results = [pack_metrics("val", true_val, pred_val), pack_metrics("test", true_test, pred_test)]

    val_weekly = evaluate_weekly(val_df, pred_val)
    test_weekly = evaluate_weekly(test_df, pred_test)
    val_complete  = val_weekly[~val_weekly["is_partial_week"]].copy()
    test_complete = test_weekly[~test_weekly["is_partial_week"]].copy()

    def safe_mae(y1, y2):
        if len(y1)==0 or len(y2)==0: return np.nan
        return mean_absolute_error(y1, y2)

    results += [
        {
            "split": "val",
            "n_total_weeks": int(len(val_weekly)),
            "n_partial_weeks": int(val_weekly["is_partial_week"].sum()),
            "MAE_weekly_sum": safe_mae(val_complete["true_weekly"], val_complete["pred_weekly"]),
            "MAE_weekly_avg": safe_mae(val_complete["true_avg_per_day"], val_complete["pred_avg_per_day"]),
        },
        {
            "split": "test",
            "n_total_weeks": int(len(test_weekly)),
            "n_partial_weeks": int(test_weekly["is_partial_week"].sum()),
            "MAE_weekly_sum": safe_mae(test_complete["true_weekly"], test_complete["pred_weekly"]),
            "MAE_weekly_avg": safe_mae(test_complete["true_avg_per_day"], test_complete["pred_avg_per_day"]),
        },
    ]

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=json_safe)

    val_weekly.to_csv(os.path.join(args.out, "val_weekly.csv"), index=False)
    test_weekly.to_csv(os.path.join(args.out, "test_weekly.csv"), index=False)
    print(f" Saved metrics + weekly reports → {args.out}")

# -----------------------------
# Inference pipeline
# -----------------------------
def inference_pipeline(args):
    os.makedirs(args.out, exist_ok=True)

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    num_cols   = ckpt["num_cols"]
    loc_uniqs  = ckpt["loc_uniqs"]
    city_uniqs = ckpt["city_uniqs"]

    model = ResidualNetWithEmbeddings(
        num_numeric_features=len(num_cols),
        n_locations=max(len(loc_uniqs), 1),
        n_cities=max(len(city_uniqs), 1),
        hidden=128, emb_dim_loc=8, emb_dim_city=4, dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_  = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_   = scaler.scale_**2

    df_new_raw = pd.read_csv(args.inference_input, parse_dates=["date"])
    merged_new = load_and_merge_weather(df_new_raw, args.weather_dir, outputs_dir=args.out)
    merged_new.to_csv(os.path.join(args.out, "inference_merged.csv"), index=False)

    df_feat = add_calendar_features(merged_new)
    df_feat = add_rolling_features(df_feat, id_col="location_id", target_col="daily_tips")
    df_feat = add_fallback_city_ma(df_feat, id_col="location_id", city_col="city_id", target_col="daily_tips")
    feats, num_cols_chk = assemble_feature_frame(df_feat.copy())

    for c in num_cols:
        if c not in feats.columns:
            feats[c] = 0.0
    feats = feats[num_cols + ["location_id","city_id","date","daily_tips"]].copy()

    feats["loc_code"]  = _factorize_with_mappings(feats, "location_id", loc_uniqs)
    feats["city_code"] = _factorize_with_mappings(feats, "city_id",   city_uniqs)

    X = feats[num_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)

    loc = torch.tensor(feats["loc_code"].to_numpy(dtype=np.int64)).to(device)
    city= torch.tensor(feats["city_code"].to_numpy(dtype=np.int64)).to(device)

    if "ma_7_fallback" in df_feat.columns:
        base_np = df_feat["ma_7_fallback"].fillna(0).to_numpy().astype("float32")
    else:
        base_np = np.zeros(len(feats), dtype="float32")
    base = torch.tensor(base_np).unsqueeze(1).to(device)

    with torch.no_grad():
        resid = model(X, loc, city)
        pred = (base + resid).cpu().numpy().ravel()

    out_daily = feats[["location_id","city_id","date","daily_tips"]].copy()
    out_daily["pred_tips"] = pred
    out_daily.to_csv(args.inference_output, index=False)
    print(f" Wrote daily inference → {args.inference_output}")

    wk = evaluate_weekly(out_daily.rename(columns={"daily_tips":"daily_tips"}), preds_daily=out_daily["pred_tips"].to_numpy())
    wk.to_csv(os.path.join(args.out, "inference_weekly.csv"), index=False)
    print(f" Wrote weekly inference → {os.path.join(args.out,'inference_weekly.csv')}")

# -----------------------------
# CLI
# -----------------------------
def build_arg_parser():
    here = os.path.dirname(__file__)
    data_path    = os.path.join(os.path.dirname(here), "data", "daily_tips_sales_test.csv")
    outputs_dir  = os.path.join(os.path.dirname(here), "outputs_test_without_events")
    weather_dir  = os.path.join(os.path.dirname(here), "data","weather")
    infer_in     = os.path.join(os.path.dirname(here), "data", "daily_tips_sales_test.csv")
    infer_out    = os.path.join(os.path.dirname(here), "outputs_test_without_events", "inference_daily.csv")

    p = argparse.ArgumentParser(description="Train / Inference for tip forecasting")
    p.add_argument("--mode", choices=["train","infer"], default="infer",
                   help="train: fit model; infer: score new CSV with saved model")
    p.add_argument("--data", type=str, default=data_path, help="Training CSV (train mode)")
    p.add_argument("--out", type=str, default=outputs_dir, help="Output directory for artifacts/results")
    p.add_argument("--weather_dir", type=str, default=weather_dir, help="Directory with weather_*.csv files")
    p.add_argument("--target_mode", type=str, default="tips", choices=["tips","residual"])
    p.add_argument("--underweight", type=float, default=1.5, help="Weight for underpredictions in asymmetric loss/metric")
    p.add_argument("--epochs", type=int, default=200)

    p.add_argument("--model_dir", type=str, default=outputs_dir, help="Directory containing best_model.pt (infer)")
    p.add_argument("--inference_input", type=str, default=infer_in, help="New CSV to score (infer)")
    p.add_argument("--inference_output", type=str, default=infer_out, help="Daily predictions CSV (infer)")
    return p

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.mode == "train":
        train_pipeline(args)
    else:
        inference_pipeline(args)

