import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error


here = os.path.dirname(__file__)
out_dir = os.path.join(os.path.dirname(here), "outputs_test_without_events")
plot_dir = os.path.join(out_dir, "weekly_plots")
os.makedirs(plot_dir, exist_ok=True)

# -----------------------------
# Load metrics for context
# -----------------------------
#metrics_path = os.path.join(out_dir, "metrics.json")
#if os.path.exists(metrics_path):
#    with open(metrics_path) as f:
#        metrics = json.load(f)
#    print("📊 Evaluation summary:")
#    for m in metrics:
#        print(m)
#else:
#    print("[warn] No metrics.json found.")

# -----------------------------
# Load val_weekly.csv
# -----------------------------
val_path = os.path.join(out_dir, "inference_weekly.csv")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"{val_path} not found.")

df = pd.read_csv(val_path, parse_dates=["week"])
print(f"Loaded {len(df)} rows from val_weekly.csv")

# -----------------------------
# Sanity check for columns
# -----------------------------
expected_cols = {"location_id", "week", "true_weekly", "pred_weekly", "true_avg_per_day", "pred_avg_per_day"}
available = set(df.columns)
if not {"true_weekly", "pred_weekly"} <= available:
    raise KeyError(f"Expected 'true_weekly' and 'pred_weekly' columns in {val_path}, found: {available}")

# -----------------------------
# Plot per-location trends
# -----------------------------
unique_locs = df["location_id"].unique()
print(f" Generating plots for {len(unique_locs)} locations...")

for loc in unique_locs:
    loc_df = df[df["location_id"] == loc].sort_values("week")

    if loc_df.empty:
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(loc_df["week"], loc_df["true_weekly"], marker="o", label="Actual Weekly Tips", linewidth=2, linestyle="--")
    plt.plot(loc_df["week"], loc_df["pred_weekly"], marker="x", label="Predicted Weekly Tips", linewidth=2)

    if "true_avg_per_day" in loc_df.columns and "pred_avg_per_day" in loc_df.columns:
        plt.plot(loc_df["week"], loc_df["true_avg_per_day"], color="green", alpha=0.3, label="True Avg/Day")
        plt.plot(loc_df["week"], loc_df["pred_avg_per_day"], color="orange", alpha=0.3, label="Pred Avg/Day")

    plt.title(f"Weekly Tip Forecast vs Actuals — Location {loc}")
    plt.xlabel("Week")
    plt.ylabel("Tips ($)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, f"weekly_comparison_loc_{loc}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

print(f" Saved all plots to: {plot_dir}")

# -----------------------------
# Optional summary plot
# -----------------------------

plt.figure(figsize=(8, 6))
plt.scatter(df["true_avg_per_day"], df["pred_avg_per_day"], alpha=0.4, s=30)
plt.plot([0, df["true_avg_per_day"].max()], [0, df["true_avg_per_day"].max()], "k--", lw=1)
plt.title("Predicted vs True Average Tips per Day (Validation)")
plt.xlabel("True Avg per Day ($)")
plt.ylabel("Pred Avg per Day ($)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "val_avg_scatter.png"), dpi=150)
plt.close()
"""

plt.figure(figsize=(8, 6))
plt.scatter(df["pct_open_week"], np.abs(df["true_weekly"] - df["pred_weekly"]))
plt.xlabel("% of week open")
plt.ylabel("Absolute weekly error ($)")
plt.title("Error vs. Operational Days")
plt.grid(True, linestyle="--", alpha=0.3)
plt.savefig(os.path.join(plot_dir, "pct.png"), dpi=150)
plt.close()

"""

print(" Saved overall scatter: val_avg_scatter.png")
# -----------------------------
# -----------------------------
df_complete = df.copy()  # keep all weeks, including partial
print(f"[info] Using all {len(df_complete)} rows (partial weeks retained)")

# Compute per-location performance with normalized metrics
perf = (
    df_complete.groupby("location_id")
    .apply(lambda g: pd.Series({
        "n_weeks": len(g),
        "avg_days_per_week": g["n_days"].mean() if "n_days" in g.columns else np.nan,
        "partial_ratio": np.mean(g["is_partial_week"]) if "is_partial_week" in g.columns else np.nan,
        # Absolute and relative MAE based on weekly totals
        "mae_weekly": mean_absolute_error(g["true_weekly"], g["pred_weekly"]),
        "mean_true_weekly": g["true_weekly"].mean(),
        "rel_mae_weekly": mean_absolute_error(g["true_weekly"], g["pred_weekly"]) / (g["true_weekly"].mean() + 1e-8),
        # Normalized MAE based on averages per day
        "mae_avg_per_day": mean_absolute_error(g["true_avg_per_day"], g["pred_avg_per_day"]),
        "mean_true_avg_day": g["true_avg_per_day"].mean(),
        "rel_mae_avg_day": mean_absolute_error(g["true_avg_per_day"], g["pred_avg_per_day"]) / (g["true_avg_per_day"].mean() + 1e-8),
    }))
    .reset_index()
)

# Sort by per-day MAE
perf = perf.sort_values("mae_avg_per_day")
perf.to_csv(os.path.join(out_dir, "val_weekly_location_perf.csv"), index=False)
print("Saved per-location performance summary → val_weekly_location_perf.csv")

# Identify best & worst 10
best5 = perf.nsmallest(10, "mae_avg_per_day")["location_id"].tolist()
worst5 = perf.nlargest(10, "mae_avg_per_day")["location_id"].tolist()

print("\nTop 10 Best Locations (Avg/Day MAE):", best5)
print("⚠Worst 10 Locations (Avg/Day MAE):", worst5)

pdf_path = os.path.join(out_dir, "val_weekly_top_bottom5.pdf")
with PdfPages(pdf_path) as pdf:
    for label, locs in [("Top 10 (Best)", best5), ("Bottom 10 (Worst)", worst5)]:
        for loc in locs:
            loc_df = df_complete[df_complete["location_id"] == loc].sort_values("week")
            if loc_df.empty:
                continue

            mae_val = perf.loc[perf.location_id == loc, "mae_avg_per_day"].values[0]
            rel_val = perf.loc[perf.location_id == loc, "rel_mae_avg_day"].values[0] * 100  # %
            avg_days = perf.loc[perf.location_id == loc, "avg_days_per_week"].values[0]
            plt.figure(figsize=(10, 5))
            plt.plot(loc_df["week"], loc_df["true_avg_per_day"], marker="o", label="Actual Avg/Day", linewidth=2, linestyle="--")
            plt.plot(loc_df["week"], loc_df["pred_avg_per_day"], marker="x", label="Predicted Avg/Day", linewidth=2)

            plt.title(f"{label} — Loc {loc}\nMAE=${mae_val:.1f} | Rel MAE={rel_val:.1f}% | Avg Days={avg_days:.1f}")
            plt.xlabel("Week")
            plt.ylabel("Tips ($/Day)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

print(f"Combined PDF saved → {pdf_path}")

# -----------------------------
# Scatter: Absolute vs Relative MAE
# -----------------------------
plt.figure(figsize=(9, 6))
plt.scatter(perf["mae_avg_per_day"], perf["rel_mae_avg_day"] * 100, alpha=0.7, s=40)
plt.xlabel("Absolute MAE (Avg Tips/Day)")
plt.ylabel("Relative MAE (% of Mean Tips/Day)")
plt.title("Absolute vs Relative MAE — All Locations")
plt.grid(True, linestyle="--", alpha=0.4)

top_outliers = perf.nlargest(5, "rel_mae_avg_day")
for _, row in top_outliers.iterrows():
    plt.annotate(
        f"Loc {int(row['location_id'])}",
        (row["mae_avg_per_day"], row["rel_mae_avg_day"] * 100),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        alpha=0.8
    )

plt.tight_layout()
scatter_path = os.path.join(out_dir, "mae_vs_relmae_avgday.png")
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"Saved scatter plot: {scatter_path}")

'''
# python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(__file__)
out_dir = os.path.join(os.path.dirname(here), "outputs")
os.makedirs(out_dir, exist_ok=True)

def residual_summary(df):
    resid_cols = [c for c in df.columns if c.endswith("_resid")]
    if not resid_cols:
        raise KeyError("No residual columns found (expected columns ending with `_resid`).")
    long = df.melt(id_vars=["location_id"], value_vars=resid_cols, var_name="resid_col", value_name="resid")
    summary = long.groupby("location_id")["resid"].agg(
        mean_resid="mean",
        median_resid="median",
        std_resid="std",
        mae_resid=lambda x: np.mean(np.abs(x))
    ).reset_index()
    return summary.sort_values("mae_resid", ascending=False)

# Load metrics for context
with open(os.path.join(out_dir, "metrics.json")) as f:
    metrics = json.load(f)
print("📊 Evaluation summary:")
for m in metrics:
    print(m)

# Load weekly predictions (already produced by train_eval)
val_weekly_true = pd.read_csv(os.path.join(out_dir, "val_weekly_true.csv"))
val_weekly_pred = pd.read_csv(os.path.join(out_dir, "val_weekly_pred.csv"))

# Merge for all locations (inner join so weeks present in both are compared)
merged = val_weekly_true.merge(val_weekly_pred, on=["location_id", "week"], suffixes=("_true", "_pred"))

# Create residual columns for every matching `*_true` / `*_pred` pair (applies to entire merged df)
true_cols = [c for c in merged.columns if c.endswith("_true")]
resid_cols = []
for true_col in true_cols:
    base = true_col[:-5]  # remove trailing `_true`
    pred_col = base + "_pred"
    if pred_col in merged.columns:
        resid_col = base + "_resid"
        merged[resid_col] = merged[true_col] - merged[pred_col]
        resid_cols.append(resid_col)

if not resid_cols:
    raise KeyError("No matching `*_true` / `*_pred` column pairs found in merged data.")

# save all residual rows and the merged table
merged.to_csv(os.path.join(out_dir, "residuals_by_location.csv"), index=False)

# aggregated summary (uses all residual columns)
summary = residual_summary(merged)
summary.to_csv(os.path.join(out_dir, "residual_summary_by_location.csv"), index=False)

# Plot and save a PNG for every location_id
unique_locs = merged["location_id"].unique()
for loc in unique_locs:
    loc_df = merged[merged["location_id"] == loc].sort_values("week")
    if loc_df.empty:
        continue

    # pick last available true/pred pair for plotting (fallback)
    local_true_cols = [c for c in true_cols if c in loc_df.columns]
    if not local_true_cols:
        continue
    last_true = sorted(local_true_cols)[-1]
    base = last_true[:-5]
    last_pred = base + "_pred"
    if last_pred not in loc_df.columns:
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(loc_df["week"], loc_df[last_pred], label="Predicted Weekly Tips")
    plt.plot(loc_df["week"], loc_df[last_true], label="Actual Weekly Tips", linestyle="--")
    plt.title(f"Weekly Tip Forecast vs Actuals (Location {loc})")
    plt.xlabel("Week")
    plt.ylabel("Total Tips ($)")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"weekly_comparison_loc_{loc}.png")
    plt.savefig(out_path)
    plt.close()
'''