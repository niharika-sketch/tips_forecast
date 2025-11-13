## Tip Forecasting Pipeline

An ML solution for forecasting restaurant tips using historical sales, calendar patterns, and local weather signals.
Fully automated — from data ingestion to model inference — and optimized for reproducible, multi-city forecasting experiments.

---
## Directory Structure
src/
│
├── features.py              # Feature engineering: calendar, rolling, fallback logic
├── model.py                 # PyTorch residual net with city/location embeddings
├── weather.py               # ECCC/GeoMet weather fetch + normalization
├── train_eval_infer.py      # Unified training + inference entrypoint
├── plot_weekly_comparison.py# Weekly performance visualization
└── utils/prepare_merged_data.py(not used - for sanity check) & Data_extraction.py (Used to grab data from postgres via django container) 


## Pipeline Summary
1. Weather Fetching - weather.py downloads and cleans Environment Canada (ECCC/GeoMet) data for each city, performs coverage checks, and fills missing values.
2. Feature Engineering - features.py adds calendar, rolling (7/28-day MA), EWMA, and fallback city-level baselines.
3. Model Training - train_eval_infer.py merges data, engineers features, trains the network, evaluates validation/test splits, and logs metrics.
4. Inference - The same script runs inference mode using a saved checkpoint.
5. Visualization - plot_weekly_comparison.py compares actual vs predicted weekly tips and produces PNG diagnostics.

## Quick Start
1. Install dependencies
pip install -r requirements.txt

2. Fetch weather data
python src/weather.py

3. Train the model
python src/train_eval_infer.py --mode train \
    --data data/daily_tips_sales_train.csv \
    --weather_dir data/weather \
    --out outputs_train \
    --epochs 200

4. Run inference
python src/train_eval_infer.py --mode infer \
    --model_dir outputs_train \
    --inference_input data/daily_tips_sales_test.csv \
    --inference_output outputs_infer/inference_daily.csv \
    --weather_dir data/weather \
    --out outputs_infer

5. Visualize performance
python src/plot_weekly_comparison.py \
    --val outputs_train/val_weekly.csv \
    --test outputs_train/test_weekly.csv \
    --out outputs_train/plots/

## Model Details
Architecture - Residual MLP combining numeric features with location and city embeddings
Loss Function - Asymmetric Smooth L1 — penalizes underpredictions more heavily
Optimizer - Adam (lr = 1e-3, weight decay = 1e-5)
Regularization - Dropout (0.1)
Batch Sizes	Train = 256 , Eval = 512
Embeddings - Location = 8 dims , City = 4 dims

## Outputs
best_model.pt - Saved PyTorch model
metrics.json - Daily + weekly MAE/RMSE/aMAE metrics
feature_sanity_report.csv - Summary of mean/std/zero-rate per feature
tip_rate_summary.csv - Per-location tip-rate stats & imputations
val_weekly.csv, test_weekly.csv - Weekly aggregated actual vs predictions
inference_daily.csv, inference_weekly.csv - Model predictions for new data

## Feature Groups
Rolling History	 - ma_7_fallback, ma_28, ewma_03
Weather - temperature, precip_mm, rain_flag, storm_flag
Calendar - dow_sin, dow_cos, woy_sin, woy_cos, mon_sin, mon_cos
Baseline - baseline = ma_dow_4 or ma_7_fallback

## Tech Stack
Python 3.9+
PyTorch
pandas / numpy
scikit-learn
aiohttp, geopy (for weather + geolocation)
matplotlib (for visualization)

## Notes
train_eval_infer.py is the only required entrypoint — it handles merging, feature generation, and model logic internally.
City IDs are zero-indexed and consistent across all modules.
Imputation and scaling are automatic.
Weather fallback logic ensures full coverage even for missing stations.

## Future extensions include:
-Addition of unit tests + continuous code clean up
-Holiday signals
-Joint multi-task forecasting (tips + sales)
-Regional attention layers or meta-learning for cross-city adaptation
-Improvement in accuracy by modify the model architecture or hyperparameters
