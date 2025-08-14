# Ukrainian Bank Insolvency Prediction 

> For markers: models, scripts, and outputs are under `thesis_export/`.

## Final modeling dataset

- `/home/fj/UkrainianInsolvencyPredictor/data/processed/model_ready/model_lag1_next1m_target_20250812_112532_SANITIZED_EW_DROPPF_BASE.csv`
  - Early-warning target (next-month failure), post-failure months removed

## Thesis export models

- Scripts (run from repo root):
  - `thesis_export/run_catboost_h1m_all.py`
  - `thesis_export/run_sarimax_all.py`

### How to run

```bash
# CatBoost (falls back to logistic if CatBoost unavailable)
python3 thesis_export/run_catboost_h1m_all.py \
  --csv /home/fj/UkrainianInsolvencyPredictor/data/processed/model_ready/model_lag1_next1m_target_20250812_112532_SANITIZED_EW_DROPPF_BASE.csv

# SARIMAX + L1 logistic (cleaned, no leakage)
python3 thesis_export/run_sarimax_all.py \
  --csv /home/fj/UkrainianInsolvencyPredictor/data/processed/model_ready/model_lag1_next1m_target_20250812_112532_SANITIZED_EW_DROPPF_BASE.csv \
  --split_date 2022-08-31
```

## Outputs

- Metrics and predictions
  - `export/` (CatBoost script)
  - `thesis_export/outputs/` (SARIMAX script)

That’s it. Everything else is in the code and generated outputs above.

