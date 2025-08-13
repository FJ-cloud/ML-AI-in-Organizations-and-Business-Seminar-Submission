### Ukrainian Insolvency Predictor — Final Summary (last runs)

- Data pipeline
  - Files processed: 76; Unique banks: 67; Observations: 5,092
  - Date range: 2019-01 to 2025-04; Columns: 118
  - Data availability: 76.0%; Missing rate: 23.67%
  - Consolidation resolved 304 duplicates; success: true
  - Artifacts: `output_final/ukrainian_banks_panel_dataset_FINAL.csv`, dataset metadata and summaries

- Class balance (model-ready)
  - Negatives: 4,260 (83.66%); Positives: 832 (16.34%)

- Modeling setup
  - Hybrid: SARIMAX (order 0,0,0) + L1 Logistic
  - Split: 2022-08-31; Data: `data/processed/model_ready/model_lag1_20250809_212523.csv`

- Best configuration (precision-focused)
  - From: `config/experiments/best_precision_config_20250810_005924.json`
  - K = 20 features
  - Test: ROC 0.6324, PR 0.6137, Brier 0.2116 (n=2,144)
  - Train: ROC 0.9971, PR 0.9502, Brier 0.0113 (n=2,948)
  - Key features: short-term BIS liabilities; int’l reserves; 3y yield levels/spread and lags; WAR; Capital/NPL/ROA/ROE; BIS cross-border loans/deposits

- K-sweep (test)
  - K=10: ROC 0.588, PR 0.585, Brier 0.2168
  - K=20: ROC 0.632, PR 0.614, Brier 0.2116
  - K=30: ROC 0.646, PR 0.613, Brier 0.2173

- Precision/Recall at alert rates (test)
  - Top-20 features:
    - 5% (k=107): Precision 1.000, Recall 0.174
    - 10% (k=214): Precision 1.000, Recall 0.348
    - 20% (k=429): Precision 0.664, Recall 0.463
  - Comparators @20% flagged:
    - All features: Prec 0.601, Rec 0.420
    - Top-30: Prec 0.643, Rec 0.449

- Pipeline report (latest)
  - `output_final/pipeline_report_20250809_212518.json` — success true; warnings about early-month duplicates; consolidated to 67 banks, 5,092 rows

- Data dictionary
  - `docs/data_dictionary_model_ready_20250809_213404.md` and `.json` describe all 118 columns, including CAMELS ratios, macro series, and lagged variants

