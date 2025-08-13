### Precision-Floor Next-Month Failure Detector vs Baseline (HGB)

- **Dataset**: `data/processed/model_ready/model_lag1_next1m_target_20250812_112532.csv`
- **Target**: `will_fail_within_1m`
- **Fold logic**: time-based monthly validation with min 18 months training

#### New approach: precision-floor optimization
- File: `precision70_summary_20250813_164359.json`
- OOF probability quality:
  - PR AUC: 0.012
  - ROC AUC: 0.678
  - Brier: 0.00533
  - n_valid: 375 (positives: 2)
- Threshold (maximize recall subject to precision ≥ 0.70):
  - Feasible: false
  - Selected threshold: 0.000339
  - Precision: 0.0138
  - Recall: 1.00
  - Confusion: TP=2 FP=143 TN=230 FN=0
- Artifacts:
  - OOF: `oof_predictions_20250813_164359.csv`
  - Top-K (last val month): `next_month_high_risk_topk_20250813_164359.csv`

Interpretation: with only 2 positives in OOF months, achieving precision ≥ 0.70 is infeasible; the threshold that maximizes recall yields 143 alerts for 2 true positives (1.38% precision).

#### Baseline PR pipeline (HGB)
- File: `metrics_hgb_20250813_164420.json`
- Test probability quality:
  - PR AUC: 0.019
  - ROC AUC: 0.630
  - Brier: 0.0079
- Top-10 per month:
  - Recall: 0.333
  - Precision: 0.013
  - Captured: 4/12, Alerts: 320

#### Takeaways
- On this split and horizon, positives are extremely rare; precision-floor ≥ 0.70 is not attainable with current features/labels.
- Baseline provides slightly higher PR AUC on its holdout; both methods yield similarly low precision at high recall.

#### Next steps
- Increase positive signal density: alternate split or extend window of evaluation months.
- Add early-warning engineered features (trend/volatility, survival filtering), and try gradient boosted trees with class-balanced sampling.
- Consider cost-aware thresholding: optimize F-beta/balanced cost rather than a hard precision floor. 