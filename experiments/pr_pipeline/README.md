### PR-optimized pipeline (separate, non-invasive)

Goal: maximize average precision (PR AUC) for next-month failure (will_fail_within_1m), with post-failure months dropped. This mirrors rare-event alerting where precision matters under limited alert budgets.

Design choices:
- Data: reuse model-ready CSV; drop post-failure; target = will_fail_within_1m.
- Features: all available numeric features; optionally add simple trend/volatility (3,6 month) if desired.
- Models: baseline Logistic (class_weight=balanced) and HistGradientBoostingClassifier (tree-based, robust to monotonic transforms), both probability outputs.
- CV: time series split on months; objective = average_precision_score on validation.
- Hyperparams:
  - Logistic: C grid (L2), option to standardize.
  - HGB: learning_rate, max_depth, max_leaf_nodes, min_samples_leaf.
- Selection: pick best by mean PR AUC; retrain on full train; evaluate on test.
- Outputs: metrics.json, predictions.csv, reliability.png, top-N capture per month.

Rationale:
- PR AUC is appropriate under heavy imbalance and aligns with limited alert capacity.
- HGB can capture nonlinearities and interactions with limited tuning and works well with mixed-scale features.
- Time-based CV reduces look-ahead risks; dropping post-failure avoids leakage. 