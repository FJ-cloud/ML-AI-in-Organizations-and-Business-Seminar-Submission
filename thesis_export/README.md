# Thesis Exports: SARIMAX + L1 Logistic Runners

This folder contains three self-contained Python scripts to reproduce the key model results with different feature counts (K):

- run_sarimax_k10.py (default top_k=10)
- run_sarimax_k20.py (default top_k=20)
- run_sarimax_k30.py (default top_k=30)

They will run on the latest modeling CSV and write results locally, without needing the full repository.

## What you need
1) Python 3.10+
2) Install dependencies (recommended in a virtual environment):
```bash
a) cd thesis_exports
b) python -m venv .venv && source .venv/bin/activate
c) pip install -r requirements.txt
```
3) Modeling CSV named like: model_lag1_YYYYMMDD_HHMMSS.csv

Place the CSV next to these scripts (same folder) for easiest use.

## How to run
- K10 (top 10 features):
```bash
python run_sarimax_k10.py
```
- K20 (top 20 features):
```bash
python run_sarimax_k20.py
```
- K30 (top 30 features):
```bash
python run_sarimax_k30.py
```

Optional flags:
- Use a specific CSV path:
```bash
python run_sarimax_k10.py --csv /path/to/model_lag1_*.csv
```
- Change split date or SARIMAX order:
```bash
python run_sarimax_k10.py --split_date 2022-08-31 --sarimax_order 0,0,0
```

## Outputs
The scripts save to:
- export/ (if detected repo root) OR
- outputs/ (local folder created next to these scripts)

Artifacts:
- feature_importance_l1_*.csv
- hybrid_sarimax_l1_metrics_*.json

## Notes
- Statsmodels warnings about convergence or index are expected and do not prevent results.
- Runtime depends on number of banks and time steps; please allow time for per‑bank SARIMAX fits. 