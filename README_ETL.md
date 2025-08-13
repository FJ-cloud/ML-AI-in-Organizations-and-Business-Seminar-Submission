## Unified ETL to Model-Ready CSV

Goal: Produce `data/processed/model_ready/model_lag1_*.csv` from Excel inputs with a single command and full logging.

### One-command run

```bash
python3 pipeline/make_all_datasets.py
```

Logs: written to `logs/unified_pipeline_*.log`.

### Stages
- Raw Excel → Panel: `run_etl_pipeline.py` → `output_final/ukrainian_banks_panel_dataset_FINAL.csv`
- Macro build (new): `scripts/data_processing/process_new_macro_data.py` → `data/processed/macro/macro_only_wide_*.csv`
- Optional filter ≥2019: `scripts/data_processing/filter_macro_outputs_from_2019.py`
- Merge & balance: `scripts/merge_camels_with_macro_wide.py` → `data/processed/merged/camels_with_macro_wide_balanced_*.csv`
- Model-ready: `scripts/prepare_model_datasets.py` → `data/processed/model_ready/model_{nolag,lag1}_*.csv`

### WAR variable
- Defined as a binary dummy in `scripts/prepare_model_datasets.py`:
  - `WAR = 1` for months ≥ 2022-03-01, else 0.

### Minimal directories
- Inputs: `data/raw_balance_sheets/*.xlsx`, `data/newData/` (macro sources)
- Outputs: `output_final/`, `data/processed/macro/`, `data/processed/merged/`, `data/processed/model_ready/`

### Notes
- Legacy macro scripts are archived under `legacy/` (to be removed later).
- For full component map see `ETL_DATA_MAP.md`. 