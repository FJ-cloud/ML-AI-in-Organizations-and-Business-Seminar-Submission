# Ukrainian Insolvency Project — End-to-End ETL Map and Inventory

Last updated: 2025-08-09

## Overview

## How to run (Unified)
- Single command: `python3 pipeline/make_all_datasets.py`
- Logs: `logs/unified_pipeline_*.log`
- Run manifest (inputs/outputs per stage): `logs/run_manifest_*.json`
- Final model-ready output: `data/processed/model_ready/model_lag1_*.csv`

## Minimal pipeline components
- `run_etl_pipeline.py` (raw Excel → panel)
- `scripts/data_processing/process_new_macro_data.py` (macro-only datasets)
- `scripts/data_processing/filter_macro_outputs_from_2019.py` (optional filter ≥2019)
- `scripts/merge_camels_with_macro_wide.py` (merge panel+macro, compute CAMELS 8, prune)
- `scripts/prepare_model_datasets.py` (restrict to IDs, WAR, CAMELS 8, macro; build nolag/lag1)
- `pipeline/make_all_datasets.py` (orchestrates all stages with logging)

## Cleanup completed
- Removed: `legacy/scripts/` and `macro_etl_pipeline/` (unused scaffolds and legacy scripts)
- Docs that mentioned legacy files remain for historical context, but the unified runner is authoritative.
- Non-pipeline analysis (e.g., `analysis_scripts/`, `eda/`, modeling) kept for research, not used by the pipeline.


From raw Excel balance sheets and macro sources to final model-ready datasets. This document maps the full flow, inventories inputs/outputs, and cites the code responsible for each stage.

## High-Level Flow

1) Raw data discovery and validation (Excel monthly aggregation files)
2) Extraction from Excel (per-sheet: assets, liabilities, equity, financial_results)
3) Transformation and cleaning (canonicalize bank names, clean numerics, add prefixes)
4) Panel dataset creation (full rectangular bank×month panel; presence flags)
5) Failure flagging (detect failure dates with consecutive missing months)
6) Final save + metadata (CSV, summaries, reports)
7) Post-processing and feature generation (CAMELS features; merged macro; model-ready splits)

## Key Code Entry Points

- Orchestrator (end-to-end): `etl_pipeline/core/orchestrator.py`
- Extractor (Excel → DataFrame): `etl_pipeline/core/extractor.py`
- Transformer (cleaning, panel build, failure flags): `etl_pipeline/core/transformer.py`
- Loader (saving, metadata): `etl_pipeline/core/loader.py`
- CLI runner: `run_etl_pipeline.py`

### Selected Code References

```1:31:/home/fj/UkrainianInsolvencyPredictor/etl_pipeline/core/orchestrator.py
class ETLOrchestrator:
    ...
```

```120:171:/home/fj/UkrainianInsolvencyPredictor/etl_pipeline/core/extractor.py
with pd.ExcelFile(file_path) as excel_file:
    sheet_names = excel_file.sheet_names
    ...
    df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    header_row = self._detect_header_row_working(df_raw)
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
```

```71:89:/home/fj/UkrainianInsolvencyPredictor/etl_pipeline/core/transformer.py
# Canonicalize names → clean numerics → add prefixes → standardize dates
df_transformed, canon_stats = self.canonicalizer.process_dataframe(df_transformed, 'bank_name')
```

```276:288:/home/fj/UkrainianInsolvencyPredictor/etl_pipeline/core/transformer.py
# Build full rectangular panel and add presence flags
panel_df = self._consolidate_bank_names(panel_df)
panel_df = self._final_cleanup(panel_df)
panel_df = panel_df.sort_values(['bank_name', 'date']).reset_index(drop=True)
```

```547:591:/home/fj/UkrainianInsolvencyPredictor/etl_pipeline/core/transformer.py
# Add failure flags via consecutive missing months after last reporting
panel_df['failed'] = 0
panel_df['failure_date'] = pd.NaT
panel_df['last_reporting_date'] = pd.NaT
```

```46:75:/home/fj/UkrainianInsolvencyPredictor/etl_pipeline/core/loader.py
# Save FINAL panel + timestamped backup + metadata + summaries
main_filename = "ukrainian_banks_panel_dataset_FINAL.csv"
panel_df.to_csv(main_path, index=False)
```

## Inputs and Sources

- Balance sheets (primary raw):
  - Directory: `data/raw_balance_sheets/`
  - Files: 76 monthly `.xlsx` + 1 `.csv`
  - Pattern: `aggregation_YYYY-MM-01_eng.xlsx`

- Macro sources (raw/ingested):
  - Directory: `data/MacroEconomicData/`
  - Files: IMF/STA CSV snapshots (RES_WEO, STA_CPI, STA_EER, STA_IRFCL)
  - Directory: `data/newData/`
  - Files: `DGS3.csv`, `Ukraine 3-Year Bond Yield Historical Data(...).csv`, `7caca..._Data.csv`

## Core Pipeline Outputs

- Final panel and reports (authoritative):
  - Directory: `output_final/` (27 files currently)
  - Main: `ukrainian_banks_panel_dataset_FINAL.csv`
  - Backups: `ukrainian_banks_panel_dataset_YYYYMMDD_HHMMSS.csv`
  - Post-processed: `ukrainian_banks_panel_dataset_CONSOLIDATED.csv`
  - CAMELS: `camels_features_*.{csv,parquet}`, `ukrainian_banks_with_camels_*.csv`, `unified_camels_dataset_*.csv`
  - Metadata/Reports: `dataset_metadata_*.{json,txt}`, `pipeline_report_*.json`, `comprehensive_camels_statistics_*.json`, `camels_methodology_*.md`, `camels_transformation_report_*.txt`

- Macro processed (intermediate):
  - Directory: `data/processed/macro/` (31 files)
  - Examples: `jehd_monthly_{long,wide}_*.csv`, `macro_only_{long,wide}_*.csv`, `us_dgs3_monthly_*.csv`, `ukraine_3y_monthly_*.csv`, `final_macro_indicators.csv`

- Merged CAMELS+macro (intermediate):
  - Directory: `data/processed/merged/` (10 files)
  - Pattern: `camels_with_macro_wide_balanced_*.csv`, `merged_camels_macro_*.csv`

- Model-ready datasets (intermediate):
  - Directory: `data/processed/model_ready/` (7 files)
  - Files: `model_lag1_*.csv`, `model_nolag_*.csv` (+ schema JSON)

## Post-Processing and Auxiliary Scripts

- Bank name consolidation (manual post step used previously):
  - `analysis_scripts/consolidate_remaining_bank_names.py` (reads FINAL → writes CONSOLIDATED)

- CAMELS features and unified datasets:
  - `camels/camels_main.py` (reads FINAL; writes CAMELS features and unified datasets under `output_final/`)

- Macro processing and merges:
  - `scripts/process_imf_data.py`, `scripts/process_macro_data*.py`, `scripts/process_new_macro_data.py`
  - `scripts/merge_camels_macro*.py`, `scripts/merge_camels_with_macro_wide.py`, `scripts/prepare_model_datasets.py`

## Data Lineage Summary

- Source → `data/raw_balance_sheets/*.xlsx`
- Orchestrator → Extractor (per-sheet) → Transformer (clean+prefix+panel) → Failure flags → Loader
- Final outputs → `output_final/ukrainian_banks_panel_dataset_FINAL.csv` (+metadata, summaries)
- Downstream (optional) → CAMELS features/unified datasets → Macro merge → Model-ready CSVs

## Notes and Known Items

- Failure detection uses a conservative rule: min consecutive missing months (default 6) after last reporting → mark failure.
- Macro profitability analysis (separate) forward-fills macro variables only within that analysis script.
- Macro ETL scaffold exists (`macro_etl_pipeline/`), but the actual macro processing currently lives under `scripts/`.

## Counts (as of last update)

- `data/raw_balance_sheets/`: 76 `.xlsx`, 1 `.csv`
- `data/processed/macro/`: 31 files
- `data/processed/merged/`: 10 files
- `data/processed/model_ready/`: 7 files
- `output_final/`: 27 files

## Suggested Cleanup Targets

- Integrate remaining bank-name consolidation into the main transformer; drop post-processing duplication.
- Move macro processing from `scripts/` into `macro_etl_pipeline/` and reference from a single entrypoint.
- Standardize base paths (avoid hard-coded absolute paths); centralize configuration.
- Add a single CLI for generating downstream merged/model-ready datasets.
- Add a manifest (JSON) for each pipeline run capturing inputs, outputs, and config. 