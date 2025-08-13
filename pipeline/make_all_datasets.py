#!/usr/bin/env python3
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import glob

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
LOGS = BASE / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = logging.getLogger('unified_pipeline')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOGS / f'unified_pipeline_{RUN_TS}.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

manifest = {
    'run_started_at': datetime.now().isoformat(),
    'run_ts': RUN_TS,
    'stages': [],
    'artifacts': {},
}


def run(cmd: list, stage: str, cwd: Path | None = None):
    logger.info(f"===== START STAGE: {stage} =====")
    logger.info(f"RUN: {' '.join(cmd)} (cwd={cwd or BASE})")
    try:
        result = subprocess.run(cmd, cwd=str(cwd or BASE), check=True, capture_output=True, text=True)
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info(line)
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.warning(line)
        logger.info(f"===== END STAGE: {stage} (ok) =====")
        manifest['stages'].append({'name': stage, 'ok': True, 'cmd': cmd})
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {' '.join(cmd)}")
        if e.stdout:
            logger.error(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        manifest['stages'].append({'name': stage, 'ok': False, 'cmd': cmd, 'error': e.stderr})
        raise


def latest(path_glob: str) -> str | None:
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None


def capture_artifacts():
    # Final panel
    manifest['artifacts']['panel_final'] = latest(str(BASE / 'output_final' / 'ukrainian_banks_panel_dataset_FINAL.csv'))
    manifest['artifacts']['panel_backup'] = latest(str(BASE / 'output_final' / 'ukrainian_banks_panel_dataset_*.csv'))
    # Pipeline report and metadata
    manifest['artifacts']['pipeline_report'] = latest(str(BASE / 'output_final' / 'pipeline_report_*.json'))
    manifest['artifacts']['dataset_metadata_json'] = latest(str(BASE / 'output_final' / 'dataset_metadata_*.json'))
    # Macro
    manifest['artifacts']['macro_manifest'] = latest(str(BASE / 'data/processed/macro' / 'new_macro_outputs_*.json'))
    manifest['artifacts']['macro_only_wide'] = latest(str(BASE / 'data/processed/macro' / 'macro_only_wide_*.csv'))
    # Merged
    manifest['artifacts']['merged_balanced'] = latest(str(BASE / 'data/processed/merged' / 'camels_with_macro_wide_balanced_*.csv'))
    # Model-ready
    manifest['artifacts']['model_nolag'] = latest(str(BASE / 'data/processed/model_ready' / 'model_nolag_*.csv'))
    manifest['artifacts']['model_lag1'] = latest(str(BASE / 'data/processed/model_ready' / 'model_lag1_*.csv'))


def write_manifest():
    capture_artifacts()
    manifest['run_finished_at'] = datetime.now().isoformat()
    out = LOGS / f'run_manifest_{RUN_TS}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Run manifest written: {out}")


def main():
    # 1) Raw Excel → Final panel
    run([sys.executable, 'run_etl_pipeline.py'], stage='ETL: raw→panel')

    # 2) Macro (new pipeline)
    run([sys.executable, 'scripts/data_processing/process_new_macro_data.py'], stage='MACRO: build macro-only datasets')
    # Optional filter to >=2019 (kept enabled for consistency)
    try:
        run([sys.executable, 'scripts/data_processing/filter_macro_outputs_from_2019.py'], stage='MACRO: filter >=2019')
    except Exception as e:
        logger.warning(f"filter_macro_outputs_from_2019 skipped: {e}")

    # 3) Merge and balance (includes CAMELS 8 compute + prune)
    run([sys.executable, 'scripts/merge_camels_with_macro_wide.py'], stage='MERGE: panel+macro → balanced+CAMELS8')

    # 4) Build model-ready datasets (nolag + lag1, constrained)
    run([sys.executable, 'scripts/prepare_model_datasets.py'], stage='MODEL: build model-ready CSVs')

    write_manifest()
    logger.info('Unified pipeline completed successfully.')


if __name__ == '__main__':
    main() 