#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
MR_DIR = BASE / 'data' / 'processed' / 'model_ready'
EDA_DIR = BASE / 'eda'
EDA_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)

CAMELS8 = [
    'Capital_Ratio','NPL_Ratio','Cost_Income_Ratio','ROA','ROE',
    'Liquid_Assets_Ratio','Loan_Deposit_Ratio','Net_Open_FX_Ratio'
]

DEFAULT_FEATURES = BASE / 'config' / 'feature_lists' / 'top20_features.txt'


def latest_model_ready() -> Path:
    files = sorted(MR_DIR.glob('model_lag1_*.csv'))
    if not files:
        raise SystemExit('No model_lag1_*.csv found. Run pipeline/make_all_datasets.py first.')
    return files[-1]


def run_hybrid(csv: Path, split_date: str, features_txt: Path | None, sarimax_order: str) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = EDA_DIR / f'hybrid_sarimax_l1_metrics_{ts}.json'
    cmd = [
        'python3', str(BASE / 'machineLearning' / 'SARIMAX' / 'train_l1_sarimax.py'),
        '--csv', str(csv),
        '--split_date', split_date,
        '--sarimax_order', sarimax_order,
    ]
    if features_txt and features_txt.exists():
        cmd += ['--features', str(features_txt)]
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_date', default='2022-08-31', help='YYYY-MM-DD time split date')
    ap.add_argument('--sarimax_order', default='0,0,0', help='p, d, q (e.g., 0,0,0)')
    ap.add_argument('--features', default=str(DEFAULT_FEATURES), help='Optional features .txt path (defaults to top20)')
    args = ap.parse_args()

    csv = latest_model_ready()
    features_txt = Path(args.features) if args.features else None

    # Refit hybrid model
    metrics_path = run_hybrid(csv, args.split_date, features_txt, args.sarimax_order)

    # Optionally aggregate or copy artifacts; here we just report latest
    payload = {
        'run_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_ready_csv': str(csv),
        'split_date': args.split_date,
        'sarimax_order': args.sarimax_order,
        'camels8': CAMELS8,
        'features_file': str(features_txt) if features_txt else None,
        'metrics_json': str(metrics_path),
    }
    out = MODELS_DIR / f'refit_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out.write_text(json.dumps(payload, indent=2))
    print('WROTE:', out)


if __name__ == '__main__':
    main() 