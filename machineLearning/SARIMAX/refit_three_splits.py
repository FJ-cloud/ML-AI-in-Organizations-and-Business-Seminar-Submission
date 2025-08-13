#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess
import glob

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
MR_DIR = BASE / 'data' / 'processed' / 'model_ready'
EDA_DIR = BASE / 'eda'
EDA_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_SPLITS = ['2022-02-28', '2022-08-31', '2024-01-31']
DEFAULT_ORDER = '0,0,0'


def latest_model_ready() -> Path:
    files = sorted(MR_DIR.glob('model_lag1_*.csv'))
    if not files:
        raise SystemExit('No model_lag1_*.csv found. Run pipeline/make_all_datasets.py first.')
    return files[-1]


def run_once(csv: Path, split_date: str, order: str, features: Path | None) -> Path:
    # Run training
    cmd = [
        'python3', str(BASE / 'machineLearning' / 'SARIMAX' / 'train_l1_sarimax.py'),
        '--csv', str(csv),
        '--split_date', split_date,
        '--sarimax_order', order,
    ]
    if features and features.exists():
        cmd += ['--features', str(features)]
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    # Pick the latest generated metrics json
    files = sorted(EDA_DIR.glob('hybrid_sarimax_l1_metrics_*.json'))
    if not files:
        raise SystemExit('No metrics JSON found after training run')
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default=','.join(DEFAULT_SPLITS), help='Comma-separated split dates (YYYY-MM-DD)')
    ap.add_argument('--sarimax_order', default=DEFAULT_ORDER, help='p,d,q')
    ap.add_argument('--features', default='', help='Optional features .txt path')
    args = ap.parse_args()

    csv = latest_model_ready()
    features = Path(args.features) if args.features else None
    split_list = [s.strip() for s in args.splits.split(',') if s.strip()]

    results = []
    for s in split_list:
        metrics_json = run_once(csv, s, args.sarimax_order, features)
        payload = json.loads(Path(metrics_json).read_text())
        results.append({'split_date': s, 'metrics_path': str(metrics_json), 'metrics': payload})

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = MODELS_DIR / f'multi_split_metrics_{ts}.json'
    out_json.write_text(json.dumps({'model_ready_csv': str(csv), 'runs': results}, indent=2))

    # Simple Markdown summary
    lines = [f'# Multi-split Metrics ({csv.name})', '']
    lines.append('| Split Date | Test ROC AUC | Test PR AUC | Test Brier |')
    lines.append('|---|---:|---:|---:|')
    for r in results:
        m = r['metrics']
        te = m.get('test', {})
        lines.append(f"| {r['split_date']} | {te.get('roc_auc', float('nan')):.3f} | {te.get('pr_auc', float('nan')):.3f} | {te.get('brier', float('nan')):.4f} |")
    out_md = MODELS_DIR / f'multi_split_metrics_{ts}.md'
    out_md.write_text('\n'.join(lines))

    print('WROTE:', out_json)
    print('WROTE:', out_md)


if __name__ == '__main__':
    main() 