#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import argparse
import json
import subprocess

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
MR_DIR = BASE / 'data' / 'processed' / 'model_ready'
EDA_DIR = BASE / 'eda'
MODELS_DIR = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_SPLIT = '2022-08-31'
DEFAULT_ORDER = '0,0,0'
K_LIST = [10, 20, 30]


def latest_model_ready() -> Path:
    files = sorted(MR_DIR.glob('model_lag1_*.csv'))
    if not files:
        raise SystemExit('No model_lag1_*.csv found. Run pipeline/make_all_datasets.py first.')
    return files[-1]


def run_k(csv: Path, split_date: str, order: str, k: int) -> Path:
    cmd = [
        'python3', str(BASE / 'machineLearning' / 'SARIMAX' / 'train_l1_sarimax.py'),
        '--csv', str(csv), '--split_date', split_date, '--sarimax_order', order, '--top_k', str(k)
    ]
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    files = sorted(EDA_DIR.glob('hybrid_sarimax_l1_metrics_*.json'))
    if not files:
        raise SystemExit('No metrics JSON found after K run')
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_date', default=DEFAULT_SPLIT)
    ap.add_argument('--sarimax_order', default=DEFAULT_ORDER)
    args = ap.parse_args()

    csv = latest_model_ready()
    results = []
    for k in K_LIST:
        metrics_json = run_k(csv, args.split_date, args.sarimax_order, k)
        payload = json.loads(Path(metrics_json).read_text())
        results.append({'k': k, 'metrics': payload})

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = MODELS_DIR / f'k_sweep_{ts}.json'
    out_json.write_text(json.dumps({'split_date': args.split_date, 'order': args.sarimax_order, 'results': results}, indent=2))

    # Markdown summary
    lines = [f'# K-sweep ({csv.name}) split={args.split_date} order={args.sarimax_order}', '']
    lines.append('| K | Test ROC | Test PR | Test Brier |')
    lines.append('|---:|---:|---:|---:|')
    for r in results:
        te = r['metrics'].get('metrics_top') or r['metrics'].get('metrics_all')
        lines.append(f"| {r['k']} | {te['test']['roc_auc']:.3f} | {te['test']['pr_auc']:.3f} | {te['test']['brier']:.4f} |")
    out_md = MODELS_DIR / f'k_sweep_{ts}.md'
    out_md.write_text('\n'.join(lines))

    print('WROTE:', out_json)
    print('WROTE:', out_md)


if __name__ == '__main__':
    main() 