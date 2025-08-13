#!/usr/bin/env python3
import json
import glob
from pathlib import Path
from datetime import datetime

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
OUT_DIR = BASE / 'experiments' / 'new_pipeline' / 'outputs'


def load_results():
    files = sorted(glob.glob(str(OUT_DIR / 'results_*.json')))
    runs = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                runs.append((f, json.load(fp)))
        except Exception:
            continue
    return runs


def format_metrics(m):
    t = m.get('test', {})
    tr = m.get('train', {})
    tr_line = f"Train: ROC={tr.get('roc_auc'):.3f} PR={tr.get('pr_auc'):.3f} Brier={tr.get('brier'):.4f} (n={tr.get('n')})"
    te_line = f"Test:  ROC={t.get('roc_auc'):.3f} PR={t.get('pr_auc'):.3f} Brier={t.get('brier'):.4f} (n={t.get('n')})"
    return tr_line + "\n" + te_line


def main():
    runs = load_results()
    if not runs:
        print('No results_*.json found in outputs/')
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = OUT_DIR / f'report_{ts}.md'

    # Identify specific variants by heuristics (selected_features_count length)
    rows = []
    for path, payload in runs:
        cfg = payload.get('config', {})
        split = payload.get('split_date')
        csv = payload.get('csv')
        feats_n = payload.get('selected_features_count')
        feats = payload.get('selected_features', [])
        stability_top = payload.get('stability_counts_top', {})
        metrics = payload.get('metrics', {})
        label = 'unknown'
        model = 'CatBoost (weighted Logloss)' if cfg.get('use_catboost', False) else 'Logistic (balanced)'
        if feats_n is not None and feats_n > 0:
            label = f"Top-{feats_n}"
        if feats_n and feats_n > 200:
            label = 'All (engineered)'
        if not feats:
            label = 'All (engineered)'
        rows.append({
            'path': path,
            'label': label,
            'csv': csv,
            'split': split,
            'model': model,
            'features_count': feats_n if feats_n is not None else len(feats),
            'features': feats,
            'stability_top': stability_top,
            'metrics': metrics,
        })

    def sort_key(r):
        if r['label'].startswith('Top-'):
            try:
                return (0, int(r['label'].split('-')[1]))
            except Exception:
                return (0, 0)
        if r['label'].startswith('All'):
            return (1, 0)
        return (2, 0)
    rows.sort(key=sort_key)

    lines = []
    lines.append('# Experimental Pipeline Report')
    lines.append('')
    lines.append('## Overview')
    lines.append('- Data: latest `model_lag1_*.csv` (cleaned lag logic)')
    lines.append('- Split: time-based, default 2022-08-31')
    lines.append('- Macro canonicalization: drop z-variants where base exists; drop duplicate `_lag1.1`')
    lines.append('- Feature engineering: per-bank rolling mean/std for windows 3 and 6 months')
    lines.append('- Stability selection: 5 time slices; keep features appearing in ≥60% of slices')
    lines.append('- Models: CatBoost (class-weighted) by default; fallback Logistic (balanced)')
    lines.append('')

    lines.append('## Runs and Metrics')
    for r in rows:
        lines.append(f"### {r['label']}")
        lines.append(f"- Results file: `{r['path']}`")
        lines.append(f"- Data: `{r['csv']}`")
        lines.append(f"- Split: {r['split']}")
        lines.append(f"- Model: {r['model']}")
        lines.append(f"- Selected features: {r['features_count']}")
        if r['features']:
            preview = ', '.join(r['features'][:15])
            lines.append(f"  - Preview: {preview}{' ...' if len(r['features'])>15 else ''}")
        if r['stability_top']:
            top_pairs = sorted(r['stability_top'].items(), key=lambda kv: -kv[1])[:10]
            top_str = ', '.join([f"{k}({v})" for k, v in top_pairs])
            lines.append(f"- Top stability counts: {top_str}")
        lines.append('')
        lines.append('```')
        lines.append(format_metrics(r['metrics']))
        lines.append('```')
        lines.append('')

    lines.append('## Pipeline Sequence (Detailed)')
    lines.append('1. Load latest `model_lag1_*.csv`')
    lines.append('2. Canonicalize macro columns:')
    lines.append('   - If both base and z-variant exist, drop z-variant (restandardize later if needed)')
    lines.append('   - Drop duplicate naming artifacts: *_lag1.1')
    lines.append('3. Assemble base features (exclude ids/presence columns)')
    lines.append('4. Add rolling features per bank:')
    lines.append('   - For each feature f and window w in {3, 6}: f_ma{w} = rolling mean, f_std{w} = rolling std')
    lines.append('5. Split by `split_date` for train/test')
    lines.append('6. Stability selection on train:')
    lines.append('   - Partition train months into 5 time slices; per slice fit L1-logistic with median imputation + scaling')
    lines.append('   - Count features with nonzero coefficients per slice; select those appearing in ≥60%')
    lines.append('   - Optionally keep top-K by frequency')
    lines.append('7. Train model:')
    lines.append('   - CatBoost (weighted Logloss) if available; else logistic (balanced) with median imputation + scaling')
    lines.append('8. Score metrics on probabilities (ROC, PR, Brier)')
    lines.append('')

    report.write_text('\n'.join(lines))
    print('Wrote report:', report)


if __name__ == '__main__':
    main() 