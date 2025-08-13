#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
except Exception:
    sm = None

SCRIPT_DIR = Path(__file__).resolve().parent

# Determine output directory: prefer repo export/, else local outputs/
def _detect_repo_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / 'data').exists() and (p / 'export').exists():
            return p
    return None

_REPO_ROOT = _detect_repo_root(SCRIPT_DIR)
if _REPO_ROOT is not None:
    OUT_DIR = _REPO_ROOT / 'export'
else:
    OUT_DIR = SCRIPT_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AMOUNT_COLS = [
    'assets_total_assets',
    'liabilities_total_liabilities',
    'equity_total_equity_capital',
    'financial_results_profit_loss_after_tax',
    '24_International reserves (excluding gold)',
    '12_Liabilities to BIS banks (cons.), short term',
    '22_Liabilities to BIS banks, locational, total',
    '01_Cross-border loans from BIS reporting banks',
    '27_Cross-border deposits with BIS rep. banks',
]


def _candidate_model_ready_dirs() -> list[Path]:
    roots: list[Path] = []
    for p in [SCRIPT_DIR, Path.cwd(), *SCRIPT_DIR.parents]:
        d = p / 'data' / 'processed' / 'model_ready'
        if d not in roots:
            roots.append(d)
    return roots


def _latest_in_dir(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def latest_model_lag1() -> Path:
    # 1) Prefer CSV alongside this script
    local = _latest_in_dir(SCRIPT_DIR, 'model_lag1_*.csv')
    if local is not None:
        return local
    # 2) Fallback: search repo-style locations
    candidates: list[Path] = []
    for d in _candidate_model_ready_dirs():
        if d.exists():
            candidates.extend(sorted(d.glob('model_lag1_*.csv')))
    if not candidates:
        raise SystemExit('No model_lag1_*.csv found. Place it next to this script or under data/processed/model_ready.')
    return sorted(candidates)[-1]


@dataclass
class Config:
    split_date: str = '2022-08-31'
    sarimax_order: tuple[int, int, int] = (0, 0, 0)
    top_k: int = 10


def load_panel(csv_path: str | None) -> pd.DataFrame:
    csv = csv_path or latest_model_lag1()
    df = pd.read_csv(csv, parse_dates=['date_m'])
    return df


def select_features(df: pd.DataFrame):
    id_cols = [c for c in ['bank_name', 'date_m', 'year_month', 'failed', 'processing_timestamp'] if c in df.columns]
    y_col = 'failed' if 'failed' in df.columns else None
    all_feats = [c for c in df.columns if c not in id_cols]
    feat_numeric = [c for c in all_feats if pd.api.types.is_numeric_dtype(df[c])]
    return id_cols, y_col, feat_numeric


def time_split(df: pd.DataFrame, split_date: str):
    dt = pd.Timestamp(split_date)
    return df[df['date_m'] <= dt].copy(), df[df['date_m'] > dt].copy()


def build_preprocessor(feat_cols: list[str]) -> tuple[Pipeline, list[str], list[str]]:
    amount_cols = [c for c in DEFAULT_AMOUNT_COLS if c in feat_cols]
    ratio_cols = [c for c in feat_cols if c not in amount_cols]
    log1p = FunctionTransformer(np.log1p, feature_names_out='one-to-one')
    amount_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('log', log1p),
        ('scale', StandardScaler()),
    ])
    ratio_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    pre = ColumnTransformer(
        transformers=[
            ('amt', amount_pipe, amount_cols),
            ('ratio', ratio_pipe, ratio_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    full_pre = Pipeline([
        ('pre', pre),
        ('final_impute', SimpleImputer(strategy='constant', fill_value=0.0)),
    ])
    return full_pre, amount_cols, ratio_cols


def fit_bank_sarimax(y: pd.Series, X: pd.DataFrame | np.ndarray, order=(1, 0, 0)):
    if sm is None:
        return None
    try:
        model = sm.tsa.statespace.SARIMAX(
            endog=y.astype(float),
            exog=np.asarray(X, dtype=float),
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, method='lbfgs', maxiter=200)
        return res
    except Exception:
        return None


def predict_bank_sarimax(res, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if res is None:
        return np.zeros(len(X))
    try:
        p = res.predict(start=0, end=len(X) - 1, exog=np.asarray(X, dtype=float))
        return np.asarray(p)
    except Exception:
        return np.zeros(len(X))


def run_pipeline(df_tr, df_te, y_col, feat_cols, sarimax_order):
    pre, amt_cols, ratio_cols = build_preprocessor(feat_cols)
    X_tr = pre.fit_transform(df_tr[feat_cols])
    X_te = pre.transform(df_te[feat_cols])
    feat_names = list(pre.named_steps['pre'].get_feature_names_out())
    X_tr_df = pd.DataFrame(X_tr, index=df_tr.index, columns=feat_names)
    X_te_df = pd.DataFrame(X_te, index=df_te.index, columns=feat_names)
    y_tr = df_tr[y_col].astype(float).values
    y_te = df_te[y_col].astype(float).values

    banks = sorted(df_tr['bank_name'].unique())
    total_banks = len(banks)
    print(f"Fitting per-bank SARIMAX for {total_banks} banks...")

    sarimax_pred_tr = np.zeros_like(y_tr)
    sarimax_pred_te = np.zeros_like(y_te)
    for i, bank in enumerate(banks, start=1):
        tr_idx = df_tr['bank_name'] == bank
        te_idx = df_te['bank_name'] == bank
        if tr_idx.sum() < 12:
            if i % 10 == 0 or i == total_banks:
                print(f"Processed {i}/{total_banks} banks (skipped short series)", flush=True)
            continue
        yb_tr = df_tr.loc[tr_idx, y_col]
        Xb_tr = X_tr_df.loc[tr_idx, feat_names]
        res = fit_bank_sarimax(yb_tr, Xb_tr, order=sarimax_order)
        sarimax_pred_tr[tr_idx.values] = predict_bank_sarimax(res, Xb_tr)
        if te_idx.any():
            Xb_te = X_te_df.loc[te_idx, feat_names]
            sarimax_pred_te[te_idx.values] = predict_bank_sarimax(res, Xb_te)
        if i % 10 == 0 or i == total_banks:
            print(f"Processed {i}/{total_banks} banks", flush=True)

    X_tr_aug = np.c_[X_tr, sarimax_pred_tr]
    X_te_aug = np.c_[X_te, sarimax_pred_te]
    feat_names_aug = feat_names + ['sarimax_score']

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting L1-logistic CV...")
    start_cv = time.time()
    logreg = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='l1',
        solver='liblinear',
        scoring='roc_auc',
        max_iter=2000,
        n_jobs=-1,
        refit=True,
    ).fit(X_tr_aug, y_tr)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished L1-logistic CV in {time.time()-start_cv:.1f}s")

    prob_tr = logreg.predict_proba(X_tr_aug)[:, 1]
    prob_te = logreg.predict_proba(X_te_aug)[:, 1]

    def safe_auc(y_true, y_score):
        try:
            return roc_auc_score(y_true, y_score)
        except Exception:
            return np.nan

    def safe_pr(y_true, y_score):
        try:
            return average_precision_score(y_true, y_score)
        except Exception:
            return np.nan

    metrics = {
        'n_features': len(feat_cols),
        'features_after_transform': feat_names_aug,
        'amount_cols_used': amt_cols,
        'ratio_cols_used': ratio_cols,
        'train': {
            'roc_auc': float(safe_auc(y_tr, prob_tr)),
            'pr_auc': float(safe_pr(y_tr, prob_tr)),
            'brier': float(brier_score_loss(y_tr, prob_tr)),
            'n': int(len(y_tr)),
        },
        'test': {
            'roc_auc': float(safe_auc(y_te, prob_te)),
            'pr_auc': float(safe_pr(y_te, prob_te)),
            'brier': float(brier_score_loss(y_te, prob_te)),
            'n': int(len(y_te)),
        }
    }
    return metrics, feat_names_aug, logreg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=None, help='Path to lag1 modeling CSV (defaults to nearby model_lag1_*.csv)')
    p.add_argument('--split_date', type=str, default='2022-08-31')
    p.add_argument('--sarimax_order', type=str, default='0,0,0', help='p,d,q')
    p.add_argument('--top_k', type=int, default=10, help='If >0, compute and report top-K features by abs(coef) and refit metrics')
    args = p.parse_args()

    df = load_panel(args.csv)
    id_cols, y_col, feat_cols = select_features(df)
    assert y_col is not None, 'failed column missing'
    assert len(feat_cols) > 0, 'No features selected after filtering'
    order_parts = [int(x.strip()) for x in args.sarimax_order.split(',')]
    assert len(order_parts) == 3, 'sarimax_order must be three comma-separated integers, e.g., 0,0,0'
    sarimax_order = tuple(order_parts)
    df_tr, df_te = time_split(df, args.split_date)
    metrics_all, feat_names_all, logreg_all = run_pipeline(df_tr, df_te, y_col, feat_cols, sarimax_order)

    # Save feature importance CSV
    coefs = np.asarray(logreg_all.coef_).ravel()
    imp = pd.DataFrame({'feature': feat_names_all, 'coef': coefs})
    imp['abs_coef'] = imp['coef'].abs()
    imp = imp.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    imp_path = OUT_DIR / f'feature_importance_l1_{ts}.csv'
    imp.to_csv(imp_path, index=False)
    print('Feature importance saved to:', imp_path)

    top_features = None
    metrics_top = None
    if args.top_k and args.top_k > 0:
        ranked = imp[imp['feature'] != 'sarimax_score']
        top_features = ranked['feature'].head(args.top_k).tolist()
        if top_features:
            print(f"Refitting with top-{args.top_k} features...")
            metrics_top, feat_names_top, logreg_top = run_pipeline(df_tr, df_te, y_col, top_features, sarimax_order)

    payload = {
        'split_date': args.split_date,
        'sarimax_order': sarimax_order,
        'features_raw': feat_cols,
        'metrics_all': metrics_all,
        'top_k': int(args.top_k or 0),
        'top_features': top_features,
        'metrics_top': metrics_top,
    }

    out_path = OUT_DIR / f'hybrid_sarimax_l1_metrics_{ts}.json'
    out_path.write_text(pd.Series(payload).to_json(indent=2))
    print('Saved:', out_path)
    print('All features -> Test ROC=%.3f PR=%.3f Brier=%.4f (n=%d)' % (
        metrics_all['test']['roc_auc'], metrics_all['test']['pr_auc'], metrics_all['test']['brier'], metrics_all['test']['n']))
    if metrics_top:
        print('Top-%d features -> Test ROC=%.3f PR=%.3f Brier=%.4f (n=%d)' % (
            args.top_k, metrics_top['test']['roc_auc'], metrics_top['test']['pr_auc'], metrics_top['test']['brier'], metrics_top['test']['n']))


if __name__ == '__main__':
    main() 