#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
except Exception:
    sm = None

# Reuse utilities from existing pipeline
from machineLearning.SARIMAX.train_l1_sarimax import (
    load_panel,
    load_feature_list,
    select_features,
    time_split,
    build_preprocessor,
    BASE_DIR,
)

EDA_DIR = BASE_DIR / 'eda'
EDA_DIR.mkdir(exist_ok=True)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


def safe_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float('nan')


def fit_residual_arima_per_bank(
    df_tr: pd.DataFrame,
    df_te: pd.DataFrame,
    bank_col: str,
    residual_tr: np.ndarray,
    resid_order: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ARIMA/SARIMAX on residuals per bank (train only), forecast test residuals.

    Returns:
        r_tr_hat: in-sample fitted residuals aligned to train rows
        r_te_hat: out-of-sample forecast residuals aligned to test rows
    """
    r_tr_hat = np.zeros_like(residual_tr, dtype=float)
    r_te_hat = np.zeros(len(df_te), dtype=float)

    banks = sorted(df_tr[bank_col].unique())
    for bank in banks:
        tr_idx_mask = (df_tr[bank_col] == bank)
        te_idx_mask = (df_te[bank_col] == bank)
        n_tr = int(tr_idx_mask.sum())
        n_te = int(te_idx_mask.sum())
        if n_tr < 8:  # need enough history for ARIMA stability
            continue
        series_tr = pd.Series(residual_tr[tr_idx_mask.values], index=np.arange(n_tr))
        try:
            if sm is None:
                continue
            model = sm.tsa.statespace.SARIMAX(
                endog=series_tr.astype(float),
                order=resid_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, method='lbfgs', maxiter=200)
            # In-sample one-step-ahead predictions for train
            fitted = res.get_prediction().predicted_mean
            if len(fitted) == n_tr:
                r_tr_hat[tr_idx_mask.values] = np.asarray(fitted)
            # Forecast for test horizon
            if n_te > 0:
                fc = res.get_forecast(steps=n_te).predicted_mean
                r_te_hat[te_idx_mask.values] = np.asarray(fc)
        except Exception:
            # leave zeros for this bank
            continue

    return r_tr_hat, r_te_hat


def run_two_stage(df_tr: pd.DataFrame, df_te: pd.DataFrame, y_col: str, feat_cols: list[str], resid_order: tuple[int, int, int]):
    # Preprocess features (log-scale for big amounts, scale others)
    pre, _, _ = build_preprocessor(feat_cols)
    X_tr = pre.fit_transform(df_tr[feat_cols])
    X_te = pre.transform(df_te[feat_cols])

    y_tr = df_tr[y_col].astype(float).values
    y_te = df_te[y_col].astype(float).values

    # Base L1-Logistic
    logreg = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='l1',
        solver='liblinear',
        scoring='roc_auc',
        max_iter=2000,
        n_jobs=None,
        refit=True,
    ).fit(X_tr, y_tr)

    prob_tr = logreg.predict_proba(X_tr)[:, 1]
    prob_te = logreg.predict_proba(X_te)[:, 1]

    # Residuals on train
    resid_tr = y_tr - prob_tr

    # Per-bank ARIMA on residuals; forecast test residuals
    r_tr_hat, r_te_hat = fit_residual_arima_per_bank(
        df_tr=df_tr,
        df_te=df_te,
        bank_col='bank_name',
        residual_tr=resid_tr,
        resid_order=resid_order,
    )

    # Adjusted probabilities
    prob_tr_final = np.clip(prob_tr + r_tr_hat, 0.0, 1.0)
    prob_te_final = np.clip(prob_te + r_te_hat, 0.0, 1.0)

    # Metrics: base and adjusted
    base_metrics = {
        'train': {
            'roc_auc': safe_auc(y_tr, prob_tr),
            'pr_auc': safe_pr(y_tr, prob_tr),
            'brier': float(brier_score_loss(y_tr, prob_tr)),
            'n': int(len(y_tr)),
        },
        'test': {
            'roc_auc': safe_auc(y_te, prob_te),
            'pr_auc': safe_pr(y_te, prob_te),
            'brier': float(brier_score_loss(y_te, prob_te)),
            'n': int(len(y_te)),
        }
    }

    adjusted_metrics = {
        'train': {
            'roc_auc': safe_auc(y_tr, prob_tr_final),
            'pr_auc': safe_pr(y_tr, prob_tr_final),
            'brier': float(brier_score_loss(y_tr, prob_tr_final)),
            'n': int(len(y_tr)),
        },
        'test': {
            'roc_auc': safe_auc(y_te, prob_te_final),
            'pr_auc': safe_pr(y_te, prob_te_final),
            'brier': float(brier_score_loss(y_te, prob_te_final)),
            'n': int(len(y_te)),
        }
    }

    return base_metrics, adjusted_metrics


def train_and_eval(args):
    df = load_panel(args.csv)
    feat_list = load_feature_list(args.features)
    id_cols, y_col, feat_cols = select_features(df, feat_list)
    assert y_col is not None, 'failed column missing'
    assert len(feat_cols) > 0, 'No features selected after filtering'

    # Split
    df_tr, df_te = time_split(df, args.split_date)

    # Residual ARIMA order
    parts = [int(x.strip()) for x in args.resid_order.split(',')]
    assert len(parts) == 3, 'resid_order must be p,d,q (e.g., 1,0,0)'
    resid_order = tuple(parts)

    # Run two-stage
    base_metrics, adjusted_metrics = run_two_stage(df_tr, df_te, y_col, feat_cols, resid_order)

    # Save payload
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    payload = {
        'model': 'logistic + residual_arima',
        'split_date': args.split_date,
        'resid_order': resid_order,
        'features_raw': feat_cols,
        'metrics_base': base_metrics,
        'metrics_adjusted': adjusted_metrics,
    }
    out_path = EDA_DIR / f'logit_resid_arima_metrics_{ts}.json'
    Path(out_path).write_text(pd.Series(payload).to_json(indent=2))

    print('\nL1-Logistic + Residual ARIMA (per bank)')
    print('Split at:', args.split_date)
    print('Residual ARIMA order:', resid_order)
    print('Base    -> Train n=%d ROC=%.3f PR=%.3f Brier=%.4f | Test n=%d ROC=%.3f PR=%.3f Brier=%.4f' % (
        base_metrics['train']['n'], base_metrics['train']['roc_auc'], base_metrics['train']['pr_auc'], base_metrics['train']['brier'],
        base_metrics['test']['n'], base_metrics['test']['roc_auc'], base_metrics['test']['pr_auc'], base_metrics['test']['brier']
    ))
    print('Adjusted-> Train n=%d ROC=%.3f PR=%.3f Brier=%.4f | Test n=%d ROC=%.3f PR=%.3f Brier=%.4f' % (
        adjusted_metrics['train']['n'], adjusted_metrics['train']['roc_auc'], adjusted_metrics['train']['pr_auc'], adjusted_metrics['train']['brier'],
        adjusted_metrics['test']['n'], adjusted_metrics['test']['roc_auc'], adjusted_metrics['test']['pr_auc'], adjusted_metrics['test']['brier']
    ))
    print('Saved metrics to:', out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=None, help='Path to lag1 modeling CSV (defaults to latest model_lag1_*.csv)')
    p.add_argument('--split_date', type=str, default='2022-08-31', help='Time-based split date (YYYY-MM-DD)')
    p.add_argument('--features', type=str, default=None, help='Path to a text file listing feature names (one per line, or "name: score")')
    p.add_argument('--resid_order', type=str, default='1,0,0', help='ARIMA order p,d,q for residual modeling (e.g., 1,0,0)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_and_eval(args) 