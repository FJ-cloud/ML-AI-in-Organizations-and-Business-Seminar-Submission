#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path
from datetime import datetime
import warnings

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
except Exception as e:
    sm = None

BASE_DIR = Path('/home/fj/UkrainianInsolvencyPredictor')
DEFAULT_DATA_DIR = BASE_DIR / 'data' / 'processed' / 'model_ready'
EDA_DIR = BASE_DIR / 'eda'
EDA_DIR.mkdir(exist_ok=True)

# Default big-magnitude columns likely to be highly skewed
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


def latest(path_glob: str) -> str:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f'No files matched: {path_glob}')
    return files[-1]


def load_panel(csv_path: str | None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = latest(str(DEFAULT_DATA_DIR / 'model_lag1_*.csv'))
    df = pd.read_csv(csv_path, parse_dates=['date_m'])
    return df


def load_feature_list(path: str | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    names = []
    for line in p.read_text().splitlines():
        t = line.strip()
        if not t or t.startswith('#'):
            continue
        t = t.split(':')[0].strip()
        names.append(t)
    return names or None


def select_features(df: pd.DataFrame, feature_names: list[str] | None):
    id_cols = [c for c in ['bank_name','date_m','year_month','failed','processing_timestamp'] if c in df.columns]
    y_col = 'failed' if 'failed' in df.columns else None
    all_feats = [c for c in df.columns if c not in id_cols]
    feat_numeric = [c for c in all_feats if pd.api.types.is_numeric_dtype(df[c])]
    if feature_names:
        feat_numeric = [c for c in feature_names if c in feat_numeric]
    return id_cols, y_col, feat_numeric


def time_split(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = pd.Timestamp(split_date)
    df_tr = df[df['date_m'] <= dt].copy()
    df_te = df[df['date_m'] > dt].copy()
    return df_tr, df_te


def build_preprocessor(feat_cols: list[str]) -> tuple[Pipeline, list[str], list[str]]:
    # Intersect default amount list with selected features
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

    # Final fallback to remove any remaining NaNs (e.g., columns entirely missing in train)
    full_pre = Pipeline([
        ('pre', pre),
        ('final_impute', SimpleImputer(strategy='constant', fill_value=0.0)),
    ])

    return full_pre, amount_cols, ratio_cols


def fit_bank_sarimax(y: pd.Series, X: pd.DataFrame | np.ndarray, order=(1,0,0)):
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
        p = res.predict(start=0, end=len(X)-1, exog=np.asarray(X, dtype=float))
        return np.asarray(p)
    except Exception:
        return np.zeros(len(X))


def compute_importance(feat_names: list[str], coef: np.ndarray) -> pd.DataFrame:
    coefs = np.asarray(coef).ravel()
    imp = pd.DataFrame({'feature': feat_names, 'coef': coefs})
    imp['abs_coef'] = imp['coef'].abs()
    imp = imp.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = EDA_DIR / f'feature_importance_l1_{ts}.csv'
    imp.to_csv(out_csv, index=False)
    print('Feature importance saved to:', out_csv)
    return imp


def run_pipeline(df_tr, df_te, y_col, feat_cols, sarimax_order):
    # Column-wise preprocessing (log1p+scale for amounts; scale for ratios)
    pre, amt_cols, ratio_cols = build_preprocessor(feat_cols)
    X_tr = pre.fit_transform(df_tr[feat_cols])
    X_te = pre.transform(df_te[feat_cols])
    feat_names = list(pre.named_steps['pre'].get_feature_names_out())

    # For SARIMAX, build DataFrames aligned to rows
    X_tr_df = pd.DataFrame(X_tr, index=df_tr.index, columns=feat_names)
    X_te_df = pd.DataFrame(X_te, index=df_te.index, columns=feat_names)

    y_tr = df_tr[y_col].astype(float).values
    y_te = df_te[y_col].astype(float).values

    # Per-bank SARIMAX on train, then forecast on test
    sarimax_pred_tr = np.zeros_like(y_tr)
    sarimax_pred_te = np.zeros_like(y_te)

    for bank in sorted(df_tr['bank_name'].unique()):
        tr_idx = df_tr['bank_name'] == bank
        te_idx = df_te['bank_name'] == bank
        if tr_idx.sum() < 12:
            continue
        yb_tr = df_tr.loc[tr_idx, y_col]
        Xb_tr = X_tr_df.loc[tr_idx, feat_names]
        res = fit_bank_sarimax(yb_tr, Xb_tr, order=sarimax_order)
        sarimax_pred_tr[tr_idx.values] = predict_bank_sarimax(res, Xb_tr)
        if te_idx.any():
            Xb_te = X_te_df.loc[te_idx, feat_names]
            sarimax_pred_te[te_idx.values] = predict_bank_sarimax(res, Xb_te)

    # Augment features with SARIMAX score
    X_tr_aug = np.c_[X_tr, sarimax_pred_tr]
    X_te_aug = np.c_[X_te, sarimax_pred_te]
    feat_names_aug = feat_names + ['sarimax_score']

    # L1-penalized Logistic Regression with CV
    logreg = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='l1',
        solver='liblinear',
        scoring='roc_auc',
        max_iter=2000,
        n_jobs=None,
        refit=True,
    ).fit(X_tr_aug, y_tr)

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


def train_and_eval(args):
    df = load_panel(args.csv)
    feat_list = load_feature_list(args.features)
    id_cols, y_col, feat_cols = select_features(df, feat_list)
    assert y_col is not None, 'failed column missing'
    assert len(feat_cols) > 0, 'No features selected after filtering'

    # Parse SARIMAX order
    order_parts = [int(x.strip()) for x in args.sarimax_order.split(',')]
    assert len(order_parts) == 3, 'sarimax_order must be three comma-separated integers, e.g., 0,0,0'
    sarimax_order = tuple(order_parts)

    # Split
    df_tr, df_te = time_split(df, args.split_date)

    # First pass: all features
    metrics_all, feat_names_all, logreg_all = run_pipeline(df_tr, df_te, y_col, feat_cols, sarimax_order)

    # Importance on scaled features (exclude SARIMAX score from top-K selection)
    imp = compute_importance(feat_names_all, logreg_all.coef_)

    # Optionally refit with top-K features
    metrics_top = None
    top_features = None
    if args.top_k and args.top_k > 0:
        ranked = imp[imp['feature'] != 'sarimax_score']
        top_features = ranked['feature'].head(args.top_k).tolist()
        metrics_top, feat_names_top, logreg_top = run_pipeline(df_tr, df_te, y_col, top_features, sarimax_order)

    # Package metrics
    payload = {
        'split_date': args.split_date,
        'sarimax_order': sarimax_order,
        'features_raw': feat_cols,
        'metrics_all': metrics_all,
        'top_k': int(args.top_k or 0),
        'top_features': top_features,
        'metrics_top': metrics_top,
    }

    print('\nHYBRID SARIMAX + L1-Logistic')
    print('Split at:', args.split_date)
    print('SARIMAX order:', sarimax_order)
    print('All features -> Train n=%d ROC=%.3f PR=%.3f Brier=%.4f | Test n=%d ROC=%.3f PR=%.3f Brier=%.4f' % (
        metrics_all['train']['n'], metrics_all['train']['roc_auc'], metrics_all['train']['pr_auc'], metrics_all['train']['brier'],
        metrics_all['test']['n'], metrics_all['test']['roc_auc'], metrics_all['test']['pr_auc'], metrics_all['test']['brier']
    ))
    if metrics_top:
        print('Top-%d features -> Train n=%d ROC=%.3f PR=%.3f Brier=%.4f | Test n=%d ROC=%.3f PR=%.3f Brier=%.4f' % (
            args.top_k,
            metrics_top['train']['n'], metrics_top['train']['roc_auc'], metrics_top['train']['pr_auc'], metrics_top['train']['brier'],
            metrics_top['test']['n'], metrics_top['test']['roc_auc'], metrics_top['test']['pr_auc'], metrics_top['test']['brier']
        ))

    out_path = BASE_DIR / 'eda' / f'hybrid_sarimax_l1_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out_path.write_text(pd.Series(payload).to_json(indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default=None, help='Path to lag1 modeling CSV (defaults to latest model_lag1_*.csv)')
    p.add_argument('--split_date', type=str, default='2022-08-31', help='Time-based split date (YYYY-MM-DD)')
    p.add_argument('--features', type=str, default=None, help='Path to a text file listing feature names (one per line, or "name: score")')
    p.add_argument('--sarimax_order', type=str, default='0,0,0', help='ARIMA order p,d,q for SARIMAX, e.g., 0,0,0')
    p.add_argument('--top_k', type=int, default=0, help='If >0, refit using only top-K features by L1 abs(coef) after scaling')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_and_eval(args) 