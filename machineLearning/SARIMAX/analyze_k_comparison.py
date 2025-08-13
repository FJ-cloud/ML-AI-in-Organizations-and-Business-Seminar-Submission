#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
MR_DIR = BASE / 'data' / 'processed' / 'model_ready'
MODELS_DIR = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)

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

try:
    import statsmodels.api as sm
except Exception:
    sm = None


def latest_model_ready() -> Path:
    files = sorted(MR_DIR.glob('model_lag1_*.csv'))
    if not files:
        raise SystemExit('No model_lag1_*.csv found. Run pipeline first.')
    return files[-1]


def load_panel(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, parse_dates=['date_m'])


def select_features(df: pd.DataFrame, feature_names=None):
    id_cols = [c for c in ['bank_name','date_m','year_month','failed','processing_timestamp'] if c in df.columns]
    y_col = 'failed'
    all_feats = [c for c in df.columns if c not in id_cols]
    feat_numeric = [c for c in all_feats if pd.api.types.is_numeric_dtype(df[c])]
    if feature_names:
        feat_numeric = [c for c in feature_names if c in feat_numeric]
    return y_col, feat_numeric


def time_split(df: pd.DataFrame, split_date: str):
    dt = pd.Timestamp(split_date)
    return df[df['date_m'] <= dt].copy(), df[df['date_m'] > dt].copy()


def build_preprocessor(feat_cols):
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
    return full_pre


def fit_bank_sarimax(y: pd.Series, X: pd.DataFrame | np.ndarray, order=(0,0,0)):
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


def run_pipeline(df_tr, df_te, y_col, feat_cols, sarimax_order):
    pre = build_preprocessor(feat_cols)
    X_tr = pre.fit_transform(df_tr[feat_cols])
    X_te = pre.transform(df_te[feat_cols])
    feat_names = list(pre.named_steps['pre'].get_feature_names_out())

    X_tr_df = pd.DataFrame(X_tr, index=df_tr.index, columns=feat_names)
    X_te_df = pd.DataFrame(X_te, index=df_te.index, columns=feat_names)

    y_tr = df_tr[y_col].astype(float).values
    y_te = df_te[y_col].astype(float).values

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

    X_tr_aug = np.c_[X_tr, sarimax_pred_tr]
    X_te_aug = np.c_[X_te, sarimax_pred_te]
    feat_names_aug = feat_names + ['sarimax_score']

    logreg = LogisticRegressionCV(Cs=10, cv=5, penalty='l1', solver='liblinear',
                                  scoring='roc_auc', max_iter=2000, refit=True).fit(X_tr_aug, y_tr)
    prob_tr = logreg.predict_proba(X_tr_aug)[:, 1]
    prob_te = logreg.predict_proba(X_te_aug)[:, 1]

    return (y_tr, prob_tr), (y_te, prob_te), feat_names_aug, logreg


def rank_features(feat_names_aug, coef):
    coefs = np.asarray(coef).ravel()
    imp = pd.DataFrame({'feature': feat_names_aug, 'coef': coefs})
    imp['abs_coef'] = imp['coef'].abs()
    imp = imp[imp['feature'] != 'sarimax_score']
    imp = imp.sort_values('abs_coef', ascending=False).reset_index(drop=True)
    return imp


def precision_recall_at_percentiles(y_true, y_score, percents=(0.05, 0.10, 0.20)):
    n = len(y_true)
    res = []
    order = np.argsort(-y_score)
    for p in percents:
        k = max(1, int(round(p * n)))
        sel = order[:k]
        tp = y_true[sel].sum()
        precision = float(tp / k)
        recall = float(tp / y_true.sum()) if y_true.sum() > 0 else np.nan
        res.append({'percent': p, 'k': k, 'precision': precision, 'recall': recall})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_date', default='2022-08-31')
    ap.add_argument('--sarimax_order', default='0,0,0')
    ap.add_argument('--k_values', default='10,30')
    args = ap.parse_args()

    sarimax_order = tuple(int(x) for x in args.sarimax_order.split(','))

    csv = latest_model_ready()
    df = load_panel(csv)
    y_col, feat_cols_all = select_features(df)
    df_tr, df_te = time_split(df, args.split_date)

    # Fit on all features to get ranking
    (ytr_all, ptr_all), (yte_all, pte_all), feat_names_aug, logreg_all = run_pipeline(
        df_tr, df_te, y_col, feat_cols_all, sarimax_order
    )
    ranking = rank_features(feat_names_aug, logreg_all.coef_)

    # Prepare plots
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pr_png = MODELS_DIR / f'k_compare_pr_curve_{ts}.png'
    cal_png = MODELS_DIR / f'k_compare_calibration_{ts}.png'
    md_path = MODELS_DIR / f'k_compare_summary_{ts}.md'

    plt.figure(figsize=(6,4))
    # Baseline (all)
    prec_all, rec_all, _ = precision_recall_curve(yte_all, pte_all)
    ap_all = average_precision_score(yte_all, pte_all)
    plt.plot(rec_all, prec_all, label=f'All (AP={ap_all:.3f})', lw=2)

    results = []

    for k in [int(x) for x in args.k_values.split(',') if x.strip()]:
        topk = ranking['feature'].head(k).tolist()
        # Refit on top-K
        (ytr_k, ptr_k), (yte_k, pte_k), _, _ = run_pipeline(df_tr, df_te, y_col, topk, sarimax_order)
        prec_k, rec_k, _ = precision_recall_curve(yte_k, pte_k)
        ap_k = average_precision_score(yte_k, pte_k)
        roc_k = roc_auc_score(yte_k, pte_k)
        results.append({'k': k, 'ap': float(ap_k), 'roc': float(roc_k)})
        plt.plot(rec_k, prec_k, label=f'Top-{k} (AP={ap_k:.3f})', lw=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_png, dpi=150)

    # Calibration curves
    plt.figure(figsize=(6,4))
    prob_true_all, prob_pred_all = calibration_curve(yte_all, pte_all, n_bins=10)
    plt.plot(prob_pred_all, prob_true_all, marker='o', label='All')
    for k in [int(x) for x in args.k_values.split(',') if x.strip()]:
        topk = ranking['feature'].head(k).tolist()
        (_, _), (yte_k, pte_k), _, _ = run_pipeline(df_tr, df_te, y_col, topk, sarimax_order)
        prob_true, prob_pred = calibration_curve(yte_k, pte_k, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'Top-{k}')
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel('Mean predicted prob')
    plt.ylabel('Empirical prob')
    plt.title('Calibration (Test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(cal_png, dpi=150)

    # Precision/Recall at percentiles summary
    lines = ['# K comparison (test set)', '', f'- Split: {args.split_date}', f'- Order: {args.sarimax_order}', '']
    # Baseline all
    prk_all = precision_recall_at_percentiles(yte_all, pte_all)
    lines.append('## All features')
    lines.append('| % | k | Precision | Recall |')
    lines.append('|---:|---:|---:|---:|')
    for r in prk_all:
        lines.append(f"| {int(r['percent']*100)} | {r['k']} | {r['precision']:.3f} | {r['recall']:.3f} |")

    for k in [int(x) for x in args.k_values.split(',') if x.strip()]:
        topk = ranking['feature'].head(k).tolist()
        (_, _), (yte_k, pte_k), _, _ = run_pipeline(df_tr, df_te, y_col, topk, sarimax_order)
        prk = precision_recall_at_percentiles(yte_k, pte_k)
        lines.append(f'\n## Top-{k} features')
        lines.append('| % | k | Precision | Recall |')
        lines.append('|---:|---:|---:|---:|')
        for r in prk:
            lines.append(f"| {int(r['percent']*100)} | {r['k']} | {r['precision']:.3f} | {r['recall']:.3f} |")

    # Save summary
    md_path.write_text('\n'.join(lines))

    print('WROTE:', pr_png)
    print('WROTE:', cal_png)
    print('WROTE:', md_path)


if __name__ == '__main__':
    main() 