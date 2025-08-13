#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    split_date: str = '2022-08-31'
    target_col: str = 'will_fail_within_1m'
    drop_post_failure: bool = True
    add_trend_vol: bool = False
    roll_windows: tuple[int, ...] = (3, 6)
    topk_list: tuple[int, ...] = (3, 5, 10)


def latest_model_lag1() -> Path:
    cand = sorted((BASE / 'data' / 'processed' / 'model_ready').glob('model_lag1_*.csv'))
    if not cand:
        raise SystemExit('No model_lag1_*.csv found')
    return cand[-1]


def time_split(df: pd.DataFrame, split_date: str):
    dt = pd.Timestamp(split_date)
    return df[df['date_m'] <= dt].copy(), df[df['date_m'] > dt].copy()


def get_features(df: pd.DataFrame) -> list[str]:
    non_features = {'bank_name','date_m','year_month','processing_timestamp','failure_date','last_reporting_date','post_failure','failed','will_fail_within_6m','will_fail_within_1m'}
    cols = [c for c in df.columns if c not in non_features and not str(c).endswith('_present')]
    return cols


def add_trend_volatility(df: pd.DataFrame, feat_cols: list[str], windows=(3,6)) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values(['bank_name','date_m']).copy()
    new_cols: list[str] = []
    for w in windows:
        for c in feat_cols:
            ma = f'{c}_ma{w}'
            sd = f'{c}_std{w}'
            df[ma] = df.groupby('bank_name')[c].transform(lambda s: s.rolling(w, min_periods=max(1,w//2)).mean())
            df[sd] = df.groupby('bank_name')[c].transform(lambda s: s.rolling(w, min_periods=max(1,w//2)).std())
            new_cols.extend([ma, sd])
    return df, new_cols


def score(y, p):
    return dict(
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y))>1 else np.nan,
        brier=float(brier_score_loss(y, p)),
        n=int(len(y)),
    )


def time_cv_months(train_df: pd.DataFrame, feat_cols: list[str], y_col: str, n_splits: int = 5) -> list[tuple[np.ndarray,np.ndarray]]:
    months = np.sort(train_df['date_m'].dt.to_period('M').unique().astype(str))
    idxs = np.linspace(0, len(months), n_splits+1, dtype=int)
    splits = []
    for i in range(n_splits):
        start, end = idxs[i], idxs[i+1]
        if end - start < 3:
            continue
        sel = months[start:end]
        m = train_df['date_m'].dt.to_period('M').astype(str).isin(sel).values
        y = train_df[y_col].values
        if np.unique(y[m]).size < 2:
            continue
        splits.append(np.where(m)[0])
    return [(np.array([], dtype=int), s) for s in splits]  # we only need validation indices


def evaluate_model(train_df, test_df, feat_cols, y_col, model_name: str):
    # Build design matrices
    X_tr = train_df[feat_cols].astype(float).values
    y_tr = train_df[y_col].astype(int).values
    X_te = test_df[feat_cols].astype(float).values
    y_te = test_df[y_col].astype(int).values

    # Define models/grids
    models = []
    if model_name == 'logit':
        models = [
            ('logit_std_C1', ('std', LogisticRegression(max_iter=5000, class_weight='balanced', C=1.0))),
            ('logit_std_C0.3', ('std', LogisticRegression(max_iter=5000, class_weight='balanced', C=0.3))),
            ('logit_std_C3', ('std', LogisticRegression(max_iter=5000, class_weight='balanced', C=3.0))),
        ]
    elif model_name == 'hgb':
        models = [
            ('hgb_lr0.03_d6', ('none', HistGradientBoostingClassifier(learning_rate=0.03, max_depth=6, max_leaf_nodes=31, min_samples_leaf=20))),
            ('hgb_lr0.05_d6', ('none', HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_leaf_nodes=31, min_samples_leaf=20))),
            ('hgb_lr0.02_d8', ('none', HistGradientBoostingClassifier(learning_rate=0.02, max_depth=8, max_leaf_nodes=63, min_samples_leaf=10))),
        ]
    else:
        raise ValueError('Unknown model_name')

    # Time-sliced CV on train optimizing PR AUC
    splits = time_cv_months(train_df, feat_cols, y_col, n_splits=5)
    best_name, best_model, best_pr = None, None, -1.0

    for name, (prep, model) in models:
        fold_scores = []
        for _, val_idx in splits:
            Xv = X_tr[val_idx]
            yv = y_tr[val_idx]
            Xt = np.delete(X_tr, val_idx, axis=0)
            yt = np.delete(y_tr, val_idx, axis=0)
            # Preprocess
            if prep == 'std':
                imp = SimpleImputer(strategy='median')
                Xt = imp.fit_transform(Xt)
                Xv = imp.transform(Xv)
                ss = StandardScaler(with_mean=False)
                Xt = ss.fit_transform(Xt)
                Xv = ss.transform(Xv)
            elif prep == 'none':
                imp = SimpleImputer(strategy='median')
                Xt = imp.fit_transform(Xt)
                Xv = imp.transform(Xv)
            model.fit(Xt, yt)
            pv = model.predict_proba(Xv)[:,1] if hasattr(model,'predict_proba') else model.predict_proba(Xv)
            fold_scores.append(average_precision_score(yv, pv))
        mean_pr = float(np.mean(fold_scores)) if fold_scores else -1.0
        if mean_pr > best_pr:
            best_pr = mean_pr
            best_name = name
            best_model = (prep, model)

    # Refit best on full train and evaluate on test
    prep, final_model = best_model
    if prep == 'std':
        imp = SimpleImputer(strategy='median')
        X_tr2 = imp.fit_transform(X_tr)
        X_te2 = imp.transform(X_te)
        ss = StandardScaler(with_mean=False)
        X_tr2 = ss.fit_transform(X_tr2)
        X_te2 = ss.transform(X_te2)
    else:
        imp = SimpleImputer(strategy='median')
        X_tr2 = imp.fit_transform(X_tr)
        X_te2 = imp.transform(X_te)
    final_model.fit(X_tr2, y_tr)
    p_tr = final_model.predict_proba(X_tr2)[:,1] if hasattr(final_model,'predict_proba') else final_model.predict_proba(X_tr2)
    p_te = final_model.predict_proba(X_te2)[:,1] if hasattr(final_model,'predict_proba') else final_model.predict_proba(X_te2)

    return best_name, best_pr, p_tr, p_te, y_tr, y_te


def topk_capture(test_df: pd.DataFrame, p_te: np.ndarray, y_te: np.ndarray, k_list: list[int]) -> dict:
    # Compute per-month top-K capture (recall) and precision
    df = test_df[['date_m']].copy()
    df['y'] = y_te
    df['p'] = p_te
    df['month'] = df['date_m'].dt.to_period('M')
    out = {}
    for k in k_list:
        captured = 0
        total_pos = int(df['y'].sum())
        alerts = 0
        for m, grp in df.groupby('month'):
            g = grp.sort_values('p', ascending=False)
            topk = g.head(k)
            alerts += len(topk)
            captured += int(topk['y'].sum())
        rec = captured / total_pos if total_pos > 0 else 0.0
        prec = captured / alerts if alerts > 0 else 0.0
        out[k] = {'recall': rec, 'precision': prec, 'captured': captured, 'alerts': alerts, 'positives': total_pos}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None)
    ap.add_argument('--split_date', type=str, default='2022-08-31')
    ap.add_argument('--model', type=str, default='hgb', choices=['hgb','logit'])
    ap.add_argument('--add_trend', action='store_true')
    ap.add_argument('--topk', type=str, default='3,5,10')
    args = ap.parse_args()

    cfg = Config(split_date=args.split_date, add_trend_vol=args.add_trend, topk_list=tuple(int(x) for x in args.topk.split(',')))
    csv = args.csv or str(latest_model_lag1())

    df = pd.read_csv(csv, parse_dates=['date_m'])
    # Early-warning setup
    if cfg.drop_post_failure and 'post_failure' in df.columns:
        df = df[df['post_failure']==False].copy()
    if cfg.target_col not in df.columns:
        raise SystemExit(f"Target {cfg.target_col} not in CSV")

    feat_cols = get_features(df)
    if cfg.add_trend_vol:
        df, new_cols = add_trend_volatility(df, feat_cols, windows=cfg.roll_windows)
        feat_cols += new_cols

    y_col = cfg.target_col
    df_tr, df_te = time_split(df, cfg.split_date)

    best_name, best_pr, p_tr, p_te, y_tr, y_te = evaluate_model(df_tr, df_te, feat_cols, y_col, args.model)

    metrics = {
        'model': args.model,
        'best_variant': best_name,
        'cv_pr_auc': best_pr,
        'train': score(y_tr, p_tr),
        'test': score(y_te, p_te),
        'topk': topk_capture(df_te, p_te, y_te, list(cfg.topk_list)),
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = OUT_DIR / f'metrics_{args.model}_{ts}.json'
    out_pred = OUT_DIR / f'predictions_{args.model}_{ts}.csv'
    pd.Series(metrics).to_json(out_json, indent=2)

    preds_tr = pd.DataFrame({'split':'train','y':y_tr,'p':p_tr})
    preds_te = pd.DataFrame({'split':'test','y':y_te,'p':p_te})
    # include identifiers for test rows
    preds_te['date_m'] = df_te['date_m'].values
    if 'bank_name' in df_te.columns:
        preds_te['bank_name'] = df_te['bank_name'].values
    preds = pd.concat([preds_tr, preds_te], ignore_index=True)
    preds.to_csv(out_pred, index=False)

    print('Saved:', out_json)
    print('Saved:', out_pred)
    print('Test PR=%.3f ROC=%.3f Brier=%.4f (n=%d)' % (
        metrics['test']['pr_auc'], metrics['test']['roc_auc'], metrics['test']['brier'], metrics['test']['n']))
    for k, v in metrics['topk'].items():
        print(f"Top-{k} per-month: recall={v['recall']:.3f} precision={v['precision']:.3f} captured={v['captured']}/{v['positives']} alerts={v['alerts']}")


if __name__ == '__main__':
    main() 