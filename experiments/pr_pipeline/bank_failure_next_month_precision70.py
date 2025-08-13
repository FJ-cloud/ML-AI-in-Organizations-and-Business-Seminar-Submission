# filename: bank_failure_next_month_precision70.py
# Python 3.10+   pip install pandas numpy scikit-learn
# Optional:      pip install xgboost

import argparse, json, warnings, re, math
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# Optional XGBoost; otherwise we fall back to HistGradientBoosting
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingClassifier as HGB

# Output directory
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Config / CLI
# -------------------------
@dataclass
class CFG:
    csv: str
    date_col: Optional[str]
    id_col: Optional[str]
    target_col: str
    model: str               # "hgb" or "xgb"
    min_train_months: int
    folds: int
    topk: int
    precision_floor: float   # e.g., 0.70
    add_trend: bool
    roll_windows: Tuple[int, ...]
    neg_pos_ratio: int
    drop_post_failure: bool
    optimize_for: str        # "precision_floor" | "fixed_k" | "f_beta"
    fixed_k: int
    beta: float
    random_state: int = 42

def parse_args() -> CFG:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--date_col", default=None)
    p.add_argument("--id_col", default=None)
    p.add_argument("--target_col", default="failed")
    p.add_argument("--model", default="hgb", choices=["hgb","xgb"])
    p.add_argument("--min_train_months", type=int, default=18)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--precision_floor", type=float, default=0.70)
    p.add_argument("--add_trend", action="store_true")
    p.add_argument("--roll_windows", type=str, default="3,6",
                   help="Comma-separated rolling windows, e.g., 3,6,12")
    p.add_argument("--neg_pos_ratio", type=int, default=50,
                   help="Max negatives per positive in training per fold (downsampling)")
    p.add_argument("--drop_post_failure", action="store_true")
    p.add_argument("--optimize_for", type=str, default="precision_floor",
                   choices=["precision_floor","fixed_k","f_beta"],
                   help="Optimization objective for decision rule")
    p.add_argument("--fixed_k", type=int, default=10,
                   help="K for per-month fixed-K evaluation")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Beta for F-beta optimization (precision emphasis if <1)")
    args = p.parse_args()
    windows = tuple(int(x) for x in str(args.roll_windows).split(",") if str(x).strip())
    return CFG(csv=args.csv, date_col=args.date_col, id_col=args.id_col, target_col=args.target_col,
               model=args.model, min_train_months=args.min_train_months, folds=args.folds,
               topk=args.topk, precision_floor=args.precision_floor, add_trend=bool(args.add_trend),
               roll_windows=windows, neg_pos_ratio=args.neg_pos_ratio, drop_post_failure=bool(args.drop_post_failure),
               optimize_for=args.optimize_for, fixed_k=args.fixed_k, beta=args.beta)


# -------------------------
# Utilities
# -------------------------
def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.search(r"(date|period|month|report|time|ym)", c, re.I):
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                pass
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    return None

def detect_binary_target(df: pd.DataFrame) -> Optional[str]:
    prefs = ["failed","target_next","target","label_next","label","default","event"]
    for pref in prefs:
        for c in df.columns:
            if re.fullmatch(pref, c, re.I):
                try:
                    v = set(np.unique(df[c].dropna().astype(int)))
                    if v <= {0,1}: return c
                except Exception:
                    pass
    for c in df.columns:
        try:
            v = set(np.unique(df[c].dropna().astype(int)))
            if v <= {0,1}: return c
        except Exception:
            pass
    return None

def detect_id_col(df: pd.DataFrame, date_col: str) -> Optional[str]:
    months = pd.to_datetime(df[date_col]).dt.to_period("M").nunique()
    approx_ids = max(1, int(round(len(df) / max(1, months))))
    best, best_score = None, float("inf")
    for c in df.columns:
        if c == date_col:
            continue
        uniq = df[c].nunique(dropna=True)
        if 1 < uniq < len(df)*0.95:
            score = abs(uniq - approx_ids) * (0.5 if df[c].dtype == object else 1.0)
            if score < best_score:
                best, best_score = c, score
    return best

def build_time_folds(df: pd.DataFrame, date_col: str, min_train_months: int, n_folds: int):
    months_sorted = sorted(pd.to_datetime(df[date_col]).dt.to_period("M").unique())
    n_folds = min(n_folds, max(1, len(months_sorted) - min_train_months))
    folds = []
    for i in range(n_folds):
        tr_end = months_sorted[min_train_months - 1 + i].to_timestamp("M")
        va_m   = months_sorted[min_train_months + i].to_timestamp("M")
        folds.append((tr_end, va_m))
    return folds

def add_trend_volatility(df: pd.DataFrame, feat_cols: List[str], id_col: str, date_col: str, windows: Tuple[int, ...]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.sort_values([id_col, date_col]).copy()
    new_cols: List[str] = []
    for w in windows:
        for c in feat_cols:
            ma = f"{c}_ma{w}"
            sd = f"{c}_std{w}"
            df[ma] = df.groupby(id_col)[c].transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).mean())
            df[sd] = df.groupby(id_col)[c].transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).std())
            new_cols.extend([ma, sd])
    return df, new_cols


def fit_ensemble(X_tr, y_tr, model: str, random_state=42):
    # 1) Calibrated Logistic (probability quality)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_tr)
    lr = LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        solver="lbfgs", max_iter=400
    )
    lr_cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
    lr_cal.fit(Xs, y_tr)

    # 2) Tabular learner (nonlinear patterns)
    if model == "xgb" and HAS_XGB:
        pos = (y_tr == 1).sum()
        neg = (y_tr == 0).sum()
        spw = max(1.0, neg / max(1, pos))
        tab = XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            min_child_weight=1.0, reg_lambda=2.0,
            objective="binary:logistic", tree_method="hist",
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=random_state
        )
    else:
        tab = HGB(
            max_iter=600, learning_rate=0.05,
            min_samples_leaf=20, random_state=random_state
        )
    tab.fit(X_tr, y_tr)

    return {"scaler": scaler, "lr": lr_cal, "tab": tab}


def predict_proba(models, X):
    Xs = models["scaler"].transform(X)
    p1 = models["lr"].predict_proba(Xs)[:, 1]
    p2 = models["tab"].predict_proba(X)[:, 1]
    return 0.5*p1 + 0.5*p2


def choose_threshold_precision_floor(y_true, y_prob, precision_floor: float):
    """
    Find the threshold that achieves Precision >= precision_floor and maximizes Recall.
    If infeasible, pick the point with maximal Precision (tie-break: higher Recall).
    """
    p, r, t = precision_recall_curve(y_true, y_prob)
    thr_grid = np.concatenate(([0.0], t, [1.0]))
    best = {"precision":0.0, "recall":0.0, "threshold":None}
    # feasible region
    for thr in thr_grid:
        y_hat = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec  = recall_score(y_true, y_hat, zero_division=0)
        if prec >= precision_floor and rec > best["recall"]:
            best = {"precision":float(prec), "recall":float(rec), "threshold":float(thr)}
    feasible = best["threshold"] is not None
    if feasible:
        return feasible, best

    # infeasible: pick max precision (then higher recall)
    max_prec, max_rec, max_thr = -1.0, -1.0, None
    for thr in thr_grid:
        y_hat = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec  = recall_score(y_true, y_hat, zero_division=0)
        if (prec > max_prec) or (math.isclose(prec, max_prec) and rec > max_rec):
            max_prec, max_rec, max_thr = float(prec), float(rec), float(thr)
    return False, {"precision": max_prec, "recall": max_rec, "threshold": max_thr}


def choose_threshold_fbeta(y_true: np.ndarray, y_prob: np.ndarray, beta: float):
    p, r, t = precision_recall_curve(y_true, y_prob)
    thr_grid = np.concatenate(([0.0], t, [1.0]))
    best = {"precision":0.0, "recall":0.0, "f_beta":0.0, "threshold":0.5}
    beta2 = beta * beta
    for thr in thr_grid:
        y_hat = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec  = recall_score(y_true, y_hat, zero_division=0)
        denom = (beta2 * prec + rec)
        fbeta = (1+beta2) * prec * rec / denom if denom > 0 else 0.0
        if fbeta > best["f_beta"]:
            best = {"precision":float(prec), "recall":float(rec), "f_beta":float(fbeta), "threshold":float(thr)}
    return best


def fixed_k_metrics_per_month(dates: pd.Series, y_true: np.ndarray, y_prob: np.ndarray, k: int):
    dfm = pd.DataFrame({"date": pd.to_datetime(dates).dt.to_period("M"), "y": y_true, "p": y_prob})
    out = {}
    total_pos = int(dfm["y"].sum())
    captured = 0
    alerts = 0
    per_month = []
    for m, grp in dfm.groupby("date"):
        g = grp.sort_values("p", ascending=False)
        topk = g.head(k)
        alerts += len(topk)
        cap = int(topk["y"].sum())
        captured += cap
        per_month.append({"month": str(m), "alerts": int(len(topk)), "captured": cap, "positives": int(grp["y"].sum())})
    rec = captured / total_pos if total_pos > 0 else 0.0
    prec = captured / alerts if alerts > 0 else 0.0
    out["overall"] = {"k": int(k), "recall": rec, "precision": prec, "captured": int(captured), "positives": int(total_pos), "alerts": int(alerts)}
    out["per_month"] = per_month
    return out


# -------------------------
# Main
# -------------------------

def main(cfg: CFG):
    # Load
    df = pd.read_csv(cfg.csv)

    # Detect/validate core columns
    date_col = cfg.date_col or detect_date_col(df)
    if date_col is None:
        raise ValueError("Could not detect date column. Pass --date_col.")
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)

    target_col = cfg.target_col or detect_binary_target(df)
    if target_col is None:
        raise ValueError("Could not detect binary target column. Pass --target_col.")

    id_col = cfg.id_col or detect_id_col(df, date_col)
    if id_col is None:
        # fallback: any object column that's not date/target
        obj_cols = [c for c in df.columns if df[c].dtype == object and c not in (date_col, target_col)]
        id_col = obj_cols[0] if obj_cols else df.columns[0]

    # Filter out post-failure rows (align with baseline early-warning setup)
    if cfg.drop_post_failure and 'post_failure' in df.columns:
        df = df[df['post_failure'] == False].copy()

    # Features
    drop_cols = {date_col, id_col, target_col}
    # Exclude known non-features to avoid leakage and identifiers
    non_features = {"bank_name","date","date_m","year_month","processing_timestamp","failure_date","last_reporting_date","post_failure","failed","will_fail_within_6m","will_fail_within_1m"}
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    feats     = [c for c in num_cols if c not in drop_cols and c not in non_features and not str(c).endswith("_present") and df[c].nunique(dropna=False) > 1]

    # Optional trend/volatility features
    if cfg.add_trend and feats:
        df, new_cols = add_trend_volatility(df, feats, id_col=id_col, date_col=date_col, windows=cfg.roll_windows)
        feats += new_cols

    # Time-aware folds
    folds = build_time_folds(df, date_col, cfg.min_train_months, cfg.folds)

    # Train/validate
    rng = np.random.default_rng(cfg.random_state)
    oof_prob = np.full(len(df), np.nan, dtype=float)
    fold_rows: List[dict] = []

    for i, (tr_end, va_m) in enumerate(folds, 1):
        va_mask = df[date_col].dt.to_period("M") == va_m.to_period("M")
        tr_mask = df[date_col] <= tr_end
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        if len(va_idx) == 0:
            continue
        y_tr_full = df[target_col].astype(int).values[tr_idx]
        y_va = df[target_col].astype(int).values[va_idx]
        if len(np.unique(y_tr_full)) < 2:
            continue

        # Class-balanced downsampling of negatives in training
        pos_mask = y_tr_full == 1
        neg_mask = ~pos_mask
        pos_idx = tr_idx[pos_mask]
        neg_idx = tr_idx[neg_mask]
        n_pos = len(pos_idx)
        if n_pos == 0:
            continue
        max_negs = min(len(neg_idx), cfg.neg_pos_ratio * n_pos)
        if max_negs < len(neg_idx):
            sel_neg_idx = rng.choice(neg_idx, size=max_negs, replace=False)
        else:
            sel_neg_idx = neg_idx
        sel_idx = np.concatenate([pos_idx, sel_neg_idx])
        sel_idx.sort()

        # Per-fold imputation to avoid leakage
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(df.loc[sel_idx, feats])
        X_va = imputer.transform(df.loc[va_idx, feats])
        y_tr = df[target_col].astype(int).values[sel_idx]

        models = fit_ensemble(X_tr, y_tr, cfg.model, cfg.random_state)
        p_va   = predict_proba(models, X_va)
        oof_prob[va_idx] = p_va

        # Fold-level probability quality (threshold-free)
        roc  = roc_auc_score(y_va, p_va) if len(np.unique(y_va))>1 else float("nan")
        ap   = average_precision_score(y_va, p_va) if len(np.unique(y_va))>0 else float("nan")
        br   = brier_score_loss(y_va, p_va)

        fold_rows.append({
            "fold": i, "train_end": str(tr_end.date()), "val_month": str(va_m.date()),
            "roc_auc": float(roc), "pr_auc": float(ap), "brier": float(br)
        })

    # OOF metrics (probabilities)
    valid = ~np.isnan(oof_prob)
    if valid.sum() == 0:
        raise ValueError("No valid OOF predictions. Check folds and target distribution.")
    y_true = df[target_col].astype(int).values[valid]
    y_prob = oof_prob[valid]

    base = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_valid": int(valid.sum()),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum())
    }

    artifacts = {}
    results_sections = {}

    if cfg.optimize_for == "precision_floor":
        feasible, best = choose_threshold_precision_floor(y_true, y_prob, cfg.precision_floor)
        thr = float(best["threshold"]) if best["threshold"] is not None else 0.5
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        selected = {
            "objective": "precision_floor",
            "precision_floor": cfg.precision_floor,
            "feasible": bool(feasible),
            "threshold": thr,
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
        results_sections["threshold_selection"] = selected
    elif cfg.optimize_for == "f_beta":
        best = choose_threshold_fbeta(y_true, y_prob, cfg.beta)
        thr = float(best["threshold"]) if best["threshold"] is not None else 0.5
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        selected = {
            "objective": "f_beta",
            "beta": cfg.beta,
            "threshold": thr,
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f_beta": float(best["f_beta"]),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
        results_sections["threshold_selection"] = selected
    elif cfg.optimize_for == "fixed_k":
        dates_valid = df.loc[valid, date_col]
        fk = fixed_k_metrics_per_month(dates_valid, y_true, y_prob, cfg.fixed_k)
        results_sections["fixed_k_evaluation"] = fk
    else:
        raise ValueError("Unknown optimize_for")

    # Top-K risks for the last validation month
    if fold_rows:
        last_val_month = max(pd.to_datetime(r["val_month"]).to_period("M") for r in fold_rows)
        oof_df = df.loc[valid, [id_col, date_col]].copy()
        oof_df["prob"] = y_prob
        top_mask = oof_df[date_col].dt.to_period("M") == last_val_month
        topk_df = (oof_df.loc[top_mask, [id_col, date_col, "prob"]]
                   .sort_values("prob", ascending=False).head(cfg.topk))
    else:
        topk_df = pd.DataFrame(columns=[id_col, date_col, "prob"])  # empty

    # Save JSON summary
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUT_DIR / f"precision70_summary_{ts}.json"
    topk_path = OUT_DIR / f"next_month_high_risk_topk_{ts}.csv"
    oof_path = OUT_DIR / f"oof_predictions_{ts}.csv"

    out = {
        "schema": {
            "date_col": date_col, "id_col": id_col, "target_col": target_col,
            "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
            "n_features": len(feats)
        },
        "features_used": feats,
        "fold_reports": fold_rows,
        "overall_prob_quality": base,
        **results_sections,
        "artifacts": {
            "run_summary_json": str(summary_path),
            "topk_csv": str(topk_path),
            "oof_predictions_csv": str(oof_path)
        }
    }
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)

    # Save artifacts
    topk_df.to_csv(topk_path, index=False)
    oof_export = df.loc[valid, [id_col, date_col]].copy()
    oof_export["y_true"] = y_true
    oof_export["y_prob"] = y_prob
    oof_export.to_csv(oof_path, index=False)

    # Pretty prints
    print("\n=== SCHEMA ===")
    print(json.dumps(out["schema"], indent=2))
    print("\n=== PROBABILITY QUALITY (OOF, threshold-free) ===")
    print(json.dumps(out["overall_prob_quality"], indent=2))
    if cfg.optimize_for == "precision_floor":
        print("\n=== THRESHOLD (maximize Recall | Precision >= {:.2f}) ===".format(cfg.precision_floor))
        print(json.dumps(results_sections["threshold_selection"], indent=2))
    elif cfg.optimize_for == "f_beta":
        print("\n=== THRESHOLD (maximize F-beta, beta={:.2f}) ===".format(cfg.beta))
        print(json.dumps(results_sections["threshold_selection"], indent=2))
    elif cfg.optimize_for == "fixed_k":
        print("\n=== FIXED-K (per-month, K={}) ===".format(cfg.fixed_k))
        print(json.dumps(results_sections["fixed_k_evaluation"]["overall"], indent=2))
    print("\n=== FOLDS (probability quality) ===")
    print(pd.DataFrame(fold_rows).round(4).to_string(index=False))
    print(f"\nSaved: {summary_path}, {topk_path}, {oof_path}")

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg) 