#!/usr/bin/env python3
from __future__ import annotations
import argparse
from typing import List, Tuple, Dict, Iterable, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

BANK_COL = "bank_id"
DATE_COL = "date"
TARGET_COL = "y"

# Prediction expects columns for ALL lagged features the model was trained on base features for.
# We will construct lags from raw base features in the provided new data.


def read_inference_panel(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in [BANK_COL, DATE_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in CSV.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([BANK_COL, DATE_COL]).reset_index(drop=True)
    return df


def add_lag_features_for_inference(df: pd.DataFrame, base_features: List[str], lags: Iterable[int]) -> pd.DataFrame:
    df = df.sort_values([BANK_COL, DATE_COL]).copy()
    g = df.groupby(BANK_COL, group_keys=False)
    pieces = [df[[BANK_COL, DATE_COL]].copy()]
    for c in base_features:
        if c not in df.columns:
            raise ValueError(f"Missing base feature '{c}' in new data.")
        for L in lags:
            pieces.append(g[c].shift(L).rename(f"{c}_lag{L}"))
    X = pd.concat(pieces, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.dropna().reset_index(drop=True)
    return X


def predict_with_artifact(artifact: Dict[str, Any], new_df: pd.DataFrame, score_date: pd.Timestamp | None,
                          banks_filter: list[str] | None) -> pd.DataFrame:
    lags = artifact["lags"]
    feature_cols: List[str] = artifact["feature_cols"]
    global_scaler: StandardScaler = artifact["global_scaler"]
    per_bank: Dict[str, Any] = artifact["per_bank"]
    calibrator = artifact.get("calibrator")

    # Derive base features from feature_cols names
    base_features = sorted({c.rsplit("_lag", 1)[0] for c in feature_cols})

    base_df = new_df.copy()
    X = add_lag_features_for_inference(base_df, base_features, lags)

    if banks_filter:
        X = X[X[BANK_COL].astype(str).isin(banks_filter)].copy()
        if X.empty:
            raise ValueError("After bank filter, no rows remain in new data.")

    if score_date is None:
        score_date = X[DATE_COL].max()
    X = X[X[DATE_COL] == score_date].copy()
    if X.empty:
        raise ValueError("No rows available at the score_date after lagging.")

    # Standardize features using training scaler
    exog = pd.DataFrame(global_scaler.transform(X[feature_cols].values), columns=feature_cols, index=X.index)

    preds = []
    for bank, g in X.groupby(BANK_COL):
        bank_key = str(bank)
        payload = per_bank.get(bank_key)
        if payload is None:
            # If unseen bank, skip or mark NaN
            continue
        res = payload["sarimax_results"]
        # Predict using stored SARIMAX results
        g_exog = exog.loc[g.index]
        g_exog.index = g[DATE_COL]
        fc = res.predict(start=g_exog.index[0], end=g_exog.index[-1], exog=g_exog)
        if artifact.get("add_residual_l1") and "residual" in payload:
            scaler_r: StandardScaler = payload["residual"]["scaler"]
            l1 = payload["residual"]["lasso"]
            Xv_sc = scaler_r.transform(g_exog.values)
            fc_resid = l1.predict(Xv_sc)
            fc = fc.add(pd.Series(fc_resid, index=g_exog.index), fill_value=0.0)
        tmp = g[[BANK_COL, DATE_COL]].copy()
        tmp["pred"] = fc.values
        preds.append(tmp)

    if not preds:
        raise ValueError("No predictions generated; banks may be unseen compared to training.")

    out = pd.concat(preds, ignore_index=True)
    out["prob_fail"] = out["pred"].clip(0, 1)
    if calibrator is not None:
        out["prob_fail"] = calibrator.transform(out["prob_fail"].values)
    return out[[BANK_COL, DATE_COL, "prob_fail"]]


def main():
    ap = argparse.ArgumentParser(description="Predict probabilities on new data using a saved L1+SARIMAX model artifact.")
    ap.add_argument("--model", required=True, help="Path to saved model artifact (.pkl)")
    ap.add_argument("--csv", required=True, help="Path to NEW input CSV containing base features")
    ap.add_argument("--date", help="Target date to score (YYYY-MM-DD). Defaults to latest available after lagging")
    ap.add_argument("--banks", help="Comma-separated list of bank_ids to include (optional)")
    ap.add_argument("--out", default="probabilities_newdata.csv", help="Output CSV path")

    args = ap.parse_args()

    artifact = joblib.load(args.model)
    new_df = read_inference_panel(args.csv)

    banks_filter = None
    if args.banks:
        banks_filter = [b.strip() for b in args.banks.split(',') if b.strip()]

    score_date = pd.Timestamp(args.date) if args.date else None

    pred = predict_with_artifact(artifact, new_df, score_date, banks_filter)
    pred.to_csv(args.out, index=False)
    print(f"Wrote probabilities to: {args.out}")


if __name__ == "__main__":
    main() 