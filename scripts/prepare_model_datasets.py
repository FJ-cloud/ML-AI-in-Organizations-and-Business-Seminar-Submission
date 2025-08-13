#!/usr/bin/env python3
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import re

BASE_DIR = Path('/home/fj/UkrainianInsolvencyPredictor')
MERGED_DIR = BASE_DIR / 'data' / 'processed' / 'merged'
OUT_DIR = BASE_DIR / 'data' / 'processed' / 'model_ready'
OUT_DIR.mkdir(parents=True, exist_ok=True)


LAG1_DOT_PATTERN = re.compile(r"_lag1\.\d+$")


def is_any_lag1(col: str) -> bool:
    # True for exact _lag1 or _lag1.<n>
    return col.endswith('_lag1') or bool(LAG1_DOT_PATTERN.search(col))


def latest_merged_path() -> Path:
    files = sorted(glob.glob(str(MERGED_DIR / 'camels_with_macro_wide_balanced_*.csv')))
    if not files:
        raise FileNotFoundError('No merged balanced dataset found in data/processed/merged')
    return Path(files[-1])


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    return (s - mu) / sd if sd and sd > 0 else s * 0


def engineer_basics(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure types
    df = df.copy()
    if 'date_m' in df.columns:
        df['date_m'] = pd.to_datetime(df['date_m'])
    # WAR: binary dummy (1 from Mar-2022 onward), else 0
    if 'date_m' in df.columns:
        war_start = pd.Timestamp('2022-03-01')
        df['WAR'] = (df['date_m'] >= war_start).astype(int)
    # Spread: Ukraine yield - US yield
    if {'ukraine_3y_yield','us_3y_yield'}.issubset(df.columns):
        ukr = pd.to_numeric(df['ukraine_3y_yield'], errors='coerce')
        us = pd.to_numeric(df['us_3y_yield'], errors='coerce')
        df['yield_spread_ukr_us_3y'] = ukr - us
        # Standardized versions to account for scale differences
        df['ukraine_3y_yield_z'] = _zscore(ukr)
        df['us_3y_yield_z'] = _zscore(us)
        df['yield_spread_ukr_us_3y_z'] = _zscore(df['yield_spread_ukr_us_3y'])
    # Keep change pct as-is (already monthly average of daily change)
    return df


def add_early_warning_labels(df: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    df = df.copy()
    if 'failure_date' in df.columns:
        # Normalize date types
        if not pd.api.types.is_datetime64_any_dtype(df['date_m']):
            df['date_m'] = pd.to_datetime(df['date_m'])
        fd = pd.to_datetime(df['failure_date'])
        # Post-failure indicator (for filtering if desired)
        df['post_failure'] = (fd.notna()) & (df['date_m'] >= fd)
        # Early warning: 1 if failure occurs within next H months (strictly before failure)
        next_h = df['date_m'] + pd.offsets.DateOffset(months=horizon_months)
        label = ((fd.notna()) & (df['date_m'] < fd) & (fd <= next_h)).astype(int)
        df['will_fail_within_6m'] = label if horizon_months == 6 else df.get('will_fail_within_6m', 0)
        df['will_fail_within_1m'] = label if horizon_months == 1 else df.get('will_fail_within_1m', 0)
    else:
        # If failure_date absent, default to zeros
        df['post_failure'] = False
        df['will_fail_within_6m'] = 0
        df['will_fail_within_1m'] = 0
    return df


def build_no_lag(df: pd.DataFrame) -> Path:
    # IDs
    id_cols_base = ['bank_name','date_m','year_month','failed','processing_timestamp','failure_date','last_reporting_date']
    id_cols = [c for c in id_cols_base if c in df.columns]
    # Include new labels if present
    for c in ['will_fail_within_6m','post_failure']:
        if c in df.columns and c not in id_cols:
            id_cols.append(c)
    # 8 CAMELS
    camels8 = [
        'Capital_Ratio','NPL_Ratio','Cost_Income_Ratio','ROA','ROE',
        'Liquid_Assets_Ratio','Loan_Deposit_Ratio','Net_Open_FX_Ratio'
    ]
    present_camels8 = [c for c in camels8 if c in df.columns]
    # WAR
    war_col = ['WAR'] if 'WAR' in df.columns else []
    # Macro: everything else except processing/presence flags and internals
    exclude = set(id_cols + present_camels8 + war_col + ['processing_timestamp','failure_date','last_reporting_date'])
    macro_cols = [
        c for c in df.columns
        if c not in exclude and not c.endswith('_present') and not is_any_lag1(c)
    ]
    cols = id_cols + war_col + present_camels8 + macro_cols
    out_df = df[cols].copy()
    # Drop any stray lag1.<n> columns if present by upstream merges
    drop_cols = [c for c in out_df.columns if LAG1_DOT_PATTERN.search(c)]
    if drop_cols:
        out_df = out_df.drop(columns=drop_cols)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUT_DIR / f'model_nolag_{ts}.csv'
    out_df.to_csv(out_path, index=False)
    return out_path


def build_lag1(df: pd.DataFrame) -> Path:
    # Variables to lag by 1 month (macro + optionally CAMELS)
    to_lag = [
        # Market-based
        'yield_spread_ukr_us_3y', 'us_3y_yield', 'ukraine_3y_yield', 'ukraine_3y_change_pct',
        # Macro JEHD series present in wide (subset)
        '24_International reserves (excluding gold)',
        '12_Liabilities to BIS banks (cons.), short term',
        '22_Liabilities to BIS banks, locational, total',
        '01_Cross-border loans from BIS reporting banks',
        '27_Cross-border deposits with BIS rep. banks',
    ]
    present_lag = [c for c in to_lag if c in df.columns]

    lagged = df.sort_values(['bank_name','date_m']).copy()
    for col in present_lag:
        lagged[f'{col}_lag1'] = lagged.groupby('bank_name')[col].shift(1)

    # Restrict output columns: IDs + WAR + CAMELS 8 + macro (and their _lag1)
    id_cols_base = ['bank_name','date_m','year_month','failed','processing_timestamp','failure_date','last_reporting_date']
    id_cols = [c for c in id_cols_base if c in lagged.columns]
    # Include new labels if present
    for c in ['will_fail_within_6m','post_failure']:
        if c in lagged.columns and c not in id_cols:
            id_cols.append(c)
    war_col = ['WAR'] if 'WAR' in lagged.columns else []
    camels8 = [
        'Capital_Ratio','NPL_Ratio','Cost_Income_Ratio','ROA','ROE',
        'Liquid_Assets_Ratio','Loan_Deposit_Ratio','Net_Open_FX_Ratio'
    ]
    present_camels8 = [c for c in camels8 if c in lagged.columns]
    # macro cols = everything else excluding processing/presence flags and any lag1 or lag1.<n>
    exclude = set(id_cols + war_col + present_camels8 + ['processing_timestamp','failure_date','last_reporting_date'])
    macro_cols = [c for c in lagged.columns if c not in exclude and not c.endswith('_present') and not is_any_lag1(c)]
    # plus lagged versions we just created (dedup handled by selection above)
    macro_lagged = [f'{c}_lag1' for c in present_lag]
    cols = id_cols + war_col + present_camels8 + macro_cols + macro_lagged
    out_df = lagged[cols].copy()
    # Drop any stray lag1.<n> columns if present by upstream merges
    drop_cols = [c for c in out_df.columns if LAG1_DOT_PATTERN.search(c)]
    if drop_cols:
        out_df = out_df.drop(columns=drop_cols)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUT_DIR / f'model_lag1_{ts}.csv'
    out_df.to_csv(out_path, index=False)
    return out_path


def main():
    merged_path = latest_merged_path()
    df = pd.read_csv(merged_path, parse_dates=['date_m'])
    df = engineer_basics(df)
    df = add_early_warning_labels(df, horizon_months=1)

    no_lag_path = build_no_lag(df)
    lag1_path = build_lag1(df)

    print(no_lag_path)
    print(lag1_path)


if __name__ == '__main__':
    main() 