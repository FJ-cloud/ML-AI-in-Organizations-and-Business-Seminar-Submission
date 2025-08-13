#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

BASE_DIR = Path('/home/fj/UkrainianInsolvencyPredictor')
OUTPUT_DIR = BASE_DIR / 'data' / 'processed' / 'merged'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMELS_DIR = BASE_DIR / 'output_final'
MACRO_DIR = BASE_DIR / 'data' / 'processed' / 'macro'
CUTOFF_DATE = pd.Timestamp('2025-04-30')

# Reference columns that indicate real bank data presence
REF_COLS = [
    'assets_total_assets',
    'liabilities_total_liabilities',
    'equity_total_equity_capital',
    'financial_results_profit_loss_after_tax',
]

ID_COLS = ['bank_name', 'date_m', 'year_month', 'failed', 'processing_timestamp']
EPS = 1e-9


def _latest(path_pattern: str) -> str:
    files = sorted(glob.glob(path_pattern))
    return files[-1] if files else ''


def load_camels_base() -> pd.DataFrame:
    # Always use the base panel (no enhanced CAMELS)
    fallback = CAMELS_DIR / 'ukrainian_banks_panel_dataset_FINAL.csv'
    if not fallback.exists():
        raise FileNotFoundError('Base panel not found: ukrainian_banks_panel_dataset_FINAL.csv')
    df = pd.read_csv(fallback, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df['date_m'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp('M')
    # Safety: drop any enhanced CAMELS columns if present
    camels_enhanced_cols = [c for c in df.columns if c.startswith('CAMELS_')]
    if camels_enhanced_cols:
        df = df.drop(columns=camels_enhanced_cols)
    return df


def load_macro_wide() -> pd.DataFrame:
    macro_path = _latest(str(MACRO_DIR / 'macro_only_wide_*.csv'))
    if not macro_path:
        raise FileNotFoundError('No macro_only_wide_*.csv found')
    macro = pd.read_csv(macro_path)
    macro['date'] = pd.to_datetime(macro['date'])
    # Ensure month-end
    macro['date'] = macro['date'].dt.to_period('M').dt.to_timestamp('M')
    return macro


def build_balanced_panel(camels_df: pd.DataFrame, macro_wide: pd.DataFrame) -> pd.DataFrame:
    # Clean bank names: drop NaN and blank/whitespace-only
    bank_series = camels_df['bank_name'].astype(str).str.strip()
    bank_series = bank_series[bank_series.str.len() > 0]
    banks = sorted(bank_series.unique())

    # Compute bank lifespans based on rows with real bank data
    # Consider a row as real if any reference column is > EPS
    if all(c in camels_df.columns for c in REF_COLS):
        ref_positive = camels_df[REF_COLS].apply(pd.to_numeric, errors='coerce')
        ref_present = (ref_positive > EPS).any(axis=1)
    else:
        ref_present = None

    if ref_present is not None and ref_present.any():
        lifespan_source = camels_df.loc[ref_present, ['bank_name', 'date_m']]
    else:
        lifespan_source = camels_df[['bank_name', 'date_m']]

    lifespan = (
        lifespan_source
        .dropna()
        .groupby('bank_name')['date_m']
        .agg(bank_start='min', bank_end='max')
        .reset_index()
    )

    min_month = macro_wide['date'].min()
    max_month = macro_wide['date'].max()
    months = pd.period_range(min_month, max_month, freq='M').to_timestamp('M')

    balanced = pd.MultiIndex.from_product([banks, months], names=['bank_name', 'date_m']).to_frame(index=False)

    # Merge CAMELS base columns for IDs/presence flags
    camels_cols = [c for c in camels_df.columns if c != 'date']
    merged = balanced.merge(camels_df[camels_cols], on=['bank_name', 'date_m'], how='left')

    # Merge macro by month (replicates for every bank)
    macro_cols = [c for c in macro_wide.columns if c != 'date']
    merged = merged.merge(macro_wide.rename(columns={'date': 'date_m'}), on='date_m', how='left')

    # Derive lists
    present_id_cols = [c for c in ID_COLS if c in merged.columns]
    present_ref_cols = [c for c in REF_COLS if c in merged.columns]
    present_macro_cols = macro_cols

    # Attach lifespan and blank macro outside bank lifespan (or if lifespan undefined)
    merged = merged.merge(lifespan, on='bank_name', how='left')
    outside = merged['bank_start'].isna() | (merged['date_m'] < merged['bank_start']) | (merged['date_m'] > merged['bank_end'])
    merged.loc[outside, present_macro_cols] = np.nan

    # Also blank CAMELS + references for undefined lifespan entirely
    undef = merged['bank_start'].isna()
    merged.loc[undef, present_ref_cols] = np.nan

    # Static cutoff: keep only months <= 2025-04-30
    merged = merged[merged['date_m'] <= CUTOFF_DATE]

    # Clear macro data where failed == 1.0
    if 'failed' in merged.columns:
        mask_failed = merged['failed'] == 1.0
        merged.loc[mask_failed, present_macro_cols] = np.nan

    # Additionally, clear base ref where row lacks real bank data (all refs NaN or <= EPS)
    if present_ref_cols:
        ref_vals = merged[present_ref_cols].apply(pd.to_numeric, errors='coerce')
        no_bank_data = ~(ref_vals > EPS).any(axis=1)
        merged.loc[no_bank_data, present_ref_cols] = np.nan

    # Drop lifespan helper
    merged = merged.drop(columns=['bank_start', 'bank_end'])

    # Drop any rows with missing/blank bank_name (paranoia)
    merged['bank_name'] = merged['bank_name'].astype(str).str.strip()
    merged = merged[merged['bank_name'].str.len() > 0]

    # Sort for readability
    merged = merged.sort_values(['bank_name', 'date_m']).reset_index(drop=True)
    return merged

# NEW: Compute 8 CAMELS ratios and prune base panel columns

def attach_camels_8_and_prune(merged: pd.DataFrame, camels_base: pd.DataFrame) -> pd.DataFrame:
    import sys
    sys.path.append(str(BASE_DIR / 'camels'))
    from camels_calculator import CAMELSCalculator
    calc = CAMELSCalculator()

    # Compute ratios on base panel (has 'date') and align to month-end index
    base = camels_base.copy()
    base['date_m'] = pd.to_datetime(base['date']).dt.to_period('M').dt.to_timestamp('M')
    ratios = calc.calculate_all_ratios(base)
    keep_cols = ['bank_name', 'date_m',
                 'Capital_Ratio', 'NPL_Ratio', 'Cost_Income_Ratio', 'ROA', 'ROE',
                 'Liquid_Assets_Ratio', 'Loan_Deposit_Ratio', 'Net_Open_FX_Ratio']
    camels8 = ratios[keep_cols].copy()

    # Merge CAMELS 8 into merged
    out = merged.merge(camels8, on=['bank_name', 'date_m'], how='left')

    # Prune: drop raw panel financial columns, keep IDs, failed, presence flags, CAMELS 8, macro
    drop_prefixes = ('assets_', 'liabilities_', 'equity_', 'financial_results_')
    cols = []
    for c in out.columns:
        if c.startswith(drop_prefixes):
            continue
        cols.append(c)
    out = out[cols]
    return out


def save_output(df: pd.DataFrame) -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = OUTPUT_DIR / f'camels_with_macro_wide_balanced_{ts}.csv'
    df.to_csv(out_path, index=False)
    return str(out_path)


def main():
    camels_df = load_camels_base()
    macro_wide = load_macro_wide()
    balanced = build_balanced_panel(camels_df, macro_wide)
    balanced = attach_camels_8_and_prune(balanced, camels_df)
    out_path = save_output(balanced)
    print(out_path)


if __name__ == '__main__':
    camels_df = load_camels_base()
    macro_wide = None
    macro_path = [p for p in sorted(glob.glob(str(MACRO_DIR / 'macro_only_wide_*.csv'))) ]
    if not macro_path:
        raise FileNotFoundError('No macro_only_wide_*.csv found')
    macro_wide = pd.read_csv(macro_path[-1])
    macro_wide['date'] = pd.to_datetime(macro_wide['date'])

    balanced = build_balanced_panel(camels_df, macro_wide)
    balanced = attach_camels_8_and_prune(balanced, camels_df)
    out_path = save_output(balanced)
    print(out_path) 