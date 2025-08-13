#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Try to use seaborn for nicer plots, fall back to matplotlib if unavailable
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Reuse utilities from existing pipeline where helpful
try:
    from machineLearning.SARIMAX.train_l1_sarimax import (
        load_panel,
        load_feature_list,
        select_features,
        DEFAULT_AMOUNT_COLS,
    )
except Exception:
    # Minimal fallbacks if import path differs
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

    def load_panel(csv_path: str | None) -> pd.DataFrame:
        base = Path(__file__).resolve().parents[1]  # project root
        default_dir = base / 'data' / 'processed' / 'model_ready'
        if csv_path is None:
            csvs = sorted(default_dir.glob('model_lag1_*.csv'))
            if not csvs:
                raise FileNotFoundError('No model_lag1_*.csv found')
            csv_path = str(csvs[-1])
        return pd.read_csv(csv_path, parse_dates=['date_m'])

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


# ---------------- Economist-like styling ----------------
ECON_PRIMARY = '#005F73'   # muted teal/blue
ECON_ACCENT = '#E3120B'    # Economist red for highlights
ECON_GRAY = '#6e6e6e'


def apply_economist_style():
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': ECON_GRAY,
        'axes.labelcolor': 'black',
        'text.color': 'black',
        'xtick.color': ECON_GRAY,
        'ytick.color': ECON_GRAY,
        'grid.color': '#dddddd',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'grid.linestyle': '-',
        'grid.linewidth': 0.6,
        'axes.grid': True,
        'axes.grid.axis': 'y',  # horizontal grid only
        'font.size': 11,
        'axes.titleweight': 'bold',
    })
    if sns is not None:
        sns.set_theme(style='whitegrid', rc={'grid.linestyle': '-', 'grid.linewidth': 0.6})


apply_economist_style()
# --------------------------------------------------------


def zscore(a: pd.Series) -> pd.Series:
    m, s = a.mean(), a.std(ddof=0)
    if s == 0 or np.isnan(s):
        return pd.Series(np.zeros(len(a)), index=a.index)
    return (a - m) / s


def plot_corr_heatmap(df: pd.DataFrame, feat_cols: list[str], outdir: Path) -> Path:
    # Choose ratio-like columns (non-amounts) for correlation
    amount_cols = [c for c in DEFAULT_AMOUNT_COLS if c in feat_cols]
    ratio_cols = [c for c in feat_cols if c not in amount_cols]
    corr_df = df[ratio_cols].copy()
    corr = corr_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    if sns is not None:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=False, cbar_kws={'shrink': 0.75})
    else:
        plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title('Correlation heatmap (ratio-like features)')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = outdir / f'fig_corr_heatmap_{ts}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_failure_timing(df: pd.DataFrame, split_date: str, outdir: Path) -> Path:
    # Monthly failures and cumulative sum
    monthly = df.groupby('date_m')['failed'].sum().sort_index()
    cumulative = monthly.cumsum()

    plt.figure(figsize=(12, 4))
    # Bar chart of cumulative failures
    dates = monthly.index
    vals = cumulative.values
    bar_colors = [ECON_PRIMARY] * len(dates)
    plt.bar(dates, vals, color=bar_colors, width=20)  # width ~20 days for monthly bars

    # Cutoff line
    sd = pd.Timestamp(split_date)
    plt.axvline(sd, color=ECON_ACCENT, linestyle='--', linewidth=1.5, label=f'Split {split_date}')

    plt.ylabel('Cumulative failures')
    plt.xlabel('Date')
    plt.title('Cumulative failures over time with split marker')
    plt.legend(frameon=False)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = outdir / f'fig_failure_timing_{ts}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_interbank_variability(df: pd.DataFrame, feat_cols: list[str], outdir: Path) -> tuple[Path, list[str]]:
    # Normalize features: z-score per full sample for ratios; log1p+z for common amounts
    amount_cols = [c for c in DEFAULT_AMOUNT_COLS if c in feat_cols]
    ratio_cols = [c for c in feat_cols if c not in amount_cols]

    norm = pd.DataFrame(index=df.index)
    for c in ratio_cols:
        norm[c] = zscore(df[c].astype(float))
    for c in amount_cols:
        try:
            norm[c] = zscore(np.log1p(df[c].astype(float)))
        except Exception:
            norm[c] = 0.0

    # Cross-sectional std per date, then average across dates
    cs_std = (
        norm.assign(date_m=df['date_m'], bank_name=df['bank_name'])
            .groupby('date_m')[feat_cols]
            .agg(lambda x: np.nanstd(x.values, ddof=0))
            .mean(axis=0)
            .sort_values(ascending=False)
    )
    top2 = cs_std.head(2).index.tolist()

    # Plot top 20
    topN = cs_std.head(20)[::-1]  # reverse for barh
    plt.figure(figsize=(10, 8))
    if sns is not None:
        sns.barplot(x=topN.values, y=topN.index, orient='h', color=ECON_PRIMARY)
    else:
        plt.barh(topN.index, topN.values, color=ECON_PRIMARY)
    plt.xlabel('Average cross-sectional std (z-scored)')
    plt.title('Interbank variability — top features')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = outdir / f'fig_interbank_variability_{ts}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out, top2


def plot_bank_failures_timeline(df: pd.DataFrame, split_date: str, outdir: Path) -> Path:
    # Collect one failure date per bank
    if 'failure_date' in df.columns:
        failures = (
            df.dropna(subset=['failure_date'])
              .groupby('bank_name', as_index=False)['failure_date']
              .min()
        )
        failures['failure_date'] = pd.to_datetime(failures['failure_date'])
    else:
        # Fallback: infer failure_date as the last date with failed==1 per bank
        tmp = df[df['failed'] == 1].groupby('bank_name')['date_m'].max().reset_index()
        tmp = tmp.rename(columns={'date_m': 'failure_date'})
        failures = tmp

    # Sort by date
    failures = failures.sort_values('failure_date').reset_index(drop=True)

    # Plot
    plt.figure(figsize=(10, max(6, 0.35 * len(failures))))
    y = np.arange(len(failures))
    plt.scatter(failures['failure_date'], y, color=ECON_PRIMARY, zorder=3)
    for i, name in enumerate(failures['bank_name']):
        plt.text(failures['failure_date'].iloc[i], y[i], ' ', fontsize=6)
    plt.yticks(y, failures['bank_name'])

    # Split cutoff
    sd = pd.Timestamp(split_date)
    plt.axvline(sd, color=ECON_ACCENT, linestyle='--', linewidth=1.5, label=f'Split {split_date}')

    plt.xlabel('Failure date')
    plt.title(f'Individual bank failures (n={len(failures)})')
    plt.legend(frameon=False, loc='upper left')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = outdir / f'fig_bank_failures_timeline_{ts}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_cumulative_failures_fullperiod(df: pd.DataFrame, split_date: str, outdir: Path) -> Path:
    # Monthly failures across entire period with explicit monthly index
    monthly = df.groupby('date_m')['failed'].sum().sort_index()
    if monthly.empty:
        raise ValueError('No failures data to plot')
    # Build full monthly index between min and max (use month-end to match date_m)
    start = monthly.index.min()
    end = monthly.index.max()
    full_idx = pd.date_range(start=start, end=end, freq='M')
    monthly_full = monthly.reindex(full_idx, fill_value=0)
    cumulative = monthly_full.cumsum()

    plt.figure(figsize=(12, 4))
    plt.bar(full_idx, cumulative.values, color=ECON_PRIMARY, width=20)

    # Split cutoff
    sd = pd.Timestamp(split_date)
    plt.axvline(sd, color=ECON_ACCENT, linestyle='--', linewidth=1.5, label=f'Split {split_date}')

    plt.ylabel('Cumulative failures')
    plt.xlabel('Month')
    plt.title('Cumulative bank failures per month (entire period)')
    plt.legend(frameon=False)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = outdir / f'fig_cum_failures_fullperiod_{ts}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def compute_first_failures_monthly(df: pd.DataFrame) -> pd.Series:
    """Return a monthly series (month-end index) counting first failures per bank.
    Uses failure_date if present, else the first month where failed==1.
    """
    if 'failure_date' in df.columns and df['failure_date'].notna().any():
        s = (df.dropna(subset=['failure_date'])
                .groupby('bank_name')['failure_date']
                .min())
        s = pd.to_datetime(s)
        months = s.dt.to_period('M').dt.to_timestamp('M')
    else:
        tmp = df[df['failed'] == 1].groupby('bank_name')['date_m'].min()
        months = pd.to_datetime(tmp)
    monthly_counts = pd.Series(1, index=months).groupby(level=0).sum().sort_index()
    if monthly_counts.empty:
        return monthly_counts
    # Full monthly span
    start = df['date_m'].min()
    end = df['date_m'].max()
    full_idx = pd.date_range(start=start, end=end, freq='M')
    monthly_counts = monthly_counts.reindex(full_idx, fill_value=0)
    return monthly_counts


def plot_first_failures_cumulative(df: pd.DataFrame, split_date: str, outdir: Path) -> Path:
    monthly_counts = compute_first_failures_monthly(df)
    if monthly_counts.empty:
        raise ValueError('No first-failure events found to plot')
    cumulative = monthly_counts.cumsum()

    plt.figure(figsize=(12, 4))
    plt.bar(monthly_counts.index, cumulative.values, color=ECON_PRIMARY, width=20)

    sd = pd.Timestamp(split_date)
    plt.axvline(sd, color=ECON_ACCENT, linestyle='--', linewidth=1.5, label=f'Split {split_date}')

    plt.ylabel('Cumulative unique bank failures')
    plt.xlabel('Month')
    plt.title('Cumulative unique bank failures per month (first failures only)')
    plt.legend(frameon=False)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = outdir / f'fig_first_failures_cumulative_{ts}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_first_failures_monthly_simple(df: pd.DataFrame, outdir: Path) -> Path:
    monthly_counts = compute_first_failures_monthly(df)
    if monthly_counts.empty:
        raise ValueError('No first-failure events found to plot')

    # Keep only months with at least one failure
    monthly_counts = monthly_counts[monthly_counts > 0]

    # Build categorical axis for dense spacing and clean labels (YYYY-MM)
    months = pd.Index(monthly_counts.index.to_period('M').astype(str))
    x = np.arange(len(months))
    y = monthly_counts.values

    # Square, compact red bar chart; fat bars; no grid; show only months with failures
    plt.figure(figsize=(6.0, 6.0))
    ax = plt.gca()
    ax.grid(False)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    bars = ax.bar(x, y, color='#E3120B', width=0.9)
    ax.set_ylabel('Banks failed', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_title('Monthly unique bank failures (months with failures only)', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=90, ha='center', fontsize=8)
    ax.tick_params(axis='y', labelsize=10)

    # Add numeric labels inside bars (fallback above if bar too short)
    y_max = max(y) if len(y) else 1
    pad_in = max(0.15, 0.08 * y_max)
    pad_above = max(0.1, 0.04 * y_max)

    for xi, yi, b in zip(x, y, bars):
        if yi > pad_in:
            ax.text(xi, yi - pad_in, f"{int(yi)}", ha='center', va='top', fontsize=10, color='white')
        else:
            ax.text(xi, yi + pad_above, f"{int(yi)}", ha='center', va='bottom', fontsize=10, color='black')

    # Tight layout for dense labels
    plt.tight_layout()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = outdir / f'fig_first_failures_monthly_{ts}.png'
    plt.savefig(out, dpi=240)
    plt.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None, help='Path to model_ready CSV (defaults to latest model_lag1_*.csv)')
    ap.add_argument('--features', type=str, default=None, help='Optional feature list file to restrict columns')
    ap.add_argument('--split_date', type=str, default='2022-08-31', help='Split date to annotate in failure plot')
    ap.add_argument('--outdir', type=str, default='eda', help='Output directory for figures')
    args = ap.parse_args()

    # Load data
    df = load_panel(args.csv)

    # Feature selection
    feat_list = load_feature_list(args.features)
    id_cols, y_col, feat_cols = select_features(df, feat_list)

    # Basic checks
    for col in ['bank_name', 'date_m', 'failed']:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Correlation heatmap
    heatmap_path = plot_corr_heatmap(df, feat_cols, outdir)

    # 2) Failure timing plot (cumulative bar chart based on all failed flags)
    failure_path = plot_failure_timing(df, args.split_date, outdir)

    # 2a) Full-period cumulative monthly failures (all failed flags)
    cum_full_path = plot_cumulative_failures_fullperiod(df, args.split_date, outdir)

    # 2a.1) Cumulative unique bank first-failures per month
    first_fail_cum_path = plot_first_failures_cumulative(df, args.split_date, outdir)

    # 2a.2) Simple monthly unique bank failures (no cumulative)
    first_fail_monthly_path = plot_first_failures_monthly_simple(df, outdir)

    # 2b) Individual bank failures timeline
    bank_fail_path = plot_bank_failures_timeline(df, args.split_date, outdir)

    # 3) Interbank variability
    interbank_path, top2 = plot_interbank_variability(df, feat_cols, outdir)

    # Record a small summary note
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_md = outdir / f'eda_figures_summary_{ts}.md'
    summary_md.write_text('\n'.join([
        '# EDA Figures Summary',
        f'- Correlation heatmap: {heatmap_path}',
        f'- Failure timing (cumulative) with split: {failure_path}',
        f'- Cumulative failures full period (all failed flags): {cum_full_path}',
        f'- Cumulative unique bank first-failures: {first_fail_cum_path}',
        f'- Monthly unique bank failures (no cumulative): {first_fail_monthly_path}',
        f'- Bank failures timeline: {bank_fail_path}',
        f'- Interbank variability: {interbank_path}',
        f'- Top-2 cross-sectional features: {", ".join(top2)}',
    ]))
    print('Saved:')
    print(heatmap_path)
    print(failure_path)
    print(cum_full_path)
    print(first_fail_cum_path)
    print(first_fail_monthly_path)
    print(bank_fail_path)
    print(interbank_path)
    print('Top-2 cross-sectional features:', ', '.join(top2))


if __name__ == '__main__':
    main() 