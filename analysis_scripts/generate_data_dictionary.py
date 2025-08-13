#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
MODEL_READY_DIR = BASE / 'data' / 'processed' / 'model_ready'
DOCS_DIR = BASE / 'docs'
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical sets
ID_COLS = {
    'bank_name': {
        'category': 'id', 'source': 'panel', 'description': 'Canonical bank name after consolidation.'
    },
    'date_m': {
        'category': 'id', 'source': 'panel', 'description': 'Month end date (period end, monthly granularity).'
    },
    'year_month': {
        'category': 'id', 'source': 'panel', 'description': 'YYYY-MM string for monthly period.'
    },
}

TARGET_COLS = {
    'failed': {
        'category': 'target', 'source': 'panel', 'description': 'Binary target: 1 if bank failed, else 0.'
    },
    'WAR': {
        'category': 'policy_flag', 'source': 'constructed', 'description': 'War dummy: 1 from 2022-03-01 onwards, else 0.'
    }
}

CAMELS8 = {
    'Capital_Ratio': {
        'category': 'camels', 'source': 'derived (balance sheet)',
        'description': 'Equity to Total Assets ratio (capital adequacy proxy).'
    },
    'NPL_Ratio': {
        'category': 'camels', 'source': 'derived (loan book)',
        'description': 'Non-performing loans to gross loans.'
    },
    'Cost_Income_Ratio': {
        'category': 'camels', 'source': 'derived (P&L)',
        'description': 'Operating expenses to operating income.'
    },
    'ROA': {
        'category': 'camels', 'source': 'derived (P&L / assets)',
        'description': 'Return on Assets: net income / total assets.'
    },
    'ROE': {
        'category': 'camels', 'source': 'derived (P&L / equity)',
        'description': 'Return on Equity: net income / equity.'
    },
    'Liquid_Assets_Ratio': {
        'category': 'camels', 'source': 'derived (liquidity)',
        'description': 'Liquid assets to total assets.'
    },
    'Loan_Deposit_Ratio': {
        'category': 'camels', 'source': 'derived (funding)',
        'description': 'Loans to customer deposits.'
    },
    'Net_Open_FX_Ratio': {
        'category': 'camels', 'source': 'derived (FX exposure)',
        'description': 'Net open FX position to equity.'
    },
}

MACRO_DESCRIPTIONS = {
    '24_International reserves (excluding gold)': 'International reserves excluding gold (JEHD).',
    '12_Liabilities to BIS banks (cons.), short term': 'Short-term liabilities to BIS reporting banks, consolidated (JEHD/BIS).',
    '22_Liabilities to BIS banks, locational, total': 'Total locational liabilities to BIS banks (JEHD/BIS).',
    '01_Cross-border loans from BIS reporting banks': 'Cross-border loans from BIS reporting banks (JEHD/BIS).',
    '27_Cross-border deposits with BIS rep. banks': 'Cross-border deposits with BIS reporting banks (JEHD/BIS).',
    'us_3y_yield': 'United States Treasury 3-year yield (%).',
    'ukraine_3y_yield': 'Ukraine 3-year government bond yield (%).',
    'ukraine_3y_change_pct': 'Monthly percent change in Ukraine 3-year yield.',
    'cpi_x': 'Consumer Price Index (variant X from merged macro sources).',
    'cpi_y': 'Consumer Price Index (variant Y from merged macro sources).',
    'exchange_rate': 'Exchange rate (local currency per USD) from merged macro sources.',
    'interest_rate': 'Policy or money market interest rate (from merged macro sources).',
    'gdp_yoy_pct': 'GDP year-over-year growth rate (WEO/IMF).',
    'reserves_pct_gdp': 'International reserves as % of GDP (derived from WEO/IMF).',
    'yield_spread_ukr_us_3y': 'Yield spread: Ukraine 3y minus US 3y (pp).',
    'ukraine_3y_yield_z': 'Z-scored Ukraine 3y yield (standardized).',
    'us_3y_yield_z': 'Z-scored US 3y yield (standardized).',
    'yield_spread_ukr_us_3y_z': 'Z-scored yield spread (standardized).',
}


def latest_model_ready() -> Path | None:
    files = sorted(MODEL_READY_DIR.glob('model_lag1_*.csv'))
    return files[-1] if files else None


def build_dictionary(cols: list[str]) -> list[dict]:
    entries = []

    # Helper to add if present
    def add_if_present(name: str, meta: dict):
        if name in cols:
            e = {'name': name}
            e.update(meta)
            entries.append(e)

    # IDs and targets
    for k, v in ID_COLS.items():
        add_if_present(k, v)
    for k, v in TARGET_COLS.items():
        add_if_present(k, v)

    # CAMELS 8
    for k, v in CAMELS8.items():
        add_if_present(k, v)

    # Macro/base market
    for k, desc in MACRO_DESCRIPTIONS.items():
        add_if_present(k, {'category': 'macro', 'source': 'JEHD/IMF/BIS/WEO/market', 'description': desc})

    # Lagged variables
    for c in cols:
        if c.endswith('_lag1') or c.endswith('_lag1.1'):
            base = c.replace('_lag1.1', '').replace('_lag1', '')
            entries.append({
                'name': c,
                'category': 'macro_lag',
                'source': 'constructed',
                'description': f'Lag-1 month of {base}. Duplicate suffix .1 indicates de-duplication needed in future cleanup.'
            })

    # Anything else not covered
    covered = {e['name'] for e in entries}
    for c in cols:
        if c not in covered:
            # Skip processing timestamp-like or internal
            desc = 'Derived/feature column from merged macro/panel sources.'
            entries.append({'name': c, 'category': 'other', 'source': 'merged', 'description': desc})

    # Stable ordering: by category then name
    cat_order = {k: i for i, k in enumerate(['id','target','policy_flag','camels','macro','macro_lag','other'])}
    entries.sort(key=lambda x: (cat_order.get(x.get('category','other'), 99), x['name']))
    return entries


def write_outputs(entries: list[dict], model_path: Path):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_path = DOCS_DIR / f'data_dictionary_model_ready_{ts}.md'
    json_path = DOCS_DIR / f'data_dictionary_model_ready_{ts}.json'

    # Markdown
    lines = []
    lines.append(f'# Data Dictionary — Model Ready ({model_path.name})')
    lines.append('')
    lines.append('| Name | Category | Source | Description |')
    lines.append('|---|---|---|---|')
    for e in entries:
        lines.append(f"| {e['name']} | {e.get('category','')} | {e.get('source','')} | {e.get('description','').replace('|','/')} |")
    md_path.write_text('\n'.join(lines), encoding='utf-8')

    # JSON
    payload = {
        'generated_at': ts,
        'model_ready_file': str(model_path),
        'columns': entries,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    print('WROTE:', md_path)
    print('WROTE:', json_path)


def main():
    model = latest_model_ready()
    if not model:
        raise SystemExit('No model_lag1_*.csv found')
    # Read columns
    df = pd.read_csv(model, nrows=5)
    cols = df.columns.tolist()
    entries = build_dictionary(cols)
    write_outputs(entries, model)

if __name__ == '__main__':
    main() 