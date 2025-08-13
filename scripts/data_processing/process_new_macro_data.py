import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime

NEW_DATA_DIR = "/home/fj/UkrainianInsolvencyPredictor/data/newData"
OUTPUT_DIR = "/home/fj/UkrainianInsolvencyPredictor/data/processed/macro"
MACRO_IMF_DIR = "/home/fj/UkrainianInsolvencyPredictor/data/MacroEconomicData"

JEHD_FILE = os.path.join(NEW_DATA_DIR, "7caca554-a0de-48de-9346-5f090263bcd3_Data.csv")
DGS3_FILE = os.path.join(NEW_DATA_DIR, "DGS3.csv")
UKR_3Y_FILE = os.path.join(NEW_DATA_DIR, "Ukraine 3-Year Bond Yield Historical Data(5).csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _quarter_end_date(year: int, q: int) -> pd.Timestamp:
    month = {1: 3, 2: 6, 3: 9, 4: 12}[q]
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def process_jehd_quarterly_to_monthly_long_wide(path: str) -> dict:
    # Read raw; keep strings; treat '..' as NaN
    df = pd.read_csv(path, dtype=str)

    # Drop trailing metadata rows if present
    df = df[~df["Country Name"].astype(str).str.startswith("Data from", na=False)]

    # Identify quarter columns like 2016Q4 [YR2016Q4]
    quarter_cols = [c for c in df.columns if re.match(r"^\d{4}Q[1-4] \[YR\d{4}Q[1-4]\]$", c)]
    id_cols = ["Country Name", "Country Code", "Series Name", "Series Code"]

    # Melt to long
    long_q = df.melt(id_vars=id_cols, value_vars=quarter_cols, var_name="quarter_label", value_name="value")

    # Clean value: replace '..' and blanks with NaN, convert to float
    long_q["value"] = pd.to_numeric(long_q["value"].replace({"..": np.nan, "": np.nan}), errors="coerce")

    # Parse quarter info
    # quarter_label format: 'YYYYQn [YRYYYYQn]'
    long_q["year"] = long_q["quarter_label"].str.extract(r"^(\d{4})Q[1-4]").astype(float).astype("Int64")
    long_q["q"] = long_q["quarter_label"].str.extract(r"^\d{4}Q([1-4])").astype(float).astype("Int64")

    # Create quarter end date and monthly expanded index
    long_q = long_q.dropna(subset=["year", "q"])  # only valid quarters
    long_q["quarter_end"] = [
        _quarter_end_date(int(y), int(q)) for y, q in zip(long_q["year"].astype(int), long_q["q"].astype(int))
    ]

    # For each series code, expand quarterly values to all months in that quarter
    records = []
    for (series_code, series_name), g in long_q.groupby(["Series Code", "Series Name"], dropna=False):
        for _, row in g.iterrows():
            q_end = row["quarter_end"]
            # months in quarter are end of months (q_end - 2M, q_end -1M, q_end)
            months = [q_end - pd.offsets.MonthEnd(2), q_end - pd.offsets.MonthEnd(1), q_end]
            for m in months:
                records.append({
                    "date": m,  # month-end
                    "series_code": series_code,
                    "indicator": series_name,
                    "value": row["value"],
                })

    monthly_long = pd.DataFrame.from_records(records)
    monthly_long = monthly_long.sort_values(["date", "series_code"]).reset_index(drop=True)

    # Create wide pivot
    monthly_wide = monthly_long.pivot_table(index="date", columns="series_code", values="value", aggfunc="last")
    # Flatten columns and create safe column names
    monthly_wide.columns = [str(c) for c in monthly_wide.columns]
    monthly_wide = monthly_wide.reset_index()

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_path = os.path.join(OUTPUT_DIR, f"jehd_monthly_long_{ts}.csv")
    wide_path = os.path.join(OUTPUT_DIR, f"jehd_monthly_wide_{ts}.csv")
    meta_path = os.path.join(OUTPUT_DIR, f"jehd_monthly_meta_{ts}.json")

    monthly_long.to_csv(long_path, index=False)
    monthly_wide.to_csv(wide_path, index=False)

    meta = {
        "source_file": os.path.basename(path),
        "series_count": int(monthly_long["series_code"].nunique()),
        "date_min": str(monthly_long["date"].min().date()) if not monthly_long.empty else None,
        "date_max": str(monthly_long["date"].max().date()) if not monthly_long.empty else None,
        "rows": int(len(monthly_long)),
        "columns_wide": list(monthly_wide.columns),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "long_csv": long_path,
        "wide_csv": wide_path,
        "meta": meta_path,
    }


def process_dgs3_daily_to_monthly(path: str) -> str:
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date")
    # Coerce to numeric and drop NaN rows
    df["DGS3"] = pd.to_numeric(df["DGS3"], errors="coerce")

    monthly = (
        df.set_index("observation_date")["DGS3"]
          .resample("M").mean()
          .rename("us_3y_yield")
          .to_frame()
          .reset_index()
          .rename(columns={"observation_date": "date"})
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"us_dgs3_monthly_{ts}.csv")
    monthly.to_csv(out_path, index=False)
    return out_path


def process_ukraine_3y_to_monthly(path: str) -> str:
    # Read investing.com style CSV; keep relevant cols
    df = pd.read_csv(path)

    # Parse date robustly
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)

    # Keep only needed columns
    cols_present = df.columns.tolist()
    needed = ["Date", "Open", "Change %"]
    missing = [c for c in needed if c not in cols_present]
    if missing:
        raise ValueError(f"Ukraine 3Y file missing columns: {missing}. Found: {cols_present}")

    def to_float(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace('%', '').replace(',', '').strip()
        try:
            return float(x)
        except Exception:
            return np.nan

    # Convert
    df["Open"] = df["Open"].apply(to_float)
    df["Change %"] = df["Change %"].apply(to_float)  # stays in percent units

    monthly = (
        df.sort_values("Date").set_index("Date")[['Open', 'Change %']]
          .resample("M").mean()
          .rename(columns={
              'Open': 'ukraine_3y_yield',
              'Change %': 'ukraine_3y_change_pct'
          })
          .reset_index()
          .rename(columns={"Date": "date"})
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"ukraine_3y_monthly_{ts}.csv")
    monthly.to_csv(out_path, index=False)
    return out_path


def process_imf_monthlies_dir(data_dir: str) -> str | None:
    """Process IMF/STA monthly-style CSVs (CPI, EER, IRFCL) into a single wide monthly file.

    - Detect date columns like YYYY-M##
    - Melt to long, coerce to monthly date, mean aggregate
    - Merge series on date
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    series_list = []
    for fname in sorted(files):
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig')
        except Exception:
            continue
        date_cols = [c for c in df.columns if re.match(r"^\d{4}-M\d{1,2}$", str(c))]
        if not date_cols:
            continue
        indicator = None
        if 'CPI' in fname:
            indicator = 'cpi'
        elif 'EER' in fname:
            indicator = 'exchange_rate'
        elif 'IRFCL' in fname:
            indicator = 'interest_rate'
        if not indicator:
            # skip unrecognized
            continue
        id_vars = [c for c in df.columns if c not in date_cols]
        m = df.melt(id_vars=id_vars, value_vars=date_cols, var_name='date_str', value_name=indicator)
        # parse YYYY-M## to YYYY-MM-01
        def to_date(s: str):
            try:
                year, mpart = s.split('-M')
                month = int(mpart)
                return pd.Timestamp(int(year), month, 1)
            except Exception:
                return pd.NaT
        m['date'] = m['date_str'].astype(str).map(to_date)
        m = m[['date', indicator]].copy()
        m[indicator] = pd.to_numeric(m[indicator], errors='coerce')
        m = m.dropna(subset=['date', indicator])
        m = m.groupby('date', as_index=False)[indicator].mean()
        series_list.append(m)
    if not series_list:
        return None
    merged = series_list[0]
    for s in series_list[1:]:
        merged = merged.merge(s, on='date', how='outer')
    merged = merged.sort_values('date').reset_index(drop=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(OUTPUT_DIR, f'imf_monthly_wide_{ts}.csv')
    merged.to_csv(out, index=False)
    return out


def process_weo_annuals_dir(data_dir: str) -> str | None:
    """Process WEO annual CSVs to compute:
    - GDP YoY Growth (%)
    - Reserves (% of GDP), where nominal GDP is available

    WEO files typically have year columns (4-digit), melted to long, then we compute indicators.
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and ('WEO' in f or 'RES_WEO' in f)]
    if not files:
        return None
    annual_frames = []
    for fname in sorted(files):
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig')
        except Exception:
            continue
        year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
        if not year_cols:
            continue
        id_vars = [c for c in df.columns if c not in year_cols]
        m = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='year', value_name='value')
        m['year'] = pd.to_numeric(m['year'], errors='coerce').astype('Int64')
        m['value'] = pd.to_numeric(m['value'], errors='coerce')
        m = m.dropna(subset=['year', 'value'])
        m['date'] = pd.to_datetime(m['year'].astype(int).astype(str) + '-01-01')
        # carry identifier columns if present
        cols = ['date', 'year', 'value']
        for c in ['INDICATOR', 'Series Name', 'Series Code', 'SERIES_CODE']:
            if c in m.columns:
                cols.append(c)
        m = m[cols]
        annual_frames.append(m)
    if not annual_frames:
        return None
    all_annual = pd.concat(annual_frames, ignore_index=True)
    # Identify nominal GDP rows
    def is_nominal_gdp(s: str) -> bool:
        s = s.lower()
        return ('gross domestic product' in s) and ('current' in s) and ('deflator' not in s)
    # Identify GDP (non-deflator) rows for YoY growth
    def is_gdp_non_deflator(s: str) -> bool:
        s = s.lower()
        return ('gross domestic product' in s) and ('deflator' not in s)
    # Determine indicator text
    ind_col = 'INDICATOR' if 'INDICATOR' in all_annual.columns else 'Series Name' if 'Series Name' in all_annual.columns else None
    out_series = []
    if ind_col is not None:
        # GDP YoY
        gdp = all_annual[all_annual[ind_col].apply(lambda x: is_gdp_non_deflator(str(x)))].copy()
        if not gdp.empty:
            gdp = gdp.sort_values(['year'])
            gdp['gdp_yoy_pct'] = gdp['value'].pct_change() * 100.0
            gdp_growth = gdp[['date', 'gdp_yoy_pct']].dropna()
            out_series.append(gdp_growth)
        # Reserves % GDP
        nominal_gdp = all_annual[all_annual[ind_col].apply(lambda x: is_nominal_gdp(str(x)))][['year', 'value']].rename(columns={'value': 'gdp_nominal'})
        reserves_like = all_annual[all_annual[ind_col].astype(str).str.lower().str.contains('reserve|current account', na=False)].copy()
        if not reserves_like.empty and not nominal_gdp.empty:
            res = reserves_like.merge(nominal_gdp, on='year', how='left')
            res = res.dropna(subset=['gdp_nominal'])
            res['reserves_pct_gdp'] = (res['value'] / res['gdp_nominal']) * 100.0
            reserves_pct = res[['date', 'reserves_pct_gdp']]
            out_series.append(reserves_pct)
    if not out_series:
        return None
    merged = out_series[0]
    for s in out_series[1:]:
        merged = merged.merge(s, on='date', how='outer')
    merged = merged.sort_values('date').reset_index(drop=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(OUTPUT_DIR, f'weo_final_indicators_{ts}.csv')
    merged.to_csv(out, index=False)
    return out


def build_macro_only_datasets(outputs: dict) -> dict:
    """Create macro-only consolidated datasets (long and wide) and merge all available sources.

    outputs: dict with keys 'jehd' (dict with long_csv, wide_csv), 'us_dgs3' (path), 'ukr_3y' (path)
    and optionally 'imf_wide' and 'weo_indicators'
    Returns paths for macro-only long and wide datasets
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load inputs
    jehd_long = pd.read_csv(outputs["jehd"]["long_csv"], parse_dates=["date"]) if isinstance(outputs.get("jehd"), dict) else None
    jehd_wide = pd.read_csv(outputs["jehd"]["wide_csv"], parse_dates=["date"]) if isinstance(outputs.get("jehd"), dict) else None
    dgs3 = pd.read_csv(outputs["us_dgs3"], parse_dates=["date"]) if outputs.get("us_dgs3") else None
    ukr = pd.read_csv(outputs["ukr_3y"], parse_dates=["date"]) if outputs.get("ukr_3y") else None

    # IMF monthly series (optional)
    imf_wide = pd.read_csv(outputs["imf_wide"], parse_dates=["date"]) if outputs.get("imf_wide") else None
    # WEO annual indicators (optional)
    weo_ind = pd.read_csv(outputs["weo_indicators"], parse_dates=["date"]) if outputs.get("weo_indicators") else None

    # JEHD code → indicator renaming for target codes
    target_codes = {
        "Q.1C0.1C0.C.9A.MOA.RXGT.1.ALL.MV.TO1.ALL",
        "Q.5B0.5B0.C.5A.BKC.ASTT.1.STR.MX.TO1.ALL",
        "Q.5B0.5B0.C.5A.BKL.ASTT.1.ALL.MX.TO1.ALL",
        "Q.5B0.5B0.C.5A.BKL.LDPT.1.ALL.NV.TO1.ALL",
        "Q.5B0.5B0.D.5A.ALL.DFXB.1.ALL.NV.TO1.BMA",
    }
    dynamic_map = {}
    if jehd_long is not None and not jehd_long.empty:
        mapping_df = jehd_long[["series_code", "indicator"]].dropna().drop_duplicates()
        dynamic_map = {
            code: name for code, name in zip(mapping_df["series_code"], mapping_df["indicator"]) if code in target_codes
        }
    # Apply renaming on JEHD wide before merging (if available)
    if jehd_wide is not None and not jehd_wide.empty and dynamic_map:
        cols_to_rename = {code: dynamic_map[code] for code in dynamic_map if code in jehd_wide.columns}
        if cols_to_rename:
            jehd_wide = jehd_wide.rename(columns=cols_to_rename)

    # Wide: start from JEHD wide if present, else create empty frame
    if jehd_wide is not None:
        wide = jehd_wide.copy()
    else:
        wide = pd.DataFrame({"date": pd.DatetimeIndex([])})

    # Merge additional macro series
    if dgs3 is not None:
        wide = wide.merge(dgs3, on="date", how="outer")
    if ukr is not None:
        wide = wide.merge(ukr, on="date", how="outer")
    if imf_wide is not None and not imf_wide.empty:
        wide = wide.merge(imf_wide, on='date', how='outer')
    if weo_ind is not None and not weo_ind.empty:
        wide = wide.merge(weo_ind, on='date', how='outer')

    wide = wide.sort_values("date").reset_index(drop=True)

    wide_out = os.path.join(OUTPUT_DIR, f"macro_only_wide_{ts}.csv")
    wide.to_csv(wide_out, index=False)

    # Long: start from JEHD long if present
    if jehd_long is not None:
        long_df = jehd_long[["date", "series_code", "indicator", "value"]].copy()
        # Replace series_code with indicator for target codes to present "nice" names in long
        if dynamic_map:
            long_df["series_code"] = long_df["series_code"].replace(dynamic_map)
    else:
        long_df = pd.DataFrame(columns=["date", "series_code", "indicator", "value"])

    # Append DGS3 as a series
    if dgs3 is not None and not dgs3.empty:
        dgs3_long = dgs3.rename(columns={"us_3y_yield": "value"})
        dgs3_long["series_code"] = "US_DGS3"
        dgs3_long["indicator"] = "US 3Y Treasury yield (%)"
        dgs3_long = dgs3_long[["date", "series_code", "indicator", "value"]]
        long_df = pd.concat([long_df, dgs3_long], ignore_index=True)

    # Append Ukraine 3Y yield and change pct as separate series
    if ukr is not None and not ukr.empty:
        ukr_yield = ukr.rename(columns={"ukraine_3y_yield": "value"})[["date", "value"]].copy()
        ukr_yield["series_code"] = "UKR_3Y_YIELD"
        ukr_yield["indicator"] = "Ukraine 3Y bond yield (%)"
        ukr_yield = ukr_yield[["date", "series_code", "indicator", "value"]]
        if "ukraine_3y_change_pct" in ukr.columns:
            ukr_chg = ukr.rename(columns={"ukraine_3y_change_pct": "value"})[["date", "value"]].copy()
            ukr_chg["series_code"] = "UKR_3Y_CHANGE_PCT"
            ukr_chg["indicator"] = "Ukraine 3Y bond yield change (%)"
            ukr_chg = ukr_chg[["date", "series_code", "indicator", "value"]]
            long_df = pd.concat([long_df, ukr_yield, ukr_chg], ignore_index=True)
        else:
            long_df = pd.concat([long_df, ukr_yield], ignore_index=True)

    # IMF monthly series → add as separate series in long (wide is authoritative)
    if imf_wide is not None and not imf_wide.empty:
        for col in [c for c in imf_wide.columns if c != 'date']:
            tmp = imf_wide[['date', col]].dropna().rename(columns={col: 'value'})
            tmp['series_code'] = col.upper()
            tmp['indicator'] = col.replace('_', ' ').title()
            long_df = pd.concat([long_df, tmp[['date','series_code','indicator','value']]], ignore_index=True)

    # WEO (annual) indicators → add as separate series in long
    if weo_ind is not None and not weo_ind.empty:
        for col in [c for c in weo_ind.columns if c != 'date']:
            tmp = weo_ind[['date', col]].dropna().rename(columns={col: 'value'})
            tmp['series_code'] = col.upper()
            tmp['indicator'] = col.replace('_', ' ').title()
            long_df = pd.concat([long_df, tmp[['date','series_code','indicator','value']]], ignore_index=True)

    long_df = long_df.sort_values(["date", "series_code"]).reset_index(drop=True)

    long_out = os.path.join(OUTPUT_DIR, f"macro_only_long_{ts}.csv")
    long_df.to_csv(long_out, index=False)

    return {
        "macro_only_wide": wide_out,
        "macro_only_long": long_out,
    }


def main():
    outputs = {}
    if os.path.exists(JEHD_FILE):
        outputs["jehd"] = process_jehd_quarterly_to_monthly_long_wide(JEHD_FILE)
    else:
        print(f"Warning: JEHD file not found: {JEHD_FILE}")

    if os.path.exists(DGS3_FILE):
        outputs["us_dgs3"] = process_dgs3_daily_to_monthly(DGS3_FILE)
    else:
        print(f"Warning: DGS3 file not found: {DGS3_FILE}")

    if os.path.exists(UKR_3Y_FILE):
        outputs["ukr_3y"] = process_ukraine_3y_to_monthly(UKR_3Y_FILE)
    else:
        print(f"Warning: Ukraine 3Y file not found: {UKR_3Y_FILE}")

    # NEW: Process IMF/STA monthlies and WEO annuals from MacroEconomicData
    if os.path.isdir(MACRO_IMF_DIR):
        try:
            imf_out = process_imf_monthlies_dir(MACRO_IMF_DIR)
            if imf_out:
                outputs['imf_wide'] = imf_out
        except Exception as e:
            print(f"Warning: IMF processing failed: {e}")
        try:
            weo_out = process_weo_annuals_dir(MACRO_IMF_DIR)
            if weo_out:
                outputs['weo_indicators'] = weo_out
        except Exception as e:
            print(f"Warning: WEO processing failed: {e}")

    # Build macro-only consolidated datasets (merged into JEHD+Yields+IMF/WEO)
    try:
        macro_only = build_macro_only_datasets(outputs)
        outputs["macro_only"] = macro_only
    except Exception as e:
        print(f"Warning: Failed to build macro-only datasets: {e}")

    # Write an index file describing generated outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx_path = os.path.join(OUTPUT_DIR, f"new_macro_outputs_{ts}.json")
    with open(idx_path, "w") as f:
        json.dump(outputs, f, indent=2, default=str)
    print(f"Saved output manifest to {idx_path}")


if __name__ == "__main__":
    main() 