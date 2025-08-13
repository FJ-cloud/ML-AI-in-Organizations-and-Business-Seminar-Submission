#!/usr/bin/env python3
"""
Data Transformer - Comprehensive data transformation for Ukrainian Bank data

Handles all data transformations based on real-world processing:
- Bank name canonicalization using proven mappings
- Data cleaning and standardization
- Panel dataset creation with wide format
- Failure flag engineering
- Data type prefixing and organization

Expert implementation based on successful processing scripts.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

from ..utils.bank_name_canonicalizer import BankNameCanonicalizer


class DataTransformer:
    """
    Comprehensive data transformer for Ukrainian bank data.
    
    Handles all transformations needed to convert raw extracted data
    into a clean, ML-ready panel dataset.
    """
    
    def __init__(self, config):
        """Initialize the data transformer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize bank name canonicalizer
        self.canonicalizer = BankNameCanonicalizer(config)
        
        # Data type prefixes for panel dataset
        self.data_type_prefixes = {
            'assets': 'assets_',
            'liabilities': 'liabilities_',
            'equity': 'equity_',
            'financial_results': 'financial_results_'
        }
        
        self.logger.info("🔄 Data transformer initialized")
    
    def transform_data_type(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Transform data for a specific data type (assets, liabilities, etc.).
        
        Args:
            df: DataFrame with extracted data
            data_type: Type of data (assets, liabilities, equity, financial_results)
            
        Returns:
            Transformed DataFrame
        """
        if df is None or df.empty:
            return df
        
        self.logger.debug(f"Transforming {data_type} data: {len(df)} rows")
        
        # Make a copy to avoid modifying original
        df_transformed = df.copy()
        
        # 1. Canonicalize bank names
        df_transformed, canon_stats = self.canonicalizer.process_dataframe(df_transformed, 'bank_name')
        
        # 2. Clean financial data columns
        df_transformed = self._clean_financial_columns(df_transformed)
        
        # 3. Add data type prefix to financial columns
        df_transformed = self._add_data_type_prefix(df_transformed, data_type)
        
        # 4. Standardize date columns
        df_transformed = self._standardize_dates(df_transformed)
        
        # 5. Add metadata
        df_transformed['data_type'] = data_type
        
        self.logger.debug(f"Transformed {data_type}: {len(df_transformed)} rows, {len(df_transformed.columns)} columns")
        
        return df_transformed
    
    def _clean_financial_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean financial data columns."""
        # Identify financial columns (numeric columns that aren't metadata)
        metadata_columns = {
            'bank_name', 'date', 'year_month', 'source_file', 'source_sheet', 'data_type'
        }
        
        financial_columns = [col for col in df.columns if col not in metadata_columns]
        
        for col in financial_columns:
            if col in df.columns:
                # Convert to numeric, handling various formats
                df[col] = self._clean_numeric_column(df[col])
        
        return df
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean a numeric column, handling various formats."""
        # Convert to string first to handle mixed types
        series_str = series.astype(str)
        
        # Remove common non-numeric characters
        series_clean = series_str.str.replace(',', '')  # Remove thousands separators
        series_clean = series_clean.str.replace(' ', '')  # Remove spaces
        series_clean = series_clean.str.replace('−', '-')  # Replace Unicode minus
        series_clean = series_clean.str.replace('–', '-')  # Replace en-dash
        series_clean = series_clean.str.replace('—', '-')  # Replace em-dash
        
        # Handle special cases
        series_clean = series_clean.replace('', np.nan)
        series_clean = series_clean.replace('nan', np.nan)
        series_clean = series_clean.replace('None', np.nan)
        series_clean = series_clean.replace('-', np.nan)
        
        # Convert to numeric
        series_numeric = pd.to_numeric(series_clean, errors='coerce')
        
        return series_numeric
    
    def _add_data_type_prefix(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Add data type prefix to financial columns."""
        if data_type not in self.data_type_prefixes:
            return df
        
        prefix = self.data_type_prefixes[data_type]
        
        # Identify columns to rename (exclude metadata columns)
        metadata_columns = {
            'bank_name', 'date', 'year_month', 'source_file', 'source_sheet', 'data_type'
        }
        
        rename_map = {}
        for col in df.columns:
            if col not in metadata_columns and not col.startswith(prefix):
                # Clean column name and add prefix
                clean_col = self._clean_column_name(col)
                new_col = f"{prefix}{clean_col}"
                rename_map[col] = new_col
        
        if rename_map:
            df = df.rename(columns=rename_map)
            self.logger.debug(f"Added {prefix} prefix to {len(rename_map)} columns")
        
        return df
    
    def _clean_column_name(self, col_name: str) -> str:
        """Clean a column name for consistency."""
        # Convert to string and strip
        name = str(col_name).strip()
        
        # Replace spaces with underscores
        name = re.sub(r'\s+', '_', name)
        
        # Remove special characters except underscores and dots
        name = re.sub(r'[^\w\.]', '_', name)
        
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Convert to lowercase
        name = name.lower()
        
        return name
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date columns."""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        if 'year_month' in df.columns:
            # Ensure year_month is in YYYY-MM format
            df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
        return df
    
    def create_panel_dataset(self, transformed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create the final panel dataset by merging all data types.
        
        Uses the proven logic from panel_data_creation/create_panel_dataset.py
        for proper data type separation and full rectangular panel creation.
        
        Args:
            transformed_data: Dictionary with transformed data by type
            
        Returns:
            Panel dataset in wide format
        """
        self.logger.info("📋 Creating panel dataset from transformed data")
        
        # Build bank timeline for presence detection (from proven logic)
        self._build_bank_timeline(transformed_data)
        
        # Get all unique banks and dates across all data types
        all_banks = set()
        all_dates = set()
        
        for data_type, df in transformed_data.items():
            if df is not None and not df.empty:
                all_banks.update(df['bank_name'].unique())
                all_dates.update(df['date'].unique())
        
        all_banks = sorted(list(all_banks))
        all_dates = sorted(list(all_dates))
        
        self.logger.info(f"📊 Creating full panel: {len(all_banks)} banks × {len(all_dates)} dates = {len(all_banks) * len(all_dates):,} rows")
        
        # Create lookup dictionaries for each data type (no column mixing!)
        self.logger.info("🔄 Building data lookups by type...")
        data_lookups = {}
        for data_type, df in transformed_data.items():
            if df is not None and not df.empty:
                data_lookups[data_type] = {}
                for (bank, date), group in df.groupby(['bank_name', 'date']):
                    data_lookups[data_type][(bank, date)] = group.iloc[0]  # Take first if multiple
            else:
                data_lookups[data_type] = {}
        
        # Create full rectangular panel structure
        panel_data = []
        
        self.logger.info("🔄 Creating panel structure...")
        for i, bank in enumerate(all_banks):
            if i % 20 == 0:
                self.logger.info(f"   Processing bank {i+1}/{len(all_banks)}: {bank}")
                
            for date in all_dates:
                # Start with basic information
                row = {
                    'bank_name': bank,
                    'date': date,
                    'year_month': date.strftime('%Y-%m'),
                }
                
                # Check if this bank reported any data this month
                bank_present = any((bank, date) in data_lookups[dt] for dt in self.data_type_prefixes.keys())
                row['bank_present'] = 1 if bank_present else 0
                
                # Process each data type separately (preserves original columns!)
                for data_type in self.data_type_prefixes.keys():
                    # Check if this specific data type is available
                    type_available = (bank, date) in data_lookups[data_type]
                    row[f'{data_type}_present'] = 1 if type_available else 0
                    
                    if type_available:
                        # Get the data for this bank-date-type combination
                        type_row = data_lookups[data_type][(bank, date)]
                        
                        # Add all columns from this data type
                        for col in type_row.index:
                            if col not in ['bank_name', 'date', 'year_month', 'data_type', 'source_file', 'source_sheet']:
                                # Add the column (already has data type prefix from transformation)
                                row[col] = type_row[col]
                
                panel_data.append(row)
        
        # Convert to DataFrame
        panel_df = pd.DataFrame(panel_data)
        
        # Apply comprehensive bank name consolidation (from proven scripts)
        panel_df = self._consolidate_bank_names(panel_df)
        
        # Apply final cleanup (remove problematic banks)
        panel_df = self._final_cleanup(panel_df)
        
        # Sort by bank and date
        panel_df = panel_df.sort_values(['bank_name', 'date']).reset_index(drop=True)
        
        # Add processing metadata
        panel_df['processing_timestamp'] = datetime.now().isoformat()
        
        self.logger.info(f"📊 Panel dataset created: {len(panel_df)} rows, {len(panel_df.columns)} columns")
        self.logger.info(f"📅 Date range: {panel_df['date'].min()} to {panel_df['date'].max()}")
        self.logger.info(f"🏦 Banks: {panel_df['bank_name'].nunique()}")
        
        return panel_df
    
    def _build_bank_timeline(self, transformed_data: Dict[str, pd.DataFrame]):
        """Build bank timeline for presence detection."""
        self.bank_timeline = {}
        
        for data_type, df in transformed_data.items():
            if df is not None and not df.empty:
                for bank in df['bank_name'].unique():
                    if bank not in self.bank_timeline:
                        self.bank_timeline[bank] = {
                            'first_seen': None,
                            'last_seen': None,
                            'data_types': set()
                        }
                    
                    bank_data = df[df['bank_name'] == bank]
                    first_date = bank_data['date'].min()
                    last_date = bank_data['date'].max()
                    
                    if self.bank_timeline[bank]['first_seen'] is None or first_date < self.bank_timeline[bank]['first_seen']:
                        self.bank_timeline[bank]['first_seen'] = first_date
                    
                    if self.bank_timeline[bank]['last_seen'] is None or last_date > self.bank_timeline[bank]['last_seen']:
                        self.bank_timeline[bank]['last_seen'] = last_date
                    
                    self.bank_timeline[bank]['data_types'].add(data_type)
    
    def _consolidate_bank_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidate bank names using proven logic from consolidate_bank_names.py.
        
        Handles JSC/PJSC duplicates, name variations, and business logic.
        """
        self.logger.info("🔄 Consolidating bank names using proven business logic...")
        
        # Define comprehensive bank mappings from proven script
        bank_mappings = self._create_comprehensive_bank_mappings()
        
        # Remove aggregation rows first
        aggregation_banks = [
            'Banks owned by foreign bank groups',
            'Privately owned banks', 
            'State-owned banks',
            'Insolvent Banks',
            'Insolvent banks'
        ]
        
        original_rows = len(df)
        df = df[~df['bank_name'].isin(aggregation_banks)]
        removed_agg = original_rows - len(df)
        
        if removed_agg > 0:
            self.logger.info(f"🗑️ Removed {removed_agg} aggregation rows")
        
        # Track consolidation statistics
        consolidation_stats = {
            'groups_processed': 0,
            'banks_merged': 0,
            'data_recovered': 0
        }
        
        # Process each mapping group
        for canonical_name, variants in bank_mappings.items():
            # Check if canonical and variants exist
            existing_variants = [v for v in variants if v in df['bank_name'].unique()]
            canonical_exists = canonical_name in df['bank_name'].unique()
            
            if not canonical_exists and not existing_variants:
                continue  # Skip if neither canonical nor variants exist
                
            if not existing_variants:
                continue  # Skip if no variants to merge
            
            self.logger.debug(f"🔄 Consolidating: {canonical_name} <- {existing_variants}")
            
            # Process each variant
            for variant in existing_variants:
                if canonical_exists:
                    # Merge data: fill missing months in canonical with variant data
                    recovered_count = self._merge_bank_data(df, canonical_name, variant)
                    consolidation_stats['data_recovered'] += recovered_count
                else:
                    # Rename variant to canonical
                    df.loc[df['bank_name'] == variant, 'bank_name'] = canonical_name
                    canonical_exists = True
                
                # Remove variant after merging (unless it was renamed)
                if canonical_exists and variant != canonical_name:
                    df = df[df['bank_name'] != variant]
                    consolidation_stats['banks_merged'] += 1
            
            consolidation_stats['groups_processed'] += 1
        
        self.logger.info(f"✅ Bank consolidation completed:")
        self.logger.info(f"   Groups processed: {consolidation_stats['groups_processed']}")
        self.logger.info(f"   Banks merged: {consolidation_stats['banks_merged']}")
        self.logger.info(f"   Data points recovered: {consolidation_stats['data_recovered']}")
        
        return df
    
    def _create_comprehensive_bank_mappings(self) -> dict:
        """Create comprehensive bank name mappings from proven consolidate_bank_names.py."""
        mappings = {}
        
        # JSC/PJSC duplicates (22 groups) - proven business logic
        jsc_mappings = {
            'A-Bank': ['JSC \'A - BANK\''],
            'Alfa-Bank': ['JSC \'ALFA-BANK\''],
            'ALPARI BANK': ['JSC \'ALPARI BANK\'', 'Alpari Bank'],  # ALPARI BANK is canonical
            'Arkada': ['JSC JSCB \'ARCADA\'', 'Arkada JSCB JSC'],
            'Bank Credit Dnipro': ['JSC \'BANK CREDIT DNEPR\''],
            'Zemelny Capital': ['CB Zemelny Capital', 'JSC СВ \'ZEMELNY CAPITAL\''],
            'CIB': ['JSC \'CIB\''],
            'COMINVESTBANK': ['JSC \'COMINVESTBANK\''],
            'Deutsche Bank DBU': ['JSC Deutsche Bank DBU'],
            'EIB': ['JSC \'EIB\''],
            'FUIB': ['JSC \'FUIB\''],
            'IIB': ['JSC IIB'],
            'ING Bank Ukraine': ['JSC \'ING Bank Ukraine\''],
            'Idea Bank': ['JSC \'Idea Bank\''],
            'Ukrainian Capital Bank': ['JSC \'BANK \'UKRAINIAN CAPITAL\'', 'PJSC \'BANK \'UKRAINIAN CAPITAL\''],
            'Oschadbank': ['JSC \'Oschadbank\''],
            'Ukrainian Bank for Reconstruction and Development': ['JSC \'Ukrainian Bank for reconstruction and development\''],
            'Ukreximbank': ['JSC \'Ukreximbank\''],
            'PrivatBank': ['JSC CB \'PRIVATBANK\''],
            'Lviv': ['JSCB \'Lviv\''],
            'Prominvestbank': ['PJSC Prominvestbank'],
        }
        
        # Additional name variations (9 groups)
        variation_mappings = {
            'Pivdenny Bank': ['Pivdennyi Bank'],
            'UkrSibbank': ['UkrSibbank JSС', 'JSС \'UKRSIBBANK\''],
            'Megabank': ['MEGABANK JSC'],
            'OTP Bank': ['OTP BANK JSC'],
            'OKCI Bank': ['OKCI BANK, JSC'],
            'TAScombank': ['TASCOMBANK JSC'],
            'RWS Bank': ['RwS Bank'],
            'Pravex Bank': ['CB Pravex-Bank', 'PRAVEX BANK\' JSC'],
            'Cominbank': ['Cominvestbank', 'ComInvestBank', 'COMINBANK'],
        }
        
        # Raiffeisen acquisition chain (business logic)
        raiffeisen_mappings = {
            'Raiffeisen Bank': ['Raiffeisen Bank Aval', 'Raiffeisen Bank Aval JSC']
        }
        
        # Additional specific cases
        additional_mappings = {
            'Sberbank': ['SBERBANK\' JSC'],
            'Ukrconstinvestbank': ['UKRСONSTINVESTBANK\' JSC'],
        }
        
        # Combine all mappings
        mappings.update(jsc_mappings)
        mappings.update(variation_mappings)
        mappings.update(raiffeisen_mappings)
        mappings.update(additional_mappings)
        
        return mappings
    
    def _merge_bank_data(self, df: pd.DataFrame, canonical_name: str, variant_name: str) -> int:
        """Merge variant data into canonical bank for missing months."""
        canonical_mask = df['bank_name'] == canonical_name
        variant_mask = df['bank_name'] == variant_name
        
        canonical_dates = set(df[canonical_mask]['date'])
        variant_dates = set(df[variant_mask]['date'])
        
        # Find dates where canonical is missing but variant has data
        missing_dates = variant_dates - canonical_dates
        
        if missing_dates:
            # Copy variant data for missing dates and rename to canonical
            variant_data = df[variant_mask & df['date'].isin(missing_dates)].copy()
            variant_data['bank_name'] = canonical_name
            
            # Append to dataframe
            df = pd.concat([df, variant_data], ignore_index=True)
            
            return len(missing_dates)
        
        return 0
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final cleanup using proven logic from final_cleanup.py.
        
        Removes JSC/PJSC/JSCB variants, weird names, and low-data banks.
        """
        self.logger.info("🧹 Applying final cleanup (removing problematic banks)...")
        
        bank_names = df['bank_name'].unique()
        problematic_banks = set()
        
        # 1. JSC/PJSC/CB/JSB/JSCB variants that weren't consolidated
        jsc_patterns = [
            r'^JSC\s+',
            r'^PJSC\s+', 
            r'^JSCB\s+',
            r'^CB\s+',
            r'^JSС\s+',  # Cyrillic C
            r'^JSB\s+',
            r'^JOINT\s+STOCK\s+BANK\s+',
            r'.*PrJSC.*',  # Add PrJSC pattern
        ]
        
        for name in bank_names:
            for pattern in jsc_patterns:
                if re.match(pattern, str(name), re.IGNORECASE):
                    problematic_banks.add(name)
                    break
        
        # 2. Weird/short names
        for name in bank_names:
            name_clean = str(name).strip()
            if (len(name_clean) <= 3 or 
                name_clean.isdigit() or 
                name_clean in ['3', '4', '5', 'nan', ''] or
                name_clean == ''):
                problematic_banks.add(name)
        
        # 3. Banks with very few data points (≤2)
        for name in bank_names:
            if pd.notna(name):  # Only check valid names
                data_points = df[df['bank_name'] == name]['bank_present'].sum()
                if data_points <= 2:
                    problematic_banks.add(name)
        
        # 4. Additional problematic patterns
        additional_patterns = [
            r'.*PuJSC.*',  # PuJSC variants
            r'.*\bJSC\b.*',  # Any JSC in name
            r'.*\bPJSC\b.*',  # Any PJSC in name
        ]
        
        for name in bank_names:
            for pattern in additional_patterns:
                if re.search(pattern, str(name), re.IGNORECASE):
                    problematic_banks.add(name)
                    break
        
        # Remove problematic banks
        original_banks = df['bank_name'].nunique()
        original_rows = len(df)
        
        df_clean = df[~df['bank_name'].isin(problematic_banks)].copy()
        
        final_banks = df_clean['bank_name'].nunique()
        final_rows = len(df_clean)
        
        self.logger.info(f"🧹 Final cleanup completed:")
        self.logger.info(f"   Banks: {original_banks} → {final_banks} (-{original_banks - final_banks})")
        self.logger.info(f"   Rows: {original_rows:,} → {final_rows:,} (-{original_rows - final_rows:,})")
        self.logger.info(f"   Problematic banks removed: {len(problematic_banks)}")
        
        return df_clean
    
    def add_failure_flags(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add failure flags to the panel dataset.
        
        Based on the logic from panel_data_creation scripts.
        """
        self.logger.info("🚩 Adding failure flags to panel dataset")
        
        panel_df = panel_df.copy()
        
        # Initialize failure flags
        panel_df['failed'] = 0
        panel_df['failure_date'] = pd.NaT
        panel_df['last_reporting_date'] = pd.NaT
        
        # Get failure detection configuration
        min_consecutive_missing = self.config.get('failure_detection.min_consecutive_missing', 6)
        
        # Process each bank
        for bank_name in panel_df['bank_name'].unique():
            bank_data = panel_df[panel_df['bank_name'] == bank_name].copy()
            bank_data = bank_data.sort_values('date')
            
            # Find last reporting date (last date with bank_present = 1)
            last_reporting_mask = bank_data['bank_present'] == 1
            if last_reporting_mask.any():
                last_reporting_date = bank_data[last_reporting_mask]['date'].max()
                panel_df.loc[panel_df['bank_name'] == bank_name, 'last_reporting_date'] = last_reporting_date
                
                # Check for consecutive missing periods
                failure_date = self._detect_failure_date(bank_data, last_reporting_date, min_consecutive_missing)
                
                if failure_date is not None:
                    # Mark bank as failed from failure date onwards
                    failure_mask = (panel_df['bank_name'] == bank_name) & (panel_df['date'] >= failure_date)
                    panel_df.loc[failure_mask, 'failed'] = 1
                    panel_df.loc[panel_df['bank_name'] == bank_name, 'failure_date'] = failure_date
        
        # Log failure statistics
        failed_banks = panel_df[panel_df['failed'] == 1]['bank_name'].nunique()
        total_banks = panel_df['bank_name'].nunique()
        
        self.logger.info(f"🚩 Failure flags added: {failed_banks}/{total_banks} banks marked as failed")
        
        return panel_df
    
    def _detect_failure_date(self, bank_data: pd.DataFrame, last_reporting_date: pd.Timestamp, min_consecutive_missing: int) -> Optional[pd.Timestamp]:
        """
        Detect failure date for a bank based on consecutive missing periods.
        
        Conservative approach: Bank is considered failed if it stops reporting
        for at least min_consecutive_missing consecutive months.
        """
        # Get all dates after last reporting
        future_dates = bank_data[bank_data['date'] > last_reporting_date]['date']
        
        if len(future_dates) < min_consecutive_missing:
            return None  # Not enough future periods to determine failure
        
        # Check if all future periods are missing (bank_present = 0)
        future_data = bank_data[bank_data['date'] > last_reporting_date]
        
        if len(future_data) >= min_consecutive_missing:
            # Check if first min_consecutive_missing periods are all missing
            first_periods = future_data.head(min_consecutive_missing)
            
            if (first_periods['bank_present'] == 0).all():
                # Bank failed after last reporting date
                return last_reporting_date + pd.DateOffset(months=1)
        
        return None
    
    def get_transformation_stats(self, transformed_data: Dict[str, pd.DataFrame]) -> Dict:
        """Get statistics about the transformation process."""
        stats = {
            'data_types_processed': len(transformed_data),
            'total_rows': 0,
            'total_columns': 0,
            'by_data_type': {}
        }
        
        for data_type, df in transformed_data.items():
            if df is not None and not df.empty:
                type_stats = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'banks': df['bank_name'].nunique() if 'bank_name' in df.columns else 0,
                    'date_range': {
                        'start': df['date'].min().strftime('%Y-%m') if 'date' in df.columns else None,
                        'end': df['date'].max().strftime('%Y-%m') if 'date' in df.columns else None
                    }
                }
                stats['by_data_type'][data_type] = type_stats
                stats['total_rows'] += type_stats['rows']
                stats['total_columns'] += type_stats['columns']
        
        return stats 