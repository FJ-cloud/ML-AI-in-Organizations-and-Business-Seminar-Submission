#!/usr/bin/env python3
"""
Data Validator - Comprehensive validation for Ukrainian Bank data

Validates data at multiple stages:
- Raw Excel file validation
- Extracted data validation  
- Panel dataset validation
- Bank name consolidation and duplicate resolution
- Quality checks and reporting

Expert implementation with detailed validation logic.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np


class DataValidator:
    """
    Comprehensive data validator for Ukrainian bank ETL pipeline.
    
    Validates data quality at multiple stages and provides detailed
    reporting on issues found. Now includes integrated bank name
    consolidation and duplicate resolution.
    """
    
    def __init__(self, config):
        """Initialize the data validator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.min_banks_per_file = config.get('min_banks_per_file', 10)
        self.max_banks_per_file = config.get('max_banks_per_file', 200)
        self.required_columns = config.get('required_columns', ['bank_name', 'date', 'year_month'])
        
        self.logger.info("✅ Data validator initialized")
    
    def validate_excel_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a single Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'file_path': str(file_path),
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metadata': {},
            'sheets': {}
        }
        
        try:
            # Extract date from filename
            date_match = self._extract_date_from_filename(file_path)
            if date_match:
                validation_result['metadata']['extracted_date'] = date_match
            else:
                validation_result['warnings'].append("Could not extract date from filename")
            
            # Validate Excel sheets
            sheets_validation = self._validate_excel_sheets(file_path)
            validation_result['sheets'] = sheets_validation
            
            # Check if any sheets are valid
            valid_sheets = [sheet for sheet, info in sheets_validation.items() 
                          if info.get('is_valid', False)]
            
            if not valid_sheets:
                validation_result['is_valid'] = False
                validation_result['issues'].append("No valid sheets found")
            else:
                validation_result['metadata']['valid_sheets'] = valid_sheets
                validation_result['metadata']['valid_sheet_count'] = len(valid_sheets)
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"File validation failed: {str(e)}")
            self.logger.error(f"Error validating {file_path}: {str(e)}")
        
        return validation_result
    
    def _extract_date_from_filename(self, file_path: Path) -> Optional[str]:
        """Extract date from filename using regex patterns."""
        filename = file_path.name
        
        # Pattern: aggregation_YYYY-MM-DD_eng.xlsx
        pattern = r'aggregation_(\d{4}-\d{2}-\d{2})_eng\.xlsx'
        match = re.search(pattern, filename)
        
        if match:
            return match.group(1)
        
        return None
    
    def _validate_excel_sheets(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Validate sheets within an Excel file."""
        sheets_info = {}
        
        try:
            # Read Excel file to get sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Expected sheet patterns
            expected_patterns = {
                'assets': r'assets?_?$',
                'liabilities': r'liabilities?_?$', 
                'equity': r'equity_?$',
                'financial_results': r'results?_?$'
            }
            
            for sheet_name in sheet_names:
                sheet_info = {
                    'name': sheet_name,
                    'is_valid': False,
                    'sheet_type': None,
                    'issues': [],
                    'warnings': [],
                    'bank_count': 0
                }
                
                # Skip _NC (National Currency) sheets
                if '_NC' in sheet_name:
                    sheet_info['warnings'].append("Skipped _NC sheet")
                    sheets_info[sheet_name] = sheet_info
                    continue
                
                # Determine sheet type
                for sheet_type, pattern in expected_patterns.items():
                    if re.search(pattern, sheet_name, re.IGNORECASE):
                        sheet_info['sheet_type'] = sheet_type
                        break
                
                if not sheet_info['sheet_type']:
                    sheet_info['warnings'].append("Unknown sheet type")
                
                # Validate sheet content
                try:
                    content_validation = self._validate_sheet_content(file_path, sheet_name)
                    sheet_info.update(content_validation)
                    
                    if sheet_info.get('bank_count', 0) > 0:
                        sheet_info['is_valid'] = True
                        
                except Exception as e:
                    sheet_info['issues'].append(f"Content validation failed: {str(e)}")
                
                sheets_info[sheet_name] = sheet_info
                
        except Exception as e:
            self.logger.error(f"Error reading Excel file {file_path}: {str(e)}")
            
        return sheets_info
    
    def _validate_sheet_content(self, file_path: Path, sheet_name: str) -> Dict[str, Any]:
        """Validate the content of a specific sheet."""
        validation = {
            'bank_count': 0,
            'has_bank_column': False,
            'header_row': None,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Read sheet with multiple header row attempts
            df = None
            header_row = None
            
            # Try different header rows (4 and 5, 0-indexed as 3 and 4)
            for row_idx in [3, 4]:
                try:
                    temp_df = pd.read_excel(file_path, sheet_name=sheet_name, header=row_idx)
                    
                    # Check if 'Bank' column exists (case insensitive)
                    bank_cols = [col for col in temp_df.columns if 'bank' in str(col).lower()]
                    
                    if bank_cols:
                        df = temp_df
                        header_row = row_idx + 1  # Convert to 1-indexed
                        validation['has_bank_column'] = True
                        validation['header_row'] = header_row
                        break
                        
                except Exception:
                    continue
            
            if df is not None and validation['has_bank_column']:
                # Estimate bank count
                bank_count = self._estimate_bank_count(df)
                validation['bank_count'] = bank_count
                
                if bank_count < self.min_banks_per_file:
                    validation['warnings'].append(f"Low bank count: {bank_count}")
                elif bank_count > self.max_banks_per_file:
                    validation['warnings'].append(f"High bank count: {bank_count}")
                
            else:
                validation['issues'].append("No valid bank column found")
                
        except Exception as e:
            validation['issues'].append(f"Sheet reading failed: {str(e)}")
        
        return validation
    
    def _estimate_bank_count(self, df: pd.DataFrame) -> int:
        """Estimate the number of banks in a DataFrame."""
        # Find the bank column
        bank_cols = [col for col in df.columns if 'bank' in str(col).lower()]
        
        if not bank_cols:
            return 0
        
        bank_col = bank_cols[0]
        
        # Count non-null, non-empty values that don't look like aggregation rows
        bank_values = df[bank_col].dropna()
        bank_values = bank_values[bank_values.astype(str).str.strip() != '']
        
        # Filter out aggregation rows
        aggregation_patterns = [
            'total', 'sum', 'всього', 'разом', 'банки', 'banks',
            'state-owned', 'privately', 'foreign', 'insolvent'
        ]
        
        filtered_banks = bank_values[
            ~bank_values.astype(str).str.lower().str.contains('|'.join(aggregation_patterns), na=False)
        ]
        
        return len(filtered_banks.unique())
    
    def validate_extracted_data(self, extracted_data: Dict[str, List[pd.DataFrame]]) -> Dict[str, Any]:
        """
        Validate extracted data from multiple files.
        
        Args:
            extracted_data: Dict mapping data types to lists of DataFrames
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'data_types': {}
        }
        
        total_dataframes = 0
        total_rows = 0
        
        for data_type, dataframes in extracted_data.items():
            type_validation = self._validate_dataframe_list(dataframes, data_type)
            validation_result['data_types'][data_type] = type_validation
            
            total_dataframes += len(dataframes)
            total_rows += sum(len(df) for df in dataframes if df is not None)
            
            if not type_validation['is_valid']:
                validation_result['is_valid'] = False
                validation_result['issues'].extend(type_validation['issues'])
            
            validation_result['warnings'].extend(type_validation['warnings'])
        
        validation_result['statistics'] = {
            'total_data_types': len(extracted_data),
            'total_dataframes': total_dataframes,
            'total_rows': total_rows,
            'average_rows_per_dataframe': round(total_rows / total_dataframes, 2) if total_dataframes > 0 else 0
        }
        
        return validation_result
    
    def _validate_dataframe_list(self, dataframes: List[pd.DataFrame], data_type: str) -> Dict[str, Any]:
        """Validate a list of DataFrames for a specific data type."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {
                'count': len(dataframes),
                'total_rows': 0,
                'null_dataframes': 0,
                'empty_dataframes': 0
            }
        }
        
        for i, df in enumerate(dataframes):
            if df is None:
                validation['statistics']['null_dataframes'] += 1
                validation['warnings'].append(f"{data_type} DataFrame {i} is None")
                continue
            
            if len(df) == 0:
                validation['statistics']['empty_dataframes'] += 1
                validation['warnings'].append(f"{data_type} DataFrame {i} is empty")
                continue
            
            validation['statistics']['total_rows'] += len(df)
            
            # Validate individual DataFrame
            df_validation = self._validate_dataframe(df, f"{data_type}_{i}")
            if not df_validation['is_valid']:
                validation['is_valid'] = False
                validation['issues'].extend(df_validation['issues'])
            
            validation['warnings'].extend(df_validation['warnings'])
        
        return validation
    
    def _validate_dataframe(self, df: pd.DataFrame, context: str) -> Dict[str, Any]:
        """Validate a single DataFrame."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for required columns
        if 'bank_name' not in df.columns:
            validation['is_valid'] = False
            validation['issues'].append(f"{context}: Missing 'bank_name' column")
        
        # Check for data quality
        if len(df) > 0:
            null_banks = df.get('bank_name', pd.Series()).isnull().sum()
            if null_banks > len(df) * 0.5:  # More than 50% null
                validation['warnings'].append(f"{context}: High null rate in bank_name: {null_banks}/{len(df)}")
        
        return validation
    
    def validate_panel_dataset(self, panel_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the final panel dataset.
        
        Args:
            panel_df: Panel dataset DataFrame
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'column_analysis': {},
            'quality_checks': {}
        }
        
        # Basic statistics
        validation_result['statistics'] = {
            'total_rows': len(panel_df),
            'total_columns': len(panel_df.columns),
            'unique_banks': panel_df['bank_name'].nunique() if 'bank_name' in panel_df.columns else 0,
            'date_range': {
                'start': panel_df['date'].min() if 'date' in panel_df.columns else None,
                'end': panel_df['date'].max() if 'date' in panel_df.columns else None,
                'unique_dates': panel_df['date'].nunique() if 'date' in panel_df.columns else 0
            }
        }
        
        # Missing data analysis
        missing_rate = panel_df.isnull().sum().sum() / panel_df.size
        validation_result['statistics']['missing_data_rate'] = round(missing_rate * 100, 2)
        
        # Data availability rate
        if 'bank_present' in panel_df.columns:
            availability_rate = panel_df['bank_present'].mean()
            validation_result['statistics']['data_availability_rate'] = round(availability_rate * 100, 2)
        
        # Column analysis
        validation_result['column_analysis'] = self._analyze_panel_columns(panel_df)
        
        # Quality checks
        validation_result['quality_checks'] = self._perform_quality_checks(panel_df)
        
        # Determine overall validity
        if validation_result['quality_checks']['critical_issues']:
            validation_result['is_valid'] = False
            validation_result['issues'].extend(validation_result['quality_checks']['critical_issues'])
        
        if validation_result['quality_checks']['warnings']:
            validation_result['warnings'].extend(validation_result['quality_checks']['warnings'])
        
        return validation_result
    
    def _analyze_panel_columns(self, panel_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze columns in the panel dataset."""
        analysis = {
            'metadata_columns': [],
            'financial_columns': [],
            'assets_columns': [],
            'liabilities_columns': [],
            'equity_columns': [],
            'financial_results_columns': [],
            'other_columns': []
        }
        
        metadata_cols = {'bank_name', 'date', 'year_month', 'bank_present', 'failed', 'failure_date', 'last_reporting_date', 'processing_timestamp'}
        
        for col in panel_df.columns:
            if col in metadata_cols:
                analysis['metadata_columns'].append(col)
            elif col.startswith('assets_'):
                analysis['assets_columns'].append(col)
                analysis['financial_columns'].append(col)
            elif col.startswith('liabilities_'):
                analysis['liabilities_columns'].append(col)
                analysis['financial_columns'].append(col)
            elif col.startswith('equity_'):
                analysis['equity_columns'].append(col)
                analysis['financial_columns'].append(col)
            elif col.startswith('financial_results_'):
                analysis['financial_results_columns'].append(col)
                analysis['financial_columns'].append(col)
            else:
                analysis['other_columns'].append(col)
        
        return analysis
    
    def _perform_quality_checks(self, panel_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive quality checks on panel dataset."""
        checks = {
            'critical_issues': [],
            'warnings': [],
            'passed_checks': []
        }
        
        # Check 1: Required columns
        required_cols = ['bank_name', 'date']
        missing_required = [col for col in required_cols if col not in panel_df.columns]
        
        if missing_required:
            checks['critical_issues'].extend([f"Missing required column: {col}" for col in missing_required])
        else:
            checks['passed_checks'].append("All required columns present")
        
        # Check 2: Bank name quality
        if 'bank_name' in panel_df.columns:
            null_banks = panel_df['bank_name'].isnull().sum()
            if null_banks > 0:
                checks['warnings'].append(f"{null_banks} rows with null bank names")
            
            unique_banks = panel_df['bank_name'].nunique()
            if unique_banks < 10:
                checks['warnings'].append(f"Low number of unique banks: {unique_banks}")
            else:
                checks['passed_checks'].append(f"Good bank diversity: {unique_banks} unique banks")
        
        # Check 3: Date consistency
        if 'date' in panel_df.columns:
            try:
                date_series = pd.to_datetime(panel_df['date'])
                date_nulls = date_series.isnull().sum()
                
                if date_nulls > 0:
                    checks['warnings'].append(f"{date_nulls} rows with invalid dates")
                else:
                    checks['passed_checks'].append("All dates are valid")
                
            except Exception as e:
                checks['critical_issues'].append(f"Date validation failed: {str(e)}")
        
        # Check 4: Data availability
        if 'bank_present' in panel_df.columns:
            availability_rate = panel_df['bank_present'].mean()
            
            if availability_rate < 0.1:  # Less than 10%
                checks['critical_issues'].append(f"Very low data availability: {availability_rate:.1%}")
            elif availability_rate < 0.3:  # Less than 30%
                checks['warnings'].append(f"Low data availability: {availability_rate:.1%}")
            else:
                checks['passed_checks'].append(f"Acceptable data availability: {availability_rate:.1%}")
        
        # Check 5: Financial data presence
        financial_cols = [col for col in panel_df.columns if any(col.startswith(prefix) for prefix in ['assets_', 'liabilities_', 'equity_', 'financial_results_'])]
        
        if not financial_cols:
            checks['critical_issues'].append("No financial data columns found")
        else:
            checks['passed_checks'].append(f"Financial data present: {len(financial_cols)} columns")
        
        return checks
    
    def consolidate_panel_dataset(self, panel_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Consolidate bank names and resolve duplicates in the panel dataset.
        
        This integrates the post-processing consolidation logic directly into
        the validation pipeline, ensuring all future Excel files are properly
        consolidated automatically.
        
        Args:
            panel_df: Panel dataset DataFrame
            
        Returns:
            Tuple of (consolidated_df, consolidation_report)
        """
        self.logger.info("🔄 Starting bank name consolidation and duplicate resolution")
        
        # Define the bank name mappings (from post-processing script)
        bank_mappings = {
            'Clearing House JSB': 'Clearing House',
            'Industrialbank JSB': 'Industrialbank', 
            'Pivdennyi JSB': 'Pivdenny Bank',
            'Ukrgаzbаnk JSB': 'Ukrgasbank'  # Note: contains Cyrillic characters
        }
        
        consolidation_report = {
            'original_banks': panel_df['bank_name'].nunique(),
            'original_rows': len(panel_df),
            'mappings_applied': {},
            'duplicates_resolved': 0,
            'final_banks': 0,
            'final_rows': 0,
            'issues': [],
            'warnings': []
        }
        
        # Make a copy to avoid modifying the original
        df = panel_df.copy()
        
        # Step 1: Apply bank name consolidations
        self.logger.info("🏦 Applying bank name consolidations...")
        
        for variant, canonical in bank_mappings.items():
            mask = df['bank_name'] == variant
            rows_changed = mask.sum()
            
            if rows_changed > 0:
                df.loc[mask, 'bank_name'] = canonical
                consolidation_report['mappings_applied'][variant] = {
                    'canonical': canonical,
                    'rows_changed': int(rows_changed)
                }
                self.logger.info(f"  ✅ {variant} → {canonical}: {rows_changed} rows updated")
            else:
                self.logger.debug(f"  ⚠️  {variant}: No rows found")
        
        # Step 2: Check for duplicate bank-date combinations
        self.logger.info("🔍 Checking for duplicate bank-date combinations...")
        
        duplicates = df.groupby(['bank_name', 'date']).size()
        duplicate_pairs = duplicates[duplicates > 1]
        
        if len(duplicate_pairs) > 0:
            self.logger.info(f"⚠️  Found {len(duplicate_pairs)} duplicate bank-date combinations")
            consolidation_report['duplicates_resolved'] = len(duplicate_pairs)
            
            # Log some examples
            for (bank, date), count in duplicate_pairs.head(5).items():
                self.logger.info(f"    {bank} on {date}: {count} records")
                consolidation_report['warnings'].append(f"Duplicate: {bank} on {date} ({count} records)")
            
            # Step 3: Resolve duplicates by keeping record with more complete data
            self.logger.info("🔧 Resolving duplicates by keeping records with more complete data...")
            
            def resolve_duplicates(group):
                if len(group) == 1:
                    return group
                
                # Count non-null values for each row (excluding metadata columns)
                metadata_columns = {'bank_name', 'date', 'year_month', 'processing_timestamp', 
                                  'failed', 'failure_date', 'last_reporting_date'}
                data_columns = [col for col in group.columns if col not in metadata_columns]
                
                if data_columns:
                    non_null_counts = group[data_columns].notna().sum(axis=1)
                    # Keep the row with the most non-null values
                    best_idx = non_null_counts.idxmax()
                    return group.loc[[best_idx]]
                else:
                    # If no data columns, just keep the first row
                    return group.iloc[[0]]
            
            # Apply duplicate resolution
            original_len = len(df)
            df = df.groupby(['bank_name', 'date'], group_keys=False).apply(resolve_duplicates)
            df = df.reset_index(drop=True)
            
            rows_removed = original_len - len(df)
            self.logger.info(f"✅ Resolved duplicates. Removed {rows_removed} duplicate rows")
            consolidation_report['duplicates_resolved'] = rows_removed
            
        else:
            self.logger.info("✅ No duplicate bank-date combinations found")
        
        # Step 4: Final validation and reporting
        consolidation_report['final_banks'] = df['bank_name'].nunique()
        consolidation_report['final_rows'] = len(df)
        
        banks_reduced = consolidation_report['original_banks'] - consolidation_report['final_banks']
        rows_reduced = consolidation_report['original_rows'] - consolidation_report['final_rows']
        
        self.logger.info(f"📊 Consolidation Summary:")
        self.logger.info(f"  • Banks: {consolidation_report['original_banks']} → {consolidation_report['final_banks']} (-{banks_reduced})")
        self.logger.info(f"  • Rows: {consolidation_report['original_rows']} → {consolidation_report['final_rows']} (-{rows_reduced})")
        self.logger.info(f"  • Mappings applied: {len(consolidation_report['mappings_applied'])}")
        self.logger.info(f"  • Duplicates resolved: {consolidation_report['duplicates_resolved']}")
        
        # Step 5: Validate consolidation results
        self._validate_consolidation_results(df, consolidation_report)
        
        self.logger.info("✅ Bank name consolidation and duplicate resolution completed")
        
        return df, consolidation_report
    
    def _validate_consolidation_results(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate the results of consolidation."""
        
        # Check 1: Verify no JSB variants remain
        jsb_variants = df[df['bank_name'].str.contains('JSB|JSCB', case=False, na=False)]
        if len(jsb_variants) > 0:
            remaining_jsb = jsb_variants['bank_name'].unique()
            report['issues'].append(f"JSB variants still present: {list(remaining_jsb)}")
            self.logger.warning(f"⚠️  JSB variants still present: {remaining_jsb}")
        else:
            self.logger.info("✅ No JSB variants remaining")
        
        # Check 2: Verify no duplicate bank-date combinations
        duplicates = df.groupby(['bank_name', 'date']).size()
        duplicate_count = (duplicates > 1).sum()
        
        if duplicate_count > 0:
            report['issues'].append(f"Duplicate bank-date combinations still present: {duplicate_count}")
            self.logger.warning(f"⚠️  {duplicate_count} duplicate bank-date combinations still present")
        else:
            self.logger.info("✅ No duplicate bank-date combinations")
        
        # Check 3: Validate expected bank count (should be around 67 for full dataset)
        final_banks = df['bank_name'].nunique()
        if final_banks > 70:
            report['warnings'].append(f"Higher than expected bank count: {final_banks} (expected ~67)")
            self.logger.warning(f"⚠️  Higher than expected bank count: {final_banks}")
        elif final_banks < 60:
            report['warnings'].append(f"Lower than expected bank count: {final_banks} (expected ~67)")
            self.logger.warning(f"⚠️  Lower than expected bank count: {final_banks}")
        else:
            self.logger.info(f"✅ Bank count within expected range: {final_banks}")
        
        # Check 4: Validate failure flags are preserved
        if 'failed' in df.columns:
            failed_count = df['failed'].sum()
            if failed_count > 0:
                self.logger.info(f"✅ Failure flags preserved: {failed_count} failure observations")
            else:
                report['warnings'].append("No failure flags found in dataset")
                self.logger.warning("⚠️  No failure flags found in dataset") 