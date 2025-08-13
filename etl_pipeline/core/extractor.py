#!/usr/bin/env python3
"""
Data Extractor - Extract data from Ukrainian Bank Excel files

Based on the proven working logic from src/loader.py and src/etl.py.
Uses the same header detection, sheet matching, and data extraction
approaches that successfully processed all 76 Excel files.

Expert implementation with intelligent header detection and fuzzy sheet matching.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
from rapidfuzz import fuzz, process


class DataExtractor:
    """
    Data extractor for Ukrainian bank Excel files.
    
    Uses the proven working logic from the existing successful ETL pipeline.
    """
    
    def __init__(self, config):
        """Initialize the data extractor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sheet patterns from working config
        self.sheet_patterns = {
            'assets': [
                r"(?i)^assets?$",
                r"(?i)^total[ _]+assets?$"
            ],
            'liabilities': [
                r"(?i)^liabilities?$", 
                r"(?i)^total[ _]+liab"
            ],
            'equity': [
                r"(?i)^equity$",
                r"(?i)^shareholders?[ _]+equity$",
                r"(?i)^total[ _]+equity$"
            ],
            'financial_results': [
                r"(?i)^financial[ _]+results?_?$",
                r"(?i)^fin[ _]*results?_?$",
                r"(?i)^financial.*res_?$"
            ]
        }
        
        # Fuzzy matching threshold from working config
        self.fuzzy_threshold = config.get('fuzzy_threshold', 80)
        
        self.logger.info("📖 Data extractor initialized with proven working logic")
    
    def extract_file(self, file_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Extract data from a single Excel file using proven working logic.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary with extracted DataFrames by data type
        """
        self.logger.debug(f"Extracting data from {file_path.name}")
        
        try:
            # Extract date info
            date_info = self._extract_date_from_filename(file_path)
            if not date_info:
                raise ValueError(f"Could not extract date from filename: {file_path.name}")
            
            # Get sheet names
            with pd.ExcelFile(file_path) as excel_file:
                sheet_names = excel_file.sheet_names
                
                extracted_data = {}
                
                # Process each data type using working logic
                for data_type, patterns in self.sheet_patterns.items():
                    sheet_name = self._choose_sheet(sheet_names, patterns)
                    
                    if sheet_name:
                        self.logger.debug(f"Processing {data_type} from sheet '{sheet_name}'")
                        df = self._extract_sheet_data(excel_file, sheet_name, date_info)
                        
                        if df is not None and not df.empty:
                            extracted_data[data_type] = df
                        else:
                            self.logger.warning(f"No data extracted from {data_type} sheet")
                    else:
                        self.logger.warning(f"No sheet found for {data_type}")
                
                return extracted_data
                
        except Exception as e:
            self.logger.error(f"Failed to extract {file_path.name}: {str(e)}")
            raise
    
    def _choose_sheet(self, all_sheet_names: List[str], patterns: List[str]) -> Optional[str]:
        """
        Choose sheet using exact same logic as working ETL pipeline.
        
        From src/loader.py _choose_sheet function.
        Explicitly avoids _NC (National Currency) sheets - we want USD sheets.
        """
        # Filter out _NC sheets first - we don't want National Currency sheets
        filtered_sheet_names = [name for name in all_sheet_names if not name.endswith('_NC')]
        
        # First try exact pattern matching on filtered sheets
        for pat in patterns:
            for name in filtered_sheet_names:
                if re.search(pat, name, flags=re.IGNORECASE):
                    return name
        
        # Fall back to fuzzy matching on filtered sheets
        if filtered_sheet_names:
            result = process.extractOne(" ".join(patterns), filtered_sheet_names, scorer=fuzz.token_sort_ratio)
            if result:
                best, score = result[0], result[1]
                return best if score >= self.fuzzy_threshold else None
        
        return None
    
    def _extract_sheet_data(self, excel_file, sheet_name: str, date_info: Dict) -> Optional[pd.DataFrame]:
        """
        Extract data from a sheet using the proven working header detection logic.
        
        Based on read_category_df from src/loader.py.
        """
        try:
            # First read without headers to analyze structure
            df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            
            if df_raw.empty:
                return None
            
            # Use proven header detection logic
            header_row = self._detect_header_row_working(df_raw)
            
            # Re-read with correct header
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
            
            # Apply basic cleaning using working logic
            df = self._clean_data_working(df, date_info)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract sheet {sheet_name}: {str(e)}")
            return None
    
    def _detect_header_row_working(self, df_raw: pd.DataFrame) -> int:
        """
        Header detection using improved logic based on working src/loader.py.
        
        More robust detection that handles different sheet structures.
        """
        def is_valid_header_row(row) -> bool:
            """Check if a row looks like headers (contains 'Bank' and other text)."""
            # Convert row to strings
            str_values = [str(val).strip().lower() for val in row if pd.notna(val)]
            
            # Must contain 'bank' to be a valid header
            has_bank = any('bank' in val for val in str_values)
            if not has_bank:
                return False
            
            # Check if more than 50% are meaningful text (not numbers or empty)
            meaningful_count = sum(1 for val in str_values if val and not val.isdigit() and val != 'nan')
            return meaningful_count >= len(str_values) * 0.5

        # Check rows 3-7 (0-based index) to find the best header row
        best_header_row = 4  # Default to row 5 (0-based index 4)
        
        for row_idx in range(3, min(8, len(df_raw))):
            if is_valid_header_row(df_raw.iloc[row_idx]):
                # Found a valid header row, use it
                best_header_row = row_idx
                break
        
        return best_header_row
    
    def _clean_data_working(self, df: pd.DataFrame, date_info: Dict) -> pd.DataFrame:
        """
        Clean data using the proven working logic from the existing ETL.
        
        Based on the working cleaner and basic processing logic.
        """
        # Find bank column (case insensitive) - exact same logic as working code
        bank_col = None
        for col in df.columns:
            if isinstance(col, str) and col.lower().strip() == 'bank':
                bank_col = col
                break
        
        if bank_col is None:
            self.logger.warning("No bank column found")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Rename bank column to standard name
        df = df.rename(columns={bank_col: 'bank_name'})
        
        # Clean bank names
        df['bank_name'] = df['bank_name'].astype(str).str.strip()
        
        # Remove invalid rows using proven patterns
        invalid_patterns = [
            r'^(?:total|sum|всього|разом)',
            r'^source:',
            r'^note:',
            r'^notes:',
            r'^\*',
            r'^-+$',
            r'^\s*$'
        ]
        
        mask = df['bank_name'].notna()
        for pattern in invalid_patterns:
            mask &= ~df['bank_name'].str.contains(pattern, case=False, na=False, regex=True)
        
        # Also remove rows where bank_name is empty or 'nan'
        mask &= (df['bank_name'] != '') & (df['bank_name'] != 'nan')
        
        df_cleaned = df[mask].copy()
        
        if len(df_cleaned) < len(df):
            removed = len(df) - len(df_cleaned)
            self.logger.debug(f"Removed {removed} invalid rows")
        
        # Add metadata columns
        df_cleaned['date'] = pd.to_datetime(date_info['date'])
        df_cleaned['year_month'] = date_info['year_month']
        df_cleaned['source_file'] = date_info['filename']
        df_cleaned['source_sheet'] = date_info.get('sheet_name', '')
        
        return df_cleaned.reset_index(drop=True)
    
    def _extract_date_from_filename(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract date information from filename using working regex."""
        # Use the working DATE_RE pattern from src/etl.py
        date_pattern = re.compile(r"(?P<year>\d{4})[^\d]?(?P<month>\d{2})")
        match = date_pattern.search(file_path.name)
        
        if match:
            year = match.group('year')
            month = match.group('month')
            date_obj = pd.to_datetime(f"{year}-{month}-01")
            
            return {
                'year': int(year),
                'month': int(month),
                'year_month': f"{year}-{month}",
                'date': date_obj,
                'filename': file_path.name
            }
        return None
    
    def analyze_file_coverage(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Analyze coverage of Excel files to identify missing periods.
        
        Args:
            file_paths: List of Excel file paths
            
        Returns:
            Dictionary with coverage analysis
        """
        dates = []
        for file_path in file_paths:
            date_info = self._extract_date_from_filename(file_path)
            if date_info:
                dates.append(date_info['date'])
        
        if not dates:
            return {'error': 'No valid dates found'}
        
        dates.sort()
        start_date = dates[0]
        end_date = dates[-1]
        
        # Generate expected monthly range
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Find missing dates
        found_dates = set(dates)
        expected_dates_set = set(expected_dates)
        missing_dates = expected_dates_set - found_dates
        
        return {
            'start': start_date.strftime('%Y-%m'),
            'end': end_date.strftime('%Y-%m'),
            'found_months': len(dates),
            'expected_months': len(expected_dates),
            'missing_months': [d.strftime('%Y-%m') for d in sorted(missing_dates)],
            'coverage_ratio': len(dates) / len(expected_dates)
        } 