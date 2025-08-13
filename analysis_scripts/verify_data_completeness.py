#!/usr/bin/env python3
"""
Data Completeness Verification Script

This script thoroughly verifies that our ETL pipeline is extracting all available
data from each Excel file, including all sheets and all data categories.

Usage:
    python analysis_scripts/verify_data_completeness.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
from rapidfuzz import fuzz, process

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def analyze_excel_file_structure(file_path):
    """Analyze the structure of an Excel file to understand what data is available."""
    try:
        with pd.ExcelFile(file_path) as excel_file:
            file_info = {
                'file_path': file_path,
                'sheet_names': excel_file.sheet_names,
                'sheet_analysis': {}
            }
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read first 10 rows to understand structure
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=10)
                    
                    # Look for header row (row with 'bank' column)
                    header_row = None
                    for i in range(len(df)):
                        row_values = df.iloc[i].astype(str).str.lower()
                        if any('bank' in str(val) for val in row_values):
                            header_row = i
                            break
                    
                    # Count total rows with data
                    df_full = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                    total_rows = len(df_full)
                    
                    # Count non-empty rows
                    non_empty_rows = df_full.dropna(how='all').shape[0]
                    
                    file_info['sheet_analysis'][sheet_name] = {
                        'header_row': header_row,
                        'total_rows': total_rows,
                        'non_empty_rows': non_empty_rows,
                        'columns': df.shape[1] if not df.empty else 0
                    }
                    
                except Exception as e:
                    file_info['sheet_analysis'][sheet_name] = {
                        'error': str(e)
                    }
            
            return file_info
            
    except Exception as e:
        return {'file_path': file_path, 'error': str(e)}

def categorize_sheets(sheet_names):
    """Categorize sheet names into data categories."""
    categories = {
        'assets': [],
        'liabilities': [],
        'equity': [],
        'financial_results': [],
        'unknown': []
    }
    
    patterns = {
        'assets': [r'(?i)assets?', r'(?i)активи'],
        'liabilities': [r'(?i)liabilit', r'(?i)зобов', r'(?i)пасив'],
        'equity': [r'(?i)equity', r'(?i)capital', r'(?i)капітал'],
        'financial_results': [r'(?i)financial.*result', r'(?i)fin.*result', r'(?i)прибут', r'(?i)збито']
    }
    
    for sheet_name in sheet_names:
        categorized = False
        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                if re.search(pattern, sheet_name):
                    categories[category].append(sheet_name)
                    categorized = True
                    break
            if categorized:
                break
        
        if not categorized:
            categories['unknown'].append(sheet_name)
    
    return categories

def verify_etl_coverage():
    """Verify that our ETL pipeline is capturing all available data."""
    
    print("🔍 COMPREHENSIVE DATA COMPLETENESS VERIFICATION")
    print("=" * 60)
    
    # Get all Excel files
    raw_data_dir = Path("data/raw_balance_sheets")
    excel_files = list(raw_data_dir.glob("*.xlsx"))
    
    print(f"📁 Found {len(excel_files)} Excel files to analyze")
    print()
    
    # Analyze each file
    all_file_info = []
    sheet_name_frequency = defaultdict(int)
    category_coverage = defaultdict(lambda: defaultdict(int))
    
    print("📊 ANALYZING EACH FILE...")
    print("-" * 30)
    
    for i, file_path in enumerate(sorted(excel_files)):
        print(f"[{i+1:2d}/{len(excel_files)}] {file_path.name}")
        
        file_info = analyze_excel_file_structure(file_path)
        all_file_info.append(file_info)
        
        if 'error' not in file_info:
            # Count sheet names
            for sheet_name in file_info['sheet_names']:
                sheet_name_frequency[sheet_name] += 1
            
            # Categorize sheets
            categories = categorize_sheets(file_info['sheet_names'])
            for category, sheets in categories.items():
                category_coverage[category][len(sheets)] += 1
    
    print()
    print("📈 SHEET NAME ANALYSIS")
    print("-" * 25)
    
    # Show most common sheet names
    print("Most common sheet names:")
    for sheet_name, count in sorted(sheet_name_frequency.items(), key=lambda x: x[1], reverse=True)[:15]:
        percentage = (count / len(excel_files)) * 100
        print(f"  {sheet_name:<35} {count:>3}/{len(excel_files)} ({percentage:5.1f}%)")
    
    print()
    print("📊 CATEGORY COVERAGE ANALYSIS")
    print("-" * 30)
    
    for category in ['assets', 'liabilities', 'equity', 'financial_results', 'unknown']:
        print(f"\n{category.upper()}:")
        if category_coverage[category]:
            for sheet_count, file_count in sorted(category_coverage[category].items()):
                percentage = (file_count / len(excel_files)) * 100
                print(f"  {sheet_count} sheets: {file_count:>3} files ({percentage:5.1f}%)")
        else:
            print("  No sheets found")
    
    # Check for files with missing categories
    print()
    print("⚠️  FILES WITH MISSING CATEGORIES")
    print("-" * 35)
    
    missing_categories = defaultdict(list)
    
    for file_info in all_file_info:
        if 'error' not in file_info:
            categories = categorize_sheets(file_info['sheet_names'])
            file_name = Path(file_info['file_path']).name
            
            for category in ['assets', 'liabilities', 'equity', 'financial_results']:
                if not categories[category]:
                    missing_categories[category].append(file_name)
    
    for category, files in missing_categories.items():
        if files:
            print(f"\n{category.upper()} missing in {len(files)} files:")
            for file_name in files[:5]:  # Show first 5
                print(f"  - {file_name}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
    
    # Compare with ETL pipeline output
    print()
    print("🔄 COMPARING WITH ETL PIPELINE OUTPUT")
    print("-" * 40)
    
    try:
        # Load the final dataset
        final_dataset = pd.read_csv("output_final/ukrainian_banks_panel_dataset_FINAL.csv")
        
        print(f"Final dataset shape: {final_dataset.shape}")
        print(f"Unique dates: {final_dataset['date'].nunique()}")
        print(f"Date range: {final_dataset['date'].min()} to {final_dataset['date'].max()}")
        
        # Check column categories
        columns = final_dataset.columns.tolist()
        column_categories = {
            'assets': [col for col in columns if col.startswith('assets_')],
            'liabilities': [col for col in columns if col.startswith('liabilities_')],
            'equity': [col for col in columns if col.startswith('equity_')],
            'financial_results': [col for col in columns if col.startswith('financial_results_')]
        }
        
        print("\nColumn counts by category:")
        for category, cols in column_categories.items():
            print(f"  {category:<20} {len(cols):>3} columns")
        
        # Check data completeness by date
        print("\nData completeness by date (sample):")
        date_counts = final_dataset['date'].value_counts().sort_index()
        for date in sorted(date_counts.index)[:10]:  # Show first 10 dates
            print(f"  {date}: {date_counts[date]:>3} banks")
        
        if len(date_counts) > 10:
            print(f"  ... and {len(date_counts) - 10} more dates")
            
    except Exception as e:
        print(f"❌ Could not load final dataset: {e}")
    
    print()
    print("✅ VERIFICATION COMPLETE")
    print("=" * 25)
    
    return all_file_info, sheet_name_frequency, category_coverage

def check_specific_file_details(file_path):
    """Check detailed information about a specific file."""
    print(f"\n🔍 DETAILED ANALYSIS: {Path(file_path).name}")
    print("=" * 50)
    
    try:
        with pd.ExcelFile(file_path) as excel_file:
            print(f"Sheet names ({len(excel_file.sheet_names)}):")
            
            for i, sheet_name in enumerate(excel_file.sheet_names):
                print(f"\n[{i+1}] {sheet_name}")
                
                try:
                    # Read with different header options
                    df_no_header = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=10)
                    
                    print(f"    Shape: {df_no_header.shape}")
                    
                    # Look for header row
                    for row_idx in range(min(8, len(df_no_header))):
                        row_values = df_no_header.iloc[row_idx].astype(str).str.lower()
                        if any('bank' in str(val) for val in row_values):
                            print(f"    Header row found at: {row_idx}")
                            print(f"    Header preview: {list(df_no_header.iloc[row_idx].values)[:5]}")
                            break
                    
                    # Check for data rows
                    df_full = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                    non_empty = df_full.dropna(how='all')
                    print(f"    Total rows: {len(df_full)}, Non-empty: {len(non_empty)}")
                    
                except Exception as e:
                    print(f"    ❌ Error reading sheet: {e}")
                    
    except Exception as e:
        print(f"❌ Error opening file: {e}")

if __name__ == "__main__":
    # Run comprehensive verification
    all_file_info, sheet_frequency, category_coverage = verify_etl_coverage()
    
    # Optionally check specific files
    print("\n" + "="*60)
    print("DETAILED FILE ANALYSIS (SAMPLE)")
    print("="*60)
    
    # Check a few representative files
    sample_files = [
        "data/raw_balance_sheets/aggregation_2019-01-01_eng.xlsx",
        "data/raw_balance_sheets/aggregation_2021-06-01_eng.xlsx", 
        "data/raw_balance_sheets/aggregation_2025-01-01_eng.xlsx"
    ]
    
    for file_path in sample_files:
        if Path(file_path).exists():
            check_specific_file_details(file_path)
        else:
            print(f"❌ File not found: {file_path}") 