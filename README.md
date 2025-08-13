# Ukrainian Bank Insolvency Prediction 

An ETL pipeline for processing Ukrainian bank regulatory data from NBU Excel files into ML-ready panel datasets.

## 🎯 Overview

This pipeline processes 76 monthly Excel files (2019-2025) from the National Bank of Ukraine, extracting assets, liabilities, equity, and financial results data to create a multivariate panel dataset for insolvency prediction modeling.

## 🚀 Key Features

- **Comprehensive Data Extraction**: Processes all 4 data categories from 76 Excel files
- **Intelligent Header Detection**: Handles varying Excel file structures across years
- **Advanced Bank Name Canonicalization**: 200+ proven bank name mapping
- **Full Rectangular Panel**: Creates complete time series for all banks
- **Failure Detection**: Automatically identifies and flags failed banks

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas openpyxl rapidfuzz pyyaml
```

### Basic Usage

```bash
# Run complete ETL pipeline
python run_etl_pipeline.py --data-dir data/raw_balance_sheets

# Validation only (no processing)
python run_etl_pipeline.py --data-dir data/raw_balance_sheets --validate-only

# Custom output directory
python run_etl_pipeline.py --data-dir data/raw_balance_sheets --output-dir custom_output/
```



## 📋 Output Files

The pipeline generates comprehensive output in the specified directory:

### Data Quality

- **✅ Complete Coverage**: All 76 months processed successfully
- **✅ Balanced Panel**: All banks have entries for all time periods
- **✅ Clean Names**: No JSC/PJSC variants or problematic entries
- **✅ Failure Detection**: 25/67 banks properly flagged as failed
- **✅ USD Values**: Uses USD sheets (avoids National Currency variants)

## 🔧 Advanced Configuration

Create a YAML configuration file for customization:

```yaml
# config.yml
categories:
  assets:
    patterns:
      - "(?i)^assets?$"
      - "(?i)^total[ _]+assets?$"
  
bank_canonicalization:
  enabled: true
  
failure_detection:
  min_consecutive_missing: 6
  
output:
  include_backups: true
  include_metadata: true
```

## 🏦 Bank Name Canonicalization

The pipeline includes comprehensive bank name standardization:

- **200+ Proven Mappings**: Real-world bank name variations
- **JSC/PJSC Handling**: Removes corporate structure suffixes
- **Business Logic**: Handles mergers, acquisitions, and rebranding
- **Aggregation Removal**: Excludes summary/total rows

### Examples
```
"JSC 'ALFA-BANK'" → "Alfa-Bank"
"CB PrivatBank JSC" → "PrivatBank"  
"Raiffeisen Bank Aval" → "Raiffeisen Bank"
```

## 📈 Data Verification

Run comprehensive data verification:

```bash
python analysis_scripts/verify_data_completeness.py
```

This analyzes:
- Sheet coverage across all files
- Header detection accuracy
- Data extraction completeness
- Category coverage statistics

## 🔍 Technical Details

### Sheet Processing Logic
1. **Sheet Discovery**: Finds Assets, Liabilities, Equity, Financial Results sheets
2. **_NC Filtering**: Excludes National Currency sheets (uses USD data)
3. **Header Detection**: Scans rows 3-7 for valid headers containing 'Bank'
4. **Data Extraction**: Applies proven cleaning and validation logic

### Bank Name Processing
1. **Initial Canonicalization**: 200+ direct mappings
2. **JSC/PJSC Consolidation**: Merges corporate variants

### Panel Creation
2. **Data Type Separation**: Maintains clean column prefixes
3. **Failure Detection**: Conservative approach (6+ consecutive missing months)
4. **Metadata Addition**: Complete audit trail and timestamps

