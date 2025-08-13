# Ukrainian Bank Insolvency Prediction - ETL Pipeline v2.0.0

A comprehensive, production-ready ETL pipeline for processing Ukrainian bank regulatory data from NBU Excel files into ML-ready panel datasets.

## 🎯 Overview

This pipeline processes 76 monthly Excel files (2019-2025) from the National Bank of Ukraine, extracting assets, liabilities, equity, and financial results data to create a clean, multivariate panel dataset suitable for insolvency prediction modeling.

## 🚀 Key Features

- **Comprehensive Data Extraction**: Processes all 4 data categories from 76 Excel files
- **Intelligent Header Detection**: Handles varying Excel file structures across years
- **Advanced Bank Name Canonicalization**: 200+ proven bank name mappings with business logic
- **Full Rectangular Panel**: Creates complete time series for all banks
- **Failure Detection**: Automatically identifies and flags failed banks
- **Production Ready**: Robust error handling, logging, and validation

## 📊 Pipeline Results

- **📁 Input**: 76 Excel files (2019-01 to 2025-04)
- **🏦 Banks**: 71 clean, canonicalized bank names
- **📋 Output**: 5,396 observations × 118 columns
- **📈 Coverage**: 100% data extraction success
- **⚡ Performance**: ~40 seconds end-to-end processing

## 🏗️ Architecture

```
etl_pipeline/
├── core/
│   ├── orchestrator.py     # Main pipeline coordinator
│   ├── extractor.py        # Excel data extraction
│   ├── transformer.py      # Data cleaning & transformation
│   ├── loader.py          # Output generation
│   └── pipeline.py        # High-level interface
├── utils/
│   ├── bank_name_canonicalizer.py  # Bank name standardization
│   ├── config_manager.py           # Configuration management
│   └── logger.py                   # Logging system
└── validators/
    └── data_validator.py   # Data quality validation
```

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

### Command Line Options

```bash
python run_etl_pipeline.py --help

Options:
  --data-dir PATH           Directory containing Excel files [required]
  --output-dir PATH         Output directory [default: output_final]
  --config-file PATH        YAML configuration file [optional]
  --log-level LEVEL         Logging level [default: INFO]
  --validate-only           Only validate files, don't process
  --force                   Overwrite existing output files
```

## 📋 Output Files

The pipeline generates comprehensive output in the specified directory:

### Main Dataset
- `ukrainian_banks_panel_dataset_FINAL.csv` - Main panel dataset
- `ukrainian_banks_panel_dataset_YYYYMMDD_HHMMSS.csv` - Timestamped backup

### Metadata & Documentation
- `dataset_metadata_YYYYMMDD_HHMMSS.json` - Machine-readable metadata
- `dataset_metadata_YYYYMMDD_HHMMSS.txt` - Human-readable summary
- `pipeline_report_YYYYMMDD_HHMMSS.json` - Complete processing report

### Summary Statistics
- `bank_summary_YYYYMMDD_HHMMSS.csv` - Bank-level statistics
- `monthly_summary_YYYYMMDD_HHMMSS.csv` - Monthly data coverage

## 📊 Dataset Structure

### Columns (118 total)

**Metadata (8 columns)**:
- `bank_name` - Canonicalized bank name
- `date` - Month-end date (YYYY-MM-DD)
- `year_month` - Year-month (YYYY-MM)
- `bank_present` - Bank reported data (1/0)
- `failed` - Bank failure flag (1/0)
- `failure_date` - Date of failure (if applicable)
- `last_reporting_date` - Last date bank reported
- `processing_timestamp` - ETL processing timestamp

**Financial Data (110 columns)**:
- `assets_*` (37 columns) - Asset categories
- `liabilities_*` (20 columns) - Liability categories  
- `equity_*` (11 columns) - Equity categories
- `financial_results_*` (42 columns) - Income statement items

### Data Quality

- **✅ Complete Coverage**: All 76 months processed successfully
- **✅ Balanced Panel**: All banks have entries for all time periods
- **✅ Clean Names**: No JSC/PJSC variants or problematic entries
- **✅ Failure Detection**: 25/71 banks properly flagged as failed
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
3. **Business Logic**: Handles special cases (Raiffeisen, ALPARI, etc.)
4. **Final Cleanup**: Removes problematic entries

### Panel Creation
1. **Full Rectangular Structure**: 71 banks × 76 months = 5,396 rows
2. **Data Type Separation**: Maintains clean column prefixes
3. **Failure Detection**: Conservative approach (6+ consecutive missing months)
4. **Metadata Addition**: Complete audit trail and timestamps

## 📝 Logging

The pipeline provides comprehensive logging:

- **INFO**: Progress updates and statistics
- **WARNING**: Non-critical issues (missing sheets, etc.)
- **ERROR**: Critical failures requiring attention
- **DEBUG**: Detailed processing information

Logs include timestamps, processing statistics, and data quality metrics.

## 🤝 Contributing

This is a production-ready pipeline built for Ukrainian bank insolvency prediction research. The codebase incorporates lessons learned from extensive real-world data processing.

## 📄 License

This project is part of academic research on Ukrainian banking sector insolvency prediction.

---

**Ukrainian Bank ETL Pipeline v2.0.0** - Production-ready data processing for insolvency prediction modeling. 