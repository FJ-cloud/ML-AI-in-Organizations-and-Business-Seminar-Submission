# Exploratory Data Analysis (EDA)

This folder contains exploratory data analysis scripts for the Ukrainian Banks Insolvency Predictor project.

## Scripts Overview

### 1. `failure_distribution_barchart.py`
Creates a bar chart showing the distribution of bank failures per month.

**Features:**
- Shows monthly failure counts across the entire dataset period
- Highlights peak failure months
- Includes summary statistics (total failures, peak month, etc.)
- Saves output as `bank_failures_by_month.png`

**Usage:**
```bash
cd eda
python3 failure_distribution_barchart.py
```

### 2. `bank_survival_chart.py`
Creates a comprehensive survival chart for all 67 banks showing their lifecycle.

**Features:**
- Entry point markers (when banks entered the market)
- Active period lines (duration of operation)
- Failure markers (for banks that failed)
- Color coding: green for active banks, red for failed banks
- Detailed survival statistics and analysis
- Saves output as `bank_survival_chart.png`

**Usage:**
```bash
cd eda
python3 bank_survival_chart.py
```

## Requirements

The scripts require the following Python packages:
- pandas
- matplotlib
- seaborn
- numpy

Install with:
```bash
pip install pandas matplotlib seaborn numpy
```

## Data Source

Both scripts read from: `../output_final/ukrainian_banks_panel_dataset_FINAL.csv`

Make sure you have run the ETL pipeline first to generate this dataset.

## Output Files

The scripts generate the following visualization files:
- `bank_failures_by_month.png` - Bar chart of failure distribution
- `bank_survival_chart.png` - Survival timeline chart

## Key Insights

The EDA scripts help answer questions like:
- When did most bank failures occur?
- How long do banks typically operate before failing?
- Which banks have been operating the longest?
- What's the overall survival rate in the Ukrainian banking sector?
- Are there patterns in market entry and exit timing? 