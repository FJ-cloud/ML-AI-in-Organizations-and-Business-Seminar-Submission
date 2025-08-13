# Ukrainian Bank Insolvency Predictor - Project Documentation

## Project Overview
**Objective**: Develop a comprehensive machine learning system to predict Ukrainian bank insolvency and profitability using CAMELS ratios and macroeconomic indicators.

**Period Covered**: 2019-2025 (including war period)
**Banks Analyzed**: 67 Ukrainian banks
**Total Observations**: 5,092 bank-month observations

---

## 📊 Current Status (as of 2025-08-04)

### ✅ COMPLETED COMPONENTS

#### 1. **Data Infrastructure**
- **ETL Pipeline**: Complete data extraction, transformation, and loading system
- **Bank Name Canonicalization**: Standardized bank naming across all datasets
- **Data Validation**: Comprehensive validation and quality checks
- **Panel Dataset**: Structured monthly panel data (2019-2025)

#### 2. **Comprehensive CAMELS System** 
- **25 New CAMELS Ratios** calculated from raw balance sheet data:
  - Core ratios (8): Net interest income, commission income, provisions, equity ratios, ROA, ROE
  - Extended ratios (17): Interest margins, cost ratios, asset quality, liquidity measures
- **9 Original CAMELS**: ROA, Capital Ratio, NPL Ratio, Cost-Income Ratio, etc.
- **Total**: 34 comprehensive banking health indicators
- **Unified System**: All calculations consolidated in `unified_camels_system.py`

#### 3. **Macroeconomic Integration**
- **GDP Data**: Annual deflator and growth rates (2016-2025)
- **CPI Data**: Monthly inflation rates (2019-2025)  
- **Smart Integration**: Preserves bank failure data while handling missing macro data
- **Lag Structure**: Uses previous year GDP for prediction (addresses forecasting constraint)

#### 4. **Analysis Results**

**Key Finding**: **Macroeconomic variables have ZERO predictive power** for Ukrainian bank profitability
- GDP growth, inflation, economic conditions → No significant effect
- Internal bank management factors dominate (cost efficiency, past performance, size)
- War economy/crisis conditions break traditional macro-banking relationships

**Model Performance**:
- Enhanced Model: R² = 64.81% (using lagged variables + bank-specific factors)
- Financial Baseline: R² = 38.71% (traditional CAMELS)
- Macro-Only Model: R² = 0.00% (complete failure)

---

## 📁 Current File Structure

### **Core Datasets**
```
output_final/
├── unified_camels_dataset_20250804_225021.csv          # ⭐ MAIN UNIFIED DATASET (34 ratios)
├── unified_camels_report_20250804_225021.json          # Comprehensive analysis report
├── unified_camels_definitions_20250804_225021.txt      # All ratio definitions & statistics
├── ukrainian_banks_panel_dataset_FINAL.csv             # Original raw panel data
└── profitability_macro_results_20250804_202302.json    # Latest profitability analysis
```

### **Analysis Scripts**
```
├── unified_camels_system.py                            # ⭐ UNIFIED CAMELS SYSTEM (34 ratios)
├── macro_profitability_panel_regression.py             # Profitability prediction models
├── cleanup_workspace.py                                # Workspace organization
├── run_etl_pipeline.py                                 # Data extraction pipeline
└── camels/camels_main.py                               # Legacy CAMELS (archived)
```

### **Data Sources**
```
data/
├── raw_balance_sheets/              # Monthly bank balance sheets (2019-2025)
├── MacroEconomicData/              # GDP and CPI data from IMF
└── interim/                        # Processed intermediate data
```

---

## 🎯 TOMORROW'S AGENDA

### **Phase 1: Enhanced CAMELS Integration** 
**Priority: HIGH**

#### A. Additional CAMELS Ratios to Implement
Based on banking literature, add these critical ratios:

**Asset Quality Indicators**:
- Non-performing loans / Total loans
- Loan loss provisions / Non-performing loans  
- Net charge-offs / Average loans
- Problem assets / Total assets

**Liquidity & Funding**:
- Liquid assets / Short-term liabilities
- Deposits / Total funding
- Wholesale funding dependency
- Cash ratio (cash / deposits)

**Efficiency & Management**:
- Operating expenses / Operating income
- Personnel expenses / Total expenses
- Non-interest income / Total income
- Productivity ratios (assets per employee, if available)

**Earnings Quality**:
- Interest spread (lending rate - deposit rate)
- Net interest margin trends
- Fee income stability
- Trading income volatility

#### B. Technical Implementation
- Extend `comprehensive_camels_calculator.py` with new ratios
- Add statistical testing for ratio stability over time
- Create ratio correlation analysis
- Implement outlier detection and treatment

### **Phase 2: Expanded Macroeconomic Analysis**
**Priority: MEDIUM**

#### A. Additional Macro Variables
Since traditional GDP/CPI failed, test alternative indicators:

**Financial Market Indicators**:
- Government bond yields
- Exchange rate volatility (UAH/USD, UAH/EUR)
- Stock market indices (if available)
- Credit default swap spreads

**Banking Sector Indicators**:
- Interbank lending rates
- Central bank policy rates
- Reserve requirements
- Banking sector concentration

**Crisis/War Indicators**:
- Conflict intensity indices
- Regional economic activity
- Infrastructure damage indicators
- Refugee/displacement impacts

**International Factors**:
- Oil prices (Ukraine is energy-dependent)
- Commodity prices (agricultural exports)
- EU/US sanctions impact measures
- International aid flows

#### B. Advanced Modeling
- Time-varying coefficient models
- Structural break analysis (pre/during/post war)
- Regional analysis (if bank location data available)
- Cross-border spillover effects

### **Phase 3: Model Enhancement & Validation**
**Priority: HIGH**






---

## 🔬 RESEARCH QUESTIONS FOR TOMORROW

### **Primary Questions**:
1. **Why do macro variables have zero effect?** Is this unique to Ukraine or common in crisis periods?
2. **What drives the 64.81% R² in the enhanced model?** Which specific factors matter most?
3. **Can we identify leading indicators** of bank distress beyond lagged performance?

### **Secondary Questions**:
4. **Regional variation**: Do banks in different regions respond differently?
5. **Ownership effects**: State vs private vs foreign-owned bank differences?
6. **Size effects**: Do large vs small banks have different risk profiles?
7. **War impact**: Clear structural breaks around Feb 2022?

---

## 🛠️ TECHNICAL PRIORITIES

### **Code Quality & Documentation**
1. **Refactor** main analysis scripts for clarity
2. **Add comprehensive docstrings** and type hints
3. **Create unit tests** for key functions
4. **Implement logging** throughout the pipeline

### **Performance Optimization**
1. **Optimize data loading** (use Parquet instead of CSV)
2. **Parallel processing** for model estimation
3. **Memory management** for large datasets
4. **Caching** for expensive calculations

### **Reproducibility**
1. **Configuration files** for all parameters
2. **Random seed management** 
3. **Environment management** (requirements.txt)
4. **Version control** for all datasets

---





---

