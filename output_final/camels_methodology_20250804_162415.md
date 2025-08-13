# CAMELS Ratios Methodology Documentation

**Generated:** 2025-08-04 16:24:17

## Overview

This document describes the methodology used to calculate 8 core CAMELS ratios for Ukrainian banks based on the IMF Financial Soundness Indicators Guide (2019).

## Reference

All ratios are implemented according to:
- IMF Financial Soundness Indicators Guide 2019
- Bank for International Settlements standards
- Ukrainian National Bank reporting requirements

## The 8 High-Impact CAMELS Ratios

### 1. Capital Ratio (Tier 1 to Assets / Leverage Ratio)

**CAMELS Pillar:** Capital

**Formula:** Equity ÷ Total Assets

**Interpretation:** First line of defence against losses; rises when a bank deleverages

**IMF Category:** Core FSI

**Healthy Range:** 8.0% - 20.0%

**Technical Implementation:**
- Numerator: `equity_total_equity_capital`
- Denominator: `assets_total_assets`

---

### 2. Non-performing Loans Ratio

**CAMELS Pillar:** Asset Quality

**Formula:** Non-performing Loans ÷ Gross Loans

**Interpretation:** Captures credit-loss pressure, especially acute in wartime economies

**IMF Category:** Core FSI

**Warning Threshold:** 5.0%

**Technical Implementation:**
- Numerator: Sum of:
  - `assets_including_provision_for_impairment_of_loans_and_receivables_from_legal_entities`
  - `assets_including_provision_for_impairment_of_loans_and_receivables_from_individuals`
- Denominator: `assets_loans_and_receivables_from_customers`

---

### 3. Cost-to-Income Ratio

**CAMELS Pillar:** Management (Efficiency)

**Formula:** Operating expenses ÷ Gross income

**Interpretation:** Shows how well management controls overheads when revenue dips

**IMF Category:** Core FSI

**Healthy Range:** 40.0% - 60.0%

**Technical Implementation:**
- Numerator: `financial_results_total_expenses`
- Denominator: `financial_results_total_income`

---

### 4. Return on Assets (ROA)

**CAMELS Pillar:** Earnings

**Formula:** Net profit ÷ Total assets

**Interpretation:** Headline profitability metric used by IMF, BIS and investors

**IMF Category:** Core FSI

**Healthy Range:** 0.5% - 2.0%

**Technical Implementation:**
- Numerator: `financial_results_profit_loss_after_tax`
- Denominator: `assets_total_assets`

---

### 5. Return on Equity (ROE)

**CAMELS Pillar:** Earnings

**Formula:** Net profit ÷ Total equity

**Interpretation:** Complements ROA by showing rewards to shareholders

**IMF Category:** Core FSI

**Healthy Range:** 10.0% - 15.0%

**Technical Implementation:**
- Numerator: `financial_results_profit_loss_after_tax`
- Denominator: `equity_total_equity_capital`

---

### 6. Liquid Assets Ratio

**CAMELS Pillar:** Liquidity

**Formula:** Liquid Assets ÷ Total Assets

**Interpretation:** Immediate buffer against deposit runs or payment shocks

**IMF Category:** Core FSI

**Healthy Range:** 15.0% - 30.0%

**Technical Implementation:**
- Numerator: `assets_cash_and_cash_equivalents`
- Denominator: `assets_total_assets`

---

### 7. Loan-to-Deposit Ratio (LDR)

**CAMELS Pillar:** Liquidity

**Formula:** Loans ÷ Customer deposits

**Interpretation:** Tests funding tightness; high values spell liquidity strain

**IMF Category:** Core FSI

**Warning Threshold:** 100.0%

**Technical Implementation:**
- Numerator: `assets_loans_and_receivables_from_customers`
- Denominator: `liabilities_amounts_due_to_customers`

---

### 8. Net Open FX Position ÷ Capital

**CAMELS Pillar:** Sensitivity to Market Risk

**Formula:** Net Open FX Position ÷ Capital

**Interpretation:** Flags vulnerability to hryvnia swings; war amplifies this channel

**IMF Category:** Core FSI

**Technical Implementation:**
- Numerator: `FX_Gap`
- Denominator: `equity_total_equity_capital`

---

## Additional Features

### War Period Indicator
- **WAR**: Binary indicator (1 for post-February 24, 2022)
- Captures the impact of Russian invasion on banking sector

### Temporal Features
- **ROA_lag1**: 1-month lagged Return on Assets
- **NPL_ratio_lag1**: 1-month lagged NPL ratio
- **Capital_ratio_lag1**: 1-month lagged Capital ratio

### War Interaction Terms
- **WAR_x_NPL_Ratio**: War period × NPL ratio (amplified credit risk)
- **WAR_x_Net_Open_FX_Ratio**: War period × FX exposure (amplified FX risk)

## Data Quality Controls

1. **Missing Values**: Excluded from ratio calculations
2. **Division by Zero**: Handled by setting result to NaN
3. **Outliers**: Winsorized at 1st and 99th percentiles
4. **Coverage**: Minimum 75% valid observations required
5. **Extreme Values**: Flagged when outside reasonable bounds

## Usage Notes

- All ratios are in decimal form (not percentages)
- Monthly frequency data (2019-01 to 2025-04)
- 67 Ukrainian banks covered
- Panel dataset structure maintained
