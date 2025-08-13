# Data Dictionary — Model Ready (model_lag1_20250809_212523.csv)

| Name | Category | Source | Description |
|---|---|---|---|
| bank_name | id | panel | Canonical bank name after consolidation. |
| date_m | id | panel | Month end date (period end, monthly granularity). |
| year_month | id | panel | YYYY-MM string for monthly period. |
| failed | target | panel | Binary target: 1 if bank failed, else 0. |
| WAR | policy_flag | constructed | War dummy: 1 from 2022-03-01 onwards, else 0. |
| Capital_Ratio | camels | derived (balance sheet) | Equity to Total Assets ratio (capital adequacy proxy). |
| Cost_Income_Ratio | camels | derived (P&L) | Operating expenses to operating income. |
| Liquid_Assets_Ratio | camels | derived (liquidity) | Liquid assets to total assets. |
| Loan_Deposit_Ratio | camels | derived (funding) | Loans to customer deposits. |
| NPL_Ratio | camels | derived (loan book) | Non-performing loans to gross loans. |
| Net_Open_FX_Ratio | camels | derived (FX exposure) | Net open FX position to equity. |
| ROA | camels | derived (P&L / assets) | Return on Assets: net income / total assets. |
| ROE | camels | derived (P&L / equity) | Return on Equity: net income / equity. |
| 01_Cross-border loans from BIS reporting banks | macro | JEHD/IMF/BIS/WEO/market | Cross-border loans from BIS reporting banks (JEHD/BIS). |
| 12_Liabilities to BIS banks (cons.), short term | macro | JEHD/IMF/BIS/WEO/market | Short-term liabilities to BIS reporting banks, consolidated (JEHD/BIS). |
| 22_Liabilities to BIS banks, locational, total | macro | JEHD/IMF/BIS/WEO/market | Total locational liabilities to BIS banks (JEHD/BIS). |
| 24_International reserves (excluding gold) | macro | JEHD/IMF/BIS/WEO/market | International reserves excluding gold (JEHD). |
| 27_Cross-border deposits with BIS rep. banks | macro | JEHD/IMF/BIS/WEO/market | Cross-border deposits with BIS reporting banks (JEHD/BIS). |
| cpi_x | macro | JEHD/IMF/BIS/WEO/market | Consumer Price Index (variant X from merged macro sources). |
| cpi_y | macro | JEHD/IMF/BIS/WEO/market | Consumer Price Index (variant Y from merged macro sources). |
| exchange_rate | macro | JEHD/IMF/BIS/WEO/market | Exchange rate (local currency per USD) from merged macro sources. |
| gdp_yoy_pct | macro | JEHD/IMF/BIS/WEO/market | GDP year-over-year growth rate (WEO/IMF). |
| interest_rate | macro | JEHD/IMF/BIS/WEO/market | Policy or money market interest rate (from merged macro sources). |
| reserves_pct_gdp | macro | JEHD/IMF/BIS/WEO/market | International reserves as % of GDP (derived from WEO/IMF). |
| ukraine_3y_change_pct | macro | JEHD/IMF/BIS/WEO/market | Monthly percent change in Ukraine 3-year yield. |
| ukraine_3y_yield | macro | JEHD/IMF/BIS/WEO/market | Ukraine 3-year government bond yield (%). |
| ukraine_3y_yield_z | macro | JEHD/IMF/BIS/WEO/market | Z-scored Ukraine 3y yield (standardized). |
| us_3y_yield | macro | JEHD/IMF/BIS/WEO/market | United States Treasury 3-year yield (%). |
| us_3y_yield_z | macro | JEHD/IMF/BIS/WEO/market | Z-scored US 3y yield (standardized). |
| yield_spread_ukr_us_3y | macro | JEHD/IMF/BIS/WEO/market | Yield spread: Ukraine 3y minus US 3y (pp). |
| yield_spread_ukr_us_3y_z | macro | JEHD/IMF/BIS/WEO/market | Z-scored yield spread (standardized). |
| 01_Cross-border loans from BIS reporting banks_lag1 | macro_lag | constructed | Lag-1 month of 01_Cross-border loans from BIS reporting banks. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 01_Cross-border loans from BIS reporting banks_lag1.1 | macro_lag | constructed | Lag-1 month of 01_Cross-border loans from BIS reporting banks. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 12_Liabilities to BIS banks (cons.), short term_lag1 | macro_lag | constructed | Lag-1 month of 12_Liabilities to BIS banks (cons.), short term. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 12_Liabilities to BIS banks (cons.), short term_lag1.1 | macro_lag | constructed | Lag-1 month of 12_Liabilities to BIS banks (cons.), short term. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 22_Liabilities to BIS banks, locational, total_lag1 | macro_lag | constructed | Lag-1 month of 22_Liabilities to BIS banks, locational, total. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 22_Liabilities to BIS banks, locational, total_lag1.1 | macro_lag | constructed | Lag-1 month of 22_Liabilities to BIS banks, locational, total. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 24_International reserves (excluding gold)_lag1 | macro_lag | constructed | Lag-1 month of 24_International reserves (excluding gold). Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 24_International reserves (excluding gold)_lag1.1 | macro_lag | constructed | Lag-1 month of 24_International reserves (excluding gold). Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 27_Cross-border deposits with BIS rep. banks_lag1 | macro_lag | constructed | Lag-1 month of 27_Cross-border deposits with BIS rep. banks. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| 27_Cross-border deposits with BIS rep. banks_lag1.1 | macro_lag | constructed | Lag-1 month of 27_Cross-border deposits with BIS rep. banks. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| ukraine_3y_change_pct_lag1 | macro_lag | constructed | Lag-1 month of ukraine_3y_change_pct. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| ukraine_3y_change_pct_lag1.1 | macro_lag | constructed | Lag-1 month of ukraine_3y_change_pct. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| ukraine_3y_yield_lag1 | macro_lag | constructed | Lag-1 month of ukraine_3y_yield. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| ukraine_3y_yield_lag1.1 | macro_lag | constructed | Lag-1 month of ukraine_3y_yield. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| us_3y_yield_lag1 | macro_lag | constructed | Lag-1 month of us_3y_yield. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| us_3y_yield_lag1.1 | macro_lag | constructed | Lag-1 month of us_3y_yield. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| yield_spread_ukr_us_3y_lag1 | macro_lag | constructed | Lag-1 month of yield_spread_ukr_us_3y. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| yield_spread_ukr_us_3y_lag1.1 | macro_lag | constructed | Lag-1 month of yield_spread_ukr_us_3y. Duplicate suffix .1 indicates de-duplication needed in future cleanup. |
| failure_date | other | merged | Derived/feature column from merged macro/panel sources. |
| last_reporting_date | other | merged | Derived/feature column from merged macro/panel sources. |