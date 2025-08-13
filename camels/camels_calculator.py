#!/usr/bin/env python3
"""
CAMELS Calculator Module

This module implements the calculation of all 8 core CAMELS ratios for Ukrainian banks
based on the IMF Financial Soundness Indicators Guide and the mapping configuration.

Author: Ukrainian Insolvency Predictor Project
Date: 2025-01-04
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

class CAMELSCalculator:
    """
    Calculator for CAMELS ratios based on Ukrainian banks balance sheet data.
    
    Implements the 8 high-impact ratios:
    1. Capital Ratio (Capital Adequacy)
    2. NPL Ratio (Asset Quality)
    3. Cost-Income Ratio (Management Efficiency)
    4. ROA (Earnings)
    5. ROE (Earnings)
    6. Liquid Assets Ratio (Liquidity)
    7. Loan-to-Deposit Ratio (Liquidity)
    8. Net Open FX Ratio (Market Risk Sensitivity)
    """
    
    def __init__(self, mapping_file: str = 'camels_mapping.yml'):
        """Initialize calculator with mapping configuration."""
        self.mapping_file = Path(__file__).parent / mapping_file
        self.config = self._load_config()
        self.ratios = self.config['ratios']
        self.implementation = self.config['implementation']
        
    def _load_config(self) -> Dict:
        """Load CAMELS mapping configuration."""
        try:
            with open(self.mapping_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"CAMELS mapping file not found: {self.mapping_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing CAMELS mapping file: {e}")
    
    def calculate_capital_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Capital Ratio (Tier 1 to Assets / Leverage Ratio)."""
        config = self.ratios['capital_ratio']
        numerator = df[config['numerator']]
        denominator = df[config['denominator']]
        
        # Handle division by zero and missing values
        valid_mask = (pd.notna(numerator) & pd.notna(denominator) & (denominator != 0))
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return result
    
    def calculate_npl_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Non-performing Loans Ratio."""
        config = self.ratios['npl_ratio']
        
        # Sum provisions for legal entities and individuals
        numerator_components = config['numerator_components']
        numerator = df[numerator_components].sum(axis=1)
        denominator = df[config['denominator']]
        
        # Handle division by zero and missing values
        valid_mask = (pd.notna(numerator) & pd.notna(denominator) & (denominator != 0))
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return result
    
    def calculate_cost_income_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Cost-to-Income Ratio."""
        config = self.ratios['cost_income_ratio']
        numerator = df[config['numerator']]
        denominator = df[config['denominator']]
        
        # Handle division by zero and missing values
        valid_mask = (pd.notna(numerator) & pd.notna(denominator) & (denominator != 0))
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return result
    
    def get_roa(self, df: pd.DataFrame) -> pd.Series:
        """Get ROA (already calculated in dataset)."""
        if 'ROA' in df.columns:
            return df['ROA']
        else:
            # Fallback calculation if ROA column doesn't exist
            config = self.ratios['roa']
            numerator = df[config['numerator']]
            denominator = df[config['denominator']]
            
            valid_mask = (pd.notna(numerator) & pd.notna(denominator) & (denominator != 0))
            result = pd.Series(np.nan, index=df.index)
            result.loc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            
            return result
    
    def calculate_roe(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Return on Equity (ROE)."""
        config = self.ratios['roe']
        numerator = df[config['numerator']]
        denominator = df[config['denominator']]
        
        # Handle division by zero and missing values
        valid_mask = (pd.notna(numerator) & pd.notna(denominator) & (denominator != 0))
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return result
    
    def calculate_liquid_assets_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Liquid Assets Ratio."""
        config = self.ratios['liquid_assets_ratio']
        numerator = df[config['numerator']]
        denominator = df[config['denominator']]
        
        # Handle division by zero and missing values
        valid_mask = (pd.notna(numerator) & pd.notna(denominator) & (denominator != 0))
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return result
    
    def calculate_loan_deposit_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Loan-to-Deposit Ratio (LDR)."""
        config = self.ratios['loan_deposit_ratio']
        numerator = df[config['numerator']]
        denominator = df[config['denominator']]
        
        # Handle division by zero and missing values
        valid_mask = (pd.notna(numerator) & pd.notna(denominator) & (denominator != 0))
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return result
    
    def calculate_fx_gap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate FX Gap (Net Open FX Position).
        
        This is a simplified approximation since detailed FX position data may not be available.
        We'll use available foreign currency related columns as a proxy.
        """
        # TODO: Implement proper FX gap calculation once we identify relevant columns
        # For now, return a placeholder series with zeros
        warnings.warn("FX Gap calculation not fully implemented - using placeholder zeros")
        return pd.Series(0.0, index=df.index)
    
    def calculate_net_open_fx_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Net Open FX Position ÷ Capital."""
        config = self.ratios['net_open_fx_ratio']
        
        # Calculate FX gap (numerator)
        fx_gap = self.calculate_fx_gap(df)
        denominator = df[config['denominator']]
        
        # Handle division by zero and missing values
        valid_mask = (pd.notna(fx_gap) & pd.notna(denominator) & (denominator != 0))
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = fx_gap[valid_mask] / denominator[valid_mask]
        
        return result
    
    def calculate_war_dummy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate WAR dummy variable (1 for post-invasion period)."""
        war_config = self.implementation['war_period']
        invasion_date = pd.to_datetime(war_config['start_date'])
        
        # Convert date column to datetime if it's not already
        dates = pd.to_datetime(df['date'])
        war_dummy = (dates >= invasion_date).astype(int)
        
        return war_dummy
    
    def calculate_all_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all CAMELS ratios and add them to the dataframe."""
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        print("Calculating CAMELS ratios...")
        
        # 1. Capital Ratio
        print("  - Capital Ratio")
        result_df['Capital_Ratio'] = self.calculate_capital_ratio(df)
        
        # 2. NPL Ratio  
        print("  - NPL Ratio")
        result_df['NPL_Ratio'] = self.calculate_npl_ratio(df)
        
        # 3. Cost-Income Ratio
        print("  - Cost-Income Ratio")
        result_df['Cost_Income_Ratio'] = self.calculate_cost_income_ratio(df)
        
        # 4. ROA (already exists or calculate)
        print("  - ROA")
        result_df['ROA'] = self.get_roa(df)
        
        # 5. ROE
        print("  - ROE")
        result_df['ROE'] = self.calculate_roe(df)
        
        # 6. Liquid Assets Ratio
        print("  - Liquid Assets Ratio")
        result_df['Liquid_Assets_Ratio'] = self.calculate_liquid_assets_ratio(df)
        
        # 7. Loan-to-Deposit Ratio
        print("  - Loan-to-Deposit Ratio")
        result_df['Loan_Deposit_Ratio'] = self.calculate_loan_deposit_ratio(df)
        
        # 8. Net Open FX Ratio
        print("  - Net Open FX Ratio")
        result_df['Net_Open_FX_Ratio'] = self.calculate_net_open_fx_ratio(df)
        
        # Add WAR dummy
        print("  - WAR dummy variable")
        result_df['WAR'] = self.calculate_war_dummy(df)
        
        return result_df
    
    def add_lagged_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged variables for key ratios."""
        
        lag_config = self.implementation['temporal_features']
        lag_periods = lag_config['periods']
        
        result_df = df.copy()
        
        print(f"Adding lagged variables (lag={lag_periods})...")
        
        # Sort by bank and date for proper lagging
        result_df = result_df.sort_values(['bank_name', 'date'])
        
        for lag_var in lag_config['lags']:
            base_var = lag_var.replace('_lag1', '')  # Remove suffix to get base variable name
            
            if base_var == 'Capital_ratio':
                base_var = 'Capital_Ratio'  # Match actual column name
            
            if base_var in result_df.columns:
                print(f"  - {lag_var}")
                result_df[lag_var] = result_df.groupby('bank_name')[base_var].shift(lag_periods)
            else:
                warnings.warn(f"Base variable {base_var} not found for lagged variable {lag_var}")
        
        return result_df
    
    def add_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add war interaction terms."""
        
        interaction_config = self.implementation['interaction_terms']['war_interactions']
        result_df = df.copy()
        
        print("Adding war interaction terms...")
        
        for interaction in interaction_config:
            formula = interaction['formula']
            description = interaction['description']
            
            if formula == "WAR × NPL_ratio":
                if 'WAR' in result_df.columns and 'NPL_Ratio' in result_df.columns:
                    result_df['WAR_x_NPL_Ratio'] = result_df['WAR'] * result_df['NPL_Ratio']
                    print(f"  - WAR_x_NPL_Ratio: {description}")
                    
            elif formula == "WAR × Net_Open_FX_ratio":
                if 'WAR' in result_df.columns and 'Net_Open_FX_Ratio' in result_df.columns:
                    result_df['WAR_x_Net_Open_FX_Ratio'] = result_df['WAR'] * result_df['Net_Open_FX_Ratio']
                    print(f"  - WAR_x_Net_Open_FX_Ratio: {description}")
        
        return result_df
    
    def apply_winsorization(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply winsorization to handle outliers."""
        
        outlier_config = self.implementation['outlier_handling']
        if outlier_config['method'] != 'winsorize':
            return df
            
        percentiles = outlier_config['percentiles']
        lower_p, upper_p = percentiles[0], percentiles[1]
        
        result_df = df.copy()
        
        print(f"Applying winsorization ({lower_p*100}%-{upper_p*100}%) to CAMELS ratios...")
        
        for col in columns:
            if col in result_df.columns:
                lower_bound = result_df[col].quantile(lower_p)
                upper_bound = result_df[col].quantile(upper_p)
                
                # Count outliers before winsorization
                outliers_low = (result_df[col] < lower_bound).sum()
                outliers_high = (result_df[col] > upper_bound).sum()
                
                if outliers_low > 0 or outliers_high > 0:
                    print(f"  - {col}: {outliers_low} low outliers, {outliers_high} high outliers")
                    result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return result_df
    
    def get_camels_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for all CAMELS ratios."""
        
        camels_columns = [
            'Capital_Ratio', 'NPL_Ratio', 'Cost_Income_Ratio', 'ROA', 'ROE',
            'Liquid_Assets_Ratio', 'Loan_Deposit_Ratio', 'Net_Open_FX_Ratio'
        ]
        
        available_columns = [col for col in camels_columns if col in df.columns]
        
        if not available_columns:
            print("No CAMELS ratio columns found for summary statistics.")
            return pd.DataFrame()
        
        summary = df[available_columns].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        
        # Add additional statistics
        summary.loc['missing_count'] = df[available_columns].isna().sum()
        summary.loc['missing_pct'] = (df[available_columns].isna().sum() / len(df)) * 100
        
        return summary
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality for CAMELS calculations."""
        
        quality_config = self.config['quality_checks']
        required_coverage = quality_config['required_coverage']
        
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': []
        }
        
        # Check coverage for each ratio
        camels_columns = [
            'Capital_Ratio', 'NPL_Ratio', 'Cost_Income_Ratio', 'ROA', 'ROE',
            'Liquid_Assets_Ratio', 'Loan_Deposit_Ratio', 'Net_Open_FX_Ratio'
        ]
        
        for col in camels_columns:
            if col in df.columns:
                coverage = df[col].notna().sum() / len(df)
                if coverage < required_coverage:
                    validation_results['warnings'].append(
                        f"{col}: Coverage {coverage:.1%} below required {required_coverage:.1%}"
                    )
        
        # Check for extreme values
        extreme_bounds = quality_config['extreme_value_flags']
        
        for ratio, bounds in extreme_bounds.items():
            col_name = ratio.replace('_bounds', '').replace('_', '_').title().replace(' ', '_')
            # Map to actual column names
            if ratio == 'capital_ratio_bounds':
                col_name = 'Capital_Ratio'
            elif ratio == 'roa_bounds':
                col_name = 'ROA'
            elif ratio == 'roe_bounds':
                col_name = 'ROE'
                
            if col_name in df.columns:
                extreme_low = (df[col_name] < bounds[0]).sum()
                extreme_high = (df[col_name] > bounds[1]).sum()
                
                if extreme_low > 0 or extreme_high > 0:
                    validation_results['warnings'].append(
                        f"{col_name}: {extreme_low} values below {bounds[0]}, {extreme_high} values above {bounds[1]}"
                    )
        
        return validation_results 