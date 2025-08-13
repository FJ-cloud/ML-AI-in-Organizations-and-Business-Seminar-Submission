#!/usr/bin/env python3
"""
Macro-Enhanced Bank Profitability Prediction for Ukrainian Banks
Focus: Predict ROA/profitability of surviving banks using macro + financial variables
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from linearmodels import PanelOLS
from scipy import stats

warnings.filterwarnings('ignore')

class MacroProfitabilityAnalysis:
    def __init__(self, data_dir='/home/fj/UkrainianInsolvencyPredictor'):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / 'output_final'
        self.macro_dir = self.data_dir / 'data' / 'MacroEconomicData'
        
        # File paths
        self.camels_file = self.output_dir / 'ukrainian_banks_panel_dataset_FINAL.csv'
        self.gdp_file = self.macro_dir / 'dataset_2025-08-04T14_41_20.658129229Z_DEFAULT_INTEGRATION_IMF.RES_WEO_6.0.0.csv'
        self.cpi_file = self.macro_dir / 'dataset_2025-08-04T14_46_21.991284889Z_DEFAULT_INTEGRATION_IMF.STA_CPI_4.0.0.csv'
        
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_and_process_macro_data(self):
        """Load and process both GDP and CPI data"""
        print("Processing macroeconomic data...")
        
        # GDP data
        gdp_df = pd.read_csv(self.gdp_file)
        year_cols = [col for col in gdp_df.columns if col.isdigit()]
        gdp_values = gdp_df[year_cols].iloc[0].to_dict()
        
        gdp_long = []
        for year_str, gdp_value in gdp_values.items():
            year = int(year_str)
            if year >= 2017:
                prev_year = year - 1
                if str(prev_year) in gdp_values:
                    gdp_long.append({
                        'year': year,
                        'gdp_deflator': float(gdp_value),
                        'gdp_deflator_lag1': float(gdp_values[str(prev_year)]),
                        'gdp_growth_rate': (float(gdp_value) - float(gdp_values[str(prev_year)])) / float(gdp_values[str(prev_year)]) * 100
                    })
        
        gdp_processed = pd.DataFrame(gdp_long)
        
        # CPI data  
        cpi_df = pd.read_csv(self.cpi_file)
        month_cols = [col for col in cpi_df.columns if '-M' in col]
        cpi_values = cpi_df[month_cols].iloc[0].to_dict()
        
        cpi_long = []
        for month_str, cpi_value in cpi_values.items():
            if pd.notna(cpi_value) and cpi_value != '':
                year, month = month_str.split('-M')
                year_month = f"{year}-{month.zfill(2)}"
                
                cpi_long.append({
                    'year_month': year_month,
                    'year': int(year),
                    'month': int(month),
                    'cpi_change_rate': float(cpi_value)
                })
        
        cpi_processed = pd.DataFrame(cpi_long)
        
        print(f"GDP data: {len(gdp_processed)} years")
        print(f"CPI data: {len(cpi_processed)} months")
        
        return gdp_processed, cpi_processed
    
    def load_surviving_banks_data(self):
        """Load CAMELS data and filter to surviving banks only"""
        print("\nLoading surviving banks data...")
        
        # Load full dataset
        camels_df = pd.read_csv(self.camels_file, low_memory=False)
        print(f"Full CAMELS data: {camels_df.shape}")
        
        # Filter to surviving banks only (failed=0)
        surviving_df = camels_df[camels_df['failed'] == 0].copy()
        print(f"Surviving banks data: {surviving_df.shape}")
        print(f"Excluded {len(camels_df) - len(surviving_df)} failed bank observations")
        
        # Convert date and add time variables
        surviving_df['date'] = pd.to_datetime(surviving_df['date'])
        surviving_df['year'] = surviving_df['date'].dt.year
        surviving_df['quarter'] = surviving_df['date'].dt.quarter
        
        # Select key variables for profitability analysis
        profitability_vars = [
            'bank_name', 'date', 'year_month', 'year', 'quarter',
            # Profitability measures (dependent variables)
            'ROA', 'ROE', 'Net_Interest_Margin',
            # Financial control variables
            'Capital_Ratio', 'NPL_Ratio', 'Cost_Income_Ratio', 
            'Liquid_Assets_Ratio', 'Loan_Deposit_Ratio', 'WAR',
            # Size measure
            'Total_Assets_Log'
        ]
        
        # Add any available lagged variables
        lag_vars = [col for col in surviving_df.columns if col.endswith('_lag1')]
        profitability_vars.extend(lag_vars)
        
        # Create log of total assets if not available
        if 'Total_Assets_Log' not in surviving_df.columns and 'assets_total_assets' in surviving_df.columns:
            surviving_df['Total_Assets_Log'] = np.log(surviving_df['assets_total_assets'].replace(0, np.nan))
        
        # Filter to available columns
        available_vars = [var for var in profitability_vars if var in surviving_df.columns]
        surviving_subset = surviving_df[available_vars].copy()
        
        print(f"Selected variables for analysis: {len(available_vars)}")
        print(f"Available profitability measures: {[v for v in ['ROA', 'ROE', 'Net_Interest_Margin'] if v in available_vars]}")
        
        return surviving_subset
    
    def integrate_data_for_profitability(self, banks_df, gdp_df, cpi_df):
        """Integrate macro data with bank profitability data"""
        print("\nIntegrating macro data with bank profitability data...")
        
        # Merge with GDP data (by year)
        integrated_df = banks_df.merge(
            gdp_df[['year', 'gdp_deflator_lag1', 'gdp_growth_rate']], 
            on='year', 
            how='left'
        )
        print(f"After GDP merge: {integrated_df.shape}")
        
        # Merge with CPI data (by year_month)
        integrated_df = integrated_df.merge(
            cpi_df[['year_month', 'cpi_change_rate']], 
            on='year_month', 
            how='left'
        )
        print(f"After CPI merge: {integrated_df.shape}")
        
        # Handle missing macro data
        integrated_df['gdp_deflator_lag1'] = integrated_df.groupby('year')['gdp_deflator_lag1'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        integrated_df['gdp_growth_rate'] = integrated_df.groupby('year')['gdp_growth_rate'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        integrated_df['cpi_change_rate'] = integrated_df['cpi_change_rate'].fillna(method='ffill').fillna(method='bfill')
        
        # Check data coverage
        gdp_coverage = integrated_df['gdp_deflator_lag1'].notna().mean()
        cpi_coverage = integrated_df['cpi_change_rate'].notna().mean()
        
        print(f"GDP data coverage: {gdp_coverage:.2%}")
        print(f"CPI data coverage: {cpi_coverage:.2%}")
        print(f"Final integrated dataset: {integrated_df.shape}")
        
        return integrated_df
    
    def prepare_profitability_models(self, df):
        """Prepare data for profitability prediction models"""
        print("\nPreparing profitability prediction models...")
        
        # Focus on ROA as main profitability measure
        if 'ROA' not in df.columns:
            raise ValueError("ROA not available for profitability analysis")
        
        # Remove observations with missing ROA
        model_df = df.dropna(subset=['ROA']).copy()
        print(f"Observations with ROA data: {len(model_df)}")
        
        # Set panel index
        model_df = model_df.set_index(['bank_name', 'date'])
        
        # Define variable groups
        financial_vars = ['Capital_Ratio', 'NPL_Ratio', 'Cost_Income_Ratio', 
                         'Liquid_Assets_Ratio', 'Loan_Deposit_Ratio', 'WAR']
        
        macro_vars = ['gdp_deflator_lag1', 'gdp_growth_rate', 'cpi_change_rate']
        
        # Size and lagged variables
        size_vars = [col for col in model_df.columns if 'asset' in col.lower() and 'log' in col.lower()]
        lag_vars = [col for col in model_df.columns if col.endswith('_lag1') and col != 'gdp_deflator_lag1']
        
        # Filter to usable variables (sufficient data, non-constant)
        financial_vars = self.filter_usable_variables(model_df, financial_vars)
        macro_vars = self.filter_usable_variables(model_df, macro_vars)  
        size_vars = self.filter_usable_variables(model_df, size_vars)
        lag_vars = self.filter_usable_variables(model_df, lag_vars)
        
        print(f"Usable financial variables: {financial_vars}")
        print(f"Usable macro variables: {macro_vars}")
        print(f"Usable size variables: {size_vars}")
        print(f"Usable lagged variables: {lag_vars}")
        
        # Check ROA distribution
        roa_stats = model_df['ROA'].describe()
        print(f"\nROA statistics:")
        print(roa_stats)
        
        return model_df, financial_vars, macro_vars, size_vars, lag_vars
    
    def filter_usable_variables(self, df, variables):
        """Filter variables to only include usable ones"""
        usable = []
        for var in variables:
            if var in df.columns:
                # Check for sufficient non-missing values (at least 50%)
                non_missing_pct = df[var].notna().mean()
                if non_missing_pct < 0.5:
                    continue
                
                # Check for variance (not constant)
                var_std = df[var].std()
                if var_std == 0 or pd.isna(var_std):
                    continue
                
                # Check for extreme values
                if np.isinf(df[var]).any():
                    continue
                
                usable.append(var)
        
        return usable
    
    def run_profitability_regressions(self, df, financial_vars, macro_vars, size_vars, lag_vars):
        """Run panel regression models for profitability prediction"""
        print("\nRunning profitability prediction models...")
        
        models = {}
        
        # Model 1: Financial variables only (baseline)
        if len(financial_vars) >= 1:
            print(f"\n1. Financial Baseline Model ({len(financial_vars)} variables)...")
            try:
                model_data = df[financial_vars + ['ROA']].dropna()
                print(f"  Observations: {len(model_data)}")
                print(f"  Banks: {model_data.index.get_level_values(0).nunique()}")
                
                if len(model_data) > 100:  # Sufficient data
                    model1 = PanelOLS(
                        model_data['ROA'], 
                        model_data[financial_vars], 
                        entity_effects=True,
                        time_effects=True,
                        drop_absorbed=True
                    )
                    result1 = model1.fit(cov_type='clustered', cluster_entity=True)
                    models['financial_baseline'] = result1
                    print("✓ Financial baseline completed")
                    print(f"  R²: {result1.rsquared:.4f}")
                else:
                    print("✗ Insufficient observations")
            except Exception as e:
                print(f"✗ Financial baseline failed: {e}")
        
        # Model 2: Macro variables only
        if len(macro_vars) >= 1:
            print(f"\n2. Macro-Only Model ({len(macro_vars)} variables)...")
            try:
                model_data = df[macro_vars + ['ROA']].dropna()
                print(f"  Observations: {len(model_data)}")
                print(f"  Banks: {model_data.index.get_level_values(0).nunique()}")
                
                if len(model_data) > 100:
                    model2 = PanelOLS(
                        model_data['ROA'], 
                        model_data[macro_vars], 
                        entity_effects=True,
                        time_effects=True,
                        drop_absorbed=True
                    )
                    result2 = model2.fit(cov_type='clustered', cluster_entity=True)
                    models['macro_only'] = result2
                    print("✓ Macro-only model completed")
                    print(f"  R²: {result2.rsquared:.4f}")
                else:
                    print("✗ Insufficient observations")
            except Exception as e:
                print(f"✗ Macro-only model failed: {e}")
        
        # Model 3: Financial + Macro (full model)  
        combined_vars = financial_vars + macro_vars
        if len(combined_vars) >= 2:
            print(f"\n3. Financial + Macro Model ({len(combined_vars)} variables)...")
            try:
                model_data = df[combined_vars + ['ROA']].dropna()
                print(f"  Observations: {len(model_data)}")
                print(f"  Banks: {model_data.index.get_level_values(0).nunique()}")
                
                if len(model_data) > 100:
                    model3 = PanelOLS(
                        model_data['ROA'], 
                        model_data[combined_vars], 
                        entity_effects=True,
                        time_effects=True,
                        drop_absorbed=True
                    )
                    result3 = model3.fit(cov_type='clustered', cluster_entity=True)
                    models['financial_macro'] = result3
                    print("✓ Financial + Macro model completed")
                    print(f"  R²: {result3.rsquared:.4f}")
                else:
                    print("✗ Insufficient observations")
            except Exception as e:
                print(f"✗ Financial + Macro model failed: {e}")
        
        # Model 4: Enhanced with lagged variables (if available)
        if len(lag_vars) > 0:
            enhanced_vars = combined_vars + lag_vars + size_vars
            enhanced_vars = list(set(enhanced_vars))  # Remove duplicates
            
            print(f"\n4. Enhanced Model ({len(enhanced_vars)} variables)...")
            try:
                model_data = df[enhanced_vars + ['ROA']].dropna()
                print(f"  Observations: {len(model_data)}")
                print(f"  Banks: {model_data.index.get_level_values(0).nunique()}")
                
                if len(model_data) > 100:
                    model4 = PanelOLS(
                        model_data['ROA'], 
                        model_data[enhanced_vars], 
                        entity_effects=True,
                        time_effects=True,  
                        drop_absorbed=True
                    )
                    result4 = model4.fit(cov_type='clustered', cluster_entity=True)
                    models['enhanced'] = result4
                    print("✓ Enhanced model completed")
                    print(f"  R²: {result4.rsquared:.4f}")
                else:
                    print("✗ Insufficient observations")
            except Exception as e:
                print(f"✗ Enhanced model failed: {e}")
        
        return models
    
    def analyze_profitability_results(self, models):
        """Analyze profitability prediction results"""
        print("\n" + "="*60)
        print("PROFITABILITY PREDICTION RESULTS")
        print("="*60)
        
        results_summary = {}
        
        for name, model in models.items():
            print(f"\n{name.upper().replace('_', ' ')} MODEL:")
            print("-" * 40)
            print(f"R²: {model.rsquared:.4f}")
            print(f"R² (within): {model.rsquared_within:.4f}")
            print(f"R² (between): {model.rsquared_between:.4f}")
            print(f"F-statistic: {model.f_statistic.stat:.2f}")
            print(f"F p-value: {model.f_statistic.pval:.6f}")
            print(f"Observations: {model.nobs}")
            print(f"Banks: {model.entity_info.total}")
            
            # Store results
            results_summary[name] = {
                'r_squared': float(model.rsquared),
                'r_squared_within': float(model.rsquared_within),
                'r_squared_between': float(model.rsquared_between),
                'f_statistic': float(model.f_statistic.stat),
                'f_pvalue': float(model.f_statistic.pval),
                'nobs': int(model.nobs),
                'n_banks': int(model.entity_info.total),
                'coefficients': model.params.to_dict(),
                'pvalues': model.pvalues.to_dict(),
                'std_errors': model.std_errors.to_dict()
            }
            
            # Show significant coefficients
            significant_vars = model.pvalues[model.pvalues < 0.05]
            if len(significant_vars) > 0:
                print(f"\nSignificant predictors (p < 0.05):")
                for var in significant_vars.index:
                    coef = model.params[var]
                    pval = model.pvalues[var]
                    direction = "↑" if coef > 0 else "↓"
                    print(f"  {var}: {coef:.6f} {direction} (p={pval:.6f})")
            
            # Economic interpretation for macro variables
            macro_interpretation = {
                'gdp_deflator_lag1': 'GDP deflator (previous year)',
                'gdp_growth_rate': 'GDP growth rate',
                'cpi_change_rate': 'CPI inflation rate'
            }
            
            macro_effects = []
            for var, interpretation in macro_interpretation.items():
                if var in model.params and var in significant_vars.index:
                    coef = model.params[var]
                    effect = "increases" if coef > 0 else "decreases"
                    macro_effects.append(f"{interpretation} {effect} bank profitability")
            
            if macro_effects:
                print(f"\nMacroeconomic effects:")
                for effect in macro_effects:
                    print(f"  • {effect}")
        
        return results_summary
    
    def create_profitability_visualizations(self, df, models, results_summary):
        """Create visualizations for profitability analysis"""
        print("\nCreating profitability visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bank Profitability vs Macroeconomic Conditions', fontsize=16, fontweight='bold')
        
        # Plot 1: Model Performance Comparison
        ax1 = axes[0, 0]
        model_names = list(results_summary.keys())
        r_squared_values = [results_summary[name]['r_squared'] for name in model_names]
        
        bars = ax1.bar([name.replace('_', '\n').title() for name in model_names], r_squared_values, color='skyblue', alpha=0.7)
        ax1.set_title('Model Performance (R²)', fontweight='bold')
        ax1.set_ylabel('R-squared')
        ax1.set_ylim(0, max(r_squared_values) * 1.1 if r_squared_values else 1)
        
        for bar, value in zip(bars, r_squared_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: ROA vs GDP Growth Over Time
        ax2 = axes[0, 1]
        df_temp = df.reset_index()
        df_temp['year'] = df_temp['date'].dt.year
        
        # Monthly averages
        monthly_data = df_temp.groupby('date').agg({
            'ROA': 'mean',
            'gdp_growth_rate': 'first'
        }).reset_index()
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(monthly_data['date'], monthly_data['ROA'] * 100, 'b-', linewidth=2, label='Average ROA (%)')
        line2 = ax2_twin.plot(monthly_data['date'], monthly_data['gdp_growth_rate'], 'r-', linewidth=2, alpha=0.7, label='GDP Growth Rate (%)')
        
        ax2.set_title('Bank ROA vs GDP Growth', fontweight='bold')
        ax2.set_ylabel('ROA (%)', color='blue')
        ax2_twin.set_ylabel('GDP Growth Rate (%)', color='red')
        ax2.tick_params(axis='x', rotation=45)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        # Plot 3: ROA vs CPI Inflation
        ax3 = axes[1, 0]
        
        # Scatter plot of ROA vs CPI change
        scatter_data = df_temp.dropna(subset=['ROA', 'cpi_change_rate'])
        if len(scatter_data) > 0:
            ax3.scatter(scatter_data['cpi_change_rate'], scatter_data['ROA'] * 100, alpha=0.3, s=10)
            
            # Add trend line
            z = np.polyfit(scatter_data['cpi_change_rate'], scatter_data['ROA'] * 100, 1)
            p = np.poly1d(z)
            ax3.plot(scatter_data['cpi_change_rate'], p(scatter_data['cpi_change_rate']), "r--", alpha=0.8)
        
        ax3.set_title('Bank ROA vs CPI Inflation', fontweight='bold')
        ax3.set_xlabel('CPI Change Rate (%)')
        ax3.set_ylabel('ROA (%)')
        
        # Plot 4: Residual analysis for best model
        ax4 = axes[1, 1]
        if models:
            best_model_name = max(results_summary.keys(), key=lambda x: results_summary[x]['r_squared'])
            best_model = models[best_model_name]
            
            residuals = best_model.resids
            fitted = best_model.fitted_values
            
            ax4.scatter(fitted, residuals, alpha=0.5, s=10)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_title(f'Residuals vs Fitted\n({best_model_name.replace("_", " ").title()})', fontweight='bold')
            ax4.set_xlabel('Fitted Values')
            ax4.set_ylabel('Residuals')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'profitability_macro_analysis_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {plot_path}")
        
        return plot_path
    
    def run_analysis(self):
        """Run the complete profitability analysis"""
        print("=" * 80)
        print("MACRO-ENHANCED BANK PROFITABILITY ANALYSIS")
        print("=" * 80)
        print(f"Analysis timestamp: {self.timestamp}")
        
        try:
            # Load and process data
            gdp_df, cpi_df = self.load_and_process_macro_data()
            banks_df = self.load_surviving_banks_data()
            
            # Integrate datasets
            integrated_df = self.integrate_data_for_profitability(banks_df, gdp_df, cpi_df)
            
            # Prepare for analysis
            model_df, financial_vars, macro_vars, size_vars, lag_vars = self.prepare_profitability_models(integrated_df)
            
            # Run regressions
            models = self.run_profitability_regressions(model_df, financial_vars, macro_vars, size_vars, lag_vars)
            
            if not models:
                raise ValueError("No models were successfully estimated")
            
            # Analyze results
            results_summary = self.analyze_profitability_results(models)
            
            # Create visualizations
            plot_path = self.create_profitability_visualizations(model_df, models, results_summary)
            
            # Save results
            results_file = self.output_dir / f'profitability_macro_results_{self.timestamp}.json'
            dataset_file = self.output_dir / f'profitability_macro_dataset_{self.timestamp}.csv'
            
            # Detailed results with metadata
            results_with_meta = {
                'metadata': {
                    'analysis_type': 'macro_enhanced_profitability_prediction',
                    'timestamp': self.timestamp,
                    'focus': 'ROA prediction for surviving banks',
                    'original_data_shape': banks_df.shape,
                    'integrated_data_shape': integrated_df.shape,
                    'model_data_shape': model_df.shape,
                    'unique_banks': integrated_df['bank_name'].nunique(),
                    'date_range': {
                        'start': str(integrated_df['date'].min()),
                        'end': str(integrated_df['date'].max())
                    }
                },
                'model_results': results_summary,
                'macro_variable_effects': self.extract_macro_effects(results_summary)
            }
            
            with open(results_file, 'w') as f:
                json.dump(results_with_meta, f, indent=2, default=str)
            
            # Save dataset
            integrated_df.to_csv(dataset_file, index=False)
            
            print(f"\n{'='*80}")
            print("PROFITABILITY ANALYSIS COMPLETED!")
            print(f"{'='*80}")
            print(f"Results file: {results_file}")
            print(f"Dataset file: {dataset_file}")
            print(f"Visualization: {plot_path}")
            
            return {
                'models': models,
                'results_summary': results_summary,
                'integrated_data': integrated_df,
                'model_data': model_df
            }
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_macro_effects(self, results_summary):
        """Extract macro variable effects for summary"""
        macro_effects = {}
        
        for model_name, results in results_summary.items():
            macro_effects[model_name] = {}
            
            for var in ['gdp_deflator_lag1', 'gdp_growth_rate', 'cpi_change_rate']:
                if var in results['coefficients'] and var in results['pvalues']:
                    coef = results['coefficients'][var]
                    pval = results['pvalues'][var]
                    significant = pval < 0.05
                    
                    macro_effects[model_name][var] = {
                        'coefficient': coef,
                        'p_value': pval,
                        'significant': significant,
                        'effect': 'positive' if coef > 0 else 'negative'
                    }
        
        return macro_effects

def main():
    """Main execution function"""
    analyzer = MacroProfitabilityAnalysis()
    results = analyzer.run_analysis()
    return results

if __name__ == "__main__":
    results = main() 