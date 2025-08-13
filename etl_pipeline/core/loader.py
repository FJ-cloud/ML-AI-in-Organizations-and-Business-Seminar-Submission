#!/usr/bin/env python3
"""
Data Loader - Save processed data and generate outputs

Handles saving the final panel dataset and other outputs with:
- Multiple output formats (CSV, compressed)
- Comprehensive metadata generation
- File organization and management
- Backup and versioning support

Expert implementation for production use.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd


class DataLoader:
    """
    Data loader for saving processed Ukrainian bank data.
    
    Handles all output operations including saving datasets,
    generating metadata, and organizing files.
    """
    
    def __init__(self, config, output_dir: Path):
        """Initialize the data loader."""
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.save_intermediate = config.get('save_intermediate', True)
        self.save_metadata = config.get('save_metadata', True)
        self.compression = config.get('compression', None)
        
        self.logger.info(f"💾 Data loader initialized: {self.output_dir}")
    
    def save_panel_dataset(self, panel_df: pd.DataFrame) -> List[str]:
        """
        Save the final panel dataset with metadata.
        
        Args:
            panel_df: Final panel dataset
            
        Returns:
            List of saved file paths
        """
        self.logger.info("💾 Saving final panel dataset")
        
        saved_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main dataset file
        main_filename = "ukrainian_banks_panel_dataset_FINAL.csv"
        main_path = self.output_dir / main_filename
        
        # Save with optional compression
        if self.compression == 'gzip':
            main_path = main_path.with_suffix('.csv.gz')
            panel_df.to_csv(main_path, index=False, compression='gzip')
        else:
            panel_df.to_csv(main_path, index=False)
        
        saved_files.append(str(main_path))
        self.logger.info(f"📄 Saved main dataset: {main_path}")
        
        # Timestamped backup
        backup_filename = f"ukrainian_banks_panel_dataset_{timestamp}.csv"
        backup_path = self.output_dir / backup_filename
        
        if self.compression == 'gzip':
            backup_path = backup_path.with_suffix('.csv.gz')
            panel_df.to_csv(backup_path, index=False, compression='gzip')
        else:
            panel_df.to_csv(backup_path, index=False)
        
        saved_files.append(str(backup_path))
        self.logger.info(f"📄 Saved backup: {backup_path}")
        
        # Save metadata if enabled
        if self.save_metadata:
            metadata_files = self._save_dataset_metadata(panel_df, timestamp)
            saved_files.extend(metadata_files)
        
        # Save summary statistics
        summary_files = self._save_summary_statistics(panel_df, timestamp)
        saved_files.extend(summary_files)
        
        self.logger.info(f"✅ Panel dataset saved: {len(saved_files)} files created")
        
        return saved_files
    
    def _save_dataset_metadata(self, panel_df: pd.DataFrame, timestamp: str) -> List[str]:
        """Save comprehensive metadata about the dataset."""
        metadata_files = []
        
        # Generate metadata
        metadata = self._generate_dataset_metadata(panel_df)
        
        # Save as JSON
        json_path = self.output_dir / f"dataset_metadata_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        metadata_files.append(str(json_path))
        
        # Save as human-readable text
        txt_path = self.output_dir / f"dataset_metadata_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self._format_metadata_text(metadata))
        metadata_files.append(str(txt_path))
        
        self.logger.info(f"📋 Saved metadata: {len(metadata_files)} files")
        
        return metadata_files
    
    def _generate_dataset_metadata(self, panel_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive metadata about the dataset."""
        # Basic statistics
        basic_stats = {
            'total_rows': len(panel_df),
            'total_columns': len(panel_df.columns),
            'unique_banks': panel_df['bank_name'].nunique(),
            'date_range': {
                'start': panel_df['date'].min().strftime('%Y-%m-%d'),
                'end': panel_df['date'].max().strftime('%Y-%m-%d'),
                'months': panel_df['year_month'].nunique()
            }
        }
        
        # Bank statistics
        bank_stats = {
            'total_banks': panel_df['bank_name'].nunique(),
            'active_banks': len(panel_df[panel_df['bank_present'] == 1]['bank_name'].unique()),
            'failed_banks': len(panel_df[panel_df['failed'] == 1]['bank_name'].unique()) if 'failed' in panel_df.columns else 0
        }
        
        # Data availability
        if 'bank_present' in panel_df.columns:
            total_possible = len(panel_df)
            actual_present = panel_df['bank_present'].sum()
            availability_rate = (actual_present / total_possible) * 100 if total_possible > 0 else 0
        else:
            availability_rate = 0
        
        data_availability = {
            'total_possible_observations': len(panel_df),
            'actual_observations': panel_df['bank_present'].sum() if 'bank_present' in panel_df.columns else len(panel_df),
            'availability_rate_percent': round(availability_rate, 2),
            'missing_rate_percent': round(100 - availability_rate, 2)
        }
        
        # Column analysis
        column_types = {
            'metadata_columns': [],
            'assets_columns': [],
            'liabilities_columns': [],
            'equity_columns': [],
            'financial_results_columns': [],
            'other_columns': []
        }
        
        metadata_cols = {'bank_name', 'date', 'year_month', 'bank_present', 'failed', 'failure_date', 'last_reporting_date', 'processing_timestamp'}
        
        for col in panel_df.columns:
            if col in metadata_cols:
                column_types['metadata_columns'].append(col)
            elif col.startswith('assets_'):
                column_types['assets_columns'].append(col)
            elif col.startswith('liabilities_'):
                column_types['liabilities_columns'].append(col)
            elif col.startswith('equity_'):
                column_types['equity_columns'].append(col)
            elif col.startswith('financial_results_'):
                column_types['financial_results_columns'].append(col)
            else:
                column_types['other_columns'].append(col)
        
        # Processing information
        processing_info = {
            'processing_timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.0.0',
            'configuration': self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        }
        
        return {
            'basic_statistics': basic_stats,
            'bank_statistics': bank_stats,
            'data_availability': data_availability,
            'column_analysis': column_types,
            'processing_information': processing_info
        }
    
    def _format_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as human-readable text."""
        lines = []
        lines.append("Ukrainian Banks Panel Dataset - Metadata")
        lines.append("=" * 50)
        lines.append("")
        
        # Basic statistics
        basic = metadata['basic_statistics']
        lines.append("BASIC STATISTICS")
        lines.append("-" * 16)
        lines.append(f"Total rows: {basic['total_rows']:,}")
        lines.append(f"Total columns: {basic['total_columns']}")
        lines.append(f"Unique banks: {basic['unique_banks']}")
        lines.append(f"Date range: {basic['date_range']['start']} to {basic['date_range']['end']}")
        lines.append(f"Months covered: {basic['date_range']['months']}")
        lines.append("")
        
        # Bank statistics
        bank = metadata['bank_statistics']
        lines.append("BANK STATISTICS")
        lines.append("-" * 15)
        lines.append(f"Total banks: {bank['total_banks']}")
        lines.append(f"Active banks: {bank['active_banks']}")
        lines.append(f"Failed banks: {bank['failed_banks']}")
        lines.append("")
        
        # Data availability
        avail = metadata['data_availability']
        lines.append("DATA AVAILABILITY")
        lines.append("-" * 17)
        lines.append(f"Total possible observations: {avail['total_possible_observations']:,}")
        lines.append(f"Actual observations: {avail['actual_observations']:,}")
        lines.append(f"Availability rate: {avail['availability_rate_percent']}%")
        lines.append(f"Missing rate: {avail['missing_rate_percent']}%")
        lines.append("")
        
        # Column analysis
        cols = metadata['column_analysis']
        lines.append("COLUMN ANALYSIS")
        lines.append("-" * 15)
        lines.append(f"Metadata columns: {len(cols['metadata_columns'])}")
        lines.append(f"Assets columns: {len(cols['assets_columns'])}")
        lines.append(f"Liabilities columns: {len(cols['liabilities_columns'])}")
        lines.append(f"Equity columns: {len(cols['equity_columns'])}")
        lines.append(f"Financial results columns: {len(cols['financial_results_columns'])}")
        lines.append(f"Other columns: {len(cols['other_columns'])}")
        lines.append("")
        
        # Processing info
        proc = metadata['processing_information']
        lines.append("PROCESSING INFORMATION")
        lines.append("-" * 21)
        lines.append(f"Processing timestamp: {proc['processing_timestamp']}")
        lines.append(f"Pipeline version: {proc['pipeline_version']}")
        lines.append("")
        
        lines.append("✅ FINAL CLEAN DATASET - READY FOR MACHINE LEARNING!")
        
        return "\n".join(lines)
    
    def _save_summary_statistics(self, panel_df: pd.DataFrame, timestamp: str) -> List[str]:
        """Save summary statistics and analysis."""
        summary_files = []
        
        # Bank summary
        bank_summary = self._create_bank_summary(panel_df)
        bank_summary_path = self.output_dir / f"bank_summary_{timestamp}.csv"
        bank_summary.to_csv(bank_summary_path, index=False)
        summary_files.append(str(bank_summary_path))
        
        # Monthly summary
        monthly_summary = self._create_monthly_summary(panel_df)
        monthly_summary_path = self.output_dir / f"monthly_summary_{timestamp}.csv"
        monthly_summary.to_csv(monthly_summary_path, index=False)
        summary_files.append(str(monthly_summary_path))
        
        self.logger.info(f"📊 Saved summary statistics: {len(summary_files)} files")
        
        return summary_files
    
    def _create_bank_summary(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics by bank."""
        summary_data = []
        
        for bank_name in panel_df['bank_name'].unique():
            bank_data = panel_df[panel_df['bank_name'] == bank_name]
            
            summary = {
                'bank_name': bank_name,
                'total_observations': len(bank_data),
                'present_observations': bank_data['bank_present'].sum() if 'bank_present' in bank_data.columns else len(bank_data),
                'first_date': bank_data['date'].min(),
                'last_date': bank_data['date'].max(),
                'months_covered': bank_data['year_month'].nunique(),
                'is_failed': bank_data['failed'].max() if 'failed' in bank_data.columns else 0,
                'failure_date': bank_data['failure_date'].iloc[0] if 'failure_date' in bank_data.columns and bank_data['failure_date'].notna().any() else None,
                'last_reporting_date': bank_data['last_reporting_date'].iloc[0] if 'last_reporting_date' in bank_data.columns and bank_data['last_reporting_date'].notna().any() else None
            }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def _create_monthly_summary(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics by month."""
        summary_data = []
        
        for year_month in sorted(panel_df['year_month'].unique()):
            month_data = panel_df[panel_df['year_month'] == year_month]
            
            summary = {
                'year_month': year_month,
                'total_banks': len(month_data),
                'present_banks': month_data['bank_present'].sum() if 'bank_present' in month_data.columns else len(month_data),
                'failed_banks': month_data['failed'].sum() if 'failed' in month_data.columns else 0,
                'availability_rate': (month_data['bank_present'].sum() / len(month_data)) * 100 if 'bank_present' in month_data.columns and len(month_data) > 0 else 0
            }
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_intermediate_data(self, data: pd.DataFrame, name: str, data_type: str = "") -> Optional[str]:
        """
        Save intermediate data if enabled.
        
        Args:
            data: DataFrame to save
            name: Name for the file
            data_type: Optional data type prefix
            
        Returns:
            Path to saved file or None if not saved
        """
        if not self.save_intermediate:
            return None
        
        # Create intermediate directory
        intermediate_dir = self.output_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if data_type:
            filename = f"{data_type}_{name}_{timestamp}.csv"
        else:
            filename = f"{name}_{timestamp}.csv"
        
        file_path = intermediate_dir / filename
        
        # Save with optional compression
        if self.compression == 'gzip':
            file_path = file_path.with_suffix('.csv.gz')
            data.to_csv(file_path, index=False, compression='gzip')
        else:
            data.to_csv(file_path, index=False)
        
        self.logger.debug(f"💾 Saved intermediate data: {file_path}")
        
        return str(file_path)
    
    def cleanup_old_files(self, keep_days: int = 7) -> int:
        """
        Clean up old files in the output directory.
        
        Args:
            keep_days: Number of days to keep files
            
        Returns:
            Number of files removed
        """
        if keep_days <= 0:
            return 0
        
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        removed_count = 0
        
        # Clean up timestamped files
        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file() and '_' in file_path.stem:
                # Check if file has timestamp pattern
                if any(part.isdigit() and len(part) >= 8 for part in file_path.stem.split('_')):
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"🧹 Cleaned up {removed_count} old files")
        
        return removed_count 