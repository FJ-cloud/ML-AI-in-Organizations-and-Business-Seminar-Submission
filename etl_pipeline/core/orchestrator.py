#!/usr/bin/env python3
"""
ETL Orchestrator - Main entry point for the Ukrainian Bank ETL Pipeline

This orchestrator manages the complete pipeline:
1. Raw Excel extraction and processing
2. Bank name canonicalization 
3. Data cleaning and transformation
4. Panel dataset creation with failure flags
5. Quality validation and reporting

Expert-level implementation with comprehensive error handling and logging.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

from .extractor import DataExtractor
from .transformer import DataTransformer
from .loader import DataLoader
from ..utils.logger import setup_logging
from ..utils.config_manager import ConfigManager
from ..validators.data_validator import DataValidator


class ETLOrchestrator:
    """
    Main orchestrator for the Ukrainian Bank ETL pipeline.
    
    Coordinates extraction, transformation, and loading of Ukrainian bank data
    from raw Excel files to a clean multivariate panel dataset ready for ML.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize the ETL orchestrator.
        
        Args:
            config_path: Path to configuration file
            output_dir: Output directory for processed data
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.start_time = datetime.now()
        
        # Setup logging
        self.logger = setup_logging(log_level)
        self.logger.info("🏦 Initializing Ukrainian Bank ETL Pipeline v2.0.0")
        
        # Load configuration
        self.config = ConfigManager(config_path)
        self.output_dir = Path(output_dir or self.config.get('output_dir', 'output_final'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = DataExtractor(self.config)
        self.transformer = DataTransformer(self.config)
        self.loader = DataLoader(self.config, self.output_dir)
        self.validator = DataValidator(self.config)
        
        # Pipeline state
        self.pipeline_stats = {
            'files_processed': 0,
            'banks_found': 0,
            'total_observations': 0,
            'errors': [],
            'warnings': []
        }
        
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        
    def run_complete_pipeline(self, 
                            raw_data_dir: str,
                            incremental: bool = False,
                            validate_only: bool = False) -> Dict:
        """
        Execute the complete ETL pipeline.
        
        Args:
            raw_data_dir: Directory containing raw Excel files
            incremental: If True, only process new/modified files
            validate_only: If True, only run validation without processing
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        try:
            self.logger.info("🚀 Starting complete ETL pipeline execution")
            self.logger.info("=" * 60)
            
            # Phase 1: Discovery and Validation
            self.logger.info("📊 Phase 1: Raw data discovery and validation")
            raw_files = self._discover_raw_files(raw_data_dir)
            
            if validate_only:
                return self._validate_raw_files(raw_files)
            
            # Phase 2: Extraction
            self.logger.info("📖 Phase 2: Data extraction from Excel files")
            extracted_data = self._extract_all_data(raw_files, incremental)
            
            # Phase 3: Transformation
            self.logger.info("🔄 Phase 3: Data transformation and cleaning")
            transformed_data = self._transform_all_data(extracted_data)
            
            # Phase 4: Panel Creation
            self.logger.info("📋 Phase 4: Panel dataset creation")
            panel_dataset = self._create_panel_dataset(transformed_data)
            
            # Phase 5: Final Validation and Loading
            self.logger.info("✅ Phase 5: Final validation and data loading")
            final_results = self._finalize_and_save(panel_dataset)
            
            # Phase 6: Reporting
            self.logger.info("📈 Phase 6: Pipeline reporting and statistics")
            report = self._generate_pipeline_report(final_results)
            
            elapsed = datetime.now() - self.start_time
            self.logger.info(f"🎉 Pipeline completed successfully in {elapsed}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            self.pipeline_stats['errors'].append(str(e))
            raise
    
    def _discover_raw_files(self, raw_data_dir: str) -> List[Path]:
        """Discover and validate raw Excel files."""
        raw_dir = Path(raw_data_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
        
        # Find Excel files
        excel_files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
        excel_files = [f for f in excel_files if not f.name.startswith('.~lock')]
        excel_files.sort()
        
        self.logger.info(f"📁 Found {len(excel_files)} Excel files in {raw_dir}")
        
        if not excel_files:
            raise ValueError(f"No Excel files found in {raw_dir}")
        
        # Basic validation
        date_range = self.extractor.analyze_file_coverage(excel_files)
        self.logger.info(f"📅 Date coverage: {date_range['start']} to {date_range['end']}")
        self.logger.info(f"📊 Expected months: {date_range['expected_months']}, Found: {date_range['found_months']}")
        
        if date_range['missing_months']:
            self.logger.warning(f"⚠️ Missing months: {date_range['missing_months']}")
            self.pipeline_stats['warnings'].append(f"Missing months: {date_range['missing_months']}")
        
        return excel_files
    
    def _validate_raw_files(self, raw_files: List[Path]) -> Dict:
        """Validate raw files without processing."""
        self.logger.info("🔍 Running validation-only mode")
        
        validation_results = {
            'total_files': len(raw_files),
            'valid_files': 0,
            'invalid_files': 0,
            'file_details': [],
            'sheet_analysis': {},
            'bank_coverage': {}
        }
        
        for file_path in raw_files:
            try:
                file_info = self.validator.validate_excel_file(file_path)
                validation_results['file_details'].append(file_info)
                
                if file_info['is_valid']:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['invalid_files'] += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Validation failed for {file_path.name}: {e}")
                validation_results['invalid_files'] += 1
                self.pipeline_stats['errors'].append(f"Validation failed for {file_path.name}: {e}")
        
        self.logger.info(f"✅ Validation complete: {validation_results['valid_files']}/{validation_results['total_files']} valid files")
        
        # Create report structure similar to full pipeline
        elapsed = datetime.now() - self.start_time
        
        report = {
            'pipeline_version': '2.0.0',
            'execution_time': str(elapsed),
            'timestamp': datetime.now().isoformat(),
            'statistics': self.pipeline_stats,
            'validation_results': validation_results,
            'success': len(self.pipeline_stats['errors']) == 0,
            'mode': 'validation_only'
        }
        
        return report
    
    def _extract_all_data(self, raw_files: List[Path], incremental: bool) -> Dict:
        """Extract data from all Excel files."""
        extracted_data = {
            'assets': [],
            'liabilities': [], 
            'equity': [],
            'financial_results': [],
            'metadata': []
        }
        
        total_files = len(raw_files)
        
        for i, file_path in enumerate(raw_files, 1):
            self.logger.info(f"[{i}/{total_files}] Processing {file_path.name}")
            
            try:
                # Extract data from this file
                file_data = self.extractor.extract_file(file_path)
                
                # Merge into main data structure
                for data_type in extracted_data.keys():
                    if data_type in file_data and file_data[data_type] is not None:
                        if data_type == 'metadata':
                            extracted_data[data_type].append(file_data[data_type])
                        else:
                            extracted_data[data_type].append(file_data[data_type])
                
                self.pipeline_stats['files_processed'] += 1
                
            except Exception as e:
                error_msg = f"Failed to extract {file_path.name}: {str(e)}"
                self.logger.error(f"❌ {error_msg}")
                self.pipeline_stats['errors'].append(error_msg)
                continue
        
        # Consolidate extracted data
        for data_type in ['assets', 'liabilities', 'equity', 'financial_results']:
            if extracted_data[data_type]:
                extracted_data[data_type] = pd.concat(extracted_data[data_type], ignore_index=True)
                self.logger.info(f"📊 {data_type}: {len(extracted_data[data_type])} records")
            else:
                self.logger.warning(f"⚠️ No data extracted for {data_type}")
        
        return extracted_data
    
    def _transform_all_data(self, extracted_data: Dict) -> Dict:
        """Transform and clean all extracted data."""
        transformed_data = {}
        
        for data_type, df in extracted_data.items():
            if data_type == 'metadata':
                continue
                
            if df is not None and not df.empty:
                self.logger.info(f"🔄 Transforming {data_type} data")
                
                try:
                    # Apply transformations
                    transformed_df = self.transformer.transform_data_type(df, data_type)
                    transformed_data[data_type] = transformed_df
                    
                    self.logger.info(f"✅ {data_type}: {len(transformed_df)} records after transformation")
                    
                except Exception as e:
                    error_msg = f"Transformation failed for {data_type}: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    self.pipeline_stats['errors'].append(error_msg)
        
        return transformed_data
    
    def _create_panel_dataset(self, transformed_data: Dict) -> pd.DataFrame:
        """Create the final panel dataset."""
        self.logger.info("📋 Creating multivariate panel dataset")
        
        try:
            panel_df = self.transformer.create_panel_dataset(transformed_data)
            
            # Add failure flags
            panel_df = self.transformer.add_failure_flags(panel_df)
            
            self.pipeline_stats['total_observations'] = len(panel_df)
            self.pipeline_stats['banks_found'] = panel_df['bank_name'].nunique()
            
            self.logger.info(f"📊 Panel dataset created: {len(panel_df)} observations, {panel_df['bank_name'].nunique()} banks")
            
            return panel_df
            
        except Exception as e:
            error_msg = f"Panel creation failed: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.pipeline_stats['errors'].append(error_msg)
            raise
    
    def _finalize_and_save(self, panel_dataset: pd.DataFrame) -> Dict:
        """Final validation, consolidation, and save results."""
        
        # Step 1: Bank name consolidation and duplicate resolution
        self.logger.info("🔄 Starting bank name consolidation and duplicate resolution")
        consolidated_dataset, consolidation_report = self.validator.consolidate_panel_dataset(panel_dataset)
        
        # Update pipeline stats with consolidation results
        self.pipeline_stats['consolidation'] = consolidation_report
        self.pipeline_stats['banks_found'] = consolidation_report['final_banks']
        self.pipeline_stats['total_observations'] = consolidation_report['final_rows']
        
        # Log consolidation summary
        self.logger.info(f"📊 Consolidation completed:")
        self.logger.info(f"  • Banks: {consolidation_report['original_banks']} → {consolidation_report['final_banks']}")
        self.logger.info(f"  • Rows: {consolidation_report['original_rows']} → {consolidation_report['final_rows']}")
        self.logger.info(f"  • Mappings applied: {len(consolidation_report['mappings_applied'])}")
        
        # Add consolidation warnings to pipeline stats
        if consolidation_report['warnings']:
            self.pipeline_stats['warnings'].extend(consolidation_report['warnings'])
        
        if consolidation_report['issues']:
            self.pipeline_stats['errors'].extend(consolidation_report['issues'])
        
        # Step 2: Final validation on consolidated dataset
        validation_results = self.validator.validate_panel_dataset(consolidated_dataset)
        
        if not validation_results['is_valid']:
            self.logger.warning("⚠️ Panel dataset validation warnings found")
            for warning in validation_results['warnings']:
                self.logger.warning(f"  - {warning}")
                self.pipeline_stats['warnings'].append(warning)
        
        # Step 3: Save consolidated results
        save_results = self.loader.save_panel_dataset(consolidated_dataset)
        
        return {
            'panel_dataset': consolidated_dataset,
            'consolidation_report': consolidation_report,
            'validation_results': validation_results,
            'save_results': save_results
        }
    
    def _generate_pipeline_report(self, final_results: Dict) -> Dict:
        """Generate comprehensive pipeline report."""
        elapsed = datetime.now() - self.start_time
        
        report = {
            'pipeline_version': '2.0.0',
            'execution_time': str(elapsed),
            'timestamp': datetime.now().isoformat(),
            'statistics': self.pipeline_stats,
            'data_summary': {
                'total_observations': self.pipeline_stats['total_observations'],
                'unique_banks': self.pipeline_stats['banks_found'],
                'files_processed': self.pipeline_stats['files_processed'],
            },
            'consolidation_summary': final_results.get('consolidation_report', {}),
            'validation_summary': final_results['validation_results'],
            'output_files': final_results['save_results'],
            'success': len(self.pipeline_stats['errors']) == 0
        }
        
        # Save report
        report_path = self.output_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Pipeline report saved: {report_path}")
        
        return report 