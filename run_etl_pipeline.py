#!/usr/bin/env python3
"""
Ukrainian Bank ETL Pipeline - Simple Execution

A comprehensive, production-ready ETL pipeline for processing Ukrainian bank
regulatory data from raw Excel files to ML-ready panel datasets.

Usage:
    python run_etl_pipeline.py

Author: Expert ETL Developer
Version: 2.0.0
"""

import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from etl_pipeline import ETLOrchestrator


def print_banner():
    """Print the pipeline banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                Ukrainian Bank ETL Pipeline v2.0.0           ║
    ║                                                              ║
    ║  Comprehensive data processing for Ukrainian bank regulatory ║
    ║  data from raw Excel files to ML-ready panel datasets       ║
    ║                                                              ║
    ║  🏦 Intelligent header detection                             ║
    ║  🔄 Advanced bank name canonicalization & consolidation     ║
    ║  📊 Multivariate panel dataset creation                      ║
    ║  ✅ Comprehensive validation and quality checks              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main execution function with sensible defaults."""
    
    # Print banner
    print_banner()
    
    # Default configuration
    data_dir = "data/raw_balance_sheets"
    output_dir = "output_final"
    log_level = "INFO"
    
    print(f"🚀 Starting ETL Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"📁 Raw data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Log level: {log_level}")
    print()

    try:
        # Initialize and run the ETL orchestrator
        orchestrator = ETLOrchestrator(
            output_dir=output_dir,
            log_level=log_level
        )
        
        # Check if data directory exists
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Error: Data directory '{data_dir}' not found!")
            print(f"   Please ensure your Excel files are in: {data_path.absolute()}")
            return 1
        
        # Run the complete pipeline
        results = orchestrator.run_complete_pipeline(str(data_path))
        
        # Print results summary
        print("\n" + "=" * 80)
        print("📈 PIPELINE RESULTS SUMMARY")
        print("=" * 80)
        
        if results['success']:
            print("✅ Pipeline completed successfully!")
            
            # Execution stats
            print(f"⏱️  Execution time: {results.get('execution_time', 'Unknown')}")
            print(f"📊 Files processed: {results['statistics'].get('files_processed', 0)}")
            print(f"🏦 Banks found: {results['statistics'].get('banks_found', 0)}")
            print(f"📋 Total observations: {results['statistics'].get('total_observations', 0)}")
            
            # Consolidation results
            if 'consolidation' in results['statistics']:
                consolidation = results['statistics']['consolidation']
                print(f"🔄 Bank consolidation:")
                print(f"   • Original banks: {consolidation.get('original_banks', 0)}")
                print(f"   • Final banks: {consolidation.get('final_banks', 0)}")
                print(f"   • Mappings applied: {len(consolidation.get('mappings_applied', {}))}")
                print(f"   • Duplicates resolved: {consolidation.get('duplicates_resolved', 0)}")
            
            # Warnings
            warnings = results['statistics'].get('warnings', [])
            if warnings:
                print(f"⚠️  Warnings: {len(warnings)}")
                for i, warning in enumerate(warnings[:3], 1):  # Show first 3
                    print(f"   {i}. {warning}")
                if len(warnings) > 3:
                    print(f"   ... and {len(warnings) - 3} more warnings")
            
            # Output files
            output_files = results.get('output_files', [])
            if output_files:
                print("📄 Output files:")
                for file_path in output_files:
                    print(f"   - {file_path}")
            
            print(f"\n🎉 Ukrainian Bank ETL Pipeline completed!")
            return 0
            
        else:
            print("❌ Pipeline failed!")
            errors = results['statistics'].get('errors', [])
            if errors:
                print("Errors encountered:")
                for i, error in enumerate(errors, 1):
                    print(f"   {i}. {error}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 