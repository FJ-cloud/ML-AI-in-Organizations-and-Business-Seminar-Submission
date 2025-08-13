#!/usr/bin/env python3
"""
ETL Component Overview - Detailed Analysis of Each Script

This script provides a comprehensive overview of what each component does
in the current ETL process, in execution order.
"""

import ast
import re
from pathlib import Path
from datetime import datetime

def analyze_file_structure(file_path):
    """Analyze a Python file's structure and extract key information."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {
                'classes': [],
                'functions': [],
                'imports': [],
                'docstring': None,
                'lines': len(content.split('\n'))
            }
        
        classes = []
        functions = []
        imports = []
        docstring = ast.get_docstring(tree)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    'name': node.name,
                    'methods': class_methods,
                    'docstring': ast.get_docstring(node)
                })
            elif isinstance(node, ast.FunctionDef) and not any(node in cls.body for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)):
                functions.append({
                    'name': node.name,
                    'docstring': ast.get_docstring(node)
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return {
            'classes': classes,
            'functions': functions,
            'imports': imports[:10],  # Limit to first 10 imports
            'docstring': docstring,
            'lines': len(content.split('\n'))
        }
    except Exception as e:
        return {
            'error': str(e),
            'lines': 0
        }

def overview_etl_components():
    """Provide detailed overview of each ETL component in execution order."""
    
    print("🔍 ETL COMPONENT OVERVIEW - EXECUTION ORDER")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define components in execution order
    components = [
        {
            "order": 1,
            "name": "Main Entry Point",
            "file": "run_etl_pipeline.py",
            "purpose": "Command-line interface and pipeline orchestration",
            "role": "Entry point that parses arguments and starts the ETL process"
        },
        {
            "order": 2,
            "name": "ETL Orchestrator",
            "file": "etl_pipeline/core/orchestrator.py",
            "purpose": "Main coordinator of the entire ETL pipeline",
            "role": "Manages all phases: discovery, extraction, transformation, loading"
        },
        {
            "order": 3,
            "name": "Data Validator (Pre-processing)",
            "file": "etl_pipeline/validators/data_validator.py",
            "purpose": "Validates raw Excel files before processing",
            "role": "Ensures Excel files are readable and contain expected sheets"
        },
        {
            "order": 4,
            "name": "Data Extractor",
            "file": "etl_pipeline/core/extractor.py",
            "purpose": "Extracts data from raw Excel files",
            "role": "Handles sheet selection, header detection, and data extraction"
        },
        {
            "order": 5,
            "name": "Data Transformer",
            "file": "etl_pipeline/core/transformer.py",
            "purpose": "Cleans and transforms extracted data",
            "role": "Bank name canonicalization, data cleaning, panel creation"
        },
        {
            "order": 6,
            "name": "Bank Name Canonicalizer",
            "file": "etl_pipeline/utils/bank_name_canonicalizer.py",
            "purpose": "Standardizes bank names (partial)",
            "role": "Maps bank name variants to canonical forms"
        },
        {
            "order": 7,
            "name": "Data Validator (Post-processing)",
            "file": "etl_pipeline/validators/data_validator.py",
            "purpose": "Validates transformed data",
            "role": "Quality checks on panel dataset before loading"
        },
        {
            "order": 8,
            "name": "Data Loader",
            "file": "etl_pipeline/core/loader.py",
            "purpose": "Saves processed data and generates metadata",
            "role": "Writes CSV files, metadata, and summary reports"
        },
        {
            "order": 9,
            "name": "POST-PROCESSING: Bank Name Consolidation",
            "file": "analysis_scripts/consolidate_remaining_bank_names.py",
            "purpose": "Final bank name consolidation (manual step)",
            "role": "Handles remaining JSB variants and duplicate resolution"
        }
    ]
    
    # Analyze each component
    for component in components:
        print(f"\n{'='*70}")
        print(f"🔧 {component['order']}. {component['name']}")
        print(f"{'='*70}")
        print(f"📁 File: {component['file']}")
        print(f"🎯 Purpose: {component['purpose']}")
        print(f"⚙️  Role: {component['role']}")
        
        # Analyze file structure
        file_path = Path(component['file'])
        if file_path.exists():
            analysis = analyze_file_structure(file_path)
            
            if 'error' not in analysis:
                print(f"📊 Lines of Code: {analysis['lines']}")
                
                if analysis['docstring']:
                    print(f"📝 Description: {analysis['docstring'][:200]}...")
                
                if analysis['classes']:
                    print(f"🏗️  Classes ({len(analysis['classes'])}):")
                    for cls in analysis['classes']:
                        methods_count = len(cls['methods'])
                        print(f"   • {cls['name']} ({methods_count} methods)")
                        if cls['methods']:
                            key_methods = cls['methods'][:5]  # Show first 5 methods
                            print(f"     Key methods: {', '.join(key_methods)}")
                
                if analysis['functions']:
                    print(f"🔧 Functions ({len(analysis['functions'])}):")
                    for func in analysis['functions'][:5]:  # Show first 5 functions
                        print(f"   • {func['name']}")
                
                if analysis['imports']:
                    key_imports = [imp for imp in analysis['imports'] if not imp.startswith('_')][:5]
                    if key_imports:
                        print(f"📦 Key Imports: {', '.join(key_imports)}")
            else:
                print(f"❌ Error analyzing file: {analysis['error']}")
        else:
            print(f"❌ File not found: {component['file']}")
        
        # Add specific insights for each component
        print(f"\n💡 Key Functionality:")
        if component['order'] == 1:
            print("   • Parses command-line arguments (data directory, output directory)")
            print("   • Sets up logging and configuration")
            print("   • Initializes and runs ETLOrchestrator")
            print("   • Handles errors and exit codes")
        
        elif component['order'] == 2:
            print("   • Discovers raw Excel files in input directory")
            print("   • Orchestrates all pipeline phases in sequence")
            print("   • Manages error handling and rollback")
            print("   • Generates comprehensive pipeline reports")
        
        elif component['order'] == 3:
            print("   • Validates Excel file readability")
            print("   • Checks for required sheets (assets, liabilities, etc.)")
            print("   • Estimates bank count per file")
            print("   • Pre-flight validation before extraction")
        
        elif component['order'] == 4:
            print("   • Intelligent sheet selection (fuzzy matching)")
            print("   • Dynamic header detection (row 4 vs 5)")
            print("   • Handles _NC (National Currency) sheet filtering")
            print("   • Date extraction from filenames")
        
        elif component['order'] == 5:
            print("   • Data type categorization (assets, liabilities, equity, results)")
            print("   • Column name cleaning and prefixing")
            print("   • Panel dataset creation (wide format)")
            print("   • Bank name canonicalization (partial)")
        
        elif component['order'] == 6:
            print("   • 200+ bank name mappings")
            print("   • Handles JSC/PJSC/CB variants (partial)")
            print("   • Removes aggregation rows")
            print("   • ⚠️  Missing 4 critical JSB variants")
        
        elif component['order'] == 7:
            print("   • Panel dataset structure validation")
            print("   • Data quality checks (missing values, duplicates)")
            print("   • Column analysis and statistics")
            print("   • Bank-date combination validation")
        
        elif component['order'] == 8:
            print("   • Saves main panel dataset CSV")
            print("   • Generates metadata (JSON and text)")
            print("   • Creates summary statistics")
            print("   • Pipeline execution reports")
        
        elif component['order'] == 9:
            print("   • 🚨 MANUAL POST-PROCESSING STEP")
            print("   • Consolidates 4 remaining JSB bank pairs")
            print("   • Resolves duplicate bank-date combinations")
            print("   • Creates final CONSOLIDATED dataset")
            print("   • ⚠️  This should be integrated into main pipeline!")

def analyze_data_flow():
    """Analyze the data flow through the pipeline."""
    
    print(f"\n{'='*70}")
    print("📊 DATA FLOW ANALYSIS")
    print(f"{'='*70}")
    
    flow_stages = [
        {
            "stage": "Input",
            "description": "Raw Excel files",
            "location": "data/raw_balance_sheets/*.xlsx",
            "format": "Excel workbooks with multiple sheets",
            "count": "~76 files (2019-2025)"
        },
        {
            "stage": "Discovery",
            "description": "File validation and discovery",
            "location": "ETL Orchestrator",
            "format": "File path list",
            "count": "~76 valid Excel files"
        },
        {
            "stage": "Extraction",
            "description": "Raw DataFrames per file",
            "location": "Data Extractor",
            "format": "pandas DataFrames",
            "count": "~304 DataFrames (4 types × 76 files)"
        },
        {
            "stage": "Transformation",
            "description": "Cleaned and categorized data",
            "location": "Data Transformer",
            "format": "Cleaned DataFrames with prefixed columns",
            "count": "~304 cleaned DataFrames"
        },
        {
            "stage": "Canonicalization",
            "description": "Partially standardized bank names",
            "location": "Bank Name Canonicalizer",
            "format": "DataFrames with canonical names",
            "count": "~200 bank name mappings applied"
        },
        {
            "stage": "Panel Creation",
            "description": "Wide panel dataset",
            "location": "Data Transformer",
            "format": "Single wide DataFrame",
            "count": "5,396 rows × 150+ columns"
        },
        {
            "stage": "Main Output",
            "description": "ETL pipeline output",
            "location": "output_final/ukrainian_banks_panel_dataset_FINAL.csv",
            "format": "CSV file",
            "count": "71 banks, 5,396 rows"
        },
        {
            "stage": "POST-PROCESSING",
            "description": "Manual consolidation step",
            "location": "analysis_scripts/consolidate_remaining_bank_names.py",
            "format": "Python script",
            "count": "4 bank pairs consolidated"
        },
        {
            "stage": "Final Output",
            "description": "Fully consolidated dataset",
            "location": "output_final/ukrainian_banks_panel_dataset_CONSOLIDATED.csv",
            "format": "CSV file",
            "count": "67 banks, 5,092 rows"
        }
    ]
    
    for i, stage in enumerate(flow_stages, 1):
        print(f"\n{i}. {stage['stage']}: {stage['description']}")
        print(f"   📍 Location: {stage['location']}")
        print(f"   📄 Format: {stage['format']}")
        print(f"   📊 Count: {stage['count']}")
        
        if stage['stage'] == "POST-PROCESSING":
            print("   🚨 ISSUE: This should be integrated into main pipeline!")

def identify_integration_points():
    """Identify specific points where integration is needed."""
    
    print(f"\n{'='*70}")
    print("🎯 INTEGRATION POINTS")
    print(f"{'='*70}")
    
    integration_points = [
        {
            "component": "Bank Name Canonicalizer",
            "file": "etl_pipeline/utils/bank_name_canonicalizer.py",
            "current_state": "Handles ~200 mappings but missing 4 JSB variants",
            "needed_integration": "Add the 4 missing JSB mappings from post-processing script",
            "specific_mappings": [
                "'Clearing House JSB' → 'Clearing House'",
                "'Industrialbank JSB' → 'Industrialbank'", 
                "'Pivdennyi JSB' → 'Pivdenny Bank'",
                "'Ukrgаzbаnk JSB' → 'Ukrgasbank'"
            ]
        },
        {
            "component": "Data Transformer",
            "file": "etl_pipeline/core/transformer.py",
            "current_state": "Creates panel dataset but doesn't resolve duplicates",
            "needed_integration": "Add duplicate resolution logic from post-processing script",
            "specific_functionality": [
                "Detect duplicate bank-date combinations",
                "Resolve by keeping record with more complete data",
                "Update row counts and statistics"
            ]
        },
        {
            "component": "Data Validator",
            "file": "etl_pipeline/validators/data_validator.py",
            "current_state": "Validates structure but not consolidated names",
            "needed_integration": "Add validation for consolidated bank names",
            "specific_checks": [
                "Verify no JSB variants remain",
                "Check for duplicate bank-date combinations",
                "Validate final bank count (should be 67)"
            ]
        }
    ]
    
    for point in integration_points:
        print(f"\n🔧 {point['component']}")
        print(f"   📁 File: {point['file']}")
        print(f"   📊 Current: {point['current_state']}")
        print(f"   🎯 Needed: {point['needed_integration']}")
        
        if 'specific_mappings' in point:
            print(f"   📝 Specific mappings to add:")
            for mapping in point['specific_mappings']:
                print(f"      • {mapping}")
        
        if 'specific_functionality' in point:
            print(f"   ⚙️  Specific functionality to add:")
            for func in point['specific_functionality']:
                print(f"      • {func}")
        
        if 'specific_checks' in point:
            print(f"   ✅ Specific checks to add:")
            for check in point['specific_checks']:
                print(f"      • {check}")

if __name__ == "__main__":
    print("🚀 Ukrainian Bank ETL - Component Overview")
    print("=" * 70)
    
    # Run comprehensive analysis
    overview_etl_components()
    analyze_data_flow()
    identify_integration_points()
    
    print(f"\n{'='*70}")
    print("✅ SUMMARY")
    print(f"{'='*70}")
    print("Current ETL Pipeline: 8 components + 1 manual post-processing step")
    print("Goal: Integrate post-processing into main pipeline for single-command execution")
    print("Key Integration: Bank name mappings + duplicate resolution logic")
    print("Expected Result: Direct output of CONSOLIDATED dataset (67 banks, ~5,092 rows)") 