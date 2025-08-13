#!/usr/bin/env python3
"""
Consolidate remaining bank name variants in the final panel dataset.
Merges 4 pairs of banks under their canonical names without data loss.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def consolidate_bank_names():
    """Consolidate the 4 remaining bank name pairs."""
    
    # Define the bank name mappings (variant -> canonical)
    bank_mappings = {
        'Clearing House JSB': 'Clearing House',
        'Industrialbank JSB': 'Industrialbank', 
        'Pivdennyi JSB': 'Pivdenny Bank',
        'Ukrgаzbаnk JSB': 'Ukrgasbank'  # Note: contains Cyrillic characters
    }
    
    print("🏦 Bank Name Consolidation Script")
    print("=" * 50)
    
    # Load the dataset
    input_file = Path("output_final/ukrainian_banks_panel_dataset_FINAL.csv")
    print(f"📂 Loading dataset: {input_file}")
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        return False
    
    df = pd.read_csv(input_file)
    print(f"📊 Loaded {len(df):,} rows with {df['bank_name'].nunique()} unique banks")
    
    # Show current counts for banks to be consolidated
    print("\n🔍 Current bank counts (before consolidation):")
    for variant, canonical in bank_mappings.items():
        variant_count = (df['bank_name'] == variant).sum()
        canonical_count = (df['bank_name'] == canonical).sum()
        print(f"  {variant}: {variant_count:,} rows")
        print(f"  {canonical}: {canonical_count:,} rows")
        print()
    
    # Apply the name mappings
    print("🔄 Applying bank name consolidations...")
    original_bank_count = df['bank_name'].nunique()
    
    for variant, canonical in bank_mappings.items():
        mask = df['bank_name'] == variant
        rows_changed = mask.sum()
        if rows_changed > 0:
            df.loc[mask, 'bank_name'] = canonical
            print(f"  ✅ {variant} → {canonical}: {rows_changed:,} rows updated")
        else:
            print(f"  ⚠️  {variant}: No rows found")
    
    # Check for duplicate bank-date combinations after consolidation
    print("\n🔍 Checking for duplicate bank-date combinations...")
    duplicates = df.groupby(['bank_name', 'date']).size()
    duplicate_pairs = duplicates[duplicates > 1]
    
    if len(duplicate_pairs) > 0:
        print(f"⚠️  Found {len(duplicate_pairs)} duplicate bank-date combinations:")
        for (bank, date), count in duplicate_pairs.head(10).items():
            print(f"    {bank} on {date}: {count} records")
        
        # Resolve duplicates by keeping the record with more non-null values
        print("🔧 Resolving duplicates by keeping records with more complete data...")
        
        def resolve_duplicates(group):
            if len(group) == 1:
                return group
            
            # Count non-null values for each row (excluding metadata columns)
            data_columns = [col for col in group.columns 
                          if not col.startswith(('processing_timestamp', 'bank_name', 'date'))]
            non_null_counts = group[data_columns].notna().sum(axis=1)
            
            # Keep the row with the most non-null values
            best_idx = non_null_counts.idxmax()
            return group.loc[[best_idx]]
        
        df = df.groupby(['bank_name', 'date'], group_keys=False).apply(resolve_duplicates)
        df = df.reset_index(drop=True)
        
        print(f"✅ Resolved duplicates. Dataset now has {len(df):,} rows")
    else:
        print("✅ No duplicate bank-date combinations found")
    
    # Show final counts after consolidation
    print("\n📊 Final bank counts (after consolidation):")
    for canonical in set(bank_mappings.values()):
        count = (df['bank_name'] == canonical).sum()
        print(f"  {canonical}: {count:,} rows")
    
    final_bank_count = df['bank_name'].nunique()
    print(f"\n📈 Bank count reduced from {original_bank_count} to {final_bank_count} unique banks")
    
    # Save the consolidated dataset
    output_file = Path("output_final/ukrainian_banks_panel_dataset_CONSOLIDATED.csv")
    print(f"\n💾 Saving consolidated dataset: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Create backup of original
    backup_file = Path("output_final/ukrainian_banks_panel_dataset_FINAL_backup.csv")
    if not backup_file.exists():
        print(f"📋 Creating backup: {backup_file}")
        import shutil
        shutil.copy2(input_file, backup_file)
    
    # Generate summary report
    print("\n📋 Consolidation Summary:")
    print(f"  • Original dataset: {len(pd.read_csv(input_file)):,} rows")
    print(f"  • Consolidated dataset: {len(df):,} rows")
    print(f"  • Banks consolidated: {len(bank_mappings)} pairs")
    print(f"  • Unique banks: {original_bank_count} → {final_bank_count}")
    
    # Verify no data loss in key metrics
    original_df = pd.read_csv(input_file)
    original_failed = original_df['failed'].sum()
    new_failed = df['failed'].sum()
    
    print(f"  • Failed banks preserved: {original_failed} → {new_failed}")
    
    if new_failed != original_failed:
        print("⚠️  Warning: Failed bank count changed during consolidation!")
    
    print("\n✅ Bank name consolidation completed successfully!")
    return True

if __name__ == "__main__":
    success = consolidate_bank_names()
    sys.exit(0 if success else 1) 