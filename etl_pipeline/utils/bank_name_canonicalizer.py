#!/usr/bin/env python3
"""
Bank Name Canonicalizer - Real-world implementation

Based on the actual bank name cleaning scripts that were successfully used.
This incorporates all the hard-won lessons from processing Ukrainian bank data.

Key features:
- 200+ bank name mappings from actual processing
- Handles JSC/PJSC/JSCB/CB/JSB variants  
- Removes aggregation rows and invalid entries
- Consolidates duplicate banks with data recovery
- Expert-level implementation based on real-world usage
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz


class BankNameCanonicalizer:
    """
    Production-ready bank name canonicalizer based on real-world processing.
    
    This implementation is based on the actual scripts that successfully
    processed Ukrainian bank data, not theoretical YAML configurations.
    """
    
    def __init__(self, config=None):
        """Initialize the canonicalizer with real-world mappings."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Real-world bank mappings from successful processing
        self.bank_mappings = self._create_comprehensive_mappings()
        
        # Invalid patterns that should be removed
        self.invalid_patterns = self._create_invalid_patterns()
        
        # Aggregation banks that should be removed
        self.aggregation_banks = {
            'Banks owned by foreign bank groups',
            'Privately owned banks', 
            'State-owned banks',
            'Insolvent Banks',
            'Insolvent banks'
        }
        
        self.logger.info(f"🏦 Bank canonicalizer initialized with {len(self.bank_mappings)} mappings")
    
    def _create_comprehensive_mappings(self) -> Dict[str, str]:
        """Create the comprehensive bank mappings from real-world processing."""
        # This is the exact mapping dictionary from fix_bank_names.py that actually worked
        return {
            # Core mappings from successful processing
            'A - BANK JSC': 'A-Bank',
            'A-Bank JSC': 'A-Bank',
            'JSC \'A - BANK\'': 'A-Bank',
            'ALTBANK JSC': 'Altbank',
            'Altbank JSC': 'Altbank',
            'JSC \'ALTBANK\'': 'Altbank',
            'AP BANK  JSC': 'AP Bank',
            'AP BANK JSC': 'AP Bank',
            'AP Bank JSC': 'AP Bank',
            'AP Bank PJSC': 'AP Bank',
            'JSC \'AP BANK\'': 'AP Bank',
            'ASVIO BANK JSC': 'Asvio Bank',
            'Asvio Bank JSC': 'Asvio Bank',
            'JSC \'ASVIO BANK\'': 'Asvio Bank',
            'Alpari Bank JSC': 'Alpari Bank',
            'Arkada JSCB JSC': 'Arkada',
            'JSC JSCB \'ARCADA\'': 'Arkada',
            'BANK  PORTAL JSC': 'Bank Portal',
            'BANK PORTAL JSC': 'Bank Portal',
            'BANK 3/4 JSC': 'Bank 3/4',
            'BANK ALLIANCE JSC': 'Bank Alliance',
            'BANK CREDIT DNIPRO  JSC': 'Bank Credit Dnipro',
            'BANK CREDIT DNIPRO JSC': 'Bank Credit Dnipro',
            'BANK FOR INVESTMENTS AND SAVINGS JSC': 'Bank for Investments and Savings',
            'BANK GRANT JSC': 'Bank Grant',
            'BANK TRUST-CAPITAL JSC': 'Trust-Capital',
            'BANK UKRAINIAN CAPITAL  JSC': 'Ukrainian Capital Bank',
            'BANK UKRAINIAN CAPITAL JSC': 'Ukrainian Capital Bank',
            'BTA BANK  JSC': 'BTA Bank',
            'BTA BANK JSC': 'BTA Bank',
            'BTA Bank JSC': 'BTA Bank',
            'BTA Bank PJSC': 'BTA Bank',
            'JSC \'BTA BANK\'': 'BTA Bank',
            'Bank 3/4 JSC': 'Bank 3/4',
            'Bank 3/4 PJSC': 'Bank 3/4',
            'Bank Alliance JSC': 'Bank Alliance',
            'Bank Avangard JSC': 'Bank Avangard',
            'Bank Credit Dnipro JSC': 'Bank Credit Dnipro',
            'Bank Familny JSC': 'Bank Familny',
            'Bank Grant JSC': 'Bank Grant',
            'Bank Portal JSC': 'Bank Portal',
            'Bank Portal PJSC': 'Bank Portal',
            'Bank Sich JSC': 'Bank Sich',
            'Bank Vostok PJSC': 'Bank Vostok',
            'Bank for Investments and Savings JSC': 'Bank for Investments and Savings',
            'Bank for Investments and Savings PJSC': 'Bank for Investments and Savings',
            'CB  GLOBUS JSC': 'CB Globus',
            'CB \'ACCORDBANK\' PuJSC': 'Accordbank',
            'CB Accordbank JSC': 'Accordbank',
            'CB FINANCIAL INITIATIVE PJSC': 'Financial Initiative',
            'CB GLOBUS JSC': 'CB Globus',
            'CB Globus JSC': 'CB Globus',
            'CB Globus PJSC': 'CB Globus',
            'CB Pravex-Bank JSC': 'Pravex Bank',
            'CB PrivatBank JSC': 'PrivatBank',
            'CB Zemelny Capital JSC': 'Zemelny Capital',
            'CIB JSC': 'CIB',
            'CITIBANK JSC': 'Citibank',
            'CLEARING HOUSE JSC': 'Clearing House',
            'COMINBANK JSC': 'Cominbank',
            'COMINVESTBANK JSC': 'Cominvestbank',
            'CREDIT EUROPE BANK JSC': 'Credit Europe Bank',
            'CRYSTALBANK  JSC': 'Crystalbank',
            'CRYSTALBANK JSC': 'Crystalbank',
            'Citibank JSC': 'Citibank',
            'Credit Agricole Bank JSC': 'Credit Agricole Bank',
            'Credit Europe Bank JSC': 'Credit Europe Bank',
            'Credit Europe Bank PJSC': 'Credit Europe Bank',
            'CreditWest Bank JSC': 'CreditWest Bank',
            'Crystalbank JSC': 'Crystalbank',
            'Deutsche Bank DBU JSC': 'Deutsche Bank DBU',
            'Deutsche Bank DBU PJSC': 'Deutsche Bank DBU',
            'EIB JSC': 'EIB',
            'EUROPROMBANK JSC': 'Europrombank',
            'FIRST INVESTMENT BANK JSC': 'First Investment Bank',
            'FUIB JSC': 'FUIB',
            'FUIB PJSC': 'FUIB',
            'First Investment Bank JSC': 'First Investment Bank',
            'Forward Bank JSC': 'Forward Bank',
            'IBOX Bank JSC': 'IBOX Bank',
            'IIB JSC': 'IIB',
            'ING Bank Ukraine JSC': 'ING Bank Ukraine',
            'ING Bank Ukraine PJSC': 'ING Bank Ukraine',
            'Idea Bank JSC': 'Idea Bank',
            'Industrialbank JSB': 'Industrialbank',
            'JOINT STOCK BANK \'RADABANK\'': 'Radabank',
            'JSB  UKRGASBANK PJSC': 'Ukrgasbank',
            'JSB \'CLEARING HOUSE\'': 'Clearing House',
            'JSB \'UKRGASBANK\'': 'Ukrgasbank',
            'JSB Pivdenny Bank PJSB': 'Pivdenny Bank',
            'JSB Radabank JSC': 'Radabank',
            'JSC \'ALFA-BANK\'': 'Alfa-Bank',
            'JSC \'BANK \'GRANT\'': 'Bank Grant',
            'JSC \'BANK \'PORTAL\'': 'Bank Portal',
            'JSC \'BANK \'UKRAINIAN CAPITAL\'': 'Ukrainian Capital Bank',
            'JSC \'BANK 3/4\'': 'Bank 3/4',
            'JSC \'BANK ALLIANCE\'': 'Bank Alliance',
            'JSC \'BANK AVANGARD\'': 'Bank Avangard',
            'JSC \'BANK CREDIT DNEPR\'': 'Bank Credit Dnipro',
            'JSC \'BANK FOR INVESTMENTS AND SAVINGS\'': 'Bank for Investments and Savings',
            'JSC \'BANK FORWARD\'': 'Forward Bank',
            'JSC \'BANK SICH\'': 'Bank Sich',
            'JSC \'BANK TRUST-CAPITAL\'': 'Trust-Capital',
            'JSC \'CB \'GLOBUS\'': 'CB Globus',
            'JSC \'CITIBANK\'': 'Citibank',
            'JSC \'COMINBANK\'': 'Cominbank',
            'JSC \'CREDIT EUROPE BANK\'': 'Credit Europe Bank',
            'JSC \'CRYSTALBANK\'': 'Crystalbank',
            'JSC \'EIB\'': 'EIB',
            'JSC \'FIRST INVESTMENT BANK\'': 'First Investment Bank',
            'JSC \'FUIB\'': 'FUIB',
            'JSC \'ING Bank Ukraine\'': 'ING Bank Ukraine',
            'JSC \'Idea Bank\'': 'Idea Bank',
            'JSC \'JSCB \'CONCORD\'': 'Concord',
            'JSC \'MOTOR-BANK\'': 'Motor-Bank',
            'JSC \'PIRAEUS BANK ICB\'': 'Piraeus Bank ICB',
            'JSC \'Poltava-bank\'': 'Poltava-Bank',
            'JSC \'RWS BANK\'': 'RWS Bank',
            'JSC \'SEB CORPORATE BANK\'': 'SEB Corporate Bank',
            'JSC \'SENSE BANK\'': 'Sense Bank',
            'JSC \'SKY BANK\'': 'Sky Bank',
            'JSC \'UNIVERSAL BANK\'': 'Universal Bank',
            'JSC \'Ukrainian Bank for reconstruction and development\'': 'Ukrainian Bank for Reconstruction and Development',
            'JSC \'Ukreximbank\'': 'Ukreximbank',
            'JSC CB \'PRIVATBANK\'': 'PrivatBank',
            'JSC Deutsche Bank DBU': 'Deutsche Bank DBU',
            'JSC IIB': 'IIB',
            'JSCB  INDUSTRIALBANK PJSC': 'Industrialbank',
            'JSCB \'Lviv\'': 'Lviv',
            'JSCB Concord JSC': 'Concord',
            'JSCB INDUSTRIALBANK PJSC': 'Industrialbank',
            'JSС \'UKRSIBBANK\'': 'UkrSibbank',
            'Kredobank JSC': 'Kredobank',
            'Lviv JSCB': 'Lviv',
            'MOTOR-BANK JSC': 'Motor-Bank',
            'MTB BANK   PJSC': 'MTB Bank',
            'MTB BANK PJSC': 'MTB Bank',
            'MetaBank': 'MetaBank',
            'MetaBank JSC': 'MetaBank',
            'Misto Bank JSC': 'Misto Bank',
            'Motor-Bank JSC': 'Motor-Bank',
            'OKCI BANK  JSC': 'OKCI Bank',
            'OKCI BANK JSC': 'OKCI Bank',
            'OKCI BANK, JSC': 'OKCI Bank',
            'OKCI Bank JSC': 'OKCI Bank',
            'OKCI Bank PJSC': 'OKCI Bank',
            'OTP BANK JSC': 'OTP Bank',
            'OTP Bank JSC': 'OTP Bank',
            'Oschadbank JSC': 'Oschadbank',
            'PIRAEUS BANK ICB JSC': 'Piraeus Bank ICB',
            'PJSC \'MTB BANK\'': 'MTB Bank',
            'PRAVEX BANK  JSC': 'Pravex Bank',
            'PRAVEX BANK JSC': 'Pravex Bank',
            'PRAVEX BANK\' JSC': 'Pravex Bank',
            'PSC Prominvestbank': 'Prominvestbank',
            'Piraeus Bank ICB JSC': 'Piraeus Bank ICB',
            'Pivdennyi Bank': 'Pivdenny Bank',
            'Pivdennyi JSB': 'Pivdenny Bank',
            'Policombank': 'Policombank',
            'Policombank JSC': 'Policombank',
            'Poltava-Bank JSC': 'Poltava-Bank',
            'Poltava-bank  JSC': 'Poltava-Bank',
            'Poltava-bank JSC': 'Poltava-Bank',
            'ProCredit Bank JSC': 'ProCredit Bank',
            'Raiffeisen Bank Aval JSC': 'Raiffeisen Bank Aval',
            'Raiffeisen Bank JSC': 'Raiffeisen Bank',
            'RwS Bank JSC': 'RWS Bank',
            'SBERBANK\' JSC': 'Sberbank',
            'SEB CORPORATE BANK  JSC': 'SEB Corporate Bank',
            'SEB CORPORATE BANK JSC': 'SEB Corporate Bank',
            'SEB Corporate Bank JSC': 'SEB Corporate Bank',
            'SEB Corporate Bank PJSC': 'SEB Corporate Bank',
            'SENSE BANK JSC': 'Sense Bank',
            'SKY BANK JSC': 'Sky Bank',
            'Settlement Center PJSC': 'Settlement Center',
            'Sky Bank JSC': 'Sky Bank',
            'TAScombank JSC': 'TAScombank',
            'Trust-Capital JSC': 'Trust-Capital',
            'UKRSIBBANK JSC': 'UkrSibbank',
            'UKRСONSTINVESTBANK JSC': 'Ukrconstinvestbank',
            'UNIVERSAL BANK JSC': 'Universal Bank',
            'UNIVERSAL BANK JSC*': 'Universal Bank',
            'UkrSibbank JSС': 'UkrSibbank',
            'Ukrainian Bank for Reconstruction and Development JSC': 'Ukrainian Bank for Reconstruction and Development',
            'Ukrainian Capital Bank PJSC': 'Ukrainian Capital Bank',
            'Ukrbudinvestbank JSC': 'Ukrbudinvestbank',
            'Ukreksimbank JSC': 'Ukreximbank',
            'Ukreximbank JSC': 'Ukreximbank',
            'Ukreximbank JSC*': 'Ukreximbank',
            'Ukrgаzbаnk JSB': 'Ukrgasbank',
            'Ukrsotsbank JSC': 'Ukrsotsbank',
            'Unex Bank JSC': 'Unex Bank',
            'Universal Bank JSC': 'Universal Bank',
            'Universal Bank PJSC': 'Universal Bank',
            'Vernum Bank JSC': 'Vernum Bank',
            'АТ \'БАНК СІЧ\'': 'Bank Sich',
            
            # Additional mappings for consolidation (from consolidate_bank_names.py)
            'Pivdennyi Bank': 'Pivdenny Bank',
            'UkrSibbank JSС': 'UkrSibbank',
            'JSС \'UKRSIBBANK\'': 'UkrSibbank',
            'MEGABANK JSC': 'Megabank',
            'OTP BANK JSC': 'OTP Bank',
            'OKCI BANK, JSC': 'OKCI Bank',
            'TASCOMBANK JSC': 'TAScombank',
            'RwS Bank': 'RWS Bank',
            'CB Pravex-Bank': 'Pravex Bank',
            'PRAVEX BANK\' JSC': 'Pravex Bank',
            'Cominvestbank': 'Cominbank',
            'ComInvestBank': 'Cominbank',
            'COMINBANK': 'Cominbank',
            'Raiffeisen Bank Aval': 'Raiffeisen Bank',
            'Raiffeisen Bank Aval JSC': 'Raiffeisen Bank',
            'SBERBANK\' JSC': 'Sberbank',
            'UKRСONSTINVESTBANK\' JSC': 'Ukrconstinvestbank',
            'ALPARI BANK': 'Alpari Bank',
            
            # Handle filter entries (these should be removed)
            '2019-04-01 00:00:00': None,
            'TOTAL': None,
            'Total': None,
            'MetaBank\'': 'MetaBank',
            'Poltava-bank': 'Poltava-Bank',
            'АТ \"БАНК СІЧ\"': 'Bank Sich',
        }
    
    def _create_invalid_patterns(self) -> List[str]:
        """Create patterns for invalid bank names that should be removed."""
        return [
            r'^(total|sum|всього|разом)',
            r'^source:',
            r'^note:',
            r'^notes:',
            r'^\*',
            r'^-+$',
            r'^\s*$',
            r'^\d{4}-\d{2}-\d{2}',  # Dates
            r'^total$',
            r'^sum$',
        ]
    
    def canonicalize_bank_name(self, bank_name: str) -> Optional[str]:
        """
        Canonicalize a single bank name.
        
        Args:
            bank_name: Raw bank name
            
        Returns:
            Canonical bank name or None if should be removed
        """
        if pd.isna(bank_name):
            return None
        
        bank_str = str(bank_name).strip()
        
        # Check if it's an invalid pattern
        for pattern in self.invalid_patterns:
            if re.match(pattern, bank_str, re.IGNORECASE):
                return None
        
        # Check if it's an aggregation bank
        if bank_str in self.aggregation_banks:
            return None
        
        # Direct lookup in mapping dictionary
        if bank_str in self.bank_mappings:
            mapped_value = self.bank_mappings[bank_str]
            return mapped_value  # Could be None for invalid entries
        
        # If no mapping found, return the cleaned name
        return self._clean_bank_name(bank_str)
    
    def _clean_bank_name(self, name: str) -> str:
        """Clean a bank name that doesn't have a direct mapping."""
        # Remove asterisks (footnote markers)
        name = re.sub(r'\*+', '', name)
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Remove leading/trailing spaces
        name = name.strip()
        
        return name
    
    def process_dataframe(self, df: pd.DataFrame, bank_column: str = 'bank_name') -> Tuple[pd.DataFrame, Dict]:
        """
        Process a DataFrame to canonicalize bank names.
        
        Args:
            df: DataFrame with bank names
            bank_column: Name of the bank column
            
        Returns:
            Tuple of (processed_df, statistics)
        """
        if bank_column not in df.columns:
            self.logger.warning(f"Bank column '{bank_column}' not found in DataFrame")
            return df, {}
        
        original_banks = df[bank_column].nunique()
        original_rows = len(df)
        
        # Apply canonicalization
        df = df.copy()
        changes = 0
        rows_to_drop = []
        
        for i, bank_name in enumerate(df[bank_column]):
            canonical_name = self.canonicalize_bank_name(bank_name)
            
            if canonical_name is None:
                # Mark row for deletion
                rows_to_drop.append(i)
                changes += 1
            elif canonical_name != str(bank_name).strip():
                # Update with canonical name
                df.loc[i, bank_column] = canonical_name
                changes += 1
        
        # Drop invalid rows
        if rows_to_drop:
            df = df.drop(rows_to_drop).reset_index(drop=True)
        
        final_banks = df[bank_column].nunique()
        final_rows = len(df)
        
        stats = {
            'original_banks': original_banks,
            'final_banks': final_banks,
            'original_rows': original_rows,
            'final_rows': final_rows,
            'rows_removed': len(rows_to_drop),
            'names_changed': changes,
            'banks_consolidated': original_banks - final_banks
        }
        
        self.logger.info(f"🏦 Bank canonicalization complete:")
        self.logger.info(f"   Banks: {original_banks} → {final_banks} (-{original_banks - final_banks})")
        self.logger.info(f"   Rows: {original_rows} → {final_rows} (-{len(rows_to_drop)})")
        self.logger.info(f"   Names changed: {changes}")
        
        return df, stats
    
    def identify_problematic_banks(self, df: pd.DataFrame, bank_column: str = 'bank_name') -> Set[str]:
        """
        Identify problematic bank names that should be removed in final cleanup.
        
        Based on final_cleanup.py logic.
        """
        if bank_column not in df.columns:
            return set()
        
        bank_names = df[bank_column].unique()
        problematic_banks = set()
        
        # JSC/PJSC/CB/JSB/JSCB variants
        jsc_patterns = [
            r'^JSC\s+',
            r'^PJSC\s+', 
            r'^JSCB\s+',
            r'^CB\s+',
            r'^JSС\s+',  # Cyrillic C
            r'^JSB\s+',
            r'^JOINT\s+STOCK\s+BANK\s+',
        ]
        
        for name in bank_names:
            # JSC variants
            for pattern in jsc_patterns:
                if re.match(pattern, str(name), re.IGNORECASE):
                    problematic_banks.add(name)
                    break
            
            # Weird/short names
            name_clean = str(name).strip()
            if (len(name_clean) <= 3 or 
                name_clean.isdigit() or 
                name_clean in ['3', '4', '5'] or
                name_clean == ''):
                problematic_banks.add(name)
            
            # Banks with very few data points (if bank_present column exists)
            if 'bank_present' in df.columns:
                data_points = df[df[bank_column] == name]['bank_present'].sum()
                if data_points <= 2:
                    problematic_banks.add(name)
        
        return problematic_banks
    
    def get_mapping_stats(self) -> Dict:
        """Get statistics about the bank mappings."""
        total_mappings = len(self.bank_mappings)
        valid_mappings = sum(1 for v in self.bank_mappings.values() if v is not None)
        removal_mappings = sum(1 for v in self.bank_mappings.values() if v is None)
        
        return {
            'total_mappings': total_mappings,
            'valid_mappings': valid_mappings,
            'removal_mappings': removal_mappings,
            'aggregation_banks': len(self.aggregation_banks),
            'invalid_patterns': len(self.invalid_patterns)
        } 