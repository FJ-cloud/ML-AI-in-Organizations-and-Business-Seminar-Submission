#!/usr/bin/env python3
"""
Configuration Manager for Ukrainian Bank ETL Pipeline

Handles configuration loading, validation, and defaults.
Expert implementation with comprehensive error handling.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


class ConfigManager:
    """
    Configuration manager with defaults and validation.
    
    Provides a unified interface for accessing configuration values
    with intelligent defaults and type validation.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else None
        self._config = {}
        self._defaults = self._get_default_config()
        
        # Load configuration
        if self.config_path and self.config_path.exists():
            self._load_config()
        else:
            self.logger.info("Using default configuration (no config file provided)")
            self._config = self._defaults.copy()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration based on working pipeline_full_data.yml."""
        return {
            # Categories with working patterns
            'categories': {
                'financial_results': {
                    'patterns': [
                        r"(?i)^financial[ _]+results?_?$",
                        r"(?i)^fin[ _]*results?_?$",
                        r"(?i)^financial.*res_?$"
                    ]
                },
                'assets': {
                    'patterns': [
                        r"(?i)^assets?$",
                        r"(?i)^total[ _]+assets?$"
                    ]
                },
                'liabilities': {
                    'patterns': [
                        r"(?i)^liabilities?$",
                        r"(?i)^total[ _]+liab"
                    ]
                },
                'equity': {
                    'patterns': [
                        r"(?i)^equity$",
                        r"(?i)^shareholders?[ _]+equity$",
                        r"(?i)^total[ _]+equity$"
                    ]
                }
            },
            
            # Data cleaning rules from working config
            'cleaning': {
                'header_rows': 4,
                'skip_footer': 0,
                'drop_rows': [
                    0,
                    {'pattern': '^Source:'}
                ],
                'drop_columns': [
                    {'pattern': r'^Unnamed:\s*\d+_level_\d+$'},
                    {'pattern': r'^Unnamed$'},
                    'Notes',
                    '№',
                    'NKB'
                ],
                'row_filters': [
                    {
                        'column': 'bank',
                        'patterns': [
                            r'(?i)^source:.*',
                            r'(?i)^note:.*',
                            r'(?i)^notes:.*',
                            r'(?i)^\*.*',
                            r'(?i)^-+$',
                            r'(?i)^\s*$',
                            r'(?i)^total[ _]+number[ _]+of[ _]+banks.*',
                            r'(?i)^grand[ _]+total.*'
                        ]
                    }
                ],
                'dtype': {}
            },
            
            # Working fuzzy threshold
            'fuzzy_threshold': 80,
            
            # Processing settings
            'output_dir': 'output_final',
            'save_intermediate': True,
            'save_metadata': True,
            'compression': None,
            
            # Validation settings
            'min_banks_per_file': 10,
            'max_banks_per_file': 200,
            'required_columns': ['bank_name', 'date', 'year_month'],
            
            # Failure detection
            'failure_detection': {
                'min_consecutive_missing': 6
            }
        }
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            self.logger.info(f"Loading configuration from {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            if not isinstance(loaded_config, dict):
                raise ValueError("Configuration file must contain a dictionary")
            
            # Merge with defaults (loaded config takes precedence)
            self._config = self._defaults.copy()
            self._merge_config(self._config, loaded_config)
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            self.logger.info("Using default configuration")
            self._config = self._defaults.copy()
    
    def _merge_config(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Support dot notation for nested keys
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary with section configuration
        """
        return self.get(section, {})
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration and return validation results.
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        # Check required sections
        required_sections = ['sheet_patterns', 'failure_detection']
        for section in required_sections:
            if section not in self._config:
                issues.append(f"Missing required section: {section}")
        
        # Validate numeric values
        numeric_configs = {
            'fuzzy_threshold': (0, 100),
            'min_banks_per_file': (1, 1000),
            'max_banks_per_file': (1, 1000),
            'canonicalization_threshold': (0, 100)
        }
        
        for key, (min_val, max_val) in numeric_configs.items():
            value = self.get(key)
            if value is not None:
                if not isinstance(value, (int, float)):
                    issues.append(f"{key} must be numeric, got {type(value)}")
                elif not (min_val <= value <= max_val):
                    issues.append(f"{key} must be between {min_val} and {max_val}, got {value}")
        
        # Validate paths
        output_dir = self.get('output_dir')
        if output_dir:
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.append(f"Cannot create output directory {output_dir}: {str(e)}")
        
        # Validate log level
        log_level = self.get('log_level', '').upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            warnings.append(f"Invalid log level: {log_level}. Using INFO.")
            self.set('log_level', 'INFO')
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def save(self, output_path: Optional[Union[str, Path]] = None):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration (defaults to original path)
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        if not save_path:
            raise ValueError("No output path specified and no original config path available")
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(path={self.config_path}, sections={list(self._config.keys())})" 