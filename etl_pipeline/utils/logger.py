#!/usr/bin/env python3
"""
Logging utility for Ukrainian Bank ETL Pipeline

Provides structured logging with multiple output formats and levels.
Expert implementation with performance considerations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: Optional[str] = None) -> logging.Logger:
    """
    Setup comprehensive logging for the ETL pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Optional log directory
        
    Returns:
        Configured logger instance
    """
    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file or log_dir:
        if not log_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"etl_pipeline_{timestamp}.log"
        
        if log_dir:
            log_path = Path(log_dir) / log_file
        else:
            log_path = Path(log_file)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        print(f"📄 Logging to file: {log_path}")
    
    # Return the root logger
    return root_logger


class ETLLogger:
    """
    Specialized logger for ETL operations with structured logging.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.stats = {
            'errors': 0,
            'warnings': 0,
            'processed_files': 0,
            'processed_records': 0
        }
    
    def log_file_processing(self, filename: str, status: str, records: int = 0):
        """Log file processing with statistics."""
        if status == 'success':
            self.logger.info(f"✅ Processed {filename}: {records} records")
            self.stats['processed_files'] += 1
            self.stats['processed_records'] += records
        elif status == 'error':
            self.logger.error(f"❌ Failed to process {filename}")
            self.stats['errors'] += 1
        elif status == 'warning':
            self.logger.warning(f"⚠️ Warning processing {filename}: {records} records")
            self.stats['warnings'] += 1
            self.stats['processed_files'] += 1
            self.stats['processed_records'] += records
    
    def log_transformation(self, data_type: str, input_records: int, output_records: int):
        """Log transformation statistics."""
        if output_records < input_records:
            self.logger.info(f"🔄 {data_type}: {input_records} → {output_records} records ({input_records - output_records} filtered)")
        else:
            self.logger.info(f"🔄 {data_type}: {input_records} → {output_records} records")
    
    def log_validation(self, data_type: str, issues: list):
        """Log validation results."""
        if issues:
            self.logger.warning(f"⚠️ {data_type} validation issues: {len(issues)}")
            for issue in issues[:5]:  # Log first 5 issues
                self.logger.warning(f"  - {issue}")
            if len(issues) > 5:
                self.logger.warning(f"  ... and {len(issues) - 5} more issues")
        else:
            self.logger.info(f"✅ {data_type} validation passed")
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'errors': 0,
            'warnings': 0,
            'processed_files': 0,
            'processed_records': 0
        } 