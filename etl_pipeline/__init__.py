"""
Ukrainian Bank Insolvency Prediction - ETL Pipeline

A comprehensive, production-ready ETL pipeline that processes Ukrainian bank
regulatory data from raw Excel files to a clean multivariate panel dataset.

Author: Expert ETL Developer
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Expert ETL Developer"

from .core.orchestrator import ETLOrchestrator

__all__ = ['ETLOrchestrator'] 