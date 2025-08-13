#!/usr/bin/env python3
import subprocess
from pathlib import Path

ROOT = Path('/home/fj/UkrainianInsolvencyPredictor')
CSV = sorted((ROOT / 'data/processed/model_ready').glob('model_lag1_*.csv'))[-1]
cmd = [
    'python3', str(ROOT / 'machineLearning/SARIMAX/train_l1_sarimax.py'),
    '--csv', str(CSV),
    '--split_date', '2022-08-31',
    '--sarimax_order', '0,0,0'
]
print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True) 