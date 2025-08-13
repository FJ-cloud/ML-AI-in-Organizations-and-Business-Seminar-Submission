#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime

BASE = Path('/home/fj/UkrainianInsolvencyPredictor')
DIRS = {
    "raw_balance_sheets": BASE / 'data' / 'raw_balance_sheets',
    "macro_raw": BASE / 'data' / 'MacroEconomicData',
    "new_data": BASE / 'data' / 'newData',
    "macro_processed": BASE / 'data' / 'processed' / 'macro',
    "merged": BASE / 'data' / 'processed' / 'merged',
    "model_ready": BASE / 'data' / 'processed' / 'model_ready',
    "output_final": BASE / 'output_final',
}

EXTRA = {
    "etl_logs": [BASE / 'imf_data_processing.log', BASE / 'macro_data_processing.log']
}


def list_dir(d: Path):
    out = []
    if not d.exists():
        return out
    for p in sorted(d.iterdir()):
        if p.is_file():
            try:
                stat = p.stat()
                out.append({
                    "path": str(p),
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "name": p.name,
                    "ext": p.suffix.lower(),
                })
            except Exception:
                continue
    return out


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    inventory = {
        "generated_at": datetime.now().isoformat(),
        "base": str(BASE),
        "directories": {},
        "extra_files": {},
    }
    for key, d in DIRS.items():
        inventory["directories"][key] = {
            "dir": str(d),
            "exists": d.exists(),
            "files": list_dir(d),
        }
    for key, files in EXTRA.items():
        items = []
        for f in files:
            if f.exists():
                stat = f.stat()
                items.append({
                    "path": str(f),
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "name": f.name,
                })
        inventory["extra_files"][key] = items

    out_path = BASE / f'ETL_INVENTORY_{ts}.json'
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(inventory, fh, indent=2)
    print(f"Wrote inventory: {out_path}")


if __name__ == '__main__':
    main() 