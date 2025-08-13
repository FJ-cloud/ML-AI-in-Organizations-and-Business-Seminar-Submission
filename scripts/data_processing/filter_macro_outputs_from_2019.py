import os
import json
import glob
import pandas as pd
from datetime import datetime

BASE_DIR = "/home/fj/UkrainianInsolvencyPredictor"
MACRO_DIR = os.path.join(BASE_DIR, "data", "processed", "macro")
CUTOFF = pd.Timestamp("2019-01-01")


def find_latest_manifest() -> str:
    manifests = sorted(glob.glob(os.path.join(MACRO_DIR, "new_macro_outputs_*.json")))
    if not manifests:
        raise FileNotFoundError("No macro outputs manifest found.")
    return manifests[-1]


def filter_csv_in_place(path: str) -> int:
    if not os.path.exists(path):
        print(f"Skip missing file: {path}")
        return 0
    df = pd.read_csv(path, parse_dates=["date"]) if path.endswith('.csv') else None
    if df is None or df.empty:
        print(f"Empty or unsupported file: {path}")
        return 0
    before = len(df)
    df = df[df["date"] >= CUTOFF].copy()
    df.to_csv(path, index=False)
    after = len(df)
    print(f"Filtered {os.path.basename(path)}: {before} -> {after}")
    return before - after


def main():
    manifest_path = find_latest_manifest()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    removed_total = 0

    # JEHD long/wide
    jehd = manifest.get("jehd")
    if isinstance(jehd, dict):
        removed_total += filter_csv_in_place(jehd.get("long_csv", ""))
        removed_total += filter_csv_in_place(jehd.get("wide_csv", ""))

    # Individual series
    if manifest.get("us_dgs3"):
        removed_total += filter_csv_in_place(manifest["us_dgs3"])
    if manifest.get("ukr_3y"):
        removed_total += filter_csv_in_place(manifest["ukr_3y"])

    # Macro-only datasets
    macro_only = manifest.get("macro_only")
    if isinstance(macro_only, dict):
        removed_total += filter_csv_in_place(macro_only.get("macro_only_wide", ""))
        removed_total += filter_csv_in_place(macro_only.get("macro_only_long", ""))

    print(f"Total rows removed (pre-2019): {removed_total}")
    print(f"Updated files per manifest: {manifest_path}")


if __name__ == "__main__":
    main() 