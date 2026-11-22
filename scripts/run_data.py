"""
CLI script: Stage 1 - Data Acquisition, Cleaning & DOI-Grouped Splitting.
"""

import argparse
import json
import logging
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    download_dataset,
    parse_and_standardize,
    clean_and_filter,
    doi_grouped_train_val_test_split
)
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_data')


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Perovskite Data Acquisition & Splitting")
    parser.add_argument('--raw-dir', type=str, default='data/raw')
    parser.add_argument('--processed-dir', type=str, default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    proc_dir = Path(args.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download / Load
    raw_csv = download_dataset(raw_dir)
    df_raw = pd.read_csv(raw_csv, low_memory=False)
    logger.info(f"Loaded raw dataset: {len(df_raw)} records.")

    # 2. Parse & Standardize
    df_std = parse_and_standardize(df_raw)

    # 3. Clean & Filter
    df_clean = clean_and_filter(df_std)

    # 4. Strict DOI-grouped split (70 / 15 / 15)
    train_df, val_df, test_df = doi_grouped_train_val_test_split(
        df_clean, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )

    # Save CSVs
    train_df.to_csv(proc_dir / 'train.csv', index=False)
    val_df.to_csv(proc_dir / 'val.csv', index=False)
    test_df.to_csv(proc_dir / 'test.csv', index=False)

    stats = {
        'raw_rows': len(df_raw),
        'clean_rows': len(df_clean),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'train_dois': int(train_df['Ref_DOI'].nunique()),
        'val_dois': int(val_df['Ref_DOI'].nunique()),
        'test_dois': int(test_df['Ref_DOI'].nunique()),
    }
    with open(proc_dir / 'data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Data stage successfully completed. Stats: {stats}")


if __name__ == '__main__':
    main()
