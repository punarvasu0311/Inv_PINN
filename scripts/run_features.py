"""
CLI script: Stage 2 - Physics-Based Feature Engineering & Leakage-Free Preprocessing.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.physics import PhysicsFeatureEngineer
from src.data.dataset import LeakageFreePreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_features')


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Physics Feature Engineering")
    parser.add_argument('--processed-dir', type=str, default='data/processed')
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)
    train_df = pd.read_csv(proc_dir / 'train.csv')
    val_df = pd.read_csv(proc_dir / 'val.csv')
    test_df = pd.read_csv(proc_dir / 'test.csv')

    logger.info("Computing physics features for train, val, and test partitions...")
    pfe = PhysicsFeatureEngineer()
    train_feat = pfe.transform_df(train_df)
    val_feat = pfe.transform_df(val_df)
    test_feat = pfe.transform_df(test_df)

    # Save feature-enriched DataFrames
    train_feat.to_csv(proc_dir / 'train_features.csv', index=False)
    val_feat.to_csv(proc_dir / 'val_features.csv', index=False)
    test_feat.to_csv(proc_dir / 'test_features.csv', index=False)

    # Determine input features
    comp_cols = ['x_FA', 'x_MA', 'x_Cs', 'x_Pb', 'x_Sn', 'x_I', 'x_Br', 'x_Cl']
    phys_cols = ['r_A_eff', 'r_B_eff', 'r_X_eff', 'tau', 'mu', 'tau_mu', 'S_A',
                 'Delta_chi_BX', 'Delta_chi_AX', 'Eg_vegard', 'eta_SQ', 'Voc_SQ', 'Jsc_max_SQ',
                 'x_FA_x_I', 'x_Sn_x_I', 'x_FA_x_Br', 'x_Pb_x_I']
    device_cols = ['absorber_thickness', 'etl_thickness', 'htl_thickness', 'Delta_H', 'Delta_L', 'is_pin']

    feature_cols = comp_cols + phys_cols + device_cols

    # Fit preprocessor strictly on training data only
    preprocessor = LeakageFreePreprocessor()
    preprocessor.fit(train_feat, feature_cols=feature_cols)
    preprocessor.save(proc_dir / 'preprocessor.pkl')

    # Transform all splits
    X_train, y_train = preprocessor.transform(train_feat)
    X_val, y_val = preprocessor.transform(val_feat)
    X_test, y_test = preprocessor.transform(test_feat)

    # Save standardized numpy arrays
    np.save(proc_dir / 'X_train.npy', X_train)
    np.save(proc_dir / 'y_train.npy', y_train)
    np.save(proc_dir / 'X_val.npy', X_val)
    np.save(proc_dir / 'y_val.npy', y_val)
    np.save(proc_dir / 'X_test.npy', X_test)
    np.save(proc_dir / 'y_test.npy', y_test)

    # Save bandgap and Jsc_max arrays for physics losses
    np.save(proc_dir / 'Eg_train.npy', train_feat['Eg_vegard'].values.astype(np.float32))
    np.save(proc_dir / 'Jsc_max_train.npy', train_feat['Jsc_max_SQ'].values.astype(np.float32))
    np.save(proc_dir / 'Eg_test.npy', test_feat['Eg_vegard'].values.astype(np.float32))

    feat_info = {
        'n_features': X_train.shape[1],
        'feature_names': feature_cols,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
    }
    with open(proc_dir / 'feature_info.json', 'w') as f:
        json.dump(feat_info, f, indent=2)

    logger.info(f"Feature engineering completed: {X_train.shape[1]} features extracted.")


if __name__ == '__main__':
    main()
