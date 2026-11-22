"""
Master CLI pipeline runner:
Orchestrates end-to-end execution of all phases of the framework:
1. data: Data acquisition, cleaning, DOI-grouped splitting
2. features: Physics feature engineering & zero-leakage preprocessing
3. forward: Forward surrogate benchmarking (RF, XGBoost, LightGBM, MLP) + SHAP
4. pinn: Residual PINN training, K=5 Deep Ensemble, uncertainty calibration, OOD detection
5. inverse: Constrained cVAE + CFG, Normalizing Flow, Optuna BO refinement, Pareto optimization
"""

import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('Inv_PINN_Pipeline')


def run_cmd(cmd: list):
    logger.info(f"Executing: {' '.join(cmd)}")
    res = subprocess.run([sys.executable] + cmd, check=True)
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="End-to-End Perovskite Inverse Design Pipeline")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'data', 'features', 'forward', 'pinn', 'inverse'],
                        help="Specific pipeline stage to execute (or 'all')")
    parser.add_argument('--processed-dir', type=str, default='data/processed')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    stages = ['data', 'features', 'forward', 'pinn', 'inverse'] if args.stage == 'all' else [args.stage]

    if 'data' in stages:
        logger.info("\n=== STAGE 1: Data Acquisition & DOI-Grouped Splitting ===")
        run_cmd(['scripts/run_data.py', '--processed-dir', args.processed_dir, '--seed', str(args.seed)])

    if 'features' in stages:
        logger.info("\n=== STAGE 2: Physics-Based Feature Engineering ===")
        run_cmd(['scripts/run_features.py', '--processed-dir', args.processed_dir])

    if 'forward' in stages:
        logger.info("\n=== STAGE 3: Forward Surrogate Benchmark & Explainability ===")
        run_cmd(['scripts/run_forward.py', '--processed-dir', args.processed_dir, '--results-dir', args.results_dir])

    if 'pinn' in stages:
        logger.info("\n=== STAGE 4: PINN Deep Ensemble & Uncertainty Calibration ===")
        run_cmd(['scripts/run_pinn.py', '--processed-dir', args.processed_dir, '--results-dir', args.results_dir,
                 '--epochs', str(args.epochs)])

    if 'inverse' in stages:
        logger.info("\n=== STAGE 5 & 6: Generative Inverse Design & Multi-Objective Pareto Refinement ===")
        run_cmd(['scripts/run_inverse.py', '--processed-dir', args.processed_dir, '--results-dir', args.results_dir,
                 '--epochs', str(args.epochs), '--seed', str(args.seed)])

    logger.info("\nPipeline execution complete! All metrics and publication figures are in the results/ directory.")


if __name__ == '__main__':
    main()
