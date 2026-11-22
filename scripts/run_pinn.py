"""
CLI script: Stage 4 - Residual Physics-Informed Deep Ensemble (K=5).
Trains 5 independently-seeded PINN models with curriculum physics loss,
uncertainty calibration on validation set, and OOD detection.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.pinn import PINNDeepEnsemble, UncertaintyCalibrator, OODDetector
from src.evaluation.metrics import compute_forward_benchmark, compute_physics_violation_rate
from src.evaluation.plotting import plot_forward_parity, plot_uncertainty_calibration

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_pinn')


def main():
    parser = argparse.ArgumentParser(description="Stage 4: PINN Deep Ensemble Training & Calibration")
    parser.add_argument('--processed-dir', type=str, default='data/processed')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--k-ensemble', type=int, default=5)
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)
    res_dir = Path(args.results_dir)
    metrics_dir = res_dir / 'metrics'
    fig_dir = res_dir / 'figures'
    ckpt_dir = res_dir / 'checkpoints' / 'pinn_ensemble'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(proc_dir / 'X_train.npy')
    y_train = np.load(proc_dir / 'y_train.npy')
    X_val = np.load(proc_dir / 'X_val.npy')
    y_val = np.load(proc_dir / 'y_val.npy')
    X_test = np.load(proc_dir / 'X_test.npy')
    y_test = np.load(proc_dir / 'y_test.npy')

    Eg_train = np.load(proc_dir / 'Eg_train.npy')
    Jsc_max_train = np.load(proc_dir / 'Jsc_max_train.npy')
    Eg_test = np.load(proc_dir / 'Eg_test.npy')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using compute device: {device}")

    # 1. Initialize and train K=5 Deep Ensemble
    ensemble = PINNDeepEnsemble(
        input_dim=X_train.shape[1],
        K=args.k_ensemble,
        hidden_dim=256,
        n_blocks=3,
        dropout=0.1,
        device=device
    )
    ensemble.train_ensemble(
        X_train=X_train,
        y_train=y_train,
        Eg_train=Eg_train,
        Jsc_max_train=Jsc_max_train,
        epochs=args.epochs,
        batch_size=128,
        lr=1.2e-3
    )
    ensemble.save(str(ckpt_dir))

    # 2. Uncertainty calibration on validation set
    mu_val, sigma_val = ensemble.predict(X_val)
    calibrator = UncertaintyCalibrator()
    calibrator.fit(y_val, mu_val, sigma_val)

    # 3. Predict on test set
    mu_test, sigma_test = ensemble.predict(X_test)
    sigma_cal = calibrator.calibrate(sigma_test)

    # 4. Fit OOD detector on training set
    ood_detector = OODDetector()
    ood_detector.fit(X_train)
    test_ood_scores = ood_detector.score(X_test)

    # 5. Evaluate benchmark metrics
    m_pinn = compute_forward_benchmark(y_test, mu_test)
    v_pinn = compute_physics_violation_rate(mu_test, Eg_test)
    cov_dict = calibrator.evaluate_coverage(y_test, mu_test, sigma_cal)
    nll = calibrator.compute_nll(y_test, mu_test, sigma_cal)

    pinn_results = {
        **m_pinn,
        **v_pinn,
        'NLL': nll,
        'calibration_coverage': cov_dict,
        'scale_factors': calibrator.scale_factors.tolist(),
        'mean_test_ood_score': float(np.mean(test_ood_scores)),
        'test_ood_pct': float(np.mean(ood_detector.is_ood(X_test)) * 100.0)
    }

    logger.info(f"PINN Ensemble Mean R^2: {m_pinn['Mean_R2']:.4f}, PCE R^2: {m_pinn['PCE_R2']:.4f}")
    logger.info(f"Total physics violation rate: {v_pinn['total_physics_violation_pct']:.2f}% (vs data baseline)")
    logger.info(f"90% Confidence coverage: {cov_dict['coverage_90%']}")

    with open(metrics_dir / 'pinn_ensemble_metrics.json', 'w') as f:
        json.dump(pinn_results, f, indent=2)

    # 6. Plots
    plot_forward_parity(y_test, mu_test, save_path=fig_dir / 'parity_pinn_ensemble.png')
    plot_uncertainty_calibration(cov_dict, save_path=fig_dir / 'uncertainty_calibration.png')
    logger.info(f"Saved parity and calibration figures in {fig_dir}")


if __name__ == '__main__':
    main()
