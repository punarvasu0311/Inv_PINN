"""
CLI script: Stage 3 - Forward Surrogate Benchmark (RF, XGBoost, LightGBM, Data-only MLP).
Evaluates all models on identical DOI-grouped split with SHAP explainability.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.forward import ForwardRF, ForwardXGBoost, ForwardLightGBM, ForwardMLPWrapper, compute_metrics
from src.evaluation.metrics import compute_physics_violation_rate
from src.evaluation.plotting import plot_forward_parity

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_forward')


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Forward Surrogate Benchmark")
    parser.add_argument('--processed-dir', type=str, default='data/processed')
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)
    res_dir = Path(args.results_dir)
    metrics_dir = res_dir / 'metrics'
    fig_dir = res_dir / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(proc_dir / 'X_train.npy')
    y_train = np.load(proc_dir / 'y_train.npy')
    X_test = np.load(proc_dir / 'X_test.npy')
    y_test = np.load(proc_dir / 'y_test.npy')
    Eg_test = np.load(proc_dir / 'Eg_test.npy')

    with open(proc_dir / 'feature_info.json') as f:
        feat_info = json.load(f)
    feature_names = feat_info['feature_names']

    benchmark_results = {}

    # 1. Random Forest
    logger.info("Training Multi-Output Random Forest...")
    rf = ForwardRF(n_estimators=250, max_depth=16, random_state=42).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    m_rf = compute_metrics(y_test, pred_rf)
    v_rf = compute_physics_violation_rate(pred_rf, Eg_test)
    benchmark_results['RandomForest'] = {**m_rf, **v_rf}
    logger.info(f"RF Mean R^2: {m_rf['Mean_R2']:.4f}, Total Violations: {v_rf['total_physics_violation_pct']:.2f}%")

    # 2. XGBoost
    logger.info("Training Multi-Output XGBoost...")
    xgb_model = ForwardXGBoost(n_estimators=350, max_depth=6, learning_rate=0.08, random_state=42).fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)
    m_xgb = compute_metrics(y_test, pred_xgb)
    v_xgb = compute_physics_violation_rate(pred_xgb, Eg_test)
    benchmark_results['XGBoost'] = {**m_xgb, **v_xgb}
    logger.info(f"XGBoost Mean R^2: {m_xgb['Mean_R2']:.4f}, Total Violations: {v_xgb['total_physics_violation_pct']:.2f}%")
    plot_forward_parity(y_test, pred_xgb, save_path=fig_dir / 'parity_xgboost.png')

    # 3. LightGBM
    logger.info("Training Multi-Output LightGBM...")
    lgb_model = ForwardLightGBM(n_estimators=350, max_depth=8, learning_rate=0.08, random_state=42).fit(X_train, y_train)
    pred_lgb = lgb_model.predict(X_test)
    m_lgb = compute_metrics(y_test, pred_lgb)
    v_lgb = compute_physics_violation_rate(pred_lgb, Eg_test)
    benchmark_results['LightGBM'] = {**m_lgb, **v_lgb}
    logger.info(f"LightGBM Mean R^2: {m_lgb['Mean_R2']:.4f}, Total Violations: {v_lgb['total_physics_violation_pct']:.2f}%")

    # 4. Data-only MLP Baseline
    logger.info("Training Data-only MLP Baseline...")
    mlp = ForwardMLPWrapper(input_dim=X_train.shape[1], epochs=60, batch_size=128, lr=1e-3).fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)
    m_mlp = compute_metrics(y_test, pred_mlp)
    v_mlp = compute_physics_violation_rate(pred_mlp, Eg_test)
    benchmark_results['DataOnlyMLP'] = {**m_mlp, **v_mlp}
    logger.info(f"MLP Mean R^2: {m_mlp['Mean_R2']:.4f}, Total Violations: {v_mlp['total_physics_violation_pct']:.2f}%")

    # 5. SHAP feature importance on primary target PCE
    logger.info("Computing SHAP feature importance for XGBoost...")
    try:
        sample_indices = np.random.RandomState(42).choice(len(X_test), size=min(400, len(X_test)), replace=False)
        _, mean_abs_shap = xgb_model.explain_shap(X_test[sample_indices], feature_names)
        top_indices = np.argsort(mean_abs_shap)[::-1][:15]
        top_shap = {feature_names[i]: float(mean_abs_shap[i]) for i in top_indices}
        benchmark_results['SHAP_Top_Features'] = top_shap
        logger.info(f"Top 5 SHAP drivers of PCE: {list(top_shap.keys())[:5]}")
    except Exception as e:
        logger.warning(f"SHAP explanation computation skipped: {e}")

    # Save benchmark table
    with open(metrics_dir / 'forward_surrogate_benchmark.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(f"Forward benchmark saved to {metrics_dir / 'forward_surrogate_benchmark.json'}")


if __name__ == '__main__':
    main()
