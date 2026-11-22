"""
CLI script: Stage 5 & 6 - Constrained Generative Inverse Design & Multi-Objective Pareto Refinement.
Compares:
1. Random Sampling
2. Unguided cVAE (w = 0)
3. Guided cVAE + CFG (w = 1.5, and sweep w in {0, 0.5, 1.0, 1.5, 2.0, 3.0})
4. Conditional Normalizing Flow (RealNVP)
5. Generator + Bayesian Optimization (Optuna) Refinement
Extracts multi-objective Pareto front (PCE vs. Pb vs. Uncertainty) and computes Pareto Hypervolume.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import LeakageFreePreprocessor
from src.models.pinn import PINNDeepEnsemble, UncertaintyCalibrator, OODDetector
from src.inverse.cvae import ConstrainedFiLMcVAE, compute_cvae_loss
from src.inverse.flow import ConditionalRealNVP, transform_to_unconstrained
from src.inverse.refinement import (
    candidate_dict_to_dataframe,
    CandidateEvaluator,
    bayesian_refinement_optuna,
    extract_pareto_front_3d
)
from src.evaluation.metrics import (
    compute_chemical_validity,
    compute_uniqueness,
    compute_novelty,
    compute_diversity,
    compute_hit_rate,
    compute_ks_tests,
    compute_pareto_hypervolume
)
from src.evaluation.plotting import (
    plot_cfg_guidance_sweep,
    plot_marginal_ks_checks,
    plot_ternary_a_site,
    plot_pareto_fronts
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('run_inverse')


def train_cvae(cvae, train_loader, epochs=50, device='cpu'):
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=1.2e-3, weight_decay=1e-5)
    cvae.train()
    for epoch in range(epochs):
        for bx, bc in train_loader:
            bx, bc = bx.to(device), bc.to(device)
            optimizer.zero_grad()
            out = cvae(bx, bc)
            loss, _ = compute_cvae_loss(out, bx, epoch=epoch, total_epochs=epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cvae.parameters(), 3.0)
            optimizer.step()
    return cvae


def train_flow(flow, train_loader, epochs=50, device='cpu'):
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1.0e-3, weight_decay=1e-5)
    flow.train()
    for epoch in range(epochs):
        for bx, bc in train_loader:
            bx, bc = bx.to(device), bc.to(device)
            optimizer.zero_grad()
            log_px, _ = flow.log_prob(bx, bc)
            loss = -torch.mean(log_px)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 3.0)
            optimizer.step()
    return flow


def main():
    parser = argparse.ArgumentParser(description="Stage 5 & 6: Generative Inverse Design & Optimization")
    parser.add_argument('--processed-dir', type=str, default='data/processed')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    proc_dir = Path(args.processed_dir)
    res_dir = Path(args.results_dir)
    metrics_dir = res_dir / 'metrics'
    fig_dir = res_dir / 'figures'
    ckpt_dir = res_dir / 'checkpoints'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data & fitted preprocessor
    train_features_df = pd.read_csv(proc_dir / 'train_features.csv')
    preprocessor = LeakageFreePreprocessor.load(proc_dir / 'preprocessor.pkl')
    X_train = np.load(proc_dir / 'X_train.npy')
    y_train = np.load(proc_dir / 'y_train.npy')

    # Load PINN deep ensemble and calibrate
    ensemble = PINNDeepEnsemble(input_dim=X_train.shape[1], K=5, device=device)
    ensemble.load(str(ckpt_dir / 'pinn_ensemble'))

    X_val = np.load(proc_dir / 'X_val.npy')
    y_val = np.load(proc_dir / 'y_val.npy')
    mu_val, sigma_val = ensemble.predict(X_val)
    calibrator = UncertaintyCalibrator().fit(y_val, mu_val, sigma_val)

    ood_detector = OODDetector().fit(X_train)
    evaluator = CandidateEvaluator(ensemble, preprocessor, ood_detector, calibrator)

    # Prepare design matrix for generative modeling
    # Design variables: [x_FA, x_MA, x_Cs, x_Pb, x_Sn, x_I, x_Br, x_Cl, abs_t, etl_t, htl_t, Delta_H, Delta_L, is_pin]
    design_cols = ['x_FA', 'x_MA', 'x_Cs', 'x_Pb', 'x_Sn', 'x_I', 'x_Br', 'x_Cl',
                   'absorber_thickness', 'etl_thickness', 'htl_thickness', 'Delta_H', 'Delta_L', 'is_pin']
    X_design = train_features_df[design_cols].copy().values

    # Sample training Pb_max per sample from uniform(x_Pb, 1.0) so model actually learns condition (Bug 4 fix)
    pb_train = X_design[:, 3]
    pb_max_train = np.random.uniform(pb_train, 1.0, size=len(pb_train))
    c_train = np.column_stack([y_train, pb_max_train])

    # Standardize condition vector consistently across training and inference (Bug 1 fix)
    c_mean = np.mean(c_train, axis=0)
    c_std = np.std(c_train, axis=0) + 1e-4
    c_train_norm = (c_train - c_mean) / c_std

    # Prepare cVAE dataset (scaled thicknesses) and RealNVP dataset (unconstrained logit space)
    X_cvae = X_design.copy()
    X_cvae[:, 8:11] = X_cvae[:, 8:11] / 500.0

    X_flow = transform_to_unconstrained(
        torch.from_numpy(X_design).float(),
        pb_max=torch.from_numpy(pb_max_train).float()
    ).numpy()

    train_ds_cvae = TensorDataset(torch.from_numpy(X_cvae).float(), torch.from_numpy(c_train_norm).float())
    train_loader_cvae = DataLoader(train_ds_cvae, batch_size=128, shuffle=True)

    train_ds_flow = TensorDataset(torch.from_numpy(X_flow).float(), torch.from_numpy(c_train_norm).float())
    train_loader_flow = DataLoader(train_ds_flow, batch_size=128, shuffle=True)

    # 1. Train Constrained FiLM-cVAE
    logger.info("Training Constrained FiLM-cVAE...")
    cvae = ConstrainedFiLMcVAE(design_dim=X_cvae.shape[1], cond_dim=5, latent_dim=16, hidden_dim=256).to(device)
    cvae = train_cvae(cvae, train_loader_cvae, epochs=args.epochs, device=device)
    torch.save(cvae.state_dict(), ckpt_dir / 'cvae_model.pt')

    # 2. Train Conditional RealNVP Normalizing Flow
    logger.info("Training Conditional RealNVP Normalizing Flow...")
    flow = ConditionalRealNVP(design_dim=X_flow.shape[1], cond_dim=5, n_layers=6, hidden_dim=128).to(device)
    flow = train_flow(flow, train_loader_flow, epochs=args.epochs, device=device)
    torch.save(flow.state_dict(), ckpt_dir / 'flow_model.pt')

    # 3. Target Specifications
    # Target 1: High-PCE (90th percentile of training data, Pb_max = 1.0)
    pce_90 = float(np.percentile(y_train[:, 0], 90.0))
    voc_90 = float(np.percentile(y_train[:, 1], 90.0))
    jsc_90 = float(np.percentile(y_train[:, 2], 90.0))
    ff_90 = float(np.percentile(y_train[:, 3], 90.0))
    target_high = np.array([pce_90, voc_90, jsc_90, ff_90])
    c_high_raw = np.array([pce_90, voc_90, jsc_90, ff_90, 1.0])
    c_high = torch.FloatTensor((c_high_raw - c_mean) / c_std)

    # Target 2: Low-Pb target (PCE* = 20.0%, Voc* = 1.05V, Jsc* = 24.5, FF* = 0.78, Pb_max = 0.50)
    target_low_pb = np.array([20.0, 1.05, 24.5, 0.78])
    c_low_pb_raw = np.array([20.0, 1.05, 24.5, 0.78, 0.50])
    c_low_pb = torch.FloatTensor((c_low_pb_raw - c_mean) / c_std)

    # Target 3: Pure-Sn Lead-Free target (PCE* = 18.0%, Voc* = 0.95V, Jsc* = 25.0, FF* = 0.76, Pb_max = 0.0)
    target_sn = np.array([18.0, 0.95, 25.0, 0.76])
    c_sn_raw = np.array([18.0, 0.95, 25.0, 0.76, 0.0])
    c_sn = torch.FloatTensor((c_sn_raw - c_mean) / c_std)

    targets_dict = {
        'High_PCE_90th': (target_high, c_high, 1.0),
        'Low_Pb_Cap_50%': (target_low_pb, c_low_pb, 0.50),
        'Lead_Free_Pure_Sn': (target_sn, c_sn, 0.0)
    }

    # 4. CFG Guidance Scale Sweep on High-PCE Target
    logger.info("Conducting CFG Guidance Scale Sweep w in {0, 0.5, 1.0, 1.5, 2.0, 3.0}...")
    w_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    sweep_hit_rates = []
    for w in w_values:
        cand_raw = cvae.sample(c_high, n_samples=args.n_samples, w=w, Pb_max=1.0)
        cand_df = candidate_dict_to_dataframe(cand_raw, n_samples=args.n_samples)
        evaluated = evaluator.evaluate(cand_df, target_high)
        hr = compute_hit_rate(evaluated['pred_PCE'].values, target_high[0], tolerance=1.5)
        sweep_hit_rates.append(hr)
        logger.info(f"  CFG scale w = {w:3.1f} -> Hit-Rate: {hr:5.1f}%")

    plot_cfg_guidance_sweep(w_values, sweep_hit_rates, optimal_w=1.5, save_path=fig_dir / 'cfg_guidance_sweep.png')

    # 5. Method Comparison Benchmark Table on High-PCE Target
    logger.info("Evaluating comparative benchmark across generative inverse methods...")
    methods_data = {}

    # Method 1: Random sampling from training data
    rand_idx = np.random.RandomState(42).choice(len(train_features_df), size=args.n_samples, replace=True)
    rand_df = train_features_df.iloc[rand_idx].copy().reset_index(drop=True)
    methods_data['Random_Sampling'] = evaluator.evaluate(rand_df, target_high)

    # Method 2: Unguided cVAE (w = 0)
    cvae_ung_raw = cvae.sample(c_high, n_samples=args.n_samples, w=0.0, Pb_max=1.0)
    cvae_ung_df = candidate_dict_to_dataframe(cvae_ung_raw, n_samples=args.n_samples)
    methods_data['Unguided_cVAE'] = evaluator.evaluate(cvae_ung_df, target_high)

    # Method 3: Guided cVAE + CFG (w = 1.5)
    cvae_cfg_raw = cvae.sample(c_high, n_samples=args.n_samples, w=1.5, Pb_max=1.0)
    cvae_cfg_df = candidate_dict_to_dataframe(cvae_cfg_raw, n_samples=args.n_samples)
    methods_data['cVAE_plus_CFG'] = evaluator.evaluate(cvae_cfg_df, target_high)

    # Method 4: Conditional Normalizing Flow
    flow_raw = flow.sample(c_high, n_samples=args.n_samples, Pb_max=1.0)
    flow_df = candidate_dict_to_dataframe(flow_raw, n_samples=args.n_samples)
    methods_data['Conditional_Flow'] = evaluator.evaluate(flow_df, target_high)

    # Method 5: cVAE + Bayesian Optimization Refinement
    top_cvae_candidates = methods_data['cVAE_plus_CFG'].sort_values('composite_score').head(5)
    refined_df = bayesian_refinement_optuna(top_cvae_candidates, target_high, evaluator, Pb_max=1.0, n_trials=60)
    methods_data['cVAE_plus_BO_Refinement'] = refined_df

    # Compute comprehensive metrics table
    benchmark_table = {}
    train_manifold_features = X_train

    for m_name, df_res in methods_data.items():
        pred_pce = df_res['pred_PCE'].values
        mae_target = float(np.mean(np.abs(pred_pce - target_high[0])))
        rmse_target = float(np.sqrt(np.mean((pred_pce - target_high[0]) ** 2)))
        hr = compute_hit_rate(pred_pce, target_high[0], tolerance=1.5)
        chem_valid = compute_chemical_validity(df_res)
        
        # Array of numeric design columns
        cand_arr, _ = preprocessor.transform(df_res)
        uniq = compute_uniqueness(cand_arr)
        mean_nn, pct_novel = compute_novelty(cand_arr, train_manifold_features)
        div = compute_diversity(cand_arr)

        # Multi-objective Pareto front & Hypervolume
        pareto_sub, p_mask = extract_pareto_front_3d(df_res)
        # Objectives to minimize: [target_error, x_Pb, mean_unc]
        pts = np.column_stack([
            np.abs(df_res['pred_PCE'].values - target_high[0]),
            df_res['x_Pb'].values,
            df_res['mean_unc'].values
        ])
        ref_pt = np.array([15.0, 1.10, 25.0])
        hv = compute_pareto_hypervolume(pts[p_mask], ref_pt)

        benchmark_table[m_name] = {
            'Target_MAE': round(mae_target, 3),
            'Target_RMSE': round(rmse_target, 3),
            'Hit_Rate_Pct': round(hr, 2),
            'Chemical_Validity_Pct': round(chem_valid, 2),
            'Uniqueness_Pct': round(uniq, 2),
            'Mean_NN_Distance': round(mean_nn, 3),
            'Pct_Beyond_Train_Median': round(pct_novel, 2),
            'Diversity': round(div, 3),
            'Mean_OOD_Score': round(float(df_res['ood_score'].mean()), 3),
            'Mean_Uncertainty': round(float(df_res['mean_unc'].mean()), 3),
            'Mean_Pb_Fraction': round(float(df_res['x_Pb'].mean()), 3),
            'Pareto_Candidates_Count': int(np.sum(p_mask)),
            'Pareto_Hypervolume': round(hv, 4)
        }

    with open(metrics_dir / 'inverse_design_benchmark_table.json', 'w') as f:
        json.dump(benchmark_table, f, indent=2)
    logger.info("Saved comprehensive inverse benchmark table.")

    # 6. Low-Pb and Lead-Free Inverse Generations
    for t_name, (tgt_pv, c_vec, pb_cap) in [('Low_Pb_Cap_50%', (target_low_pb, c_low_pb, 0.50)),
                                            ('Lead_Free_Pure_Sn', (target_sn, c_sn, 0.0))]:
        cand_raw = cvae.sample(c_vec, n_samples=args.n_samples, w=1.5, Pb_max=pb_cap)
        c_df = candidate_dict_to_dataframe(cand_raw, n_samples=args.n_samples)
        evaluated = evaluator.evaluate(c_df, tgt_pv)
        # Verify hard constraint: x_Pb <= Pb_max
        max_pb_found = float(evaluated['x_Pb'].max())
        assert max_pb_found <= pb_cap + 1e-4, f"Violation of hard lead cap in {t_name}: max Pb={max_pb_found}"
        evaluated.to_csv(proc_dir / f'candidates_{t_name}.csv', index=False)
        logger.info(f"Target {t_name} evaluated: mean predicted PCE = {evaluated['pred_PCE'].mean():.2f}%, max Pb = {max_pb_found:.4f}")

    # 7. Generate Publication Figures
    logger.info("Generating publication figures...")
    best_df = methods_data['cVAE_plus_CFG']
    pareto_df, pareto_mask = extract_pareto_front_3d(best_df)

    plot_marginal_ks_checks(train_features_df, best_df, save_path=fig_dir / 'marginal_ks_checks.png')
    plot_ternary_a_site(best_df, highlight_pareto_mask=pareto_mask, save_path=fig_dir / 'ternary_a_site_composition.png')
    plot_pareto_fronts(best_df, pareto_mask=pareto_mask, save_path=fig_dir / 'pareto_tradeoffs.png')

    logger.info("Inverse design stage completed successfully.")


if __name__ == '__main__':
    main()
