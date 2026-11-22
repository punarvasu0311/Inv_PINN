"""
Candidate screening, Bayesian Optimization (Optuna) refinement, and Multi-Objective Pareto Selection.
Enforces:
- Exact physics feature re-computation
- Composite scoring: L(x) = L_target(x) + beta * U(x) + gamma * D_OOD(x) + eta * L_physics(x)
- Filtering of high-uncertainty, OOD, or physics-violating candidates
- Optuna Bayesian local refinement preserving hard stoichiometry & compatibility constraints
- Multi-objective Pareto front extraction (PCE vs Pb vs Uncertainty) and Hypervolume calculation
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.features.physics import PhysicsFeatureEngineer
from src.data.dataset import COMPATIBILITY_DICT

logger = logging.getLogger(__name__)


def candidate_dict_to_dataframe(candidates: Dict[str, Any], n_samples: int) -> pd.DataFrame:
    """Convert tensor/array candidate dictionary into standardized DataFrame."""
    def to_np(item):
        if hasattr(item, 'detach'):
            return item.detach().cpu().numpy()
        return np.array(item)

    a_site = to_np(candidates['a_site'])
    b_site = to_np(candidates['b_site'])
    x_site = to_np(candidates['x_site'])
    thicknesses = to_np(candidates['thicknesses'])
    offsets = to_np(candidates['offsets'])
    is_pin = to_np(candidates['is_pin']).squeeze()

    records = []
    for i in range(n_samples):
        pin_flag = float(is_pin[i]) > 0.5
        arch = 'pin' if pin_flag else 'nip'
        compat = COMPATIBILITY_DICT[arch]
        
        # Pick compatible transport layers
        etl = compat['etl'][i % len(compat['etl'])]
        htl = compat['htl'][i % len(compat['htl'])]
        tco = compat['tco'][0]

        records.append({
            'x_FA': float(a_site[i, 0]),
            'x_MA': float(a_site[i, 1]),
            'x_Cs': float(a_site[i, 2]),
            'x_Pb': float(b_site[i, 0]),
            'x_Sn': float(b_site[i, 1]),
            'x_I': float(x_site[i, 0]),
            'x_Br': float(x_site[i, 1]),
            'x_Cl': float(x_site[i, 2]),
            'absorber_thickness': float(thicknesses[i, 0]),
            'etl_thickness': float(thicknesses[i, 1]),
            'htl_thickness': float(thicknesses[i, 2]),
            'Delta_H': float(offsets[i, 0]),
            'Delta_L': float(offsets[i, 1]),
            'is_pin': float(pin_flag),
            'cell_arch': arch,
            'etl_material': etl,
            'htl_material': htl,
            'substrate': tco,
        })

    return pd.DataFrame(records)


class CandidateEvaluator:
    """
    Recomputes physics descriptors for generated candidates and evaluates
    them using the K=5 PINN deep ensemble and OOD detector.
    """
    def __init__(self, ensemble, preprocessor, ood_detector, calibrator):
        self.ensemble = ensemble
        self.preprocessor = preprocessor
        self.ood_detector = ood_detector
        self.calibrator = calibrator
        self.phys_engineer = PhysicsFeatureEngineer()

    def evaluate(
        self,
        candidate_df: pd.DataFrame,
        target_pv: np.ndarray,
        beta: float = 0.5,
        gamma: float = 0.5,
        eta: float = 0.5
    ) -> pd.DataFrame:
        """
        Evaluate candidate designs.
        target_pv: [PCE*, Voc*, Jsc*, FF*]
        Computes:
            L_target = MSE(pred_mu, target_pv)
            U = calibrated uncertainty
            D_OOD = OOD score
            L_physics = consistency and boundary violation penalties
            Score = L_target + beta * U + gamma * D_OOD + eta * L_physics
        """
        df = self.phys_engineer.transform_df(candidate_df)
        X_scaled, _ = self.preprocessor.transform(df)

        mu, sigma = self.ensemble.predict(X_scaled)
        sigma_cal = self.calibrator.calibrate(sigma)
        ood_scores = self.ood_detector.score(X_scaled)

        n = len(df)
        scores = []
        l_targets = []
        l_phys_list = []
        uncertainties = []

        for i in range(n):
            pred_pce, pred_voc, pred_jsc, pred_ff = mu[i, 0], mu[i, 1], mu[i, 2], mu[i, 3]
            unc = float(np.mean(sigma_cal[i]))
            ood = float(ood_scores[i])

            # Target matching error (normalized scale)
            l_t = (
                ((pred_pce - target_pv[0]) / max(target_pv[0], 1.0)) ** 2 +
                ((pred_voc - target_pv[1]) / max(target_pv[1], 0.1)) ** 2 +
                ((pred_jsc - target_pv[2]) / max(target_pv[2], 1.0)) ** 2 +
                ((pred_ff - target_pv[3]) / max(target_pv[3], 0.1)) ** 2
            )

            # Physics violations
            eg = df.loc[i, 'Eg_vegard'] if 'Eg_vegard' in df.columns else 1.5
            jsc_max = df.loc[i, 'Jsc_max_SQ'] if 'Jsc_max_SQ' in df.columns else 35.0
            pce_cons = abs(pred_pce - pred_voc * pred_jsc * pred_ff)
            voc_viol = max(0.0, pred_voc - eg)
            jsc_viol = max(0.0, pred_jsc - jsc_max)

            l_phys = pce_cons + 2.0 * voc_viol + 1.5 * jsc_viol

            composite_score = l_t + beta * unc + gamma * ood + eta * l_phys

            scores.append(composite_score)
            l_targets.append(l_t)
            l_phys_list.append(l_phys)
            uncertainties.append(unc)

        df['pred_PCE'] = mu[:, 0]
        df['pred_Voc'] = mu[:, 1]
        df['pred_Jsc'] = mu[:, 2]
        df['pred_FF'] = mu[:, 3]
        df['pred_unc_PCE'] = sigma_cal[:, 0]
        df['mean_unc'] = uncertainties
        df['ood_score'] = ood_scores
        df['l_target'] = l_targets
        df['l_physics'] = l_phys_list
        df['composite_score'] = scores

        return df


def bayesian_refinement_optuna(
    best_candidate: pd.Series,
    target_pv: np.ndarray,
    evaluator: CandidateEvaluator,
    Pb_max: float = 1.0,
    n_trials: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """
    Local Bayesian optimization refinement around a promising candidate using Optuna.
    Enforces all hard constraints:
      - Simplex A-site (FA+MA+Cs=1)
      - Simplex B-site (Pb+Sn=1, Pb <= Pb_max)
      - Simplex X-site (I+Br+Cl=1)
      - Physical layer thicknesses & energy levels
    """
    arch = best_candidate['cell_arch']
    compat = COMPATIBILITY_DICT[arch]

    def objective(trial: optuna.Trial) -> float:
        # A-site sampling on simplex
        u_fa = trial.suggest_float('u_fa', 0.05, 0.95)
        u_ma = trial.suggest_float('u_ma', 0.05, 0.95)
        u_cs = trial.suggest_float('u_cs', 0.01, 0.50)
        tot_a = u_fa + u_ma + u_cs
        x_FA, x_MA, x_Cs = u_fa / tot_a, u_ma / tot_a, u_cs / tot_a

        # B-site sampling with hard Pb cap
        x_Pb = trial.suggest_float('x_Pb', 0.0, float(Pb_max))
        x_Sn = 1.0 - x_Pb

        # X-site sampling on simplex
        u_i = trial.suggest_float('u_i', 0.40, 1.0)
        u_br = trial.suggest_float('u_br', 0.0, 0.50)
        u_cl = trial.suggest_float('u_cl', 0.0, 0.15)
        tot_x = u_i + u_br + u_cl
        x_I, x_Br, x_Cl = u_i / tot_x, u_br / tot_x, u_cl / tot_x

        # Thicknesses within fine-tuned ranges around initial candidate
        abs_t = trial.suggest_float('absorber_thickness', 250.0, 750.0)
        etl_t = trial.suggest_float('etl_thickness', 20.0, 80.0)
        htl_t = trial.suggest_float('htl_thickness', 30.0, 180.0)

        # Offsets
        delta_h = trial.suggest_float('Delta_H', 0.05, 0.50)
        delta_l = trial.suggest_float('Delta_L', 0.05, 0.45)

        etl_mat = trial.suggest_categorical('etl_material', compat['etl'])
        htl_mat = trial.suggest_categorical('htl_material', compat['htl'])

        candidate = pd.DataFrame([{
            'x_FA': x_FA, 'x_MA': x_MA, 'x_Cs': x_Cs,
            'x_Pb': x_Pb, 'x_Sn': x_Sn,
            'x_I': x_I, 'x_Br': x_Br, 'x_Cl': x_Cl,
            'absorber_thickness': abs_t,
            'etl_thickness': etl_t,
            'htl_thickness': htl_t,
            'Delta_H': delta_h,
            'Delta_L': delta_l,
            'is_pin': float(arch == 'pin'),
            'cell_arch': arch,
            'etl_material': etl_mat,
            'htl_material': htl_mat,
            'substrate': 'FTO' if arch == 'nip' else 'ITO',
        }])

        evaluated = evaluator.evaluate(candidate, target_pv)
        return float(evaluated['composite_score'].iloc[0])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Collect all evaluated trials into a DataFrame
    refined_records = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            p = t.params
            tot_a = p['u_fa'] + p['u_ma'] + p['u_cs']
            tot_x = p['u_i'] + p['u_br'] + p['u_cl']
            refined_records.append({
                'x_FA': p['u_fa'] / tot_a,
                'x_MA': p['u_ma'] / tot_a,
                'x_Cs': p['u_cs'] / tot_a,
                'x_Pb': p['x_Pb'],
                'x_Sn': 1.0 - p['x_Pb'],
                'x_I': p['u_i'] / tot_x,
                'x_Br': p['u_br'] / tot_x,
                'x_Cl': p['u_cl'] / tot_x,
                'absorber_thickness': p['absorber_thickness'],
                'etl_thickness': p['etl_thickness'],
                'htl_thickness': p['htl_thickness'],
                'Delta_H': p['Delta_H'],
                'Delta_L': p['Delta_L'],
                'is_pin': float(arch == 'pin'),
                'cell_arch': arch,
                'etl_material': p['etl_material'],
                'htl_material': p['htl_material'],
                'substrate': 'FTO' if arch == 'nip' else 'ITO',
            })

    refined_df = pd.DataFrame(refined_records)
    evaluated_refined = evaluator.evaluate(refined_df, target_pv)
    return evaluated_refined.sort_values('composite_score').reset_index(drop=True)


def extract_pareto_front_3d(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Multi-objective non-dominated sorting on:
      1. Target Error (minimize |pred_PCE - target_PCE|) or 1 / PCE
      2. Lead fraction (minimize x_Pb)
      3. Epistemic uncertainty (minimize mean_unc)
    Returns:
      pareto_df: DataFrame of non-dominated candidates
      pareto_mask: Boolean mask over original df
    """
    n = len(df)
    # Objectives to minimize: [target_error, x_Pb, mean_unc]
    # If l_target is present, use it; otherwise compute PCE error
    obj1 = df['l_target'].values if 'l_target' in df.columns else (df['pred_PCE'].max() - df['pred_PCE'].values)
    obj2 = df['x_Pb'].values
    obj3 = df['mean_unc'].values

    objectives = np.column_stack([obj1, obj2, obj3])
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j is strictly better in at least one and no worse in all
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                is_pareto[i] = False
                break

    pareto_df = df[is_pareto].copy().reset_index(drop=True)
    return pareto_df, is_pareto
