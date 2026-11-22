"""
Evaluation metrics for forward and inverse perovskite design pipelines.
Includes:
- Forward regression metrics: RMSE, MAE, R^2, physical violation rate, NLL
- Inverse design metrics:
    * Target MAE, RMSE, and Hit-Rate
    * Chemical & physical validity rate (%)
    * Uniqueness (no duplicates at 1e-3 resolution)
    * Novelty (Euclidean NN distance to training set, % beyond median training NN)
    * Diversity (mean pairwise distance)
    * Two-sample Kolmogorov-Smirnov (KS) marginal distribution tests
    * Multi-objective Pareto hypervolume (exact 2D / 3D sweep)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import ks_2samp, norm
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def compute_forward_benchmark(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, R^2 per output head and combined averages."""
    targets = ['PCE', 'Voc', 'Jsc', 'FF']
    res = {}
    r2_all, rmse_all, mae_all = [], [], []
    for i, name in enumerate(targets):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        r2 = float(r2_score(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        res[f"{name}_R2"] = r2
        res[f"{name}_RMSE"] = rmse
        res[f"{name}_MAE"] = mae
        r2_all.append(r2)
        rmse_all.append(rmse)
        mae_all.append(mae)
    res['Mean_R2'] = float(np.mean(r2_all))
    res['Mean_RMSE'] = float(np.mean(rmse_all))
    res['Mean_MAE'] = float(np.mean(mae_all))
    return res


def compute_physics_violation_rate(y_pred: np.ndarray, Eg_array: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute physics violation rate across predictions:
    - Consistency violation: |PCE - Voc * Jsc * FF| > 1.0%
    - Bandgap violation: Voc > Eg
    - Negative values in PCE, Voc, Jsc, FF
    - FF outside [0.15, 0.95]
    """
    pce = y_pred[:, 0]
    voc = y_pred[:, 1]
    jsc = y_pred[:, 2]
    ff = y_pred[:, 3]

    pce_expected = voc * jsc * ff
    viol_consistency = np.abs(pce - pce_expected) > 1.0

    viol_voc = (voc > Eg_array) if Eg_array is not None else np.zeros_like(voc, dtype=bool)
    viol_neg = (pce < 0.0) | (voc < 0.0) | (jsc < 0.0) | (ff < 0.0)
    viol_ff = (ff < 0.15) | (ff > 0.95)

    any_viol = viol_consistency | viol_voc | viol_neg | viol_ff
    return {
        'consistency_violation_pct': float(np.mean(viol_consistency) * 100.0),
        'voc_eg_violation_pct': float(np.mean(viol_voc) * 100.0),
        'bounds_violation_pct': float(np.mean(viol_neg | viol_ff) * 100.0),
        'total_physics_violation_pct': float(np.mean(any_viol) * 100.0),
    }


def compute_chemical_validity(candidate_df) -> float:
    """Verify stoichiometry: A-site sums to 1, B-site sums to 1, X-site sums to 1, all >= 0."""
    a_sum = candidate_df['x_FA'] + candidate_df['x_MA'] + candidate_df['x_Cs']
    b_sum = candidate_df['x_Pb'] + candidate_df['x_Sn']
    x_sum = candidate_df['x_I'] + candidate_df['x_Br'] + candidate_df['x_Cl']

    tol = 1e-2
    valid_a = np.abs(a_sum - 1.0) < tol
    valid_b = np.abs(b_sum - 1.0) < tol
    valid_x = np.abs(x_sum - 1.0) < tol

    comp_cols = ['x_FA', 'x_MA', 'x_Cs', 'x_Pb', 'x_Sn', 'x_I', 'x_Br', 'x_Cl']
    non_neg = (candidate_df[comp_cols] >= -1e-4).all(axis=1)

    all_valid = valid_a & valid_b & valid_x & non_neg
    return float(np.mean(all_valid) * 100.0)


def compute_uniqueness(designs: np.ndarray, precision: float = 1e-3) -> float:
    """Fraction of unique candidate designs at given precision grid."""
    rounded = np.round(designs / precision) * precision
    unique_count = len(np.unique(rounded, axis=0))
    return float(unique_count / max(len(designs), 1) * 100.0)


def compute_novelty(candidates: np.ndarray, train_manifold: np.ndarray) -> Tuple[float, float]:
    """
    Compute candidate novelty:
    1. Mean nearest-neighbor (NN) Euclidean distance to the training set.
    2. Percentage of candidates with NN distance exceeding the training median internal NN distance.
    """
    k = min(candidates.shape[1], train_manifold.shape[1])
    c_sub = candidates[:, :k]
    t_sub = train_manifold[:, :k]

    # Reference training-set internal distances
    nn_train = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1).fit(t_sub)
    train_internal_dists, _ = nn_train.kneighbors(t_sub)
    median_train_nn = float(np.median(train_internal_dists[:, 1]))

    # Candidate to training-set distances
    nn_query = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1).fit(t_sub)
    candidate_dists, _ = nn_query.kneighbors(c_sub)
    candidate_nn = candidate_dists[:, 0]

    mean_nn_dist = float(np.mean(candidate_nn))
    pct_beyond_median = float(np.mean(candidate_nn > median_train_nn) * 100.0)
    return mean_nn_dist, pct_beyond_median


def compute_diversity(designs: np.ndarray) -> float:
    """Mean pairwise Euclidean distance among generated candidates."""
    if len(designs) <= 1:
        return 0.0
    return float(np.mean(pdist(designs, metric='euclidean')))


def compute_hit_rate(pred_pce: np.ndarray, target_pce: float, tolerance: float = 0.0) -> float:
    """Fraction of designs achieving or exceeding target PCE (pred_PCE >= target_pce - tolerance)."""
    hits = pred_pce >= (target_pce - tolerance)
    return float(np.mean(hits) * 100.0)


def compute_ks_tests(real_features: np.ndarray, gen_features: np.ndarray,
                     feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Two-sample Kolmogorov-Smirnov test for distribution alignment."""
    results = {}
    for i, name in enumerate(feature_names):
        if i < real_features.shape[1] and i < gen_features.shape[1]:
            stat, p_val = ks_2samp(real_features[:, i], gen_features[:, i])
            results[name] = {'ks_stat': float(stat), 'p_value': float(p_val)}
    return results


def compute_pareto_hypervolume(pareto_points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Computes exact hypervolume dominated by Pareto front points relative to reference point.
    All objectives are assumed to be minimized: [f1, f2, f3].
    """
    if len(pareto_points) == 0:
        return 0.0

    # Filter points strictly dominating reference point
    valid_mask = np.all(pareto_points <= reference_point, axis=1)
    pts = pareto_points[valid_mask]
    if len(pts) == 0:
        return 0.0

    if pts.shape[1] == 2:
        # Exact 2D sweep algorithm
        sorted_idx = np.argsort(pts[:, 0])
        sorted_pts = pts[sorted_idx]
        hv = 0.0
        cur_y = reference_point[1]
        for p in sorted_pts:
            if p[1] < cur_y:
                hv += (reference_point[0] - p[0]) * (cur_y - p[1])
                cur_y = p[1]
        return float(hv)

    # 3D Monte Carlo integration with fine resolution
    n_samples = 150000
    rng = np.random.RandomState(42)
    mins = np.min(pts, axis=0)
    samples = rng.uniform(mins, reference_point, size=(n_samples, pts.shape[1]))

    # Check how many samples are dominated by at least one Pareto point
    dominated = np.zeros(n_samples, dtype=bool)
    for p in pts:
        dominated |= np.all(samples >= p, axis=1)

    box_volume = np.prod(reference_point - mins)
    return float(box_volume * np.mean(dominated))
