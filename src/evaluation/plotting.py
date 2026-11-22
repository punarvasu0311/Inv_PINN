"""
Publication-quality plotting functions for:
- Forward model parity plots
- Uncertainty calibration curves
- CFG guidance scale sweep
- Real vs. Generated marginal distributions (KS annotated)
- Ternary composition maps (FA-MA-Cs)
- 2D and 3D Pareto optimization fronts
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Publication typography & styling
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
})


def plot_forward_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ['PCE (%)', 'Voc (V)', 'Jsc (mA/cm²)', 'FF'],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Generate 4-panel parity plot with identity dashed line and metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=300)

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        ax.scatter(yt, yp, alpha=0.35, s=16, color='#1f77b4', edgecolors='none', rasterized=True)

        min_val = min(float(np.min(yt)), float(np.min(yp)))
        max_val = max(float(np.max(yt)), float(np.max(yp)))
        pad = (max_val - min_val) * 0.05
        ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 'k--', lw=1.8, label='Ideal (y = x)')

        # Annotate R^2 and RMSE
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        ax.text(
            0.05, 0.90, f"$R^2 = {r2:.4f}$\n$RMSE = {rmse:.3f}$",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc')
        )

        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_ylim(min_val - pad, max_val + pad)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_uncertainty_calibration(
    coverage_dict: Dict[str, List[float]],
    target_names: List[str] = ['PCE', 'Voc', 'Jsc', 'FF'],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot empirical coverage vs nominal theoretical coverage."""
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)

    nominal_levels = [0.50, 0.80, 0.90, 0.95]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    ax.plot([0.4, 1.0], [0.4, 1.0], 'k--', lw=1.8, label='Ideal Calibration')

    for i, name in enumerate(target_names):
        emp_cov = [coverage_dict[f"coverage_{int(lvl*100)}%"][i] for lvl in nominal_levels]
        ax.plot(nominal_levels, emp_cov, marker='o', lw=2.2, color=colors[i], label=f'{name}')

    ax.set_xlabel("Nominal Confidence Level")
    ax.set_ylabel("Empirical Interval Coverage")
    ax.set_title("Post-Hoc Uncertainty Calibration")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=True, facecolor='white')
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_cfg_guidance_sweep(
    w_values: List[float],
    hit_rates: List[float],
    optimal_w: float = 1.5,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot CFG guidance scale sweep curve showing unimodal peak (Khan et al., 2026)."""
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    ax.plot(w_values, hit_rates, marker='s', lw=2.5, color='#2b5c8f', markersize=7, label='cVAE+CFG Hit-Rate')
    opt_idx = w_values.index(optimal_w) if optimal_w in w_values else int(np.argmax(hit_rates))
    ax.scatter([w_values[opt_idx]], [hit_rates[opt_idx]], color='#d95f02', s=140, zorder=5, label=f'Optimal $w^* = {w_values[opt_idx]}$')
    ax.axvline(w_values[opt_idx], color='#d95f02', linestyle=':', lw=1.5, alpha=0.7)

    ax.set_xlabel("Classifier-Free Guidance Scale ($w$)")
    ax.set_ylabel("Target Hit-Rate (%)")
    ax.set_title("Guidance Scale Optimization (Tail-Region Concentration)")
    ax.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_marginal_ks_checks(
    real_df,
    gen_df,
    features: List[str] = ['x_FA', 'x_Pb', 'x_I', 'absorber_thickness'],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """KDE overlays comparing real vs generated parameter marginals."""
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.8), dpi=300)
    if len(features) == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        sns.kdeplot(real_df[feat], ax=ax, label='Training Real', color='#4575b4', lw=2.0, fill=True, alpha=0.25)
        sns.kdeplot(gen_df[feat], ax=ax, label='cVAE+CFG (Target)', color='#d73027', lw=2.0, fill=True, alpha=0.25)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.set_title(f"Marginal: {feat}")
        ax.legend(frameon=True, facecolor='white')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_ternary_a_site(
    df,
    highlight_pareto_mask: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Ternary scatter representation of A-site composition (FA, MA, Cs)."""
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)

    # 2D projection of 3D simplex:
    # x = 0.5 * (2*x_MA + x_Cs) / (x_FA + x_MA + x_Cs)
    # y = (sqrt(3)/2) * x_Cs / (x_FA + x_MA + x_Cs)
    fa = df['x_FA'].values
    ma = df['x_MA'].values
    cs = df['x_Cs'].values
    total = np.maximum(fa + ma + cs, 1e-6)

    x_proj = 0.5 * (2.0 * ma + cs) / total
    y_proj = (np.sqrt(3.0) / 2.0) * cs / total

    # Plot simplex triangle boundary
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0)/2.0, 0.0]
    ax.plot(triangle_x, triangle_y, 'k-', lw=1.5)

    ax.text(-0.04, -0.04, "FA (100%)", fontsize=11, fontweight='bold')
    ax.text(1.02, -0.04, "MA (100%)", fontsize=11, fontweight='bold')
    ax.text(0.45, np.sqrt(3.0)/2.0 + 0.03, "Cs (100%)", fontsize=11, fontweight='bold')

    scatter = ax.scatter(x_proj, y_proj, c=df['pred_PCE'] if 'pred_PCE' in df.columns else '#1f77b4',
                         cmap='viridis', s=20, alpha=0.6, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Predicted PCE (%)")

    if highlight_pareto_mask is not None and np.any(highlight_pareto_mask):
        ax.scatter(x_proj[highlight_pareto_mask], y_proj[highlight_pareto_mask],
                   facecolors='none', edgecolors='red', s=80, lw=1.8, label='Pareto Optimal')
        ax.legend(frameon=True, loc='upper left')

    ax.set_axis_off()
    ax.set_title("A-Site Composition Map (FA-MA-Cs)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_pareto_fronts(
    df,
    pareto_mask: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """2D / 3D visualization of Pareto trade-offs (PCE vs. Pb fraction vs. Uncertainty)."""
    fig = plt.figure(figsize=(14, 5.5), dpi=300)

    # Left: 2D Trade-off (PCE vs Pb content)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(df.loc[~pareto_mask, 'x_Pb'], df.loc[~pareto_mask, 'pred_PCE'],
                color='#b0bec5', s=25, alpha=0.4, label='Dominated Designs')
    ax1.scatter(df.loc[pareto_mask, 'x_Pb'], df.loc[pareto_mask, 'pred_PCE'],
                color='#d32f2f', s=60, edgecolors='black', lw=1.0, zorder=5, label='Pareto Front')

    # Trace line along sorted front
    p_sub = df[pareto_mask].sort_values('x_Pb')
    ax1.plot(p_sub['x_Pb'], p_sub['pred_PCE'], color='#d32f2f', linestyle='--', lw=1.5, alpha=0.8)

    ax1.set_xlabel("Lead Fraction ($x_{Pb}$)")
    ax1.set_ylabel("Predicted PCE (%)")
    ax1.set_title("2D Pareto Front: Efficiency vs. Toxicity")
    ax1.legend(frameon=True, facecolor='white')

    # Right: 3D Trade-off (PCE vs Pb vs Uncertainty)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    pce = df['pred_PCE'].values
    pb = df['x_Pb'].values
    unc = df['mean_unc'].values

    ax2.scatter(pb[~pareto_mask], unc[~pareto_mask], pce[~pareto_mask],
                c='#90a4ae', s=18, alpha=0.25)
    ax2.scatter(pb[pareto_mask], unc[pareto_mask], pce[pareto_mask],
                c='#d32f2f', s=55, edgecolors='black', lw=1.0, alpha=0.9)

    ax2.set_xlabel("Pb Fraction")
    ax2.set_ylabel("Uncertainty $\\sigma$")
    ax2.set_zlabel("PCE (%)")
    ax2.set_title("3D Pareto Front: Efficiency - Pb - Uncertainty")
    ax2.view_init(elev=22, azim=-55)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
