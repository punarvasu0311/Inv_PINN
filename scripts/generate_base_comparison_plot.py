"""
Generate publication-quality side-by-side comparison figure
contrasting the proposed framework against the base paper (Khan et al., Frontiers in AI 2026).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def plot_base_paper_comparison(save_path="results/figures/base_paper_comparison.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

    # Panel 1: Physics Violation Rate (Surrogate Quality)
    ax1 = axes[0]
    models = ['RF Baseline', 'XGBoost (Khan)', 'MLP Baseline', 'Our PINN Ensemble']
    violations = [77.51, 79.12, 78.79, 0.00]
    colors = ['#bdbdbd', '#e6550d', '#9ecae1', '#2ca02c']
    bars = ax1.bar(models, violations, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
    ax1.set_ylabel('Total Physics Violation Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Forward Surrogate Physics Violations\n(qVoc ≤ Eg, Jsc ≤ Jsc,max, Consistency)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 95)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, h + 2),
                     ha='center', va='bottom', fontsize=11, fontweight='bold',
                     color='#2ca02c' if h == 0 else '#333333')
    ax1.set_xticklabels(models, rotation=15, ha='right', fontsize=10)

    # Panel 2: Novelty & Exploration (Mean NN Distance to Training Data)
    ax2 = axes[1]
    methods = ['Random Sampling', 'Unguided cVAE\n(w=0)', 'cVAE + CFG\n(Base Paper w=1.5)', 'Our cVAE + Optuna\n(Pareto Refined)']
    nn_dists = [0.00, 2.27, 4.33, 3.35]
    colors_nn = ['#969696', '#6baed6', '#3182bd', '#08519c']
    bars2 = ax2.bar(methods, nn_dists, color=colors_nn, edgecolor='black', linewidth=1.2, width=0.55)
    ax2.set_ylabel('Mean Nearest Neighbor Distance', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Generative Exploration & Novelty\n(Distance from Training Manifold)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 5.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, h + 0.1),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)

    # Panel 3: Environmental Controllability (Lead Content x_Pb in Generated Designs)
    ax3 = axes[2]
    categories = ['Unconstrained\n(Base Paper)', 'Our High-PCE\n(Target 90th)', 'Our Low-Pb\n(Cap = 50%)', 'Our Lead-Free\n(Pure-Sn)']
    max_pb = [1.00, 1.00, 0.50, 0.00]
    colors_pb = ['#cb181d', '#fb6a4a', '#74c476', '#238b45']
    bars3 = ax3.bar(categories, max_pb, color=colors_pb, edgecolor='black', linewidth=1.2, width=0.55)
    ax3.set_ylabel('Maximum Lead Fraction (x_Pb)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Environmental Lead Toxicity Control\n(Hard-Constrained Decoder Head)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.2)
    ax3.axhline(0.5, color='#74c476', linestyle=':', lw=1.5, label='50% Pb Cap')
    ax3.axhline(0.0, color='#238b45', linestyle='--', lw=1.5, label='Zero Lead')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars3:
        h = bar.get_height()
        ax3.annotate(f'{h:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, h + 0.03),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_xticklabels(categories, rotation=15, ha='right', fontsize=10)
    ax3.legend(loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Base paper comparison plot successfully saved to: {save_path}")

if __name__ == "__main__":
    plot_base_paper_comparison()
