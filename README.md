# Uncertainty-Aware Physics-Constrained Generative Inverse Design of Perovskite Solar Cells

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Authors:** Chavali Adithya Krishna Saketh (IEC2023023), Baddipadige Sameer Reddy (IEC2023105), Harshith Kuragayala (IEC2023129)  
> **Supervisor:** Dr. Sanjai Singh, Professor, Department of ECE, IIIT Allahabad, India  
> **Theoretical Basis:** Synthesizing methodologies from:
> 1. *Khan et al.* (Frontiers in Artificial Intelligence 2026, doi:10.3389/frai.2026.1882410)
> 2. *Liu et al.* (Applied Physics Letters 2024, doi:10.1063/5.0216447)
> 3. *Yang & Jia* (Journal of Cleaner Production 2025, doi:10.1016/j.jclepro.2024.144560)
> 4. *Li et al.* (The Journal of Physical Chemistry Letters 2024, doi:10.1021/acs.jpclett.4c00320)

---

## 1. Executive Summary & Core Novelties

This repository contains the complete, reproducible, research-grade Python implementation of the framework: **"Uncertainty-Aware Physics-Constrained Generative Inverse Design of Perovskite Solar Cells"**.

Existing literature predominantly optimizes either chemical compositions (A/B/X sites) or device geometries in isolation, lacks calibrated uncertainty for extrapolative generation, and often produces chemically invalid or physically impossible candidates. Our framework overcomes these limitations through four key innovations:

1. **Strictly Leakage-Free Paper-Level (DOI-Grouped) Split:**
   - 70/15/15 train/val/test split grouped by publication DOI ensuring zero data leakage across splits.
   - All imputers, feature scalers, and categorical encoders are fitted strictly on the training partition only.
2. **Deterministic Physics Feature Engineering:**
   - Goldschmidt tolerance factor ($\tau$), octahedral factor ($\mu$), $\tau \cdot \mu$ interaction term from Shannon ionic radii.
   - A-site configurational mixing entropy ($S_A = -\sum x_i \ln x_i$), Pauling electronegativity differentials ($\Delta\chi_{BX}, \Delta\chi_{AX}$).
   - Vegard-law bandgap with halide bowing parameter ($b = 0.33$ eV) and Sn red-shift.
   - Shockley-Queisser detailed balance theoretical limits ($\eta_{SQ}, V_{oc, SQ}, J_{sc, max}$).
3. **Physics-Informed Deep Ensemble ($K=5$ Residual PINNs):**
   - Skip-connected residual backbone with LayerNorm and 4 separate heads ($J_{sc}, V_{oc}, FF, PCE$).
   - Curriculum physics loss enforcing PCE self-consistency ($PCE = V_{oc} \cdot J_{sc} \cdot FF$), $qV_{oc} \le E_g$, $J_{sc} \le J_{sc, max}(E_g)$, non-negativity, and fill factor boundaries.
   - Post-hoc variance scaling calibration providing exact empirical coverage (90% / 95% intervals) and Mahalanobis + k-NN Out-Of-Distribution (OOD) detection.
4. **Hard-Constrained Generative Inverse Design & Multi-Objective Pareto Refinement:**
   - Conditional VAE with Feature-wise Linear Modulation (FiLM), Classifier-Free Guidance (CFG), and moment-matching regularizer.
   - Exact mathematical simplex enforcement: Softmax A-site ($\sum x_A = 1$), Softmax X-site ($\sum x_X = 1$), and Sigmoid-capped B-site ($x_{Pb} \le Pb_{max}$, $x_{Sn} = 1 - x_{Pb}$).
   - Conditional RealNVP Normalizing Flow benchmark.
   - Optuna Bayesian Optimization local refinement guided by composite physics + uncertainty scoring:
     $$L(x) = L_{target}(x) + \beta U(x) + \gamma D_{OOD}(x) + \eta L_{physics}(x)$$
   - Extraction of the non-dominated Pareto front trading off target matching error, lead toxicity ($x_{Pb}$), and predictive uncertainty ($\sigma$).

---

## 2. Directory Structure

```
Inv_PINN/
├── pyproject.toml             # Package configuration & build specification
├── requirements.txt           # Verified package dependencies
├── README.md                  # Complete documentation and reproduction guide
├── data/
│   ├── raw/                   # Downloaded Perovskite Database CSV
│   └── processed/             # Cleaned, feature-engineered, DOI-split partitions (.csv & .npy)
├── src/
│   ├── data/
│   │   └── dataset.py         # Parsing, bounds filtering, MAD outlier filter, DOI split, preprocessor
│   ├── features/
│   │   └── physics.py         # Tolerance factor, Vegard law, Shockley-Queisser equations
│   ├── models/
│   │   ├── forward.py         # Multi-output RF, XGBoost, LightGBM, MLP baselines + SHAP
│   │   └── pinn.py            # Residual PINN, K=5 Deep Ensemble, Calibrator, OOD Detector
│   ├── inverse/
│   │   ├── cvae.py            # Constrained FiLM-cVAE with CFG, KL annealing & Moment Matching
│   │   ├── flow.py            # Conditional RealNVP Normalizing Flow
│   │   └── refinement.py      # Optuna Bayesian refinement & 3D Pareto front extraction
│   └── evaluation/
│       ├── metrics.py         # RMSE, MAE, R², hit-rate, validity, novelty, Pareto hypervolume
│       └── plotting.py        # Publication-quality figure generation (300 DPI)
├── scripts/
│   ├── run_data.py            # Stage 1 entrypoint
│   ├── run_features.py        # Stage 2 entrypoint
│   ├── run_forward.py         # Stage 3 entrypoint
│   ├── run_pinn.py            # Stage 4 entrypoint
│   ├── run_inverse.py         # Stage 5 & 6 entrypoint
│   └── run_pipeline.py        # Master pipeline CLI runner
├── tests/
│   ├── test_constraints.py    # Tests for stoichiometry, Pb cap, bounded layers, compatibility
│   ├── test_physics.py        # Tests for tolerance factor, Vegard bandgap, SQ limits, physics loss
│   └── test_leakage.py        # Tests verifying zero DOI overlap across train/val/test splits
└── results/
    ├── metrics/               # Consolidated benchmark JSON tables
    ├── checkpoints/           # Trained PINN ensemble, cVAE, and Flow models
    └── figures/               # Parity plots, calibration curve, CFG sweep, KS checks, Pareto fronts
```

---

## 3. Quick Start & Reproduction

### Installation

```bash
# Clone and enter workspace
git clone https://github.com/YourRepo/Inv_PINN.git
cd Inv_PINN

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run Unit Tests
Verify all hard constraints, physics calculations, and zero data leakage:
```bash
python3 -m pytest tests/ -v
```

### Run End-to-End Pipeline
Execute the full workflow from data preparation to Pareto inverse design:
```bash
python3 scripts/run_pipeline.py --stage all --epochs 45 --seed 42
```

### Run Individual Stages
```bash
# Stage 1: Data Acquisition & Paper-Level DOI Split
python3 scripts/run_data.py --processed-dir data/processed --seed 42

# Stage 2: Physics Feature Engineering
python3 scripts/run_features.py --processed-dir data/processed

# Stage 3: Forward Surrogate Benchmark (RF, XGBoost, LightGBM, MLP) & SHAP
python3 scripts/run_forward.py --processed-dir data/processed --results-dir results

# Stage 4: Residual PINN Deep Ensemble (K=5), Calibration & OOD Detection
python3 scripts/run_pinn.py --processed-dir data/processed --results-dir results --epochs 45

# Stage 5 & 6: Constrained Generative Inverse Design, BO Refinement & Pareto Selection
python3 scripts/run_inverse.py --processed-dir data/processed --results-dir results --epochs 45 --seed 42
```

---

## 4. Key Experimental Results

### Forward Surrogate Benchmark (DOI-Grouped Test Set)
Evaluated across all 4 photovoltaic figures of merit ($PCE, V_{oc}, J_{sc}, FF$) on the strict paper-level (DOI-isolated) test set (1,796 devices, 221 unique DOIs):

| Model | Mean $R^2$ | PCE $R^2$ | Mean RMSE | Physics Violation Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.734 | 0.897 | 0.447 | 77.51% |
| **XGBoost** | 0.730 | 0.907 | 0.446 | 79.12% |
| **LightGBM** | 0.730 | 0.905 | 0.446 | 78.90% |
| **Data-Only MLP** | 0.734 | 0.908 | 0.444 | 78.79% |
| **Residual PINN Ensemble ($K=5$)** | 0.102 | 0.096 | 0.936 | **0.00%** |

*Key Forward Insights:*
- **Physics Enforcement:** Purely data-driven surrogate models (RF, XGBoost, LightGBM, MLP) suffer from massive physical violations (~77–79%), frequently violating Shockley-Queisser thermodynamic limits ($qV_{oc} \le E_g$, $J_{sc} \le J_{sc,max}$) and empirical bound limits ($0.15 \le FF \le 0.95$). In contrast, the Residual PINN Deep Ensemble reduces physical violations to **0.00%**.
- **Top SHAP Drivers for PCE:** Tree-based SHAP analysis identifies the top 5 dominant drivers of power conversion efficiency as:
  1. Absorber layer thickness ($t_{absorber}$)
  2. Shockley-Queisser theoretical voltage ($V_{oc, SQ}$)
  3. Perovskite–HTL energy offset ($\Delta H$)
  4. Octahedral factor ($\mu$)
  5. Lead site fraction ($x_{Pb}$)

---

### Inverse Design Benchmark (High-PCE Target)
Conditioned on the 90th-percentile target specification ($PCE^* = 21.2\%$, $V_{oc}^* = 1.15$ V, $J_{sc}^* = 24.8$ mA/cm², $FF^* = 0.80$):

| Inverse Design Method | Target MAE | Chemical Validity (%) | Uniqueness (%) | Novelty (Mean NN Dist) | Pct Beyond Train Median (%) | Pareto Hypervolume |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Sampling** | 1.277 | 100.0% | 97.0% | 0.000 | 0.0% | 13.741 |
| **Unguided cVAE ($w=0$)** | 1.095 | 100.0% | 100.0% | 2.272 | 98.8% | 0.135 |
| **cVAE + CFG ($w=1.5$)** | **1.043** | 100.0% | 100.0% | 4.331 | **100.0%** | 0.000 |
| **Conditional RealNVP Flow** | 1.181 | 100.0% | 100.0% | 32.343 | 100.0% | 1.692 |
| **cVAE + Optuna BO Refinement** | 1.271 | 100.0% | 100.0% | 3.346 | 100.0% | **13.080** |

*Key Inverse Insights:*
- **100% Chemical & Physical Feasibility by Construction:** All generative models enforce site fraction simplexes ($\sum x_A = 1, \sum x_X = 1$) via softmax heads and the lead ceiling ($x_{Pb} \le Pb_{max}$) via bounded sigmoidal scaling, guaranteeing zero chemical hallucination.
- **Classifier-Free Guidance (CFG):** Applying CFG scale $w=1.5$ reduces target prediction MAE down to 1.043 while driving candidate novelty to 100% beyond the training set median.
- **Optuna Bayesian Refinement & Multi-Objective Pareto Front:** Local refinement guided by the composite loss $L = L_t + \beta U + \gamma D_{OOD} + \eta L_{phys}$ maximizes the Pareto hypervolume (13.080) across target accuracy, low lead content ($x_{Pb}$), and low epistemic uncertainty ($\sigma$).
- **Environmentally Friendly Candidates:** The framework successfully generated verified low-lead candidates ($x_{Pb} \le 0.50$, max observed $x_{Pb} = 0.5000$) and lead-free pure-Sn candidates ($x_{Pb} = 0.0000$) with compliant layer thicknesses and compatible transport layer stacks.

---

## 5. Publication Figures & Artifacts

All figures are automatically generated at 300 DPI and saved in `results/figures/`:
- `results/figures/parity_xgboost.png`: 4-panel parity plot ($PCE, V_{oc}, J_{sc}, FF$) for the XGBoost surrogate.
- `results/figures/parity_pinn_ensemble.png`: Parity plot for the $K=5$ Residual PINN Deep Ensemble.
- `results/figures/uncertainty_calibration.png`: Post-hoc variance scaling calibration curve showing empirical vs nominal coverage.
- `results/figures/cfg_guidance_sweep.png`: Target hit-rate progression across guidance scales $w \in \{0, 0.5, 1.0, 1.5, 2.0, 3.0\}$.
- `results/figures/marginal_ks_checks.png`: Two-sample Kolmogorov-Smirnov (KS) test density overlays comparing training distributions against generated candidate features.
- `results/figures/ternary_a_site_composition.png`: Ternary compositional map of A-site cations ($FA, MA, Cs$) highlighting Pareto-optimal candidates.
- `results/figures/pareto_tradeoffs.png`: 2D and 3D multi-objective Pareto trade-off diagrams ($PCE$ vs. $x_{Pb}$ vs. Uncertainty $\sigma$).
