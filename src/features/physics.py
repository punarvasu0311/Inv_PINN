"""
Physics-based feature engineering for perovskite compositions and device architectures.
Implements:
- Goldschmidt tolerance factor (tau)
- Octahedral factor (mu)
- tau * mu interaction
- A-site configurational mixing entropy
- Pauling electronegativity differences
- Vegard-law bandgap estimation with halide bowing
- Shockley-Queisser limits (eta_SQ, Voc_SQ, Jsc_max)
- Key compositional interaction terms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

# Shannon ionic radii in picometers (pm)
SHANNON_RADII = {
    'FA': 253.0,   # Formamidinium (12-coordination approx)
    'MA': 217.0,   # Methylammonium (12-coordination approx)
    'Cs': 188.0,   # Cs+ (12-coordination)
    'Pb': 119.0,   # Pb2+ (6-coordination)
    'Sn': 110.0,   # Sn2+ (6-coordination)
    'I': 220.0,    # I- (6-coordination)
    'Br': 196.0,   # Br- (6-coordination)
    'Cl': 181.0,   # Cl- (6-coordination)
}

# Pauling electronegativities
ELECTRONEGATIVITIES = {
    'FA': 2.20,
    'MA': 2.30,
    'Cs': 0.79,
    'Pb': 2.33,
    'Sn': 1.96,
    'I': 2.66,
    'Br': 2.96,
    'Cl': 3.16,
}

# End-member standard bandgaps (eV)
END_MEMBER_EG = {
    'I': 1.48,   # MAPbI3
    'Br': 2.28,  # MAPbBr3
    'Cl': 2.97,  # MAPbCl3
}

# Fundamental constants
Q = 1.602176634e-19       # Elementary charge (C)
K_B = 1.380649e-23        # Boltzmann constant (J/K)
H = 6.62607015e-34        # Planck constant (J*s)
C = 299792458.0           # Speed of light (m/s)
T_STD = 300.0             # Standard operating temperature (K)
P_IN_MW = 100.0           # AM 1.5G incident optical power (mW/cm^2)


class PhysicsFeatureEngineer:
    """
    Deterministic physics feature engineering module.
    Translates chemical stoichiometry and bandgap into optoelectronic & crystallographic features.
    """
    def __init__(self, bowing_parameter: float = 0.33):
        self.bowing_parameter = bowing_parameter

    def compute_radii(self, x_FA: float, x_MA: float, x_Cs: float,
                      x_Pb: float, x_Sn: float,
                      x_I: float, x_Br: float, x_Cl: float) -> Tuple[float, float, float]:
        """Compute effective Shannon ionic radii for A, B, and X sites."""
        r_A = x_FA * SHANNON_RADII['FA'] + x_MA * SHANNON_RADII['MA'] + x_Cs * SHANNON_RADII['Cs']
        r_B = x_Pb * SHANNON_RADII['Pb'] + x_Sn * SHANNON_RADII['Sn']
        r_X = x_I * SHANNON_RADII['I'] + x_Br * SHANNON_RADII['Br'] + x_Cl * SHANNON_RADII['Cl']
        return float(r_A), float(r_B), float(r_X)

    def compute_tolerance_and_octahedral(self, r_A: float, r_B: float, r_X: float) -> Tuple[float, float, float]:
        """
        Compute Goldschmidt tolerance factor (tau), octahedral factor (mu),
        and their joint product (tau_mu).
        tau = (r_A + r_X) / (sqrt(2) * (r_B + r_X))
        mu = r_B / r_X
        """
        tau = (r_A + r_X) / (np.sqrt(2.0) * (r_B + r_X))
        mu = r_B / max(r_X, 1.0)
        tau_mu = tau * mu
        return float(tau), float(mu), float(tau_mu)

    def compute_mixing_entropy(self, x_FA: float, x_MA: float, x_Cs: float) -> float:
        """
        A-site configurational mixing entropy:
        S_A = -sum(x_i * ln(x_i)) for x_i > 0
        """
        s_a = 0.0
        for x in [x_FA, x_MA, x_Cs]:
            if x > 1e-9:
                s_a -= x * np.log(x)
        return float(s_a)

    def compute_electronegativity_diffs(self, x_FA: float, x_MA: float, x_Cs: float,
                                        x_Pb: float, x_Sn: float,
                                        x_I: float, x_Br: float, x_Cl: float) -> Tuple[float, float]:
        """Compute mean Pauling electronegativity differences Delta_chi_BX and Delta_chi_AX."""
        chi_A = x_FA * ELECTRONEGATIVITIES['FA'] + x_MA * ELECTRONEGATIVITIES['MA'] + x_Cs * ELECTRONEGATIVITIES['Cs']
        chi_B = x_Pb * ELECTRONEGATIVITIES['Pb'] + x_Sn * ELECTRONEGATIVITIES['Sn']
        chi_X = x_I * ELECTRONEGATIVITIES['I'] + x_Br * ELECTRONEGATIVITIES['Br'] + x_Cl * ELECTRONEGATIVITIES['Cl']
        delta_chi_BX = abs(chi_B - chi_X)
        delta_chi_AX = abs(chi_A - chi_X)
        return float(delta_chi_BX), float(delta_chi_AX)

    def compute_vegard_bandgap(self, x_I: float, x_Br: float, x_Cl: float,
                               x_Sn: float = 0.0) -> float:
        """
        Vegard-law bandgap estimation with halide bowing (b = 0.33 eV)
        and Sn-alloying red-shift effect.
        """
        eg_halides = (x_I * END_MEMBER_EG['I'] +
                      x_Br * END_MEMBER_EG['Br'] +
                      x_Cl * END_MEMBER_EG['Cl'] -
                      self.bowing_parameter * x_I * x_Br)
        # Sn alloying narrows bandgap
        eg_total = eg_halides - 0.35 * x_Sn * (1.0 - 0.5 * x_Sn)
        return float(np.clip(eg_total, 1.10, 3.20))

    @staticmethod
    def compute_shockley_queisser(Eg: float, T: float = T_STD) -> Tuple[float, float, float]:
        """
        Compute Shockley-Queisser detailed balance upper limits for a given bandgap Eg (in eV).
        Returns:
            eta_SQ: Maximum power conversion efficiency (in %)
            Voc_SQ: Maximum open-circuit voltage (in V)
            Jsc_max: Maximum short-circuit current density (in mA/cm^2)
        """
        if Eg <= 0.4 or Eg > 4.0:
            return 0.0, 0.0, 0.0

        # AM1.5G photon flux integration parameterization (detailed balance)
        # Empirical high-accuracy parametric fit to ASTM G173-03 AM1.5G reference spectrum
        # Jsc_max decreases monotonically with Eg:
        if Eg < 0.8:
            Jsc_max = 68.0 - 15.0 * (Eg - 0.4)
        elif Eg < 1.3:
            Jsc_max = 62.0 - 28.0 * (Eg - 0.8)
        elif Eg < 1.8:
            Jsc_max = 48.0 - 32.0 * (Eg - 1.3)
        elif Eg < 2.4:
            Jsc_max = 32.0 - 24.0 * (Eg - 1.8)
        else:
            Jsc_max = max(17.6 - 15.0 * (Eg - 2.4), 0.1)

        # Radiative recombination dark saturation current J0_rad:
        # J0_rad = (2 * pi * q * (k_B*T)^3 / (h^3 * c^2)) * exp(-q*Eg / (k_B*T)) * (Eg_norm^2 + 2*Eg_norm + 2)
        k_b_t_ev = (K_B * T) / Q
        eg_norm = Eg / k_b_t_ev
        
        # Pre-factor in mA/cm^2
        c_pre = 1.07e-9 # mA/cm^2 calibrated to 300K blackbody integration
        j0_rad = c_pre * (eg_norm**2 + 2.0 * eg_norm + 2.0) * np.exp(-min(eg_norm, 80.0))
        j0_rad = max(j0_rad, 1e-40)

        # Radiative open-circuit voltage
        voc_sq = k_b_t_ev * np.log(max(Jsc_max / j0_rad, 1.0) + 1.0)
        # Physical constraint: Voc_SQ cannot exceed Eg / q
        voc_sq = min(voc_sq, Eg - 0.25)
        voc_sq = max(voc_sq, 0.1)

        # Empirical normalized Voc for ideal fill factor
        v_norm = voc_sq / k_b_t_ev
        if v_norm > 1.0:
            ff_sq = (v_norm - np.log(v_norm + 0.72)) / (v_norm + 1.0)
        else:
            ff_sq = 0.25
        ff_sq = np.clip(ff_sq, 0.25, 0.90)

        # Maximum SQ efficiency (%) = Jsc (mA/cm^2) * Voc (V) * FF / Pin (mW/cm^2) * 100%
        eta_sq = (Jsc_max * voc_sq * ff_sq / P_IN_MW) * 100.0
        eta_sq = float(np.clip(eta_sq, 0.0, 33.7)) # 33.7% is single junction SQ peak

        return float(eta_sq), float(voc_sq), float(Jsc_max)

    def extract_features(self, row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, float]:
        """Extract all physics features from a single record."""
        x_FA = float(row.get('x_FA', 0.0))
        x_MA = float(row.get('x_MA', 0.0))
        x_Cs = float(row.get('x_Cs', 0.0))
        x_Pb = float(row.get('x_Pb', 1.0))
        x_Sn = float(row.get('x_Sn', 0.0))
        x_I = float(row.get('x_I', 1.0))
        x_Br = float(row.get('x_Br', 0.0))
        x_Cl = float(row.get('x_Cl', 0.0))

        r_A, r_B, r_X = self.compute_radii(x_FA, x_MA, x_Cs, x_Pb, x_Sn, x_I, x_Br, x_Cl)
        tau, mu, tau_mu = self.compute_tolerance_and_octahedral(r_A, r_B, r_X)
        s_a = self.compute_mixing_entropy(x_FA, x_MA, x_Cs)
        delta_chi_BX, delta_chi_AX = self.compute_electronegativity_diffs(
            x_FA, x_MA, x_Cs, x_Pb, x_Sn, x_I, x_Br, x_Cl
        )

        Eg_vegard = self.compute_vegard_bandgap(x_I, x_Br, x_Cl, x_Sn)
        Eg_actual = float(row.get('Eg', Eg_vegard))
        if np.isnan(Eg_actual) or Eg_actual <= 0.5:
            Eg_actual = Eg_vegard

        eta_sq, voc_sq, jsc_max = self.compute_shockley_queisser(Eg_actual)

        return {
            'r_A_eff': r_A,
            'r_B_eff': r_B,
            'r_X_eff': r_X,
            'tau': tau,
            'mu': mu,
            'tau_mu': tau_mu,
            'S_A': s_a,
            'Delta_chi_BX': delta_chi_BX,
            'Delta_chi_AX': delta_chi_AX,
            'Eg_vegard': Eg_vegard,
            'eta_SQ': eta_sq,
            'Voc_SQ': voc_sq,
            'Jsc_max_SQ': jsc_max,
            'x_FA_x_I': x_FA * x_I,
            'x_Sn_x_I': x_Sn * x_I,
            'x_FA_x_Br': x_FA * x_Br,
            'x_Pb_x_I': x_Pb * x_I,
        }

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physics feature engineering to an entire DataFrame."""
        out_df = df.copy()
        new_feats = []
        for idx, row in df.iterrows():
            new_feats.append(self.extract_features(row))
        feat_df = pd.DataFrame(new_feats, index=df.index)
        for col in feat_df.columns:
            out_df[col] = feat_df[col]
        return out_df
