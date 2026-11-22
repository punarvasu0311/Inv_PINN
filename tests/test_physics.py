"""
Unit tests for physics calculations:
- Shannon ionic radii, Goldschmidt tolerance factor (tau), octahedral factor (mu)
- Mixing entropy
- Vegard law bandgap with bowing
- Shockley-Queisser detailed balance upper limits
- Curriculum physics loss zero on consistent data and non-zero on violations
"""

import pytest
import numpy as np
import torch

from src.features.physics import PhysicsFeatureEngineer
from src.models.pinn import CurriculumPhysicsLoss


class TestPhysicsFeatureCalculations:
    """Verify analytical physics formulas."""

    @pytest.fixture
    def engineer(self):
        return PhysicsFeatureEngineer(bowing_parameter=0.33)

    def test_goldschmidt_tolerance_factor(self, engineer):
        # MAPbI3: MA=217, Pb=119, I=220
        # tau = (217 + 220) / (sqrt(2) * (119 + 220)) = 437 / (1.4142 * 339) ≈ 0.911
        r_A, r_B, r_X = engineer.compute_radii(0, 1, 0, 1, 0, 1, 0, 0)
        tau, mu, tau_mu = engineer.compute_tolerance_and_octahedral(r_A, r_B, r_X)
        assert 0.85 < tau < 0.95, f"MAPbI3 tolerance factor unexpected: {tau}"
        assert 0.45 < mu < 0.60, f"MAPbI3 octahedral factor unexpected: {mu}"

    def test_mixing_entropy(self, engineer):
        # Single cation: S_A = 0
        s_single = engineer.compute_mixing_entropy(1.0, 0.0, 0.0)
        assert s_single == 0.0

        # Equimolar triple cation: S_A = -3 * (1/3 * ln(1/3)) = ln(3) ≈ 1.0986
        s_triple = engineer.compute_mixing_entropy(1.0/3, 1.0/3, 1.0/3)
        np.testing.assert_allclose(s_triple, np.log(3.0), atol=1e-4)

    def test_vegard_law_with_bowing(self, engineer):
        # Pure I
        eg_i = engineer.compute_vegard_bandgap(1.0, 0.0, 0.0, x_Sn=0.0)
        np.testing.assert_allclose(eg_i, 1.48, atol=1e-3)

        # 50:50 I:Br alloy with bowing: 0.5*1.48 + 0.5*2.28 - 0.33*(0.25) = 1.88 - 0.0825 = 1.7975
        eg_half = engineer.compute_vegard_bandgap(0.5, 0.5, 0.0, x_Sn=0.0)
        np.testing.assert_allclose(eg_half, 1.7975, atol=1e-3)

        # Sn alloying reduces bandgap
        eg_sn = engineer.compute_vegard_bandgap(1.0, 0.0, 0.0, x_Sn=0.5)
        assert eg_sn < eg_i, "Sn alloying should lower bandgap"

    def test_shockley_queisser_limits(self, engineer):
        # For Eg = 1.34 eV (near single-junction peak ~33.7%)
        eta_sq, voc_sq, jsc_max = engineer.compute_shockley_queisser(1.34)
        assert 30.0 <= eta_sq <= 33.7, f"SQ efficiency unexpected: {eta_sq}"
        assert 1.0 <= voc_sq <= 1.34, f"SQ Voc cannot exceed Eg: {voc_sq}"
        assert 35.0 <= jsc_max <= 50.0, f"SQ Jsc max unexpected: {jsc_max}"


class TestPhysicsLossConsistency:
    """Verify behavior of the physics loss formulation."""

    def test_loss_zero_on_consistent_inputs(self):
        loss_fn = CurriculumPhysicsLoss(total_epochs=10, ramp_epochs=1, max_lambda=0.5)
        loss_fn.current_epoch = 5

        # Perfectly consistent predictions:
        # Voc = 1.1 V, Jsc = 25 mA/cm^2, FF = 0.80 -> PCE = 1.1 * 25 * 0.8 = 22.0 %
        pred = torch.tensor([[22.0, 1.1, 25.0, 0.80]])
        target = pred.clone()
        Eg_batch = torch.tensor([1.5])
        Jsc_max_batch = torch.tensor([35.0])

        total_loss, metrics = loss_fn(pred, target, Eg_batch, Jsc_max_batch)
        np.testing.assert_allclose(metrics['loss_consistency'], 0.0, atol=1e-5)
        np.testing.assert_allclose(metrics['loss_voc_eg'], 0.0, atol=1e-5)
        np.testing.assert_allclose(metrics['loss_jsc_sq'], 0.0, atol=1e-5)
        np.testing.assert_allclose(total_loss.item(), 0.0, atol=1e-5)

    def test_loss_positive_on_physics_violations(self):
        loss_fn = CurriculumPhysicsLoss(total_epochs=10, ramp_epochs=1, max_lambda=0.5)
        loss_fn.current_epoch = 5

        # Inconsistent prediction:
        # PCE says 30% but Voc * Jsc * FF = 1.0 * 20 * 0.8 = 16%
        # Voc (1.8 V) > Eg (1.4 eV) -> violation
        # Jsc (45 mA/cm^2) > Jsc_max (35 mA/cm^2) -> violation
        pred = torch.tensor([[30.0, 1.8, 45.0, 0.80]])
        target = torch.tensor([[16.0, 1.0, 20.0, 0.80]])
        Eg_batch = torch.tensor([1.4])
        Jsc_max_batch = torch.tensor([35.0])

        total_loss, metrics = loss_fn(pred, target, Eg_batch, Jsc_max_batch)
        assert metrics['loss_consistency'] > 1.0, "Consistency loss should be high"
        assert metrics['loss_voc_eg'] > 0.1, "Voc > Eg penalty should trigger"
        assert metrics['loss_jsc_sq'] > 1.0, "Jsc > Jsc_max penalty should trigger"
        assert total_loss.item() > 5.0


class TestPhysicalUnitsRegression:
    """Regression tests ensuring PINN, physics loss, and violation checks operate in raw physical units."""

    def test_pinn_outputs_physical_units(self):
        from src.models.pinn import ResidualPINN
        model = ResidualPINN(input_dim=15, hidden_dim=32, n_blocks=1)
        model.eval()
        x = torch.randn(20, 15)
        pred = model(x).detach().numpy()

        # PCE: >= 0
        assert np.all(pred[:, 0] >= 0.0), "PCE must be non-negative"
        # Voc: >= 0
        assert np.all(pred[:, 1] >= 0.0), "Voc must be non-negative"
        # Jsc: >= 0
        assert np.all(pred[:, 2] >= 0.0), "Jsc must be non-negative"
        # FF: in [0.15, 0.95]
        assert np.all((pred[:, 3] >= 0.15) & (pred[:, 3] <= 0.95)), "FF must be in [0.15, 0.95]"

    def test_physics_violation_rate_physical_vs_standardized(self):
        from src.evaluation.metrics import compute_physics_violation_rate

        # Physically sensible predictions: PCE=20.0%, Voc=1.0V, Jsc=25.0mA/cm^2, FF=0.80, Eg=1.5eV
        valid_phys = np.array([
            [20.0, 1.0, 25.0, 0.80],
            [16.8, 1.05, 20.0, 0.76],
        ])
        eg = np.array([1.5, 1.55])
        v_res = compute_physics_violation_rate(valid_phys, eg)
        assert v_res['total_physics_violation_pct'] == 0.0, "Valid physical predictions must have 0% violation"

        # Standardized predictions (mean 0, std 1) have negative values and fail bounds
        standardized = np.array([
            [-0.2, 0.5, -1.1, 0.1],
            [1.2, -0.4, 0.3, -0.8]
        ])
        v_std = compute_physics_violation_rate(standardized, eg)
        assert v_std['bounds_violation_pct'] == 100.0, "Standardized predictions must trigger bounds violations"
