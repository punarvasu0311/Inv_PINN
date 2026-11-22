"""
Unit tests for chemical stoichiometry, hard lead caps, physical layer bounds,
and device architecture compatibility constraints.
"""

import pytest
import numpy as np
import torch
import pandas as pd

from src.inverse.cvae import ConstrainedFiLMcVAE
from src.inverse.flow import ConditionalRealNVP
from src.inverse.refinement import candidate_dict_to_dataframe
from src.data.dataset import COMPATIBILITY_DICT


class TestChemicalStoichiometryConstraints:
    """Verify that generated candidate compositions strictly adhere to crystal chemistry."""

    @pytest.fixture
    def cvae_model(self):
        model = ConstrainedFiLMcVAE(design_dim=14, cond_dim=5, latent_dim=16, hidden_dim=64)
        model.eval()
        return model

    @pytest.fixture
    def flow_model(self):
        model = ConditionalRealNVP(design_dim=14, cond_dim=5, n_layers=4, hidden_dim=64)
        model.eval()
        return model

    def test_cvae_a_site_sum_and_positivity(self, cvae_model):
        c = torch.tensor([22.0, 1.15, 24.0, 0.80, 1.0])
        samples = cvae_model.sample(c, n_samples=300, w=1.5, Pb_max=1.0)
        a_site = samples['a_site'].cpu().numpy()

        sums = np.sum(a_site, axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="A-site fractions must sum to 1.0")
        assert np.all(a_site >= 0.0), "A-site fractions must be non-negative"

    def test_cvae_x_site_sum_and_positivity(self, cvae_model):
        c = torch.tensor([20.0, 1.10, 23.0, 0.78, 1.0])
        samples = cvae_model.sample(c, n_samples=300, w=1.5, Pb_max=1.0)
        x_site = samples['x_site'].cpu().numpy()

        sums = np.sum(x_site, axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="X-site fractions must sum to 1.0")
        assert np.all(x_site >= 0.0), "X-site fractions must be non-negative"

    def test_cvae_hard_lead_cap_partial(self, cvae_model):
        pb_max = 0.40
        c = torch.tensor([19.5, 1.05, 24.5, 0.76, pb_max])
        samples = cvae_model.sample(c, n_samples=500, w=1.5, Pb_max=pb_max)
        b_site = samples['b_site'].cpu().numpy()

        sums = np.sum(b_site, axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="B-site fractions must sum to 1.0")

        x_pb = b_site[:, 0]
        assert np.all(x_pb <= pb_max + 1e-5), f"Pb fraction exceeded Pb_max ({pb_max}): max={x_pb.max()}"
        assert np.all(x_pb >= 0.0), "Pb fraction cannot be negative"

    def test_cvae_pure_sn_lead_free(self, cvae_model):
        c = torch.tensor([17.0, 0.90, 25.0, 0.75, 0.0])
        samples = cvae_model.sample(c, n_samples=300, w=1.5, Pb_max=0.0)
        b_site = samples['b_site'].cpu().numpy()

        x_pb = b_site[:, 0]
        x_sn = b_site[:, 1]
        np.testing.assert_allclose(x_pb, 0.0, atol=1e-6, err_msg="Lead fraction must be strictly 0 for pure-Sn")
        np.testing.assert_allclose(x_sn, 1.0, atol=1e-6, err_msg="Tin fraction must be strictly 1 for pure-Sn")

    def test_flow_stoichiometry_and_pb_cap(self, flow_model):
        pb_max = 0.50
        c = torch.tensor([21.0, 1.10, 24.0, 0.80, pb_max])
        samples = flow_model.sample(c, n_samples=300, Pb_max=pb_max)

        a_site = samples['a_site'].cpu().numpy()
        b_site = samples['b_site'].cpu().numpy()
        x_site = samples['x_site'].cpu().numpy()

        np.testing.assert_allclose(np.sum(a_site, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.sum(b_site, axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.sum(x_site, axis=-1), 1.0, atol=1e-5)
        assert np.all(b_site[:, 0] <= pb_max + 1e-5)


class TestPhysicalLayerBounds:
    """Verify thickness and energy offset ranges."""

    def test_thickness_bounds(self):
        model = ConstrainedFiLMcVAE(design_dim=14, cond_dim=5, latent_dim=16, hidden_dim=64)
        c = torch.tensor([22.0, 1.15, 24.0, 0.80, 1.0])
        samples = model.sample(c, n_samples=200, w=1.0)
        t = samples['thicknesses'].cpu().numpy()

        # Absorber in [200, 800] nm
        assert np.all((t[:, 0] >= 190.0) & (t[:, 0] <= 810.0))
        # ETL in [15, 100] nm
        assert np.all((t[:, 1] >= 14.0) & (t[:, 1] <= 105.0))
        # HTL in [20, 220] nm
        assert np.all((t[:, 2] >= 18.0) & (t[:, 2] <= 225.0))


class TestDeviceArchitectureCompatibility:
    """Verify that architecture-conditioned masking strictly adheres to experimental compatibility."""

    def test_compatibility_mapping(self):
        model = ConstrainedFiLMcVAE(design_dim=14, cond_dim=5, latent_dim=16, hidden_dim=64)
        c = torch.tensor([22.0, 1.15, 24.0, 0.80, 1.0])
        samples = model.sample(c, n_samples=100, w=1.5)
        df = candidate_dict_to_dataframe(samples, n_samples=100)

        for _, row in df.iterrows():
            arch = row['cell_arch']
            valid_etls = COMPATIBILITY_DICT[arch]['etl']
            valid_htls = COMPATIBILITY_DICT[arch]['htl']
            assert row['etl_material'] in valid_etls, f"Invalid ETL {row['etl_material']} for {arch}"
            assert row['htl_material'] in valid_htls, f"Invalid HTL {row['htl_material']} for {arch}"
