"""
Unit tests for zero data leakage:
- DOI-grouped splitting guarantees zero paper overlap between train, val, and test partitions.
- LeakageFreePreprocessor fits statistics (mean, variance, imputation medians) strictly on the training partition.
"""

import pytest
import numpy as np
import pandas as pd

from src.data.dataset import (
    generate_realistic_perovskite_dataset,
    doi_grouped_train_val_test_split,
    LeakageFreePreprocessor
)


class TestZeroDataLeakage:
    """Rigorous verification that no information leaks across partitions."""

    @pytest.fixture
    def sample_data(self):
        return generate_realistic_perovskite_dataset(n_samples=2000, seed=42)

    def test_strict_doi_isolation(self, sample_data):
        train_df, val_df, test_df = doi_grouped_train_val_test_split(
            sample_data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        train_dois = set(train_df['Ref_DOI'].unique())
        val_dois = set(val_df['Ref_DOI'].unique())
        test_dois = set(test_df['Ref_DOI'].unique())

        assert len(train_dois) > 0
        assert len(val_dois) > 0
        assert len(test_dois) > 0

        # Exact set intersection must be empty
        overlap_train_val = train_dois.intersection(val_dois)
        overlap_train_test = train_dois.intersection(test_dois)
        overlap_val_test = val_dois.intersection(test_dois)

        assert len(overlap_train_val) == 0, f"Leaked DOIs between train and val: {overlap_train_val}"
        assert len(overlap_train_test) == 0, f"Leaked DOIs between train and test: {overlap_train_test}"
        assert len(overlap_val_test) == 0, f"Leaked DOIs between val and test: {overlap_val_test}"

    def test_preprocessor_fitted_on_train_only(self, sample_data):
        train_df, val_df, test_df = doi_grouped_train_val_test_split(
            sample_data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        feature_cols = ['x_FA', 'x_MA', 'x_Cs', 'x_Pb', 'x_Sn', 'x_I', 'x_Br', 'x_Cl',
                        'Eg', 'absorber_thickness', 'etl_thickness', 'htl_thickness']

        preprocessor = LeakageFreePreprocessor()
        preprocessor.fit(train_df, feature_cols=feature_cols)

        # Check scaler mean matches training set mean, NOT full dataset mean
        expected_train_mean = train_df[feature_cols].values.mean(axis=0)
        np.testing.assert_allclose(preprocessor.feature_scaler.mean_, expected_train_mean, atol=1e-5)

        full_mean = sample_data[feature_cols].values.mean(axis=0)
        # Train mean and full mean should not be identical (demonstrating proper isolation)
        diff = np.abs(preprocessor.feature_scaler.mean_ - full_mean)
        assert np.any(diff > 1e-4), "Preprocessor mean suspiciously matches full dataset mean"
