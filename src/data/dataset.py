"""
Data acquisition, parsing, cleaning, DOI-grouped splitting, and leakage-free preprocessing
for the Perovskite Solar Cell Inverse Design framework.
Operates on the official Perovskite Database Project dataset.
"""

import os
import re
import json
import logging
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical validity bounds & Experimental Knowledge Constraints (Liu et al. 2024)
# ---------------------------------------------------------------------------
PHYSICAL_BOUNDS = {
    'PCE': (0.0, 35.0),    # %
    'Voc': (0.0, 2.0),     # V
    'Jsc': (0.0, 50.0),    # mA/cm^2
    'FF': (0.15, 0.95),    # dimensionless (0-1)
}

# Compatibility dictionaries for device architectures (Liu et al. APL 2024, Table S6/S7)
COMPATIBILITY_DICT = {
    'nip': {
        'etl': ['TiO2-c', 'TiO2-mp', 'SnO2-np', 'ZnO-np', 'PCBM-60', 'In2O3', 'Nb2O5', 'CeO2'],
        'htl': ['Spiro-OMeTAD', 'PTAA', 'P3HT', 'CuSCN', 'CuI', 'NiO-cp', 'PolyTPD'],
        'tco': ['FTO', 'ITO'],
        'back_contact': ['Au', 'Ag', 'AgAl', 'Carbon', 'Cu']
    },
    'pin': {
        'etl': ['PCBM-60', 'C60', 'BCP', 'SnO2-np', 'ZnO-np', 'LiF', 'PCBM-70'],
        'htl': ['PEDOT:PSS', 'PTAA', 'NiO-cp', 'NiO-np', 'Cu2O', 'Me-4PACz', 'PolyTPD'],
        'tco': ['ITO', 'FTO'],
        'back_contact': ['Ag', 'Au', 'AgAl', 'Cu', 'Al']
    }
}

RAW_URLS = [
    "https://media.githubusercontent.com/media/Jesperkemist/perovskitedatabase_data/main/data/Perovskite_database_content_all_data.csv",
    "https://zenodo.org/records/5838596/files/Perovskite_database_content_all_data.csv",
]


def download_dataset(raw_dir: Path) -> Path:
    """Download the official Perovskite Database Project CSV."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / "Perovskite_database_content_all_data.csv"
    
    # If file already exists and is large (> 50 MB), use it directly
    if target_path.exists() and target_path.stat().st_size > 50 * 1024 * 1024:
        logger.info(f"Official Perovskite Database dataset already exists at {target_path} ({target_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return target_path

    logger.info("Downloading official Perovskite Database dataset from remote repository...")
    download_success = False
    for url in RAW_URLS:
        try:
            logger.info(f"Connecting to {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, 'wb') as f:
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 1024 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
            if target_path.stat().st_size > 10 * 1024 * 1024:
                download_success = True
                logger.info(f"Successfully downloaded {target_path.stat().st_size / 1024 / 1024:.1f} MB from {url}")
                break
        except Exception as e:
            logger.warning(f"Download attempt failed for {url}: {e}")

    if not download_success:
        if target_path.exists() and target_path.stat().st_size > 1024:
            logger.warning(f"Using existing local file at {target_path}")
        else:
            raise RuntimeError(
                f"Could not download official Perovskite Database CSV from {RAW_URLS}. "
                "Please ensure internet connectivity or place Perovskite_database_content_all_data.csv in data/raw/."
            )

    return target_path


def parse_ion_fractions(ions_str: Any, coeffs_str: Any, target_ions: List[str]) -> Dict[str, float]:
    """Parse ion and coefficient strings from Perovskite Database into normalized site fractions."""
    fractions = {ion: 0.0 for ion in target_ions}
    if pd.isna(ions_str) or not str(ions_str).strip():
        fractions[target_ions[0]] = 1.0
        return fractions

    raw_ions = [s.strip() for s in re.split(r'[;,|]', str(ions_str)) if s.strip()]
    if pd.isna(coeffs_str) or not str(coeffs_str).strip():
        raw_coeffs = [1.0] * len(raw_ions)
    else:
        raw_coeffs = []
        for c in re.split(r'[;,|]', str(coeffs_str)):
            try:
                raw_coeffs.append(float(c.strip()))
            except:
                raw_coeffs.append(1.0)

    if len(raw_coeffs) < len(raw_ions):
        raw_coeffs.extend([1.0] * (len(raw_ions) - len(raw_coeffs)))

    total = 0.0
    for ion_name, coeff in zip(raw_ions, raw_coeffs):
        clean_name = ion_name.upper().replace('+', '').replace('-', '').strip()
        for t in target_ions:
            if t.upper() in clean_name or clean_name in t.upper():
                fractions[t] += max(0.0, coeff)
                total += max(0.0, coeff)
                break

    if total > 0.0:
        for t in target_ions:
            fractions[t] /= total
    else:
        fractions[target_ions[0]] = 1.0

    return fractions


def canonicalize_etl(raw_val: Any, arch: str) -> str:
    """Map raw ETL stack sequence string to canonical vocabulary in COMPATIBILITY_DICT."""
    val = str(raw_val).lower() if not pd.isna(raw_val) else ''
    if 'pcbm' in val or 'pc61bm' in val:
        return 'PCBM-60'
    elif 'c60' in val:
        return 'C60'
    elif 'bcp' in val:
        return 'BCP'
    elif 'sno2' in val:
        return 'SnO2-np'
    elif 'tio2' in val and 'mp' in val:
        return 'TiO2-mp'
    elif 'tio2' in val:
        return 'TiO2-c'
    elif 'zno' in val:
        return 'ZnO-np'
    elif 'in2o3' in val:
        return 'In2O3'
    elif 'nb2o5' in val:
        return 'Nb2O5'
    elif 'ceo2' in val:
        return 'CeO2'
    return 'TiO2-c' if arch == 'nip' else 'PCBM-60'


def canonicalize_htl(raw_val: Any, arch: str) -> str:
    """Map raw HTL stack sequence string to canonical vocabulary in COMPATIBILITY_DICT."""
    val = str(raw_val).lower() if not pd.isna(raw_val) else ''
    if 'spiro' in val:
        return 'Spiro-OMeTAD'
    elif 'pedot' in val:
        return 'PEDOT:PSS'
    elif 'ptaa' in val:
        return 'PTAA'
    elif 'p3ht' in val:
        return 'P3HT'
    elif 'nio' in val:
        return 'NiO-cp'
    elif 'cuscn' in val:
        return 'CuSCN'
    elif 'cui' in val:
        return 'CuI'
    elif 'cu2o' in val:
        return 'Cu2O'
    elif 'poly' in val:
        return 'PolyTPD'
    elif 'pacz' in val:
        return 'Me-4PACz'
    return 'Spiro-OMeTAD' if arch == 'nip' else 'PEDOT:PSS'


def canonicalize_substrate(raw_val: Any) -> str:
    """Map raw substrate string to canonical FTO or ITO."""
    val = str(raw_val).lower() if not pd.isna(raw_val) else ''
    if 'ito' in val:
        return 'ITO'
    return 'FTO'


def parse_numeric_thickness(val: Any, default: float, lo: float, hi: float) -> float:
    """Extract numeric thickness from text string with bounds check."""
    if pd.isna(val):
        return default
    m = re.search(r'(\d+\.?\d*)', str(val))
    if m:
        v = float(m.group(1))
        if lo <= v <= hi:
            return v
    return default


def parse_and_standardize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse raw database columns into clean standardized schema.
    Handles official Perovskite Database column naming:
    Perovskite_composition_a_ions/coefficients, JV_default_PCE/Voc/Jsc/FF, etc.
    """
    df = df_raw.copy()
    initial_count = len(df)
    logger.info(f"Step 1: Input raw dataset has {initial_count} rows.")

    # 1. DOI
    if 'Ref_DOI_number' in df.columns:
        df['Ref_DOI'] = df['Ref_DOI_number'].fillna('UNKNOWN_DOI').astype(str)
    elif 'Ref_DOI' not in df.columns:
        df['Ref_DOI'] = 'UNKNOWN_DOI'

    # 2. Architecture: nip vs pin
    if 'Cell_architecture' in df.columns:
        df['cell_arch'] = df['Cell_architecture'].astype(str).str.lower().apply(
            lambda x: 'pin' if 'pin' in x or 'inverted' in x else 'nip'
        )
    else:
        df['cell_arch'] = 'nip'
    df['is_pin'] = (df['cell_arch'] == 'pin').astype(float)

    # 3. Composition parsing: A, B, X site fractions
    if 'Perovskite_composition_a_ions' in df.columns:
        logger.info("Parsing official ion composition strings...")
        a_site_list = [
            parse_ion_fractions(i, c, ['FA', 'MA', 'Cs'])
            for i, c in zip(df['Perovskite_composition_a_ions'], df['Perovskite_composition_a_ions_coefficients'])
        ]
        b_site_list = [
            parse_ion_fractions(i, c, ['Pb', 'Sn'])
            for i, c in zip(df['Perovskite_composition_b_ions'], df['Perovskite_composition_b_ions_coefficients'])
        ]
        c_site_list = [
            parse_ion_fractions(i, c, ['I', 'Br', 'Cl'])
            for i, c in zip(df['Perovskite_composition_c_ions'], df['Perovskite_composition_c_ions_coefficients'])
        ]

        df_a = pd.DataFrame(a_site_list, index=df.index)
        df_b = pd.DataFrame(b_site_list, index=df.index)
        df_c = pd.DataFrame(c_site_list, index=df.index)

        df['x_FA'] = df_a['FA']
        df['x_MA'] = df_a['MA']
        df['x_Cs'] = df_a['Cs']
        df['x_Pb'] = df_b['Pb']
        df['x_Sn'] = df_b['Sn']
        df['x_I'] = df_c['I']
        df['x_Br'] = df_c['Br']
        df['x_Cl'] = df_c['Cl']

    # Ensure required composition columns exist
    comp_cols = ['x_FA', 'x_MA', 'x_Cs', 'x_Pb', 'x_Sn', 'x_I', 'x_Br', 'x_Cl']
    for c in comp_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Ensure stoichiometry sums to 1
    a_sum = df[['x_FA', 'x_MA', 'x_Cs']].sum(axis=1).replace(0, 1.0)
    for c in ['x_FA', 'x_MA', 'x_Cs']:
        df[c] = df[c] / a_sum

    b_sum = df[['x_Pb', 'x_Sn']].sum(axis=1).replace(0, 1.0)
    for c in ['x_Pb', 'x_Sn']:
        df[c] = df[c] / b_sum

    x_sum = df[['x_I', 'x_Br', 'x_Cl']].sum(axis=1).replace(0, 1.0)
    for c in ['x_I', 'x_Br', 'x_Cl']:
        df[c] = df[c] / x_sum

    # 4. PV Targets in PHYSICAL UNITS
    col_mapping = {
        'JV_default_PCE': 'PCE',
        'JV_default_Voc': 'Voc',
        'JV_default_Jsc': 'Jsc',
        'JV_default_FF': 'FF',
    }
    for raw_c, target_c in col_mapping.items():
        if raw_c in df.columns and target_c not in df.columns:
            df[target_c] = pd.to_numeric(df[raw_c], errors='coerce')
        elif target_c in df.columns:
            df[target_c] = pd.to_numeric(df[target_c], errors='coerce')

    # Standardize FF to [0, 1] range if expressed as percentage
    if 'FF' in df.columns:
        df.loc[df['FF'] > 1.0, 'FF'] = df.loc[df['FF'] > 1.0, 'FF'] / 100.0

    # 5. Device layers & materials
    if 'ETL_stack_sequence' in df.columns:
        df['etl_material'] = [canonicalize_etl(val, arch) for val, arch in zip(df['ETL_stack_sequence'], df['cell_arch'])]
    elif 'etl_material' not in df.columns:
        df['etl_material'] = np.where(df['cell_arch'] == 'nip', 'TiO2-c', 'PCBM-60')

    if 'HTL_stack_sequence' in df.columns:
        df['htl_material'] = [canonicalize_htl(val, arch) for val, arch in zip(df['HTL_stack_sequence'], df['cell_arch'])]
    elif 'htl_material' not in df.columns:
        df['htl_material'] = np.where(df['cell_arch'] == 'nip', 'Spiro-OMeTAD', 'PEDOT:PSS')

    if 'Substrate_stack_sequence' in df.columns:
        df['substrate'] = df['Substrate_stack_sequence'].apply(canonicalize_substrate)
    elif 'substrate' not in df.columns:
        df['substrate'] = 'FTO'

    # 6. Thicknesses (nm)
    if 'Perovskite_thickness' in df.columns:
        df['absorber_thickness'] = df['Perovskite_thickness'].apply(lambda x: parse_numeric_thickness(x, default=450.0, lo=100.0, hi=1200.0))
    elif 'absorber_thickness' not in df.columns:
        df['absorber_thickness'] = 450.0

    if 'ETL_thickness' in df.columns:
        df['etl_thickness'] = df['ETL_thickness'].apply(lambda x: parse_numeric_thickness(x, default=50.0, lo=10.0, hi=300.0))
    elif 'etl_thickness' not in df.columns:
        df['etl_thickness'] = 50.0

    if 'HTL_thickness_list' in df.columns:
        df['htl_thickness'] = df['HTL_thickness_list'].apply(lambda x: parse_numeric_thickness(x, default=100.0, lo=10.0, hi=300.0))
    elif 'HTL_thickness' in df.columns:
        df['htl_thickness'] = df['HTL_thickness'].apply(lambda x: parse_numeric_thickness(x, default=100.0, lo=10.0, hi=300.0))
    elif 'htl_thickness' not in df.columns:
        df['htl_thickness'] = 100.0

    # 7. Bandgap Eg (eV)
    vegard_eg = (
        1.48 * df['x_I'] + 2.28 * df['x_Br'] + 2.97 * df['x_Cl'] -
        0.33 * df['x_I'] * df['x_Br'] - 0.35 * df['x_Sn']
    )
    if 'Perovskite_band_gap' in df.columns:
        df['Eg'] = pd.to_numeric(df['Perovskite_band_gap'], errors='coerce')
        df['Eg'] = np.where((df['Eg'] < 1.0) | (df['Eg'] > 3.5) | df['Eg'].isna(), vegard_eg, df['Eg'])
    elif 'Eg' in df.columns:
        df['Eg'] = pd.to_numeric(df['Eg'], errors='coerce')
        df['Eg'] = np.where((df['Eg'] < 1.0) | (df['Eg'] > 3.5) | df['Eg'].isna(), vegard_eg, df['Eg'])
    else:
        df['Eg'] = vegard_eg

    # 8. Energy offsets Delta_H and Delta_L
    if 'Delta_H' not in df.columns:
        df['Delta_H'] = 0.25
    if 'Delta_L' not in df.columns:
        df['Delta_L'] = 0.20

    return df


def clean_and_filter(df: pd.DataFrame, mad_z_thresh: float = 4.0) -> pd.DataFrame:
    """Filter physically impossible values, duplicates, and MAD outliers."""
    n_start = len(df)
    logger.info(f"Step 2: Filtering on {n_start} rows.")

    # 1. Deduplication
    core_cols = ['Ref_DOI', 'x_FA', 'x_MA', 'x_Cs', 'x_Pb', 'x_Sn', 'x_I', 'x_Br', 'x_Cl', 'PCE', 'Voc', 'Jsc', 'FF']
    avail_core = [c for c in core_cols if c in df.columns]
    df = df.drop_duplicates(subset=avail_core).copy()
    logger.info(f"  After deduplication: {len(df)} rows (removed {n_start - len(df)} duplicates)")

    # 2. Hard physical bounds
    n_before = len(df)
    for target, (lo, hi) in PHYSICAL_BOUNDS.items():
        if target in df.columns:
            mask = (df[target] > lo) & (df[target] <= hi) & df[target].notna()
            df = df[mask].copy()
    logger.info(f"  After physical bounds filtering: {len(df)} rows (removed {n_before - len(df)} out-of-bounds)")

    # 3. Median Absolute Deviation (MAD) outlier filtering
    n_before = len(df)
    for target in ['PCE', 'Voc', 'Jsc', 'FF']:
        if target in df.columns:
            vals = df[target].values
            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            if mad > 1e-6:
                modified_z = 0.6745 * np.abs(vals - median) / mad
                df = df[modified_z < mad_z_thresh].copy()
    logger.info(f"  After MAD outlier filtering (z < {mad_z_thresh}): {len(df)} rows (removed {n_before - len(df)} outliers)")

    return df.reset_index(drop=True)


def doi_grouped_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict paper-level (DOI-grouped) split.
    All devices from the same DOI/publication are placed in exactly one split partition.
    Guarantees zero data leakage across train, val, and test partitions.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(seed)

    if 'Ref_DOI' not in df.columns or df['Ref_DOI'].isna().all():
        logger.warning("No valid Ref_DOI column. Creating synthetic paper clusters for grouping.")
        df['Ref_DOI'] = [f"DOI_CLUSTER_{i // 5}" for i in range(len(df))]

    # Group record counts by DOI
    doi_series = df['Ref_DOI'].fillna('UNKNOWN_DOI').astype(str)
    unique_dois = np.array(sorted(doi_series.unique()))
    rng.shuffle(unique_dois)

    total_records = len(df)
    target_train = int(total_records * train_ratio)
    target_val = int(total_records * val_ratio)

    train_dois, val_dois, test_dois = set(), set(), set()
    current_train, current_val, current_test = 0, 0, 0

    doi_counts = doi_series.value_counts().to_dict()

    for doi in unique_dois:
        cnt = doi_counts.get(doi, 0)
        if current_train + cnt <= target_train or current_train < target_train:
            train_dois.add(doi)
            current_train += cnt
        elif current_val + cnt <= target_val or current_val < target_val:
            val_dois.add(doi)
            current_val += cnt
        else:
            test_dois.add(doi)
            current_test += cnt

    train_df = df[doi_series.isin(train_dois)].copy().reset_index(drop=True)
    val_df = df[doi_series.isin(val_dois)].copy().reset_index(drop=True)
    test_df = df[doi_series.isin(test_dois)].copy().reset_index(drop=True)

    # Verification: zero leakage
    leak_train_val = train_dois.intersection(val_dois)
    leak_train_test = train_dois.intersection(test_dois)
    leak_val_test = val_dois.intersection(test_dois)

    assert len(leak_train_val) == 0, f"Leakage detected between train and val: {leak_train_val}"
    assert len(leak_train_test) == 0, f"Leakage detected between train and test: {leak_train_test}"
    assert len(leak_val_test) == 0, f"Leakage detected between val and test: {leak_val_test}"

    logger.info(f"DOI-grouped Split complete:")
    logger.info(f"  Train: {len(train_df)} devices ({len(train_dois)} DOIs, {len(train_df)/total_records*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} devices ({len(val_dois)} DOIs, {len(val_df)/total_records*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} devices ({len(test_dois)} DOIs, {len(test_df)/total_records*100:.1f}%)")

    return train_df, val_df, test_df


class LeakageFreePreprocessor:
    """
    Strictly fit all imputers, scalers, and encoders on the training split only.
    Applies fitted transformations to validation, testing, and new inverse candidates.
    CRITICAL: Preserves targets in PHYSICAL UNITS (PCE in %, Voc in V, Jsc in mA/cm^2, FF in [0, 1]).
    """
    def __init__(self):
        self.feature_cols: List[str] = []
        self.numeric_feature_cols: List[str] = []
        self.cat_cols: List[str] = ['etl_material', 'htl_material', 'substrate']
        self.target_cols: List[str] = ['PCE', 'Voc', 'Jsc', 'FF']
        
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.feature_scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Target statistics (mean and std) stored for physical loss weighting, but targets are NOT standardized
        self.target_means: np.ndarray = np.zeros(4)
        self.target_stds: np.ndarray = np.ones(4)
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> 'LeakageFreePreprocessor':
        self.feature_cols = [c for c in feature_cols if c in train_df.columns]
        
        # Separate numeric and categorical
        self.numeric_feature_cols = [
            c for c in self.feature_cols if c not in self.cat_cols and pd.api.types.is_numeric_dtype(train_df[c])
        ]
        
        # Fit numeric feature transformations
        X_num = train_df[self.numeric_feature_cols].values
        self.numeric_imputer.fit(X_num)
        X_imp = self.numeric_imputer.transform(X_num)
        self.feature_scaler.fit(X_imp)

        # Fit categorical encoders on training data only
        for c in self.cat_cols:
            if c in train_df.columns:
                le = LabelEncoder()
                unique_vals = list(train_df[c].dropna().astype(str).unique()) + ['<UNKNOWN>']
                le.fit(unique_vals)
                self.label_encoders[c] = le

        # Store target statistics in physical units
        if all(t in train_df.columns for t in self.target_cols):
            y = train_df[self.target_cols].values
            self.target_means = np.mean(y, axis=0)
            self.target_stds = np.std(y, axis=0) + 1e-4

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform features and extract targets.
        Returns:
            X_transformed: (N, n_features) with standardized numeric features + encoded categorical features
            y_physical: (N, 4) targets strictly in PHYSICAL UNITS (PCE, Voc, Jsc, FF)
        """
        assert self.is_fitted, "Preprocessor must be fitted first!"

        # 1. Numeric features (imputed and standardized)
        X_num = df[self.numeric_feature_cols].values
        X_imp = self.numeric_imputer.transform(X_num)
        X_scaled = self.feature_scaler.transform(X_imp)

        # 2. Categorical features (encoded using train-fitted label encoders)
        cat_encoded_list = []
        for c in self.cat_cols:
            if c in self.label_encoders and c in df.columns:
                le = self.label_encoders[c]
                vals = df[c].fillna('<UNKNOWN>').astype(str).values
                # Map unseen values to <UNKNOWN>
                known_classes = set(le.classes_)
                vals_mapped = [v if v in known_classes else '<UNKNOWN>' for v in vals]
                encoded = le.transform(vals_mapped).reshape(-1, 1).astype(float)
                # Normalize encoded integers to [0, 1] range
                encoded = encoded / max(len(le.classes_), 1.0)
                cat_encoded_list.append(encoded)

        if cat_encoded_list:
            X_all = np.hstack([X_scaled] + cat_encoded_list)
        else:
            X_all = X_scaled

        # 3. Targets: STRICTLY IN PHYSICAL UNITS (No StandardScaler)
        if all(t in df.columns for t in self.target_cols):
            y_physical = df[self.target_cols].values.astype(np.float32)
        else:
            y_physical = None

        return X_all.astype(np.float32), y_physical

    def save(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> 'LeakageFreePreprocessor':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
