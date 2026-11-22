"""
Physics-Informed Neural Network (PINN) and Deep Ensemble (K=5) for Perovskite Solar Cells.
All outputs and physics constraints operate strictly in PHYSICAL UNITS:
- PCE in % (0 - 35)
- Voc in V (0 - 2.0)
- Jsc in mA/cm^2 (0 - 50)
- FF in dimensionless range [0.15, 0.95]

Physics constraints enforced:
1. PCE self-consistency: PCE = Voc * Jsc * FF (under standard 1 Sun / 100 mW/cm^2)
2. Bandgap ceiling: q * Voc <= Eg (in eV)
3. Shockley-Queisser spectral photocurrent limit: Jsc <= Jsc_max(Eg)
4. Empirical fill factor & non-negativity bounds
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class ResidualDenseBlock(nn.Module):
    """Residual dense block with LayerNorm, SiLU activation, and skip connection."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        h = self.dropout(h)
        return self.act(h + res)


class ResidualPINN(nn.Module):
    """
    Residual PINN for multi-output perovskite solar cell performance prediction.
    Features shared latent representation followed by 4 specialized output heads.
    Operates strictly in PHYSICAL UNITS:
      Outputs: [PCE (%), Voc (V), Jsc (mA/cm^2), FF (0-1)].
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        target_means: Optional[np.ndarray] = None,
        target_stds: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            ResidualDenseBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)
        ])

        # Default physical scales if none provided:
        # PCE ~ 16%, Voc ~ 1.0V, Jsc ~ 22 mA/cm^2, FF ~ 0.72
        t_means = target_means if target_means is not None else np.array([16.0, 1.0, 22.0, 0.72], dtype=np.float32)
        t_stds = target_stds if target_stds is not None else np.array([5.0, 0.2, 5.0, 0.10], dtype=np.float32)
        
        self.register_buffer('target_means', torch.tensor(t_means, dtype=torch.float32))
        self.register_buffer('target_stds', torch.tensor(t_stds, dtype=torch.float32))

        # Output heads
        self.head_voc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.head_jsc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.head_ff = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        self.head_pce = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)

        # Predict normalized deviations and map to physical scale
        norm_voc = self.head_voc(h)
        norm_jsc = self.head_jsc(h)
        norm_ff = self.head_ff(h)
        norm_pce = self.head_pce(h)

        voc_raw = norm_voc * self.target_stds[1] + self.target_means[1]
        jsc_raw = norm_jsc * self.target_stds[2] + self.target_means[2]
        ff_raw = norm_ff * self.target_stds[3] + self.target_means[3]
        pce_raw = norm_pce * self.target_stds[0] + self.target_means[0]

        # Enforce intrinsic positive bounds in physical units:
        voc = F.relu(voc_raw)                             # Voc >= 0 (V)
        jsc = F.relu(jsc_raw)                             # Jsc >= 0 (mA/cm^2)
        ff = torch.clamp(ff_raw, min=0.15, max=0.95)      # 0.15 <= FF <= 0.95
        pce = F.relu(pce_raw)                             # PCE >= 0 (%)

        # Stack into [PCE, Voc, Jsc, FF] (B, 4) in PHYSICAL UNITS
        return torch.cat([pce, voc, jsc, ff], dim=-1)


class CurriculumPhysicsLoss(nn.Module):
    """
    Curriculum physics loss operating on PHYSICAL UNITS:
    L_total = (1 - lambda_phys) * L_data + lambda_phys * L_physics
    Normalizes each loss component by its physical target standard deviation.
    """
    def __init__(
        self,
        total_epochs: int = 60,
        ramp_epochs: int = 20,
        max_lambda: float = 0.40,
        target_stds: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.ramp_epochs = ramp_epochs
        self.max_lambda = max_lambda
        self.current_epoch = 0
        
        t_stds = target_stds if target_stds is not None else np.array([5.0, 0.2, 5.0, 0.10], dtype=np.float32)
        self.register_buffer('target_stds', torch.tensor(t_stds, dtype=torch.float32))

    def get_lambda(self) -> float:
        if self.current_epoch < self.ramp_epochs:
            return float(self.max_lambda * (self.current_epoch / float(max(self.ramp_epochs, 1))))
        return float(self.max_lambda)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        Eg_batch: Optional[torch.Tensor] = None,
        Jsc_max_batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Data loss: normalized MSE so PCE, Voc, Jsc, FF contribute equally
        scale = self.target_stds.to(pred.device)
        norm_pred = pred / scale
        norm_target = target / scale
        l_data = F.mse_loss(norm_pred, norm_target)

        pce = pred[:, 0]
        voc = pred[:, 1]
        jsc = pred[:, 2]
        ff = pred[:, 3]

        # 2. Physics consistency in physical units: PCE = Voc * Jsc * FF
        pce_computed = voc * jsc * ff
        l_consistency = F.mse_loss(pce / scale[0], pce_computed / scale[0])

        # 3. Bandgap bound: q*Voc <= Eg (Voc in V <= Eg in eV)
        if Eg_batch is not None:
            l_voc_eg = F.relu(voc - Eg_batch).pow(2).mean() / (scale[1] ** 2)
        else:
            l_voc_eg = torch.tensor(0.0, device=pred.device)

        # 4. Spectral bound: Jsc <= Jsc_max(Eg)
        if Jsc_max_batch is not None:
            l_jsc_sq = F.relu(jsc - Jsc_max_batch).pow(2).mean() / (scale[2] ** 2)
        else:
            l_jsc_sq = torch.tensor(0.0, device=pred.device)

        # 5. Non-negativity and boundary penalties
        l_bounds = (
            F.relu(-pce).pow(2).mean() +
            F.relu(-voc).pow(2).mean() +
            F.relu(-jsc).pow(2).mean() +
            F.relu(0.15 - ff).pow(2).mean() +
            F.relu(ff - 0.95).pow(2).mean()
        )

        l_physics = l_consistency + 2.0 * l_voc_eg + 1.5 * l_jsc_sq + l_bounds

        lam = self.get_lambda()
        total_loss = (1.0 - lam) * l_data + lam * l_physics

        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_data': l_data.item(),
            'loss_physics': l_physics.item(),
            'loss_consistency': l_consistency.item(),
            'loss_voc_eg': l_voc_eg.item(),
            'loss_jsc_sq': l_jsc_sq.item(),
            'lambda_phys': lam
        }
        return total_loss, loss_dict


class PINNDeepEnsemble:
    """
    K=5 Deep Ensemble of Physics-Informed Neural Networks.
    Provides predictive mean mu(x) and epistemic uncertainty sigma(x) in PHYSICAL UNITS.
    """
    def __init__(
        self,
        input_dim: int,
        K: int = 5,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        target_means: Optional[np.ndarray] = None,
        target_stds: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ):
        self.K = K
        self.device = device
        self.input_dim = input_dim
        self.target_means = target_means
        self.target_stds = target_stds
        self.models: List[ResidualPINN] = []

        for k in range(K):
            torch.manual_seed(42 + k * 1007)
            model = ResidualPINN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_blocks=n_blocks,
                dropout=dropout,
                target_means=target_means,
                target_stds=target_stds
            ).to(device)
            self.models.append(model)

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        Eg_train: Optional[np.ndarray] = None,
        Jsc_max_train: Optional[np.ndarray] = None,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 2e-3
    ) -> List[Dict[str, float]]:
        history = []
        n_samples = len(X_train)

        # Compute target statistics if not already set
        if self.target_means is None or self.target_stds is None:
            self.target_means = np.mean(y_train, axis=0)
            self.target_stds = np.std(y_train, axis=0) + 1e-4

        for k in range(self.K):
            logger.info(f"Training PINN Member {k+1}/{self.K}...")
            model = self.models[k]
            # Update target stats buffers in model
            model.target_means.copy_(torch.tensor(self.target_means, dtype=torch.float32))
            model.target_stds.copy_(torch.tensor(self.target_stds, dtype=torch.float32))

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            loss_fn = CurriculumPhysicsLoss(
                total_epochs=epochs,
                ramp_epochs=max(5, epochs // 3),
                max_lambda=0.35,
                target_stds=self.target_stds
            ).to(self.device)

            x_t = torch.from_numpy(X_train).float()
            y_t = torch.from_numpy(y_train).float()
            eg_t = torch.from_numpy(Eg_train).float() if Eg_train is not None else torch.zeros(n_samples)
            jsc_t = torch.from_numpy(Jsc_max_train).float() if Jsc_max_train is not None else torch.zeros(n_samples)

            dataset = TensorDataset(x_t, y_t, eg_t, jsc_t)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                loss_fn.current_epoch = epoch
                model.train()
                for bx, by, beg, bjsc in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    beg = beg.to(self.device) if Eg_train is not None else None
                    bjsc = bjsc.to(self.device) if Jsc_max_train is not None else None

                    optimizer.zero_grad()
                    pred = model(bx)
                    loss, _ = loss_fn(pred, by, beg, bjsc)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()

                scheduler.step()

        return history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ensemble predictions in physical units.
        Returns:
            mu: Mean prediction (N, 4) in [PCE (%), Voc (V), Jsc (mA/cm^2), FF (0-1)]
            sigma: Epistemic standard deviation (N, 4) in matching physical units
        """
        all_preds = []
        x_t = torch.from_numpy(X).float().to(self.device)

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x_t).cpu().numpy()
                all_preds.append(pred)

        all_preds = np.stack(all_preds, axis=0)  # (K, N, 4)
        mu = np.mean(all_preds, axis=0)
        # Numerical minimum epistemic uncertainty floor
        sigma = np.maximum(np.std(all_preds, axis=0), 0.05)
        return mu, sigma

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if self.target_means is not None:
            np.save(os.path.join(directory, "target_means.npy"), self.target_means)
            np.save(os.path.join(directory, "target_stds.npy"), self.target_stds)
        for k, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(directory, f"pinn_member_{k}.pt"))

    def load(self, directory: str) -> None:
        means_p = os.path.join(directory, "target_means.npy")
        stds_p = os.path.join(directory, "target_stds.npy")
        if os.path.exists(means_p):
            self.target_means = np.load(means_p)
            self.target_stds = np.load(stds_p)
        for k, model in enumerate(self.models):
            path = os.path.join(directory, f"pinn_member_{k}.pt")
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device))
                if self.target_means is not None:
                    model.target_means.copy_(torch.tensor(self.target_means, dtype=torch.float32))
                    model.target_stds.copy_(torch.tensor(self.target_stds, dtype=torch.float32))


class UncertaintyCalibrator:
    """Post-hoc variance scaling calibration on validation set in physical units."""
    def __init__(self):
        self.scale_factors: np.ndarray = np.ones(4)

    def fit(self, y_true: np.ndarray, mu_val: np.ndarray, sigma_val: np.ndarray) -> 'UncertaintyCalibrator':
        """
        Fit scale factors s_j on physical units:
        s_j^2 = mean((y_ij - mu_ij)^2 / (sigma_ij^2 + eps))
        """
        eps = 1e-4
        residuals_sq = (y_true - mu_val) ** 2
        uncertainties_sq = sigma_val ** 2 + eps
        sq_ratios = residuals_sq / uncertainties_sq
        self.scale_factors = np.sqrt(np.mean(sq_ratios, axis=0))
        self.scale_factors = np.clip(self.scale_factors, 0.5, 60.0)
        logger.info(f"Uncertainty calibration scale factors: {self.scale_factors.round(3)}")
        return self

    def calibrate(self, sigma: np.ndarray) -> np.ndarray:
        return sigma * self.scale_factors

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma_cal: np.ndarray,
        levels: List[float] = [0.50, 0.80, 0.90, 0.95]
    ) -> Dict[str, List[float]]:
        results = {}
        for lvl in levels:
            z_crit = norm.ppf(0.5 + lvl / 2.0)
            low = mu - z_crit * sigma_cal
            high = mu + z_crit * sigma_cal
            in_bound = (y_true >= low) & (y_true <= high)
            cov = np.mean(in_bound, axis=0).tolist()
            results[f"coverage_{int(lvl*100)}%"] = cov
        return results

    def compute_nll(self, y_true: np.ndarray, mu: np.ndarray, sigma_cal: np.ndarray) -> float:
        var = sigma_cal ** 2 + 1e-4
        nll = 0.5 * np.log(2.0 * np.pi * var) + ((y_true - mu) ** 2) / (2.0 * var)
        return float(np.mean(nll))


class OODDetector:
    """
    Out-Of-Distribution (OOD) Detector combining:
    1. Mahalanobis distance in standardized feature space
    2. k-NN distance to training manifold
    """
    def __init__(self, k_neighbors: int = 15):
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.mahalanobis_threshold: float = 3.5
        self.knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', n_jobs=-1)
        self.median_train_knn: float = 1.0

    def fit(self, X_train: np.ndarray) -> 'OODDetector':
        self.mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train, rowvar=False) + 1e-3 * np.eye(X_train.shape[1])
        self.inv_cov = np.linalg.pinv(cov)

        self.knn.fit(X_train)
        dists, _ = self.knn.kneighbors(X_train)
        self.median_train_knn = float(np.median(dists[:, -1]))

        train_d_m = self.mahalanobis_distance(X_train)
        self.mahalanobis_threshold = float(np.percentile(train_d_m, 95.0))
        logger.info(f"OOD Detector fitted. Mahalanobis 95th thresh: {self.mahalanobis_threshold:.2f}, Median kNN: {self.median_train_knn:.3f}")
        return self

    def mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean
        left = np.dot(diff, self.inv_cov)
        d_m_sq = np.sum(left * diff, axis=1)
        return np.sqrt(np.maximum(d_m_sq, 0.0))

    def knn_distance(self, X: np.ndarray) -> np.ndarray:
        dists, _ = self.knn.kneighbors(X)
        return dists[:, -1]

    def score(self, X: np.ndarray) -> np.ndarray:
        d_m = self.mahalanobis_distance(X)
        d_knn = self.knn_distance(X)
        norm_m = d_m / max(self.mahalanobis_threshold, 1e-3)
        norm_knn = d_knn / max(self.median_train_knn, 1e-3)
        return 0.5 * norm_m + 0.5 * norm_knn

    def is_ood(self, X: np.ndarray, threshold: float = 1.3) -> np.ndarray:
        return self.score(X) > threshold
