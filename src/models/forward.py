"""
Forward surrogate baseline models:
- Multi-output Random Forest (RF)
- Multi-output XGBoost
- Multi-output LightGBM
- Data-only Deep Multi-Layer Perceptron (MLP)
- SHAP feature attribution
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap

logger = logging.getLogger(__name__)

TARGET_NAMES = ['PCE', 'Voc', 'Jsc', 'FF']


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = TARGET_NAMES) -> Dict[str, float]:
    """Compute RMSE, MAE, R^2 for each target and overall averages."""
    metrics = {}
    rmse_list, mae_list, r2_list = [], [], []
    for i, name in enumerate(target_names):
        if i < y_true.shape[1] and i < y_pred.shape[1]:
            yt = y_true[:, i]
            yp = y_pred[:, i]
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            mae = float(mean_absolute_error(yt, yp))
            r2 = float(r2_score(yt, yp))
            metrics[f"{name}_RMSE"] = rmse
            metrics[f"{name}_MAE"] = mae
            metrics[f"{name}_R2"] = r2
            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)
    metrics["Mean_RMSE"] = float(np.mean(rmse_list))
    metrics["Mean_MAE"] = float(np.mean(mae_list))
    metrics["Mean_R2"] = float(np.mean(r2_list))
    return metrics


class ForwardRF:
    """Random Forest multi-output forward surrogate."""
    def __init__(self, n_estimators: int = 300, max_depth: int = 18, random_state: int = 42):
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ForwardRF':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ForwardXGBoost:
    """XGBoost multi-output forward surrogate."""
    def __init__(self, n_estimators: int = 400, max_depth: int = 6, learning_rate: float = 0.08, random_state: int = 42):
        self.model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                tree_method='hist',
                n_jobs=-1
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ForwardXGBoost':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def explain_shap(self, X_sample: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values for the primary target (PCE, index 0)."""
        estimator_pce = self.model.estimators_[0]
        explainer = shap.TreeExplainer(estimator_pce)
        shap_values = explainer.shap_values(X_sample)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        return shap_values, mean_abs_shap


class ForwardLightGBM:
    """LightGBM multi-output forward surrogate."""
    def __init__(self, n_estimators: int = 400, max_depth: int = 8, learning_rate: float = 0.08, random_state: int = 42):
        self.model = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                verbosity=-1,
                n_jobs=-1
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ForwardLightGBM':
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class DataOnlyMLP(nn.Module):
    """Deep Multi-Layer Perceptron baseline without physics loss."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ForwardMLPWrapper:
    """PyTorch training wrapper for DataOnlyMLP."""
    def __init__(self, input_dim: int, epochs: int = 80, batch_size: int = 128, lr: float = 1e-3, device: str = 'cpu'):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = DataOnlyMLP(input_dim=input_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ForwardMLPWrapper':
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).float().to(self.device)
            pred = self.model(x_t).cpu().numpy()
        return pred
