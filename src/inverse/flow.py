"""
Conditional Normalizing Flow (Conditional RealNVP) for perovskite inverse design.
Learns p(design | c) directly with exact likelihood and invertible architecture.
Condition vector: c = [PCE*, Voc*, Jsc*, FF*, Pb_max].
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def transform_to_unconstrained(
    x: torch.Tensor,
    pb_max: Optional[torch.Tensor] = None,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    Transform raw physical design parameters into unconstrained space
    with small dequantization noise so RealNVP can model them without boundary singularities.
    x has shape (N, 14):
      [0:3]   -> A-site (FA, MA, Cs) summing to 1
      [3:5]   -> B-site (Pb, Sn) summing to 1
      [5:8]   -> X-site (I, Br, Cl) summing to 1
      [8:11]  -> thicknesses (absorber, ETL, HTL) in nm
      [11:13] -> offsets (Delta_H, Delta_L) in eV
      [13:14] -> is_pin (binary 0 or 1)
    """
    device = x.device
    noise = torch.randn_like(x) * 1e-3

    # 1. A-site: [0:3]
    a = torch.clamp(x[:, 0:3] + torch.abs(noise[:, 0:3]), min=eps)
    a = a / a.sum(dim=-1, keepdim=True)
    logit_a = torch.log(a) - torch.mean(torch.log(a), dim=-1, keepdim=True)

    # 2. B-site: [3:5] (Pb and Sn)
    if pb_max is not None:
        p_cap = torch.clamp(pb_max, min=0.05, max=1.0)
        if p_cap.ndim == 1:
            p_cap = p_cap.unsqueeze(-1)
        pb_norm = torch.clamp(x[:, 3:4] / p_cap, min=eps, max=1.0 - eps)
    else:
        pb_norm = torch.clamp(x[:, 3:4], min=eps, max=1.0 - eps)
    logit_pb = torch.logit(pb_norm)
    sn_norm = torch.clamp(x[:, 4:5], min=eps, max=1.0 - eps)
    logit_sn = torch.logit(sn_norm)

    # 3. X-site: [5:8]
    x_site = torch.clamp(x[:, 5:8] + torch.abs(noise[:, 5:8]), min=eps)
    x_site = x_site / x_site.sum(dim=-1, keepdim=True)
    logit_x = torch.log(x_site) - torch.mean(torch.log(x_site), dim=-1, keepdim=True)

    # 4. Thicknesses: [8:11] (absorber in [200, 800], etl in [15, 100], htl in [20, 220])
    t_abs = x[:, 8:9] * 500.0 if x[:, 8:9].mean() < 10.0 else x[:, 8:9]
    t_etl = x[:, 9:10] * 500.0 if x[:, 9:10].mean() < 10.0 else x[:, 9:10]
    t_htl = x[:, 10:11] * 500.0 if x[:, 10:11].mean() < 10.0 else x[:, 10:11]

    u_abs = torch.clamp((t_abs - 200.0) / 600.0, min=eps, max=1.0 - eps)
    u_etl = torch.clamp((t_etl - 15.0) / 85.0, min=eps, max=1.0 - eps)
    u_htl = torch.clamp((t_htl - 20.0) / 200.0, min=eps, max=1.0 - eps)
    logit_t = torch.logit(torch.cat([u_abs, u_etl, u_htl], dim=-1))

    # 5. Energy offsets: [11:13] (Delta_H in [-0.1, 0.6], Delta_L in [-0.1, 0.5])
    u_dh = torch.clamp((x[:, 11:12] - (-0.1)) / 0.7, min=eps, max=1.0 - eps)
    u_dl = torch.clamp((x[:, 12:13] - (-0.1)) / 0.6, min=eps, max=1.0 - eps)
    logit_off = torch.logit(torch.cat([u_dh, u_dl], dim=-1))

    # 6. is_pin: [13:14] (binary with dequantization)
    u_pin = torch.clamp(x[:, 13:14] * 0.8 + 0.1 + noise[:, 13:14] * 0.05, min=eps, max=1.0 - eps)
    logit_pin = torch.logit(u_pin)

    return torch.cat([logit_a, logit_pb, logit_sn, logit_x, logit_t, logit_off, logit_pin], dim=-1)


class ConditionalAffineCoupling(nn.Module):
    """
    Conditional Affine Coupling Layer:
    Splits input into x_A and x_B using alternating mask.
    Computes scale s(x_A, c) and translation t(x_A, c).
    Transforms x_B' = x_B * exp(s) + t.
    """
    def __init__(self, dim: int, cond_dim: int, hidden_dim: int = 128, mask_type: str = 'even'):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type

        # Binary mask: alternating 1 and 0
        mask = torch.zeros(dim)
        if mask_type == 'even':
            mask[0::2] = 1.0
        else:
            mask[1::2] = 1.0
        self.register_buffer('mask', mask)

        # Scale and translation networks conditioned on x_masked and c
        in_dim = dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dim)
        )
        # Initialize last layer weights near zero for identity initial mapping
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward mapping: x -> z with log-determinant."""
        x_masked = x * self.mask
        inp = torch.cat([x_masked, c], dim=-1)
        params = self.net(inp)
        s, t = torch.chunk(params, 2, dim=-1)

        # Restrict scale to prevent gradient explosion
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        z = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Inverse mapping: z -> x."""
        z_masked = z * self.mask
        inp = torch.cat([z_masked, c], dim=-1)
        params = self.net(inp)
        s, t = torch.chunk(params, 2, dim=-1)

        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        x = z_masked + (1.0 - self.mask) * ((z - t) * torch.exp(-s))
        return x


class ConditionalRealNVP(nn.Module):
    """
    Multi-layer Conditional RealNVP Normalizing Flow.
    Models p(design | c) using standard Gaussian base distribution.
    """
    def __init__(
        self,
        design_dim: int = 14,
        cond_dim: int = 5,
        cond_embed_dim: int = 64,
        n_layers: int = 6,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.design_dim = design_dim
        self.cond_dim = cond_dim

        # Condition embedding MLP
        self.cond_embedder = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.LayerNorm(cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim)
        )

        # Coupling layers with alternating masks
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(
                dim=design_dim,
                cond_dim=cond_embed_dim,
                hidden_dim=hidden_dim,
                mask_type='even' if i % 2 == 0 else 'odd'
            )
            for i in range(n_layers)
        ])

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log p(x | c) = log p(z) + sum(log |det J|).
        """
        c_embed = self.cond_embedder(c)
        z = x
        log_det_sum = torch.zeros(x.size(0), device=x.device)

        for layer in self.layers:
            z, log_det = layer(z, c_embed)
            log_det_sum = log_det_sum + log_det

        # Standard multivariate normal base density: log N(z; 0, I)
        log_pz = -0.5 * (self.design_dim * math.log(2.0 * math.pi) + torch.sum(z.pow(2), dim=-1))
        log_px = log_pz + log_det_sum
        return log_px, z

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n_samples: int = 1000, Pb_max: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Sample candidates x ~ p(x | c).
        Applies physical constraints & post-projection to chemical simplex.
        """
        self.eval()
        device = next(self.parameters()).device
        if c.ndim == 1:
            c = c.unsqueeze(0)
        c = c.to(device)

        c_expanded = c.expand(n_samples, -1)
        c_embed = self.cond_embedder(c_expanded)

        # Sample z from base standard normal
        z = torch.randn(n_samples, self.design_dim, device=device)

        # Invert through coupling layers in reverse order
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, c_embed)

        # Map unconstrained flow outputs into physically valid chemical design spaces (14 dims):
        # [0:3]   -> A-site logits
        # [3:4]   -> Pb ratio logit
        # [4:5]   -> Sn logit
        # [5:8]   -> X-site logits (I, Br, Cl)
        # [8:11]  -> thicknesses raw (absorber, ETL, HTL)
        # [11:13] -> offsets raw (Delta_H, Delta_L)
        # [13:14] -> is_pin raw

        a_site = F.softmax(x[:, 0:3], dim=-1)
        pb_ratio = torch.sigmoid(x[:, 3:4]) * float(Pb_max)
        b_site = torch.cat([pb_ratio, 1.0 - pb_ratio], dim=-1)
        x_site = F.softmax(x[:, 5:8], dim=-1)

        t_raw = torch.sigmoid(x[:, 8:11])
        thicknesses = torch.cat([
            200.0 + t_raw[:, 0:1] * 600.0,
            15.0 + t_raw[:, 1:2] * 85.0,
            20.0 + t_raw[:, 2:3] * 200.0
        ], dim=-1)

        off_raw = torch.sigmoid(x[:, 11:13])
        offsets = torch.cat([
            -0.1 + off_raw[:, 0:1] * 0.7,
            -0.1 + off_raw[:, 1:2] * 0.6
        ], dim=-1)

        is_pin = torch.sigmoid(x[:, 13:14])

        return {
            'a_site': a_site,
            'b_site': b_site,
            'x_site': x_site,
            'thicknesses': thicknesses,
            'offsets': offsets,
            'is_pin': is_pin
        }
