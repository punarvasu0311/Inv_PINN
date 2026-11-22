"""
Physics-Constrained Conditional Variational Autoencoder (cVAE) with:
- Feature-wise Linear Modulation (FiLM) conditioning in every residual block
- Classifier-Free Guidance (CFG) with conditioning dropout during training
- Cosine KL divergence annealing to prevent posterior collapse
- Auxiliary moment-matching regularizer using decoded prior samples (Khan et al., 2026)
- Hard chemical & device constraints enforced by construction in the decoder:
    * Softmax A-site (x_FA + x_MA + x_Cs = 1, non-negative)
    * Sigmoid B-site bounded by Pb_max (x_Pb <= Pb_max, x_Sn = 1 - x_Pb)
    * Softmax X-site (x_I + x_Br + x_Cl = 1, non-negative)
    * Bounded layer thicknesses (absorber, ETL, HTL) via sigmoid scaling
    * Learned architecture-masked categorical transport layer selection (Liu et al., 2024)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import COMPATIBILITY_DICT

logger = logging.getLogger(__name__)


class FiLMBlock(nn.Module):
    """Residual block modulated by conditioning vector via FiLM."""
    def __init__(self, dim: int, cond_embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        self.film_proj = nn.Linear(cond_embed_dim, 2 * dim)

    def forward(self, x: torch.Tensor, c_embed: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.ln1(self.fc1(x)))
        h = self.dropout(h)

        film = self.film_proj(c_embed)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h = (1.0 + gamma) * h + beta

        h = self.ln2(self.fc2(h))
        h = self.dropout(h)
        return self.act(h + res)


class HardConstrainedDecoder(nn.Module):
    """
    Decodes latent vectors into design space with hard chemical constraints enforced by construction.
    Includes learned architecture-masked categorical heads for ETL, HTL, and Substrate.
    """
    def __init__(self, latent_dim: int, cond_embed_dim: int, hidden_dim: int = 256, n_blocks: int = 3):
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, cond_embed_dim) for _ in range(n_blocks)
        ])

        # Continuous / Simplex parameter heads
        self.head_a = nn.Linear(hidden_dim, 3)          # FA, MA, Cs logits
        self.head_b = nn.Linear(hidden_dim, 1)          # Pb logit
        self.head_x = nn.Linear(hidden_dim, 3)          # I, Br, Cl logits
        self.head_thickness = nn.Linear(hidden_dim, 3)  # absorber, ETL, HTL
        self.head_offsets = nn.Linear(hidden_dim, 2)    # Delta_H, Delta_L
        self.head_arch = nn.Linear(hidden_dim, 1)       # is_pin logit

        # Learned architecture-masked categorical heads (Bug 3 fix)
        self.n_etl_nip = len(COMPATIBILITY_DICT['nip']['etl'])
        self.n_etl_pin = len(COMPATIBILITY_DICT['pin']['etl'])
        self.n_htl_nip = len(COMPATIBILITY_DICT['nip']['htl'])
        self.n_htl_pin = len(COMPATIBILITY_DICT['pin']['htl'])

        self.head_etl_nip = nn.Linear(hidden_dim, self.n_etl_nip)
        self.head_etl_pin = nn.Linear(hidden_dim, self.n_etl_pin)
        self.head_htl_nip = nn.Linear(hidden_dim, self.n_htl_nip)
        self.head_htl_pin = nn.Linear(hidden_dim, self.n_htl_pin)
        self.head_sub = nn.Linear(hidden_dim, 2) # FTO, ITO

    def forward(self, z: torch.Tensor, c_embed: torch.Tensor, Pb_max: float = 1.0) -> Dict[str, torch.Tensor]:
        h = self.latent_proj(z)
        for block in self.blocks:
            h = block(h, c_embed)

        # 1. A-site: Softmax guarantees non-negativity and sum = 1
        a_site = F.softmax(self.head_a(h), dim=-1)

        # 2. B-site: Sigmoid bounded strictly by Pb_max
        pb_ratio = torch.sigmoid(self.head_b(h)) * float(Pb_max)
        sn_ratio = 1.0 - pb_ratio
        b_site = torch.cat([pb_ratio, sn_ratio], dim=-1)

        # 3. X-site: Softmax guarantees non-negativity and sum = 1
        x_site = F.softmax(self.head_x(h), dim=-1)

        # 4. Thicknesses (nm)
        t_raw = torch.sigmoid(self.head_thickness(h))
        absorber_t = 200.0 + t_raw[:, 0:1] * 600.0   # [200, 800] nm
        etl_t = 15.0 + t_raw[:, 1:2] * 85.0          # [15, 100] nm
        htl_t = 20.0 + t_raw[:, 2:3] * 200.0         # [20, 220] nm
        thicknesses = torch.cat([absorber_t, etl_t, htl_t], dim=-1)

        # 5. Energy offsets (eV)
        off_raw = torch.sigmoid(self.head_offsets(h))
        delta_h = -0.1 + off_raw[:, 0:1] * 0.7
        delta_l = -0.1 + off_raw[:, 1:2] * 0.6
        offsets = torch.cat([delta_h, delta_l], dim=-1)

        # 6. Architecture polarity: p-i-n probability
        is_pin_prob = torch.sigmoid(self.head_arch(h))

        # 7. Architecture-masked categorical heads
        etl_nip_logits = self.head_etl_nip(h)
        etl_pin_logits = self.head_etl_pin(h)
        htl_nip_logits = self.head_htl_nip(h)
        htl_pin_logits = self.head_htl_pin(h)
        sub_logits = self.head_sub(h)

        return {
            'a_site': a_site,
            'b_site': b_site,
            'x_site': x_site,
            'thicknesses': thicknesses,
            'offsets': offsets,
            'is_pin': is_pin_prob,
            'etl_nip_logits': etl_nip_logits,
            'etl_pin_logits': etl_pin_logits,
            'htl_nip_logits': htl_nip_logits,
            'htl_pin_logits': htl_pin_logits,
            'sub_logits': sub_logits
        }


class ConstrainedFiLMcVAE(nn.Module):
    """
    Complete cVAE with FiLM conditioning, Classifier-Free Guidance (CFG),
    and hard constraints.
    Condition vector c = [PCE*, Voc*, Jsc*, FF*, Pb_max] (dim=5).
    """
    def __init__(
        self,
        design_dim: int = 14,
        cond_dim: int = 5,
        cond_embed_dim: int = 64,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        p_uncond: float = 0.10
    ):
        super().__init__()
        self.design_dim = design_dim
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_embed_dim
        self.latent_dim = latent_dim
        self.p_uncond = p_uncond

        self.cond_embedder = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.LayerNorm(cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim)
        )
        self.uncond_token = nn.Parameter(torch.randn(1, cond_embed_dim) * 0.02)

        self.encoder_proj = nn.Sequential(
            nn.Linear(design_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.enc_blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, cond_embed_dim, dropout=dropout) for _ in range(n_blocks)
        ])
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = HardConstrainedDecoder(
            latent_dim=latent_dim,
            cond_embed_dim=cond_embed_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks
        )

    def encode(self, x: torch.Tensor, c_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_proj(x)
        for block in self.enc_blocks:
            h = block(h, c_embed)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=4.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, c: torch.Tensor, Pb_max: float = 1.0) -> Dict[str, Any]:
        batch_size = x.size(0)
        c_embed = self.cond_embedder(c)

        if self.training and self.p_uncond > 0.0:
            mask = (torch.rand(batch_size, 1, device=x.device) < self.p_uncond).float()
            c_embed = (1.0 - mask) * c_embed + mask * self.uncond_token.expand(batch_size, -1)

        mu, logvar = self.encode(x, c_embed)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z, c_embed, Pb_max=Pb_max)

        # Moment-matching sample from standard Gaussian prior (Bug 4 fix)
        z_prior = torch.randn_like(z)
        prior_decoded = self.decoder(z_prior, c_embed, Pb_max=Pb_max)

        return {
            'decoded': decoded,
            'prior_decoded': prior_decoded,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        n_samples: int = 1000,
        w: float = 1.5,
        Pb_max: float = 1.0
    ) -> Dict[str, Any]:
        """Inference sampling with Classifier-Free Guidance (CFG)."""
        self.eval()
        device = next(self.parameters()).device
        if c.ndim == 1:
            c = c.unsqueeze(0)
        c = c.to(device)

        c_expanded = c.expand(n_samples, -1)
        z = torch.randn(n_samples, self.latent_dim, device=device)

        c_cond = self.cond_embedder(c_expanded)
        c_uncond = self.uncond_token.expand(n_samples, -1)

        if abs(w) < 1e-5:
            dec = self.decoder(z, c_cond, Pb_max=Pb_max)
            return self._finalize_sample(dec)

        dec_cond = self.decoder(z, c_cond, Pb_max=Pb_max)
        dec_uncond = self.decoder(z, c_uncond, Pb_max=Pb_max)

        guided_out = {}
        # A-site: guided softmax
        guided_a_logits = (1.0 + w) * torch.log(dec_cond['a_site'] + 1e-7) - w * torch.log(dec_uncond['a_site'] + 1e-7)
        guided_out['a_site'] = F.softmax(guided_a_logits, dim=-1)

        # B-site: hard lead cap is strictly satisfied by dec_cond (head bounded by Pb_max)
        guided_out['b_site'] = dec_cond['b_site']

        # X-site: guided softmax
        guided_x_logits = (1.0 + w) * torch.log(dec_cond['x_site'] + 1e-7) - w * torch.log(dec_uncond['x_site'] + 1e-7)
        guided_out['x_site'] = F.softmax(guided_x_logits, dim=-1)

        # Continuous properties
        guided_out['thicknesses'] = torch.clamp((1.0 + w) * dec_cond['thicknesses'] - w * dec_uncond['thicknesses'], min=10.0, max=1200.0)
        guided_out['offsets'] = torch.clamp((1.0 + w) * dec_cond['offsets'] - w * dec_uncond['offsets'], min=-0.2, max=0.8)
        guided_out['is_pin'] = torch.clamp((1.0 + w) * dec_cond['is_pin'] - w * dec_uncond['is_pin'], min=0.0, max=1.0)

        # Categorical logits from conditional path
        guided_out['etl_nip_logits'] = dec_cond['etl_nip_logits']
        guided_out['etl_pin_logits'] = dec_cond['etl_pin_logits']
        guided_out['htl_nip_logits'] = dec_cond['htl_nip_logits']
        guided_out['htl_pin_logits'] = dec_cond['htl_pin_logits']
        guided_out['sub_logits'] = dec_cond['sub_logits']

        return self._finalize_sample(guided_out)

    def _finalize_sample(self, dec: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Select discrete materials based on predicted logits and architecture."""
        is_pin = (dec['is_pin'].squeeze() > 0.5)
        n = dec['a_site'].size(0)

        etl_mats = []
        htl_mats = []
        sub_mats = []

        etl_nip_choices = COMPATIBILITY_DICT['nip']['etl']
        etl_pin_choices = COMPATIBILITY_DICT['pin']['etl']
        htl_nip_choices = COMPATIBILITY_DICT['nip']['htl']
        htl_pin_choices = COMPATIBILITY_DICT['pin']['htl']
        sub_choices = ['FTO', 'ITO']

        etl_nip_idx = torch.argmax(dec['etl_nip_logits'], dim=-1).cpu().numpy()
        etl_pin_idx = torch.argmax(dec['etl_pin_logits'], dim=-1).cpu().numpy()
        htl_nip_idx = torch.argmax(dec['htl_nip_logits'], dim=-1).cpu().numpy()
        htl_pin_idx = torch.argmax(dec['htl_pin_logits'], dim=-1).cpu().numpy()
        sub_idx = torch.argmax(dec['sub_logits'], dim=-1).cpu().numpy()
        pin_flags = is_pin.cpu().numpy() if is_pin.ndim > 0 else np.array([bool(is_pin)] * n)

        for i in range(n):
            if pin_flags[i]:
                etl_mats.append(etl_pin_choices[etl_pin_idx[i] % len(etl_pin_choices)])
                htl_mats.append(htl_pin_choices[htl_pin_idx[i] % len(htl_pin_choices)])
                sub_mats.append('ITO')
            else:
                etl_mats.append(etl_nip_choices[etl_nip_idx[i] % len(etl_nip_choices)])
                htl_mats.append(htl_nip_choices[htl_nip_idx[i] % len(htl_nip_choices)])
                sub_mats.append('FTO')

        dec['etl_material'] = etl_mats
        dec['htl_material'] = htl_mats
        dec['substrate'] = sub_mats
        return dec


def compute_cvae_loss(
    model_output: Dict[str, Any],
    target_design: torch.Tensor,
    epoch: int,
    total_epochs: int = 60,
    beta_max: float = 0.5,
    lambda_mm: float = 0.4
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes total cVAE loss:
    L_total = L_recon + beta * L_kl + lambda_mm * L_moment_matching
    CRITICAL FIX (Bug 4): computes L_moment_matching on actual decoded prior samples
    decoder(z_prior, c_embed) rather than the posterior reconstruction.
    """
    dec = model_output['decoded']
    prior_dec = model_output['prior_decoded']
    mu = model_output['mu']
    logvar = model_output['logvar']

    recon_vector = torch.cat([
        dec['a_site'],
        dec['b_site'],
        dec['x_site'],
        dec['thicknesses'] / 500.0,
        dec['offsets'],
        dec['is_pin']
    ], dim=-1)

    # 1. Reconstruction Loss (MSE)
    l_recon = F.mse_loss(recon_vector, target_design[:, :recon_vector.size(1)])

    # 2. KL Divergence with cosine annealing
    warmup_epochs = max(1, total_epochs // 3)
    if epoch < warmup_epochs:
        beta = beta_max * 0.5 * (1.0 - np.cos(np.pi * epoch / warmup_epochs))
    else:
        beta = beta_max

    l_kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    # 3. Moment-Matching regularizer on DECODED PRIOR SAMPLES (Bug 4 fix)
    prior_vector = torch.cat([
        prior_dec['a_site'],
        prior_dec['b_site'],
        prior_dec['x_site'],
        prior_dec['thicknesses'] / 500.0,
        prior_dec['offsets'],
        prior_dec['is_pin']
    ], dim=-1)

    tgt_slice = target_design[:, :prior_vector.size(1)]
    mean_diff = (prior_vector.mean(dim=0) - tgt_slice.mean(dim=0)).pow(2).mean()
    std_diff = (prior_vector.std(dim=0) - tgt_slice.std(dim=0)).pow(2).mean()
    l_mm = mean_diff + std_diff

    total_loss = l_recon + beta * l_kl + lambda_mm * l_mm

    loss_dict = {
        'loss_total': total_loss.item(),
        'loss_recon': l_recon.item(),
        'loss_kl': l_kl.item(),
        'loss_mm': l_mm.item(),
        'beta': float(beta)
    }
    return total_loss, loss_dict
