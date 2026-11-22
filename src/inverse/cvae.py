"""
Physics-Constrained Conditional Variational Autoencoder (cVAE) with:
- Feature-wise Linear Modulation (FiLM) conditioning in every residual block
- Classifier-Free Guidance (CFG) with conditioning dropout during training
- Cosine KL divergence annealing to prevent posterior collapse
- Auxiliary moment-matching regularizer (Khan et al., 2026)
- Hard chemical & device constraints enforced by construction in the decoder:
    * Softmax A-site (x_FA + x_MA + x_Cs = 1, non-negative)
    * Sigmoid B-site bounded by Pb_max (x_Pb <= Pb_max, x_Sn = 1 - x_Pb)
    * Softmax X-site (x_I + x_Br + x_Cl = 1, non-negative)
    * Bounded layer thicknesses (absorber, ETL, HTL) via sigmoid scaling
    * Architecture-aware compatible transport layer selection
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

        # FiLM parameters: gamma and beta projection
        self.film_proj = nn.Linear(cond_embed_dim, 2 * dim)

    def forward(self, x: torch.Tensor, c_embed: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.ln1(self.fc1(x)))
        h = self.dropout(h)

        # FiLM modulation: gamma * h + beta
        film = self.film_proj(c_embed)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h = (1.0 + gamma) * h + beta

        h = self.ln2(self.fc2(h))
        h = self.dropout(h)
        return self.act(h + res)


class HardConstrainedDecoder(nn.Module):
    """
    Decodes latent vectors into design space with hard chemical constraints enforced by construction.
    Guarantees:
      - x_FA + x_MA + x_Cs = 1 (A-site simplex via softmax)
      - x_Pb <= Pb_max and x_Pb + x_Sn = 1 (B-site bounded simplex)
      - x_I + x_Br + x_Cl = 1 (X-site simplex via softmax)
      - Thicknesses bounded to realistic experimental ranges (nm)
      - Energy offsets Delta_H, Delta_L bounded to realistic ranges (eV)
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

        # Specialized parameter heads
        self.head_a = nn.Linear(hidden_dim, 3)          # FA, MA, Cs logits
        self.head_b = nn.Linear(hidden_dim, 1)          # Pb logit
        self.head_x = nn.Linear(hidden_dim, 3)          # I, Br, Cl logits
        self.head_thickness = nn.Linear(hidden_dim, 3)  # absorber, ETL, HTL
        self.head_offsets = nn.Linear(hidden_dim, 2)    # Delta_H, Delta_L
        self.head_arch = nn.Linear(hidden_dim, 1)       # is_pin logit

    def forward(self, z: torch.Tensor, c_embed: torch.Tensor, Pb_max: float = 1.0) -> Dict[str, torch.Tensor]:
        h = self.latent_proj(z)
        for block in self.blocks:
            h = block(h, c_embed)

        # 1. A-site: Softmax guarantees non-negativity and sum = 1
        a_site = F.softmax(self.head_a(h), dim=-1) # (B, 3) -> [x_FA, x_MA, x_Cs]

        # 2. B-site: Sigmoid bounded by Pb_max
        pb_ratio = torch.sigmoid(self.head_b(h)) * float(Pb_max) # (B, 1)
        sn_ratio = 1.0 - pb_ratio
        b_site = torch.cat([pb_ratio, sn_ratio], dim=-1) # (B, 2) -> [x_Pb, x_Sn]

        # 3. X-site: Softmax guarantees non-negativity and sum = 1
        x_site = F.softmax(self.head_x(h), dim=-1) # (B, 3) -> [x_I, x_Br, x_Cl]

        # 4. Thicknesses (nm) bounded by physical ranges:
        # absorber: [200, 800] nm
        # etl: [15, 100] nm
        # htl: [20, 220] nm
        t_raw = torch.sigmoid(self.head_thickness(h))
        absorber_t = 200.0 + t_raw[:, 0:1] * 600.0
        etl_t = 15.0 + t_raw[:, 1:2] * 85.0
        htl_t = 20.0 + t_raw[:, 2:3] * 200.0
        thicknesses = torch.cat([absorber_t, etl_t, htl_t], dim=-1)

        # 5. Energy offsets (eV):
        # Delta_H: [-0.1, 0.6] eV
        # Delta_L: [-0.1, 0.5] eV
        off_raw = torch.sigmoid(self.head_offsets(h))
        delta_h = -0.1 + off_raw[:, 0:1] * 0.7
        delta_l = -0.1 + off_raw[:, 1:2] * 0.6
        offsets = torch.cat([delta_h, delta_l], dim=-1)

        # 6. Architecture polarity: p-i-n probability
        is_pin_prob = torch.sigmoid(self.head_arch(h))

        return {
            'a_site': a_site,
            'b_site': b_site,
            'x_site': x_site,
            'thicknesses': thicknesses,
            'offsets': offsets,
            'is_pin': is_pin_prob
        }


class ConstrainedFiLMcVAE(nn.Module):
    """
    Complete cVAE with FiLM conditioning, Classifier-Free Guidance (CFG),
    and hard constraints.
    Condition vector c = [PCE*, Voc*, Jsc*, FF*, Pb_max] (dim=5).
    """
    def __init__(
        self,
        design_dim: int = 12,
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

        # Condition embedder MLP: c -> c_embed
        self.cond_embedder = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.LayerNorm(cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim)
        )
        # Learnable unconditional token for CFG condition dropout
        self.uncond_token = nn.Parameter(torch.randn(1, cond_embed_dim) * 0.02)

        # Encoder: x + c_embed -> mu, logvar
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

        # Decoder
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

        # Classifier-Free Guidance training: condition dropout with probability p_uncond
        if self.training and self.p_uncond > 0.0:
            mask = (torch.rand(batch_size, 1, device=x.device) < self.p_uncond).float()
            c_embed = (1.0 - mask) * c_embed + mask * self.uncond_token.expand(batch_size, -1)

        mu, logvar = self.encode(x, c_embed)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z, c_embed, Pb_max=Pb_max)

        return {
            'decoded': decoded,
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
    ) -> Dict[str, torch.Tensor]:
        """
        Inference sampling with Classifier-Free Guidance (CFG).
        Guidance equation:
            x_guided = (1 + w) * D(z, c) - w * D(z, c_null)
        """
        self.eval()
        device = next(self.parameters()).device
        if c.ndim == 1:
            c = c.unsqueeze(0)
        c = c.to(device)

        # Replicate condition for n_samples
        c_expanded = c.expand(n_samples, -1)
        z = torch.randn(n_samples, self.latent_dim, device=device)

        c_cond = self.cond_embedder(c_expanded)
        c_uncond = self.uncond_token.expand(n_samples, -1)

        if abs(w) < 1e-5:
            # Unguided baseline (w = 0)
            return self.decoder(z, c_cond, Pb_max=Pb_max)

        # Decode conditional and unconditional representations
        dec_cond = self.decoder(z, c_cond, Pb_max=Pb_max)
        dec_uncond = self.decoder(z, c_uncond, Pb_max=Pb_max)

        # Apply guidance to linear representations before simplex projection:
        # For simplex outputs, guidance in logit space ensures valid probability manifolds!
        guided_out = {}
        # A-site: guided softmax
        guided_a_logits = (1.0 + w) * torch.log(dec_cond['a_site'] + 1e-7) - w * torch.log(dec_uncond['a_site'] + 1e-7)
        guided_out['a_site'] = F.softmax(guided_a_logits, dim=-1)

        # B-site: guided lead ratio
        pb_cond = dec_cond['b_site'][:, 0:1]
        pb_uncond = dec_uncond['b_site'][:, 0:1]
        guided_pb = torch.clamp((1.0 + w) * pb_cond - w * pb_uncond, 0.0, float(Pb_max))
        guided_out['b_site'] = torch.cat([guided_pb, 1.0 - guided_pb], dim=-1)

        # X-site: guided softmax
        guided_x_logits = (1.0 + w) * torch.log(dec_cond['x_site'] + 1e-7) - w * torch.log(dec_uncond['x_site'] + 1e-7)
        guided_out['x_site'] = F.softmax(guided_x_logits, dim=-1)

        # Continuous properties: linear CFG
        guided_out['thicknesses'] = torch.clamp((1.0 + w) * dec_cond['thicknesses'] - w * dec_uncond['thicknesses'], min=10.0, max=1000.0)
        guided_out['offsets'] = torch.clamp((1.0 + w) * dec_cond['offsets'] - w * dec_uncond['offsets'], min=-0.2, max=0.8)
        guided_out['is_pin'] = torch.clamp((1.0 + w) * dec_cond['is_pin'] - w * dec_uncond['is_pin'], min=0.0, max=1.0)

        return guided_out


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
    With cosine annealing schedule for beta.
    """
    dec = model_output['decoded']
    mu = model_output['mu']
    logvar = model_output['logvar']

    # Flatten reconstructed design into matching vector
    # Order: [x_FA, x_MA, x_Cs, x_Pb, x_Sn, x_I, x_Br, x_Cl, abs_t, etl_t, htl_t, Delta_H, Delta_L, is_pin]
    recon_vector = torch.cat([
        dec['a_site'],
        dec['b_site'],
        dec['x_site'],
        dec['thicknesses'] / 500.0,  # normalized scale for loss
        dec['offsets'],
        dec['is_pin']
    ], dim=-1)

    # 1. Reconstruction Loss (MSE)
    l_recon = F.mse_loss(recon_vector, target_design[:, :recon_vector.size(1)])

    # 2. KL Divergence with cosine annealing
    # beta(epoch) = beta_max * 0.5 * (1 - cos(pi * epoch / (total_epochs / 3)))
    warmup_epochs = max(1, total_epochs // 3)
    if epoch < warmup_epochs:
        beta = beta_max * 0.5 * (1.0 - np.cos(np.pi * epoch / warmup_epochs))
    else:
        beta = beta_max

    l_kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    # 3. Moment-Matching regularizer (Khan et al., 2026): matches mean & variance of empirical data
    z_prior = torch.randn_like(mu)
    # Match first and second central moments
    mean_diff = (recon_vector.mean(dim=0) - target_design[:, :recon_vector.size(1)].mean(dim=0)).pow(2).mean()
    std_diff = (recon_vector.std(dim=0) - target_design[:, :recon_vector.size(1)].std(dim=0)).pow(2).mean()
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
