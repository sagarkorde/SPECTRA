"""
SPECTRA — Module C: Variational Autoencoder (VAE)
==================================================
The VAE learns a compact latent representation z in R^d of each Bitcoin
transaction's feature vector.  The reconstruction error serves as an
unsupervised anomaly score: unusual transactions (outside the learned
distribution) incur higher ELBO loss.

Architecture
------------
Encoder q_phi(z|x)  : x → [mu(x), sigma²(x)]  via two FC hidden layers
Decoder p_θ(x|z)  : z → x̂              via symmetric hidden layers

Loss (ELBO)
-----------
  L = E_q[log p(x|z)] − beta · KL(q(z|x) ∥ p(z))
    = −||x − x̂||² − beta · (−½ sum(1 + log sigma² − mu² − sigma²))

The beta weight (default 1.0) follows beta-VAE convention; increasing beta
promotes disentanglement at the cost of reconstruction quality.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER / DECODER / VAE
# ─────────────────────────────────────────────────────────────────────────────

class VAEEncoder(nn.Module):
    """Maps input x to Gaussian parameters (mu, log sigma²) in latent space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            in_d = h
        self.net     = nn.Sequential(*layers)
        self.fc_mu   = nn.Linear(in_d, latent_dim)
        self.fc_logv = nn.Linear(in_d, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.net(x)
        mu     = self.fc_mu(h)
        log_var = self.fc_logv(h)
        return mu, log_var


class VAEDecoder(nn.Module):
    """Maps latent z back to reconstruction x̂."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            in_d = h
        layers += [nn.Linear(in_d, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """Full Variational Autoencoder with reparameterization trick."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2,
        beta_kl: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta_kl    = beta_kl
        self.encoder    = VAEEncoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder    = VAEDecoder(latent_dim, hidden_dims, input_dim, dropout)

    # ── reparameterization trick ─────────────────────────────────────────────
    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic latent mean mu (used at inference)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    # ── ELBO loss ────────────────────────────────────────────────────────────
    @staticmethod
    def elbo_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total ELBO loss, reconstruction loss, KL divergence.

        Reconstruction: mean squared error (features are continuous scalars)
        KL:  -½ sum(1 + log sigma² − mu² − sigma²)  per sample, then mean over batch
        """
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss    = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        total = recon_loss + beta * kl_loss
        return total, recon_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_vae(
    model: VAE,
    X_train: np.ndarray,
    X_val:   np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    patience: int = 10,
    seed: int = 42,
) -> dict:
    """Train the VAE with early stopping on validation ELBO.

    Returns
    -------
    history : dict with lists 'train_loss', 'val_loss'
    """
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tr), batch_size=batch_size, shuffle=True)

    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kl_loss": []}
    best_val  = float("inf")
    best_state: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = ep_recon = ep_kl = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            x_hat, mu, logv = model(xb)
            loss, r, k = VAE.elbo_loss(xb, x_hat, mu, logv, beta=model.beta_kl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss  += loss.item()
            ep_recon += r.item()
            ep_kl    += k.item()

        n_batches = max(len(loader), 1)
        train_l   = ep_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            xv = X_vl.to(device)
            xv_hat, mu_v, logv_v = model(xv)
            val_l, _, _ = VAE.elbo_loss(xv, xv_hat, mu_v, logv_v, beta=model.beta_kl)
            val_l = val_l.item()

        scheduler.step(val_l)
        history["train_loss"].append(train_l)
        history["val_loss"].append(val_l)
        history["recon_loss"].append(ep_recon / n_batches)
        history["kl_loss"].append(ep_kl / n_batches)

        if val_l < best_val - 1e-6:
            best_val   = val_l
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "VAE epoch %3d/%d | train=%.4f val=%.4f recon=%.4f kl=%.4f",
                epoch, epochs, train_l, val_l,
                ep_recon / n_batches, ep_kl / n_batches,
            )

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val)
            break

    model.load_state_dict(best_state)
    logger.info("VAE training complete — best val loss: %.4f", best_val)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# LATENT SPACE + ANOMALY SCORING
# ─────────────────────────────────────────────────────────────────────────────

def get_latent_vectors(
    model: VAE,
    X: np.ndarray,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Encode X → latent mean mu in R^{n x d} (deterministic)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    mus = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            mu, _ = model.encoder(xb)
            mus.append(mu.cpu().numpy())
    return np.vstack(mus).astype(np.float32)


def get_reconstruction_error(
    model: VAE,
    X: np.ndarray,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute per-sample MSE reconstruction error (anomaly score).

    Higher error → transaction deviates more from the learned distribution
    → more suspicious.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb    = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            x_hat = model.decoder(model.reparameterize(*model.encoder(xb)))
            err   = F.mse_loss(x_hat, xb, reduction="none").mean(dim=1)
            errors.append(err.cpu().numpy())
    return np.concatenate(errors).astype(np.float32)


def anomaly_threshold(
    errors: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Return reconstruction-error threshold at `percentile`."""
    return float(np.percentile(errors, percentile))
