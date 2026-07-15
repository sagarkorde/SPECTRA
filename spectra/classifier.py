"""
SPECTRA — Module R: GraphSAGE Transaction Authenticator
=======================================================
GNN classifier that operates on SPECTRA node embeddings to predict
the transaction-type label (6-class) and produce an authenticity confidence.

Architecture choice: GraphSAGE (Hamilton et al., NeurIPS 2017)
  Rationale:
    (1) Inductive learning — generalizes to unseen addresses without
        retraining (critical for live Bitcoin monitoring).
    (2) Neighborhood sampling scales to the millions of Bitcoin addresses
        without loading the full adjacency in GPU memory (unlike GCN).
    (3) Concatenation aggregator preserves node identity alongside
        aggregated neighbor information — important in a directed graph
        where sender and receiver roles are asymmetric.
    (4) Empirical evidence from BERT4ETH / EvolveGCN literature shows
        SAGEConv reaches competitive accuracy on Bitcoin transaction graphs
        at a fraction of GAT's computational cost.

Input node features
  spectral_embedding  (k dims)  — Module S
  aug_address_features (6 dims) — Module S
  cluster_id_onehot             — Module C
  bayesian_trust_score (1 dim)  — Module P
  markov_anomaly_score (1 dim)  — Module T

Output
  6-class softmax probability over transaction types
  + anomaly score = 1 − max(softmax)  (uncertainty = low confidence)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SPECTRAClassifier(nn.Module):
    """Two-layer GraphSAGE classifier with residual connection and dropout.

    If torch_geometric is unavailable, falls back to a plain MLP that
    ignores graph structure (useful for ablation SPECTRA-noS / CPU-only runs).
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int = 128,
        embed_dim:  int = 64,
        n_classes:  int = 6,
        dropout:    float = 0.3,
        use_graph:  bool = True,
    ):
        super().__init__()
        self.use_graph = use_graph and HAS_PYG

        if self.use_graph:
            self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
            self.conv2 = SAGEConv(hidden_dim, embed_dim, aggr="mean")
        else:
            self.conv1 = nn.Linear(in_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, embed_dim)

        self.bn1     = nn.BatchNorm1d(hidden_dim)
        self.bn2     = nn.BatchNorm1d(embed_dim)
        self.drop    = nn.Dropout(dropout)
        self.proj    = nn.Linear(in_dim, embed_dim)   # skip connection
        self.head    = nn.Linear(embed_dim, n_classes)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_graph and edge_index is not None:
            h = F.elu(self.bn1(self.conv1(x, edge_index)))
            h = self.drop(h)
            h = F.elu(self.bn2(self.conv2(h, edge_index)))
        else:
            h = F.elu(self.bn1(self.conv1(x)))
            h = self.drop(h)
            h = F.elu(self.bn2(self.conv2(h)))

        # Residual: project x to embed_dim and add
        skip = self.proj(x)
        return h + skip

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.encode(x, edge_index)
        return self.head(self.drop(emb))

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=-1)

    def anomaly_score(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """1 − max(softmax) as a confidence-based anomaly signal."""
        probs = self.predict_proba(x, edge_index)
        return 1.0 - probs.max(dim=-1).values


# ─────────────────────────────────────────────────────────────────────────────
# PYGEOMETRIC DATA BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_pyg_data(
    node_features: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    node_labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask:   np.ndarray,
    test_mask:  np.ndarray,
) -> "Data":
    """Package everything into a PyG Data object."""
    x          = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(
        np.stack([edge_src, edge_dst], axis=0), dtype=torch.long
    )
    y          = torch.tensor(node_labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask   = torch.tensor(val_mask,   dtype=torch.bool)
    data.test_mask  = torch.tensor(test_mask,  dtype=torch.bool)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_classifier(
    model:      SPECTRAClassifier,
    data:       "Data",
    epochs:     int = 100,
    lr:         float = 1e-3,
    weight_decay: float = 1e-4,
    device:     Optional[torch.device] = None,
    patience:   int = 15,
    seed:       int = 42,
) -> Dict:
    """Train the GNN classifier with cross-entropy loss + early stopping.

    Returns
    -------
    history : dict with train_loss, val_loss, train_acc, val_acc lists
    """
    torch.manual_seed(seed)
    device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    data      = data.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    stopper   = EarlyStopper(patience=patience)

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val  = float("inf")
    best_state: dict = {}

    ei = data.edge_index if model.use_graph else None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits     = model(data.x, ei)
        train_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_v = model(data.x, ei)
            val_loss = F.cross_entropy(
                logits_v[data.val_mask], data.y[data.val_mask]
            ).item()
            train_acc = (
                logits[data.train_mask].argmax(dim=-1) == data.y[data.train_mask]
            ).float().mean().item()
            val_acc = (
                logits_v[data.val_mask].argmax(dim=-1) == data.y[data.val_mask]
            ).float().mean().item()

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "GNN epoch %3d/%d | train_loss=%.4f val_loss=%.4f "
                "train_acc=%.4f val_acc=%.4f",
                epoch, epochs, train_loss.item(), val_loss, train_acc, val_acc,
            )

        if stopper.step(val_loss):
            logger.info("Early stopping at epoch %d (best val=%.4f)", epoch, best_val)
            break

    model.load_state_dict(best_state)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(
    model:  SPECTRAClassifier,
    data:   "Data",
    mask:   torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Return classification metrics on the nodes specified by mask."""
    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
        average_precision_score,
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    data = data.to(device)

    ei = data.edge_index if model.use_graph else None
    with torch.no_grad():
        probs  = model.predict_proba(data.x, ei)[mask].cpu().numpy()
        y_pred = probs.argmax(axis=1)
        y_true = data.y[mask].cpu().numpy()

    metrics: Dict[str, float] = {}
    metrics["accuracy"]          = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"]          = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"]       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["precision_macro"]   = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"]      = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["mcc"]               = float(matthews_corrcoef(y_true, y_pred))

    try:
        metrics["auc_roc"] = float(roc_auc_score(
            y_true, probs, multi_class="ovr", average="macro"
        ))
    except Exception:
        metrics["auc_roc"] = float("nan")

    try:
        metrics["auc_pr"] = float(np.mean([
            average_precision_score(
                (y_true == c).astype(int), probs[:, c]
            ) for c in range(probs.shape[1])
        ]))
    except Exception:
        metrics["auc_pr"] = float("nan")

    return metrics


def full_evaluation(
    model:  SPECTRAClassifier,
    data:   "Data",
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate on train / val / test splits."""
    return {
        "train": evaluate_classifier(model, data, data.train_mask, device),
        "val":   evaluate_classifier(model, data, data.val_mask,   device),
        "test":  evaluate_classifier(model, data, data.test_mask,  device),
    }
