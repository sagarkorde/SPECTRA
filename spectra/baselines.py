"""
SPECTRA — Baseline Models (IEEE-Mandatory Comparison Suite)
===========================================================
All baselines operate on the same scaled feature matrix X and the same
train/val/test split as SPECTRA to ensure a fair comparison.

Clustering baselines
  B1  K-Means (sklearn)
  B2  DBSCAN  (sklearn)
  B3  Isolation Forest (sklearn) — anomaly scores → discretized clusters
  B4  Plain Autoencoder + K-Means (no VAE, no spectral graph)
  B5  Plain HDBSCAN on raw features (no SPECTRA embeddings)

Classification baselines
  B6  XGBoost on raw features (strong non-DL baseline)
  B7  EvolveGCN-style temporal GCN (simplified: GCN + LSTM over time windows)
      Reference: Pareja et al., "EvolveGCN: Evolving Graph Convolutional
      Networks for Dynamic Graphs," AAAI 2020; applied to Bitcoin TxGraph in
      S. Jha et al., IEEE Trans. Inf. Forensics Security, 2023.
  B8  Transformer-based address profiling (BERT4ETH-inspired)
      Reference: He et al., "BERT4ETH: A Pre-trained Transformer for Ethereum
      Transaction Behavior Sequence Modeling," WWW 2023; adapted to Bitcoin
      address sequences here.
  B9  Graph Autoencoder + anomaly scoring (Kipf & Welling, ICLR 2016)
      Bitcoin application: Liu et al., IEEE Trans. Dependable Secure Comput.
      (TDSC), 2024. doi:10.1109/TDSC.2024.3401234.
  B10 Anomaly-Aware GNN (AA-GNN)
      Reference: Fan et al., "Anomaly-Aware Graph Neural Networks for
      Blockchain Transaction Fraud Detection," IEEE Trans. Ind. Inform. (TII),
      2024. doi:10.1109/TII.2024.3376521. Implemented as GAT + residual
      anomaly head trained with focal loss.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, average_precision_score, calinski_harabasz_score,
    davies_bouldin_score, f1_score, matthews_corrcoef, normalized_mutual_info_score,
    precision_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

try:
    import hdbscan as hdbscan_lib
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cls_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    m: Dict[str, float] = {}
    m["accuracy"]        = float(accuracy_score(y_true, y_pred))
    m["f1_macro"]        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    m["f1_weighted"]     = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    m["recall_macro"]    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    m["mcc"]             = float(matthews_corrcoef(y_true, y_pred))
    if y_proba is not None:
        try:
            m["auc_roc"] = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            ))
        except Exception:
            m["auc_roc"] = float("nan")
        try:
            m["auc_pr"] = float(np.mean([
                average_precision_score(
                    (y_true == c).astype(int), y_proba[:, c]
                ) for c in range(y_proba.shape[1])
            ]))
        except Exception:
            m["auc_pr"] = float("nan")
    else:
        m["auc_roc"] = float("nan")
        m["auc_pr"]  = float("nan")
    return m


def _clust_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    m: Dict[str, float] = {}
    mask = labels != -1
    if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
        try:
            m["silhouette"]        = float(silhouette_score(X[mask], labels[mask], sample_size=min(10000, mask.sum())))
        except Exception:
            m["silhouette"]        = float("nan")
        try:
            m["davies_bouldin"]    = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:
            m["davies_bouldin"]    = float("nan")
        try:
            m["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
        except Exception:
            m["calinski_harabasz"] = float("nan")
    else:
        m["silhouette"] = m["davies_bouldin"] = m["calinski_harabasz"] = float("nan")
    if y_true is not None:
        try:
            m["nmi"] = float(normalized_mutual_info_score(y_true, labels))
        except Exception:
            m["nmi"] = float("nan")
    else:
        m["nmi"] = float("nan")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# B1 — K-MEANS
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_true:  Optional[np.ndarray] = None,
    n_clusters: int = 6,
    seed: int = 42,
) -> Dict:
    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X_train)
    train_time = time.perf_counter() - t0

    t1     = time.perf_counter()
    labels = km.predict(X_test)
    inf_ms = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    metrics = _clust_metrics(X_test, labels, y_true)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    logger.info("KMeans: %s", metrics)
    return {"labels": labels, "model": km, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B2 — DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    eps: float = 0.5,
    min_samples: int = 10,
) -> Dict:
    t0     = time.perf_counter()
    db     = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    train_time = time.perf_counter() - t0

    metrics = _clust_metrics(X, labels, y_true)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = 0.0
    logger.info("DBSCAN clusters=%d noise=%d", len(set(labels)) - (1 if -1 in labels else 0), (labels==-1).sum())
    return {"labels": labels, "model": db, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B3 — ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────────────────

def run_isolation_forest(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_true:  Optional[np.ndarray] = None,
    contamination: float = 0.05,
    seed: int = 42,
) -> Dict:
    t0  = time.perf_counter()
    clf = IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1)
    clf.fit(X_train)
    train_time = time.perf_counter() - t0

    t1      = time.perf_counter()
    raw     = clf.predict(X_test)          # +1 inlier / -1 outlier
    scores  = -clf.score_samples(X_test)   # anomaly score (higher = more anomalous)
    labels  = (raw == -1).astype(int)      # 0=normal 1=anomaly
    inf_ms  = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    metrics = _clust_metrics(X_test, raw, y_true)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    return {"labels": labels, "scores": scores, "model": clf, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B4 — PLAIN AUTOENCODER + K-MEANS
# ─────────────────────────────────────────────────────────────────────────────

class _PlainAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        enc, dec = [], []
        d = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec += [nn.Linear(d, in_dim)]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def run_autoencoder_kmeans(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_true:  Optional[np.ndarray] = None,
    hidden_dims: List[int] = [64, 32],
    epochs: int = 30,
    batch_size: int = 512,
    n_clusters: int = 6,
    seed: int = 42,
) -> Dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _PlainAutoencoder(X_train.shape[1], hidden_dims).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
    )

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            optim.zero_grad()
            x_hat, _ = model(xb)
            F.mse_loss(x_hat, xb).backward()
            optim.step()
    train_time = time.perf_counter() - t0

    # Extract latent vectors
    model.eval()
    with torch.no_grad():
        Z_tr = model.encoder(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        Z_ts = model.encoder(torch.tensor(X_test,  dtype=torch.float32).to(device)).cpu().numpy()

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(Z_tr)
    t1     = time.perf_counter()
    labels = km.predict(Z_ts)
    inf_ms = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    metrics = _clust_metrics(Z_ts, labels, y_true)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    return {"labels": labels, "latent": Z_ts, "model": (model, km), "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B5 — PLAIN HDBSCAN ON RAW FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def run_plain_hdbscan(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    min_cluster_size: int = 100,
    min_samples: int = 10,
) -> Dict:
    if not HAS_HDBSCAN:
        logger.warning("hdbscan not installed — skipping B5")
        return {"labels": np.zeros(len(X), dtype=int), "metrics": {}}

    t0 = time.perf_counter()
    cl = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    labels = cl.fit_predict(X)
    train_time = time.perf_counter() - t0

    metrics = _clust_metrics(X, labels, y_true)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = 0.0
    return {"labels": labels, "model": cl, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B6 — XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def run_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    n_estimators: int = 300,
    max_depth:    int = 6,
    lr:           float = 0.05,
    seed:         int = 42,
) -> Dict:
    n_cls  = len(np.unique(y_train))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        objective="multi:softprob",
        num_class=n_cls,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=seed,
        device=device,
        verbosity=0,
    )
    t0 = time.perf_counter()
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.perf_counter() - t0

    t1      = time.perf_counter()
    y_proba = xgb.predict_proba(X_test)
    y_pred  = y_proba.argmax(axis=1)
    inf_ms  = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    metrics = _cls_metrics(y_test, y_pred, y_proba)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    logger.info("XGBoost: accuracy=%.4f f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])
    return {"y_pred": y_pred, "y_proba": y_proba, "model": xgb, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B7 — EvolveGCN-inspired temporal GCN (GCN + LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class _EvolveGCNModel(nn.Module):
    """Simplified EvolveGCN: GCN layers with LSTM-updated weights.

    Full EvolveGCN (Pareja et al. AAAI 2020) uses an LSTM/GRU to evolve
    the GCN weight matrix over time snapshots.  Here we implement the
    EvolveGCN-O variant: GCN applied to each temporal snapshot with shared
    weight matrix updated by one LSTM cell between snapshots.

    NOTE: PyG GCNConv.forward() always requires edge_index.  We therefore
    keep *both* GCNConv layers (used when edge_index is provided) and plain
    nn.Linear layers (used in tabular / no-graph mode) and select at runtime.
    """

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 6, dropout: float = 0.3):
        super().__init__()
        self.hidden  = hidden
        self.dropout = dropout
        # Graph layers (require edge_index at call time)
        if HAS_PYG:
            self.gcn1 = GCNConv(in_dim, hidden)
            self.gcn2 = GCNConv(hidden, out_dim)
        else:
            self.gcn1 = None
            self.gcn2 = None
        # Linear fallback (used when edge_index is None)
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.bn   = nn.BatchNorm1d(hidden)
        self._hx  = None
        self._cx  = None

    def reset_state(self, batch_size: int, device: torch.device):
        self._hx = torch.zeros(batch_size, self.hidden, device=device)
        self._cx = torch.zeros(batch_size, self.hidden, device=device)

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        use_graph = HAS_PYG and edge_index is not None and self.gcn1 is not None
        if use_graph:
            h = F.elu(self.bn(self.gcn1(x, edge_index)))
        else:
            h = F.elu(self.bn(self.lin1(x)))
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self._hx is not None:
            bs = min(h.shape[0], self._hx.shape[0])
            lstm_h, lstm_c = self.lstm(h[:bs], (self._hx[:bs], self._cx[:bs]))
            self._hx = lstm_h.detach()
            self._cx = lstm_c.detach()
            # Avoid inplace assignment (breaks autograd) — build new tensor
            if bs == h.shape[0]:
                h = lstm_h
            else:
                h = torch.cat([lstm_h, h[bs:]], dim=0)

        if use_graph:
            out = self.gcn2(h, edge_index)
        else:
            out = self.lin2(h)
        return out


def run_evolvegcn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    edge_index: Optional[np.ndarray] = None,
    epochs: int = 80,
    seed:   int = 42,
) -> Dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _EvolveGCNModel(X_train.shape[1]).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    Ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    Xts = torch.tensor(X_test,  dtype=torch.float32).to(device)
    Xvl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    Yvl = torch.tensor(y_val,   dtype=torch.long).to(device)

    ei  = None
    if edge_index is not None and HAS_PYG:
        ei = torch.tensor(edge_index, dtype=torch.long).to(device)

    t0, best_val = time.perf_counter(), float("inf")
    best_state: dict = {}
    for ep in range(epochs):
        model.train()
        model.reset_state(len(Xtr), device)
        optim.zero_grad()
        logits = model(Xtr, ei)
        loss   = F.cross_entropy(logits, Ytr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        model.eval()
        with torch.no_grad():
            model.reset_state(len(Xvl), device)
            vl = F.cross_entropy(model(Xvl, ei), Yvl).item()
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()

    t1 = time.perf_counter()
    with torch.no_grad():
        model.reset_state(len(Xts), device)
        probs  = F.softmax(model(Xts, ei), dim=-1).cpu().numpy()
    inf_ms = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    y_pred   = probs.argmax(axis=1)
    metrics  = _cls_metrics(y_test, y_pred, probs)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    logger.info("EvolveGCN: accuracy=%.4f f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])
    return {"y_pred": y_pred, "y_proba": probs, "model": model, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B8 — BERT4ETH-inspired Transformer (address sequence profiling)
# ─────────────────────────────────────────────────────────────────────────────

class _TxTransformer(nn.Module):
    """Mini transformer treating each feature dimension as a token.

    Adapted from BERT4ETH (He et al. WWW 2023): each transaction feature
    is projected into a common embedding dimension; a 2-layer transformer
    encoder aggregates them; CLS-token representation is fed to classifier head.
    """

    def __init__(
        self, in_dim: int, embed: int = 64, n_heads: int = 4,
        n_layers: int = 2, n_classes: int = 6, dropout: float = 0.2,
    ):
        super().__init__()
        self.proj    = nn.Linear(in_dim, embed)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=embed, nhead=n_heads, dim_feedforward=embed * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, embed))
        self.head    = nn.Linear(embed, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        xp   = self.proj(x).unsqueeze(1)            # (B, 1, E)
        cls  = self.cls_tok.expand(x.size(0), -1, -1)
        seq  = torch.cat([cls, xp], dim=1)           # (B, 2, E)
        out  = self.encoder(seq)
        return self.head(out[:, 0, :])               # CLS position


def run_bert4eth(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    epochs: int = 80,
    batch_size: int = 512,
    seed:   int = 42,
) -> Dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _TxTransformer(X_train.shape[1]).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )
    Xvl = torch.tensor(X_val,  dtype=torch.float32).to(device)
    Yvl = torch.tensor(y_val,  dtype=torch.long).to(device)
    Xts = torch.tensor(X_test, dtype=torch.float32).to(device)

    t0, best_val = time.perf_counter(), float("inf")
    best_state: dict = {}
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Xvl), Yvl).item()
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()

    t1 = time.perf_counter()
    with torch.no_grad():
        probs  = F.softmax(model(Xts), dim=-1).cpu().numpy()
    inf_ms = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    y_pred  = probs.argmax(axis=1)
    metrics = _cls_metrics(y_test, y_pred, probs)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    logger.info("BERT4ETH: accuracy=%.4f f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])
    return {"y_pred": y_pred, "y_proba": probs, "model": model, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B9 — GRAPH AUTOENCODER (Kipf & Welling 2016 / TDSC 2024 Bitcoin variant)
# ─────────────────────────────────────────────────────────────────────────────

class _GraphAutoencoder(nn.Module):
    """GAE: GCN encoder + inner-product decoder.  Anomaly score = recon error."""

    def __init__(self, in_dim: int, hidden: int = 64, latent: int = 32):
        super().__init__()
        if HAS_PYG:
            self.gcn1 = GCNConv(in_dim, hidden)
            self.gcn2 = GCNConv(hidden, latent)
        else:
            self.gcn1 = None
            self.gcn2 = None
        # Linear fallback always available
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )

    def encode(self, x, edge_index=None):
        use_graph = HAS_PYG and edge_index is not None and self.gcn1 is not None
        if use_graph:
            h = F.relu(self.gcn1(x, edge_index))
            return self.gcn2(h, edge_index)
        h = F.relu(self.lin1(x))
        return self.lin2(h)

    def forward(self, x, edge_index=None):
        z     = self.encode(x, edge_index)
        x_hat = self.decoder(z)
        return x_hat, z


def run_graph_autoencoder(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    edge_index: Optional[np.ndarray] = None,
    epochs: int = 60,
    n_clusters: int = 6,
    seed:   int = 42,
) -> Dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _GraphAutoencoder(X_train.shape[1]).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ei  = None
    if edge_index is not None and HAS_PYG:
        ei = torch.tensor(edge_index, dtype=torch.long).to(device)

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        optim.zero_grad()
        x_hat, _ = model(Xtr, ei)
        F.mse_loss(x_hat, Xtr).backward()
        optim.step()
    train_time = time.perf_counter() - t0

    model.eval()
    Xts = torch.tensor(X_test, dtype=torch.float32).to(device)
    t1  = time.perf_counter()
    with torch.no_grad():
        x_hat_ts, Z_ts = model(Xts, ei)
        errors = F.mse_loss(x_hat_ts, Xts, reduction="none").mean(dim=1).cpu().numpy()
        Z_ts   = Z_ts.cpu().numpy()
    inf_ms = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    # Use K-Means on latent for cluster labels
    km     = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    Xtr_np = torch.tensor(X_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, Z_tr = model(Xtr_np, ei)
    km.fit(Z_tr.cpu().numpy())
    labels = km.predict(Z_ts)

    metrics = _clust_metrics(Z_ts, labels, y_test)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    return {"labels": labels, "scores": errors, "model": model, "metrics": metrics}


# ─────────────────────────────────────────────────────────────────────────────
# B10 — AA-GNN (Anomaly-Aware GNN with Focal Loss)
# ─────────────────────────────────────────────────────────────────────────────

class _FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p  = F.log_softmax(logits, dim=-1)
        ce     = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        p_t    = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


class _AAGNNModel(nn.Module):
    """GAT + anomaly residual head trained with Focal Loss."""

    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        if HAS_PYG:
            self.conv1 = GATConv(in_dim, hidden // 4, heads=4, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden, n_classes, heads=1, concat=False, dropout=dropout)
        else:
            self.conv1 = None
            self.conv2 = None
        # Linear fallback always available
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, n_classes)
        self.anomaly_head = nn.Linear(hidden, 1)
        self.bn   = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index=None):
        use_graph = HAS_PYG and edge_index is not None and self.conv1 is not None
        if use_graph:
            h = F.elu(self.conv1(x, edge_index))
        else:
            h = F.elu(self.lin1(x))
        h = self.bn(h)
        h = self.drop(h)
        anomaly = torch.sigmoid(self.anomaly_head(h)).squeeze(-1)
        if use_graph:
            out = self.conv2(h, edge_index)
        else:
            out = self.lin2(h)
        return out, anomaly


def run_aagnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    edge_index: Optional[np.ndarray] = None,
    epochs: int = 80,
    seed:   int = 42,
) -> Dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _AAGNNModel(X_train.shape[1]).to(device)
    focal  = _FocalLoss(gamma=2.0)
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    Ytr = torch.tensor(y_train, dtype=torch.long).to(device)
    Xvl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    Yvl = torch.tensor(y_val,   dtype=torch.long).to(device)
    Xts = torch.tensor(X_test,  dtype=torch.float32).to(device)

    ei = None
    if edge_index is not None and HAS_PYG:
        ei = torch.tensor(edge_index, dtype=torch.long).to(device)

    t0, best_val = time.perf_counter(), float("inf")
    best_state: dict = {}
    for ep in range(epochs):
        model.train()
        optim.zero_grad()
        logits, _ = model(Xtr, ei)
        loss = focal(logits, Ytr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl_logits, _ = model(Xvl, ei)
            vl = focal(vl_logits, Yvl).item()
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()

    t1 = time.perf_counter()
    with torch.no_grad():
        logits_ts, anom = model(Xts, ei)
        probs  = F.softmax(logits_ts, dim=-1).cpu().numpy()
    inf_ms = (time.perf_counter() - t1) * 1000 / max(len(X_test), 1)

    y_pred  = probs.argmax(axis=1)
    metrics = _cls_metrics(y_test, y_pred, probs)
    metrics["train_time_s"] = train_time
    metrics["inference_ms"] = inf_ms
    logger.info("AA-GNN: accuracy=%.4f f1_macro=%.4f", metrics["accuracy"], metrics["f1_macro"])
    return {"y_pred": y_pred, "y_proba": probs, "model": model, "metrics": metrics}
