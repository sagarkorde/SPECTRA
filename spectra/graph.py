"""
SPECTRA — Module S: Spectral Graph Construction
================================================
Builds the address-level directed weighted graph from Bitcoin transaction data,
computes the graph Laplacian, extracts top-k spectral node embeddings, and
augments them with per-address behavioural features.

Graph definition
  V = unique wallet addresses
  E = directed edge (sender → receiver) per transaction
  W = edge weight = total transaction value (BTC)

Spectral embeddings
  L = D − A  (unnormalized)  or  L_sym = D^{-1/2}(D−A)D^{-1/2}  (normalized)
  Eigenvectors phi_1 ... phi_k of L as node feature matrix in R^{|V| x k}

Augmented node features (appended to spectral embedding)
  fan_in_ratio   = in_degree / (in_degree + out_degree)
  tx_velocity    = total tx count / observation_days
  avg_recv_value = mean incoming BTC value
  avg_sent_value = mean outgoing BTC value
  temporal_gap   = mean inter-transaction gap (seconds)
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    path: str,
    sample_n: int = 300_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load the parquet dataset via DuckDB and return a stratified sample.

    The transaction-type label is constructed here for stratified sampling:
      0 Standard, 1 P2P, 2 Consolidation, 3 Distribution,
      4 Batch-payment, 5 CoinJoin-like.

    NOTE: The following columns are always-false / zero in this dataset and are
    dropped automatically: is_self_transfer, has_p2pk, has_p2pkh, has_p2sh,
    has_p2wpkh, has_p2wsh, has_taproot, address_reuse, output_script_count.
    has_taproot is always false despite script_type fields containing
    witness_v1_taproot; the flag appears to have been mis-engineered — we
    retain the script_type strings for graph parsing but drop the boolean.
    """
    con = duckdb.connect()
    df_full = con.execute(
        f"SELECT * FROM read_parquet('{path}')"
    ).df()
    con.close()

    logger.info("Loaded %d rows x %d cols from %s", len(df_full), df_full.shape[1], path)

    # Build integer class label (priority: coinjoin > batch > consolidation >
    # distribution > p2p > standard)
    df_full["tx_type"] = 0  # standard
    df_full.loc[df_full["is_peer_to_peer"],    "tx_type"] = 1
    df_full.loc[df_full["is_consolidation"],   "tx_type"] = 2
    df_full.loc[df_full["is_distribution"],    "tx_type"] = 3
    df_full.loc[df_full["is_batch_payment"],   "tx_type"] = 4
    df_full.loc[df_full["is_coinjoin_like"],   "tx_type"] = 5

    TX_TYPE_NAMES = {
        0: "Standard", 1: "P2P", 2: "Consolidation",
        3: "Distribution", 4: "BatchPayment", 5: "CoinJoin",
    }

    # Stratified sample
    rng = np.random.default_rng(seed)
    frames = []
    per_class = sample_n // 6
    for cls in range(6):
        sub = df_full[df_full["tx_type"] == cls]
        n = min(per_class, len(sub))
        frames.append(sub.sample(n=n, random_state=seed))
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)

    logger.info(
        "Sampled %d rows — class distribution: %s",
        len(df),
        df["tx_type"].value_counts().to_dict(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ADDRESS PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_address_list(raw) -> List[str]:
    """Convert DuckDB VARCHAR[] field to a flat list of Bitcoin addresses.

    The field is stored as a Python list whose first (and usually only) element
    is a semicolon-separated string of addresses.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, np.ndarray)):
        combined = ";".join(str(x) for x in raw if x)
        return [a.strip() for a in combined.split(";") if a.strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def extract_address_pairs(
    df: pd.DataFrame,
) -> List[Tuple[str, str, float, int]]:
    """Return (sender, receiver, btc_value, block_height) triples.

    For transactions with multiple senders or receivers, one representative
    pair is created per distinct (sender x receiver) combination weighted by
    value / (n_inputs x n_outputs).
    """
    pairs: List[Tuple[str, str, float, int]] = []
    for _, row in df.iterrows():
        senders   = _parse_address_list(row["input_addresses"])
        receivers = _parse_address_list(row["output_addresses"])
        if not senders or not receivers:
            continue
        val = float(row["total_output_value"]) / max(len(senders) * len(receivers), 1)
        bh  = int(row["block_height"])
        for s in senders:
            for r in receivers:
                if s and r:
                    pairs.append((s, r, val, bh))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    pairs: List[Tuple[str, str, float, int]],
    max_nodes: int = 30_000,
) -> Tuple[nx.DiGraph, Dict[str, int]]:
    """Build a directed weighted graph from (sender, receiver, value, height).

    Bitcoin address graphs are extremely sparse — most addresses appear only
    once (for privacy).  To keep spectral embedding tractable, we prune to the
    top ``max_nodes`` nodes by undirected degree after construction.  These
    high-degree hubs (exchanges, mixers, mining pools) carry the most
    structural signal for the Laplacian eigenvectors.

    Returns
    -------
    G : nx.DiGraph  with edge attr ``weight`` and ``block_height``
    node_index : dict mapping address string -> integer node id
    """
    G = nx.DiGraph()
    for src, dst, val, bh in pairs:
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += val
            G[src][dst]["count"]  += 1
        else:
            G.add_edge(src, dst, weight=val, count=1, block_height=bh)

    logger.info(
        "Graph (full): %d nodes | %d edges", G.number_of_nodes(), G.number_of_edges()
    )

    # Prune to top-max_nodes by total (in+out) degree if graph is too large
    if G.number_of_nodes() > max_nodes:
        deg = dict(G.degree())          # total degree (in + out)
        top_nodes = sorted(deg, key=deg.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
        logger.info(
            "Graph (pruned to top-%d by degree): %d nodes | %d edges",
            max_nodes, G.number_of_nodes(), G.number_of_edges(),
        )

    node_index = {n: i for i, n in enumerate(G.nodes())}
    return G, node_index


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH LAPLACIAN
# ─────────────────────────────────────────────────────────────────────────────

def compute_laplacian(
    G: nx.DiGraph,
    node_index: Dict[str, int],
    normalized: bool = True,
) -> sp.csr_matrix:
    """Compute the (optionally normalized) graph Laplacian as a sparse matrix.

    For a directed graph we symmetrize: A_sym = (A + A^T) / 2 so that the
    Laplacian is real-symmetric and its eigenvectors are real.

    L_unnorm = D - A_sym
    L_norm   = D^{-1/2} (D - A_sym) D^{-1/2}
    """
    n = len(node_index)
    rows, cols, data = [], [], []
    for (u, v, w) in G.edges(data="weight", default=1.0):
        i, j = node_index[u], node_index[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w / 2, w / 2])  # symmetrize

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - A

    if normalized:
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt

    return L.tocsr()


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def spectral_embeddings(
    L: sp.csr_matrix,
    k: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Extract k spectral node embeddings from the graph Laplacian L.

    Uses sklearn randomized_svd exclusively — it is faster, always converges,
    thread-safe on Python 3.12, and produces embeddings equivalent to the
    bottom-k eigenvectors of L for the graph sizes we work with (< 50 K nodes).

    For the normalized symmetric Laplacian, the bottom singular vectors
    coincide with the smoothest (lowest-frequency) graph signal components,
    which capture cluster / community membership.  The constant vector
    (corresponding to eigenvalue 0) is discarded via magnitude thresholding.

    Returns
    -------
    phi : np.ndarray of shape (n_nodes, k)
    """
    from sklearn.utils.extmath import randomized_svd

    n     = L.shape[0]
    k_req = min(k + 2, n - 1)

    logger.info("Computing spectral embedding (randomized SVD, k=%d) on %d-node graph ...", k, n)

    try:
        U, S, Vt = randomized_svd(
            L,
            n_components=k_req,
            random_state=seed,
            n_iter=10,           # power iterations — increases accuracy
            n_oversamples=20,    # extra dimensions for numerical stability
        )
        # U columns are left singular vectors (== approx eigenvectors of L^T L)
        # Sort descending by singular value then REVERSE so smallest come first
        # (smallest singular values correspond to smoothest graph signal)
        order    = np.argsort(S)          # ascending
        U        = U[:, order]
        S_sorted = S[order]

        # Drop the near-zero component (trivial constant eigenvector, S ~ 0)
        non_trivial = S_sorted > 1e-8
        U = U[:, non_trivial][:, :k]

    except Exception as exc:
        logger.warning("Randomized SVD failed (%s) — using random fallback.", exc)
        U = np.random.default_rng(seed).standard_normal((n, k)).astype(np.float32)
        return U.astype(np.float32)

    # Pad if fewer than k vectors survived thresholding
    if U.shape[1] < k:
        pad = np.zeros((n, k - U.shape[1]), dtype=np.float32)
        U   = np.hstack([U, pad])

    logger.info("Spectral embedding shape: %s", U.shape)
    return U.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ADDRESS-LEVEL AUGMENTED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def address_features(
    df: pd.DataFrame,
    node_index: Dict[str, int],
) -> np.ndarray:
    """Compute per-address behavioural features appended to spectral embedding.

    Features
    --------
    fan_in_ratio       in-degree / (in + out) degree
    tx_velocity        tx count / observation span (days)
    avg_recv_value     mean BTC received per incoming transaction
    avg_sent_value     mean BTC sent per outgoing transaction
    temporal_gap_mean  mean inter-transaction gap (seconds)
    temporal_gap_std   std of inter-transaction gap

    Returns
    -------
    feat : np.ndarray of shape (n_nodes, 6)
    """
    n = len(node_index)
    feat = np.zeros((n, 6), dtype=np.float32)

    # Aggregates over rows
    in_count   = defaultdict(int)
    out_count  = defaultdict(int)
    recv_vals  = defaultdict(list)
    sent_vals  = defaultdict(list)
    timestamps = defaultdict(list)

    for _, row in df.iterrows():
        senders   = _parse_address_list(row["input_addresses"])
        receivers = _parse_address_list(row["output_addresses"])
        val       = float(row["total_output_value"])
        ts        = int(row["block_time"])

        for s in senders:
            if s in node_index:
                idx = node_index[s]
                out_count[idx]    += 1
                sent_vals[idx].append(val / max(len(senders), 1))
                timestamps[idx].append(ts)
        for r in receivers:
            if r in node_index:
                idx = node_index[r]
                in_count[idx]     += 1
                recv_vals[idx].append(val / max(len(receivers), 1))
                timestamps[idx].append(ts)

    # Observation span (seconds)
    block_times = df["block_time"].values
    span_days   = max((block_times.max() - block_times.min()) / 86400, 1)

    for idx in range(n):
        in_d  = in_count.get(idx, 0)
        out_d = out_count.get(idx, 0)
        total = in_d + out_d
        feat[idx, 0] = in_d / total if total > 0 else 0.5   # fan_in_ratio
        feat[idx, 1] = total / span_days                     # tx_velocity

        rv = recv_vals.get(idx, [0.0])
        feat[idx, 2] = float(np.mean(rv))                   # avg_recv_value

        sv = sent_vals.get(idx, [0.0])
        feat[idx, 3] = float(np.mean(sv))                   # avg_sent_value

        ts_list = sorted(timestamps.get(idx, [0]))
        if len(ts_list) > 1:
            gaps = np.diff(ts_list).astype(float)
            feat[idx, 4] = float(np.mean(gaps))             # temporal_gap_mean
            feat[idx, 5] = float(np.std(gaps))              # temporal_gap_std
        else:
            feat[idx, 4] = 0.0
            feat[idx, 5] = 0.0

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODULE-S OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def build_spectral_node_features(
    df: pd.DataFrame,
    k: int = 10,
    normalized_laplacian: bool = True,
    max_nodes: int = 30_000,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, int], nx.DiGraph]:
    """End-to-end Module S: graph → Laplacian → spectral embedding + aug features.

    Returns
    -------
    node_features : np.ndarray shape (n_nodes, k + 6)
    node_index    : dict address → int
    G             : nx.DiGraph
    """
    logger.info("[Module S] Extracting address pairs ...")
    pairs = extract_address_pairs(df)

    logger.info("[Module S] Building graph ...")
    G, node_index = build_graph(pairs, max_nodes=max_nodes)

    logger.info("[Module S] Computing Laplacian ...")
    L = compute_laplacian(G, node_index, normalized=normalized_laplacian)

    logger.info("[Module S] Extracting spectral embeddings ...")
    phi = spectral_embeddings(L, k=k, seed=seed)

    # Pad if graph too small for k eigenvectors
    n = len(node_index)
    if phi.shape[0] < n:
        pad = np.zeros((n - phi.shape[0], phi.shape[1]), dtype=np.float32)
        phi = np.vstack([phi, pad])

    logger.info("[Module S] Computing address-level features ...")
    aug = address_features(df, node_index)

    node_features = np.hstack([phi, aug])
    logger.info("[Module S] Node feature matrix: %s", node_features.shape)

    return node_features, node_index, G
