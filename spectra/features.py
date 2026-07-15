"""
SPECTRA — Module E: Information-Theoretic Feature Ranking & NMF
================================================================
Stage 1 of the ensemble clustering module:

  1. Shannon entropy per feature — identifies high-variance / informative cols
  2. Mutual Information ranking — selects the top-k features most predictive
     of the transaction-type label (discrete) or cluster assignment (after
     first clustering pass)
  3. Non-negative Matrix Factorization (NMF) — decomposes the feature matrix
     X ≈ W H into interpretable basis patterns W and coefficient matrix H

All three outputs feed into the VAE (spectra/vae.py) and the GNN classifier
(spectra/classifier.py).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS TO DROP (always-zero / identifier / redundant)
# ─────────────────────────────────────────────────────────────────────────────

_ALWAYS_ZERO = [
    "is_self_transfer", "has_p2pk", "has_p2pkh", "has_p2sh",
    "has_p2wpkh", "has_p2wsh", "has_taproot", "address_reuse",
    "output_script_count",
]
_IDENTIFIERS = [
    "txid", "input_addresses", "output_addresses",
    "input_script_types", "output_script_types",
    "op_return_data", "month_1", "timestamp",
]
_DROP_COLS = _ALWAYS_ZERO + _IDENTIFIERS

# Transaction-type flag columns kept as *labels*, not features
_TYPE_FLAG_COLS = [
    "is_consolidation", "is_distribution", "is_peer_to_peer",
    "is_batch_payment", "is_coinjoin_like",
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MATRIX PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_feature_matrix(
    df: pd.DataFrame,
    extra_drop: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Drop non-informative / identifier columns and return float32 matrix X.

    Returns
    -------
    X          : np.ndarray (n_samples, n_features)
    feat_names : list of feature column names
    """
    drop = set(_DROP_COLS + (_TYPE_FLAG_COLS if extra_drop is None else extra_drop))
    keep = [c for c in df.columns if c not in drop and c != "tx_type"]
    X = df[keep].copy()

    # Encode boolean columns as int
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    X[bool_cols] = X[bool_cols].astype(np.int8)

    # Encode any remaining non-numeric
    X = X.select_dtypes(include=[np.number])

    feat_names = X.columns.tolist()
    X_arr = X.values.astype(np.float32)

    # Clip extreme outliers (> 99.9th percentile) to reduce skew
    p999 = np.nanpercentile(X_arr, 99.9, axis=0)
    p001 = np.nanpercentile(X_arr, 0.1, axis=0)
    X_arr = np.clip(X_arr, p001, p999)

    # Replace any NaN/Inf introduced by clipping
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Feature matrix: %d × %d", X_arr.shape[0], X_arr.shape[1])
    return X_arr, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# SHANNON ENTROPY PER FEATURE
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy_per_feature(
    X: np.ndarray,
    n_bins: int = 50,
) -> np.ndarray:
    """Compute per-feature discrete Shannon entropy using histogram binning.

    H(X_j) = -Σ p(x_i) log2 p(x_i)

    Parameters
    ----------
    X      : (n_samples, n_features) float array
    n_bins : number of histogram bins

    Returns
    -------
    entropies : (n_features,) array of H values in bits
    """
    entropies = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        counts, _ = np.histogram(col, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies[j] = -float(np.sum(probs * np.log2(probs)))
    return entropies


# ─────────────────────────────────────────────────────────────────────────────
# MUTUAL INFORMATION RANKING
# ─────────────────────────────────────────────────────────────────────────────

def mutual_information_ranking(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: List[str],
    seed: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """Rank features by mutual information with discrete target y.

    Uses sklearn.feature_selection.mutual_info_classif which estimates MI
    via k-nearest-neighbour methods (non-parametric, handles non-linear deps).

    Returns
    -------
    mi_scores  : (n_features,) array of MI scores (descending order)
    feat_names : reordered feature names (descending MI)
    """
    logger.info("[Module E] Computing mutual information …")
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=seed)

    order = np.argsort(mi)[::-1]
    mi_sorted    = mi[order]
    names_sorted = [feat_names[i] for i in order]

    logger.info(
        "Top-5 features by MI: %s",
        list(zip(names_sorted[:5], mi_sorted[:5].round(4))),
    )
    return mi_sorted, names_sorted


def select_top_k_features(
    X: np.ndarray,
    mi_scores: np.ndarray,
    feat_names: List[str],
    k: int = 20,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Keep only the top-k features by MI score.

    Returns
    -------
    X_sel      : (n_samples, k) selected feature matrix
    sel_names  : list of selected feature names
    sel_indices: integer indices of selected columns in original X
    """
    order      = np.argsort(mi_scores)[::-1][:k]
    X_sel      = X[:, order]
    sel_names  = [feat_names[i] for i in order]
    logger.info("Selected %d features: %s", k, sel_names)
    return X_sel, sel_names, order


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD SCALER WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, transform train/val/test."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_val_s, X_test_s, scaler


# ─────────────────────────────────────────────────────────────────────────────
# NMF DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def nmf_decomposition(
    X: np.ndarray,
    n_components: int = 10,
    max_iter: int = 300,
    seed: int = 42,
    alpha_W: float = 0.1,
    l1_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, NMF]:
    """Factorize X ≈ W H via scikit-learn NMF.

    X must be non-negative — a MinMaxScaler is applied internally.
    NMF reveals interpretable basis patterns (rows of H correspond to
    latent transaction archetypes).

    Parameters
    ----------
    X            : (n_samples, n_features)
    n_components : rank r of the factorization
    alpha_W      : regularization strength on W
    l1_ratio     : 1 = L1, 0 = L2, 0.5 = elastic-net

    Returns
    -------
    W   : (n_samples, n_components) — coefficient matrix (sample weights)
    H   : (n_components, n_features) — basis pattern matrix
    nmf : fitted NMF object
    """
    logger.info("[Module E] Running NMF (r=%d) …", n_components)

    # MinMax to [0,1] — NMF requires non-negative input
    scaler = MinMaxScaler()
    X_nn   = scaler.fit_transform(X).astype(np.float32)
    X_nn   = np.clip(X_nn, 0.0, None)  # guard floating-point negatives

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=max_iter,
        random_state=seed,
        alpha_W=alpha_W,
        l1_ratio=l1_ratio,
        solver="cd",
    )
    W = nmf.fit_transform(X_nn).astype(np.float32)
    H = nmf.components_.astype(np.float32)

    recon_err = nmf.reconstruction_err_
    logger.info(
        "[Module E] NMF reconstruction error: %.4f | explained components: %d",
        recon_err, n_components,
    )
    return W, H, nmf
