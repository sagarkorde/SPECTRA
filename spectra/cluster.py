"""
SPECTRA — Module C: HDBSCAN Clustering & Wasserstein Drift Detection
=====================================================================
Stage 3–4 of the ensemble clustering module.

Stage 3 — HDBSCAN
  Density-based clustering on the VAE latent vectors z.
  No assumption on cluster count (unlike K-Means / GMM).
  Noise points (label = -1) are flagged as primary anomaly candidates.

Stage 4 — Wasserstein Drift Detection
  For each time window t (month / week / quarter), compute the distribution
  P_t of cluster assignments across transactions in that window.
  Drift score: W_1(P_t, P_{t-1}) via scipy.stats.wasserstein_distance.
  An alert is raised when W_1 exceeds threshold delta.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HDBSCAN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_hdbscan(
    Z: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """Run HDBSCAN on latent vectors Z.

    Parameters
    ----------
    Z                        : (n_samples, latent_dim) array
    min_cluster_size         : minimum number of points in a cluster
    min_samples              : controls how conservative cluster boundaries are
    cluster_selection_epsilon: merge clusters within this ε of each other

    Returns
    -------
    labels      : (n_samples,) int cluster IDs; -1 = noise
    soft_probs  : (n_samples,) float membership probability to assigned cluster
    clusterer   : fitted HDBSCAN object
    """
    logger.info(
        "[Module C] HDBSCAN — min_cluster_size=%d min_samples=%d ...",
        min_cluster_size, min_samples,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(Z)

    soft_probs = clusterer.probabilities_.astype(np.float32)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    logger.info(
        "[Module C] HDBSCAN found %d clusters | noise points: %d (%.1f%%)",
        n_clusters, n_noise, 100 * n_noise / len(labels),
    )
    return labels.astype(np.int32), soft_probs, clusterer


def run_hdbscan_approximate(
    Z: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    subsample_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """Approximate HDBSCAN for the O(n^2)-memory limitation (Limitations,
    item 3): fit exact HDBSCAN on a random subsample of size
    `subsample_frac * n` (mutual-reachability graph stays subsample-sized,
    not full-n-sized), then assign every remaining point via
    `hdbscan.approximate_predict` (nearest-exemplar lookup, no new O(n^2)
    computation). Returns labels/soft_probs for ALL n points, in the same
    original order as Z, so it is a drop-in replacement for run_hdbscan's
    output shape.
    """
    n = len(Z)
    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(n, size=max(min_cluster_size * 2, int(n * subsample_frac)), replace=False)
    sub_mask = np.zeros(n, dtype=bool)
    sub_mask[sub_idx] = True
    rest_idx = np.where(~sub_mask)[0]

    logger.info(
        "[Module C] Approximate HDBSCAN — fitting on %d/%d points (%.1f%%), "
        "approximate_predict for the remaining %d ...",
        len(sub_idx), n, 100 * len(sub_idx) / n, len(rest_idx),
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    sub_labels = clusterer.fit_predict(Z[sub_idx])

    labels = np.full(n, -1, dtype=np.int32)
    soft_probs = np.zeros(n, dtype=np.float32)
    labels[sub_idx] = sub_labels
    soft_probs[sub_idx] = clusterer.probabilities_.astype(np.float32)

    if len(rest_idx) > 0:
        rest_labels, rest_strengths = hdbscan.approximate_predict(clusterer, Z[rest_idx])
        labels[rest_idx] = rest_labels
        soft_probs[rest_idx] = rest_strengths.astype(np.float32)

    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info(
        "[Module C] Approximate HDBSCAN found %d clusters | noise points: %d (%.1f%%)",
        n_clusters, n_noise, 100 * n_noise / n,
    )
    return labels, soft_probs, clusterer


def cluster_summary(labels: np.ndarray) -> Dict[int, int]:
    """Return {cluster_id: count} dict (noise = -1)."""
    return dict(Counter(labels.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def clustering_metrics(
    Z: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute cluster quality metrics.

    Silhouette, Davies-Bouldin, Calinski-Harabasz require >= 2 clusters and
    >= 2 non-noise samples.  NMI is computed when y_true is provided.

    Returns
    -------
    metrics : dict with keys silhouette, davies_bouldin, calinski_harabasz,
              and optionally nmi.
    """
    from sklearn.metrics import (
        calinski_harabasz_score, davies_bouldin_score,
        normalized_mutual_info_score, silhouette_score,
    )

    # Work only with non-noise points for geometry metrics
    mask = labels != -1
    metrics: Dict[str, float] = {}

    if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
        try:
            metrics["silhouette"]         = float(silhouette_score(Z[mask], labels[mask], sample_size=min(10000, mask.sum())))
        except Exception:
            metrics["silhouette"]         = float("nan")
        try:
            metrics["davies_bouldin"]     = float(davies_bouldin_score(Z[mask], labels[mask]))
        except Exception:
            metrics["davies_bouldin"]     = float("nan")
        try:
            metrics["calinski_harabasz"]  = float(calinski_harabasz_score(Z[mask], labels[mask]))
        except Exception:
            metrics["calinski_harabasz"]  = float("nan")
    else:
        metrics["silhouette"]         = float("nan")
        metrics["davies_bouldin"]     = float("nan")
        metrics["calinski_harabasz"]  = float("nan")

    if y_true is not None:
        try:
            metrics["nmi"] = float(normalized_mutual_info_score(y_true, labels))
        except Exception:
            metrics["nmi"] = float("nan")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# WASSERSTEIN DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _window_key(ts: pd.Timestamp, window: str) -> str:
    """Convert a timestamp to a window key string."""
    if window == "week":
        return f"{ts.year}-W{ts.isocalendar()[1]:02d}"
    elif window == "month":
        return f"{ts.year}-{ts.month:02d}"
    elif window == "quarter":
        q = (ts.month - 1) // 3 + 1
        return f"{ts.year}-Q{q}"
    raise ValueError(f"Unknown window type: {window}")


def wasserstein_drift(
    labels: np.ndarray,
    timestamps: np.ndarray,
    window: str = "month",
    alert_threshold: float = 0.1,
) -> Tuple[List[str], List[float], List[bool]]:
    """Compute Wasserstein drift score between consecutive time windows.

    For each window t, build the empirical distribution P_t over cluster IDs
    (treating cluster IDs as a 1-D discrete variable) and compute W_1(P_t, P_{t-1}).

    Parameters
    ----------
    labels      : (n_samples,) HDBSCAN cluster labels
    timestamps  : (n_samples,) Unix timestamp ints (block_time)
    window      : time aggregation granularity
    alert_threshold: delta — drift score above this triggers alert flag

    Returns
    -------
    window_keys  : sorted list of window keys
    drift_scores : W_1 per consecutive pair (len = n_windows - 1)
    alerts       : bool list, True when W_1 > threshold
    """
    ts_series  = pd.to_datetime(timestamps, unit="s")
    window_map = defaultdict(list)
    for lab, ts in zip(labels.tolist(), ts_series):
        key = _window_key(ts, window)
        window_map[key].append(int(lab))

    sorted_keys = sorted(window_map.keys())
    if len(sorted_keys) < 2:
        logger.warning("Fewer than 2 time windows — cannot compute drift.")
        return sorted_keys, [], []

    # Build per-window empirical distribution over cluster labels
    all_clusters = sorted(set(labels.tolist()))

    def dist_vec(labs: List[int]) -> np.ndarray:
        counts = Counter(labs)
        v = np.array([counts.get(c, 0) for c in all_clusters], dtype=np.float64)
        s = v.sum()
        return v / s if s > 0 else v

    dist_prev    = dist_vec(window_map[sorted_keys[0]])
    drift_scores = []
    alerts       = []

    for key in sorted_keys[1:]:
        dist_curr = dist_vec(window_map[key])
        # 1-D Wasserstein uses the cluster index as position on the real line
        w1 = wasserstein_distance(
            np.arange(len(all_clusters)),
            np.arange(len(all_clusters)),
            dist_prev, dist_curr,
        )
        drift_scores.append(float(w1))
        alerts.append(w1 > alert_threshold)
        logger.debug("Drift [%s]: W1=%.4f alert=%s", key, w1, w1 > alert_threshold)
        dist_prev = dist_curr

    n_alerts = sum(alerts)
    logger.info(
        "[Module C] Wasserstein drift — %d windows, %d alerts (delta=%.3f)",
        len(sorted_keys), n_alerts, alert_threshold,
    )
    return sorted_keys, drift_scores, alerts


def per_cluster_drift(
    labels: np.ndarray,
    timestamps: np.ndarray,
    window: str = "month",
) -> Dict[int, List[float]]:
    """Compute within-cluster temporal drift: fraction of cluster members per window.

    Returns
    -------
    {cluster_id: [fraction_window_0, fraction_window_1, ...]}
    """
    ts_series  = pd.to_datetime(timestamps, unit="s")
    unique_cls = sorted(set(int(l) for l in labels if l != -1))
    sorted_wins: List[str] = []

    # Collect window → cluster → count
    data: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for lab, ts in zip(labels.tolist(), ts_series):
        key = _window_key(ts, window)
        data[key][int(lab)] += 1
    sorted_wins = sorted(data.keys())

    result: Dict[int, List[float]] = {}
    for cls in unique_cls:
        fracs = []
        for win in sorted_wins:
            total = sum(data[win].values())
            cnt   = data[win].get(cls, 0)
            fracs.append(cnt / total if total > 0 else 0.0)
        result[cls] = fracs
    return result
