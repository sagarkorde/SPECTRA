"""
SPECTRA — End-to-End Pipeline
==============================
Orchestrates all modules in the correct order and writes all outputs:
  - Checkpoint system (resume after Ctrl-C)
  - Full results table (CSV + LaTeX)
  - All 10+ publication figures
  - Ablation study (6 variants x all metrics)

Usage
-----
from spectra.pipeline import SPECTRAPipeline
pipeline = SPECTRAPipeline("configs/config.yaml")
results  = pipeline.run()
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    """Simple pickle-based checkpoint store."""

    def __init__(self, path: str = "outputs/checkpoints/spectra_ckpt.pkl"):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, Any] = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self._store, f)

    def has(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def put(self, key: str, obj: Any):
        self._store[key] = obj
        self._save()

    def clear(self):
        self._store = {}
        if os.path.exists(self.path):
            os.remove(self.path)


# ─────────────────────────────────────────────────────────────────────────────
# NODE -> TRANSACTION PREDICTION MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def build_majority_node_labels(df, node_index, y, n_nodes):
    """Assign each graph node (address) a label via majority vote across all
    transactions whose input addresses map to that node.

    BUG THIS REPLACES: the original inline code did
    `node_labels[node_index[addr]] = int(y[i])` inside a loop over all
    transactions, which is a last-write-wins overwrite - only the LAST
    transaction touching a given address (in df row order) determined that
    node's label, discarding every earlier one. High-degree hub addresses
    (exchanges, mixers) participate in many different transaction types, so
    a single arbitrary label per node made the node-classification task
    only weakly related to the real per-transaction labels, which is why
    node-level accuracy (~91%) collapsed to near-chance (~18%) once
    predictions were honestly re-scored against real transaction-level
    labels via map_node_probs_to_tx.
    """
    from collections import Counter
    from spectra.graph import _parse_address_list

    counts = [Counter() for _ in range(n_nodes)]
    for i in range(len(df)):
        for addr in _parse_address_list(df.iloc[i]["input_addresses"]):
            if addr in node_index:
                counts[node_index[addr]][int(y[i])] += 1

    node_labels = np.zeros(n_nodes, dtype=np.int64)
    for nid, counter in enumerate(counts):
        if counter:
            node_labels[nid] = counter.most_common(1)[0][0]
    return node_labels


def map_node_probs_to_tx(df, node_index, node_probs, idx_subset):
    """Map per-node class-probability predictions to transaction-level
    predictions via address matching: try input addresses first, then fall
    back to output addresses, averaging over matched nodes; transactions
    with no address in node_index (i.e. outside the pruned hub graph)
    default to a uniform prior. This is the SAME address-matching logic used
    to build gnn_probs_ts in phase2_spectra, factored out so it can also be
    used to re-score other node-level models (e.g. ablation variants)
    transaction-level for a fair, apples-to-apples comparison against the
    tabular/baseline models, which are all evaluated on the full
    transaction-level test set rather than a node-level split.
    """
    from spectra.graph import _parse_address_list

    n_classes = node_probs.shape[1]
    tx_proba = np.full((len(idx_subset), n_classes), 1.0 / n_classes, dtype=np.float32)
    for j, i in enumerate(idx_subset):
        row = df.iloc[i]
        matched = []
        for addr in _parse_address_list(row["input_addresses"]):
            if addr in node_index:
                matched.append(node_probs[node_index[addr]])
        if not matched:
            for addr in _parse_address_list(row["output_addresses"]):
                if addr in node_index:
                    matched.append(node_probs[node_index[addr]])
        if matched:
            tx_proba[j] = np.mean(matched, axis=0)
    return tx_proba


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SPECTRAPipeline:
    """Full SPECTRA pipeline with checkpointing and result aggregation."""

    CLASS_NAMES = ["Standard", "P2P", "Consolidation", "Distribution", "BatchPayment", "CoinJoin"]

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.seed   = self.cfg["seed"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt   = Checkpoint(
            os.path.join(self.cfg["output"]["checkpoints_dir"], "spectra_ckpt.pkl")
        )

        # Fix all seeds
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        logger.info("SPECTRA Pipeline initialized | device=%s | seed=%d", self.device, self.seed)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — DATA PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def phase1_data(self) -> Dict[str, Any]:
        """Load data, engineer features, build spectral embeddings, split."""
        if self.ckpt.has("phase1"):
            logger.info("[Phase 1] Loaded from checkpoint.")
            return self.ckpt.get("phase1")

        logger.info("=" * 60)
        logger.info("[Phase 1] DATA PIPELINE")
        logger.info("=" * 60)

        from spectra.graph import build_spectral_node_features, load_data
        from spectra.features import (
            mutual_information_ranking, nmf_decomposition,
            prepare_feature_matrix, scale_features, select_top_k_features,
            shannon_entropy_per_feature,
        )

        cfg_d  = self.cfg["data"]
        cfg_f  = self.cfg["features"]
        cfg_g  = self.cfg["graph"]

        # ── 1a. Load dataset ─────────────────────────────────────────────────
        df = load_data(cfg_d["path"], sample_n=cfg_d["sample_n"], seed=self.seed)

        # ── 1b. Prepare feature matrix ───────────────────────────────────────
        X_raw, feat_names = prepare_feature_matrix(df)
        y = df["tx_type"].values.astype(np.int64)

        # ── 1c. Entropy ranking ──────────────────────────────────────────────
        entropy = shannon_entropy_per_feature(X_raw)

        # ── 1d. Mutual information ranking ───────────────────────────────────
        mi_scores, mi_names = mutual_information_ranking(X_raw, y, feat_names, seed=self.seed)

        # ── 1e. Select top-k features ────────────────────────────────────────
        k = cfg_f["top_k_mi"]
        X_sel, sel_names, sel_idx = select_top_k_features(X_raw, mi_scores, feat_names, k=k)

        # Reorder entropy to match mi_scores order (already sorted)
        entropy_sorted = entropy[[feat_names.index(n) for n in mi_names[:k]]]

        # ── 1f. NMF decomposition ────────────────────────────────────────────
        cfg_nmf = self.cfg["nmf"]
        W_nmf, H_nmf, nmf_model = nmf_decomposition(
            X_sel,
            n_components=cfg_nmf["n_components"],
            max_iter=cfg_nmf["max_iter"],
            seed=self.seed,
            alpha_W=cfg_nmf["alpha_W"],
            l1_ratio=cfg_nmf["l1_ratio"],
        )

        # ── 1g. Train/val/test split ─────────────────────────────────────────
        idx_all = np.arange(len(df))
        idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
            idx_all, y, test_size=1 - cfg_d["train_frac"],
            stratify=y, random_state=self.seed,
        )
        val_ratio = cfg_d["val_frac"] / (cfg_d["val_frac"] + cfg_d["test_frac"])
        idx_vl, idx_ts, y_vl, y_ts = train_test_split(
            idx_tmp, y_tmp, test_size=1 - val_ratio,
            stratify=y_tmp, random_state=self.seed,
        )

        # ── 1h. Scale features ───────────────────────────────────────────────
        X_tr_s, X_vl_s, X_ts_s, scaler = scale_features(
            X_sel[idx_tr], X_sel[idx_vl], X_sel[idx_ts]
        )

        # ── 1i. Spectral graph (on smaller sample for tractability) ──────────
        n_graph = min(cfg_d["graph_sample_n"], len(df))
        rng     = np.random.default_rng(self.seed)
        g_idx   = rng.choice(len(df), size=n_graph, replace=False)
        df_graph = df.iloc[g_idx].reset_index(drop=True)

        logger.info("[Phase 1] Building spectral graph on %d rows ...", n_graph)
        node_feats, node_index, G = build_spectral_node_features(
            df_graph,
            k=cfg_g["k_eigenvectors"],
            normalized_laplacian=(cfg_g["laplacian_type"] == "normalized"),
            max_nodes=cfg_d.get("max_graph_nodes", 30_000),
            seed=self.seed,
        )

        out = {
            "df": df,
            "df_graph": df_graph,
            "X_raw": X_raw,
            "X_sel": X_sel,
            "X_tr_s": X_tr_s, "X_vl_s": X_vl_s, "X_ts_s": X_ts_s,
            "y": y,
            "y_tr": y_tr, "y_vl": y_vl, "y_ts": y_ts,
            "idx_tr": idx_tr, "idx_vl": idx_vl, "idx_ts": idx_ts,
            "feat_names": feat_names,
            "sel_names": sel_names,
            "sel_idx": sel_idx,
            "mi_scores": mi_scores,
            "mi_names": mi_names,
            "entropy": entropy,
            "entropy_sorted": entropy_sorted,
            "W_nmf": W_nmf, "H_nmf": H_nmf,
            "scaler": scaler,
            "node_feats": node_feats,
            "node_index": node_index,
            "G": G,
        }
        self.ckpt.put("phase1", out)
        logger.info("[Phase 1] DONE — %d samples | %d features | %d graph nodes",
                    len(df), X_sel.shape[1], node_feats.shape[0])
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — SPECTRA MODULES
    # ─────────────────────────────────────────────────────────────────────────

    def phase2_spectra(self, p1: Dict) -> Dict[str, Any]:
        """Train VAE, HDBSCAN, Bayesian profiler, Markov chain, GNN classifier."""
        if self.ckpt.has("phase2"):
            logger.info("[Phase 2] Loaded from checkpoint.")
            return self.ckpt.get("phase2")

        logger.info("=" * 60)
        logger.info("[Phase 2] SPECTRA MODULES")
        logger.info("=" * 60)

        from spectra.classifier import (
            SPECTRAClassifier, build_pyg_data,
            full_evaluation, train_classifier,
        )
        from spectra.cluster import (
            clustering_metrics, per_cluster_drift,
            run_hdbscan, wasserstein_drift,
        )
        from spectra.profiler import BayesianTrustScorer, MarkovChainProfiler
        from spectra.vae import (
            VAE, anomaly_threshold, get_latent_vectors,
            get_reconstruction_error, train_vae,
        )

        cfg_v = {k: float(v) if isinstance(v, str) and k in ("lr", "beta_kl", "dropout") else v
                 for k, v in self.cfg["vae"].items()}
        cfg_v = self.cfg["vae"]   # keep original; cast individually below
        cfg_h = self.cfg["hdbscan"]
        cfg_d = self.cfg["drift"]
        cfg_b = self.cfg["bayesian"]
        cfg_m = self.cfg["markov"]
        cfg_c = self.cfg["gnn"]

        # Defensive float-casting for values that YAML may parse as strings
        # (YAML parses scientific notation like 1e-4 as strings by default)
        for _d in (cfg_v, cfg_h, cfg_b, cfg_m, cfg_c):
            for _k in list(_d.keys()):
                if isinstance(_d[_k], str):
                    try:
                        _d[_k] = float(_d[_k])
                    except (ValueError, TypeError):
                        pass

        df       = p1["df"]
        X_tr_s   = p1["X_tr_s"]
        X_vl_s   = p1["X_vl_s"]
        X_ts_s   = p1["X_ts_s"]
        X_all_s  = p1["scaler"].transform(p1["X_sel"]).astype(np.float32)
        y        = p1["y"]
        idx_tr   = p1["idx_tr"]
        idx_vl   = p1["idx_vl"]
        idx_ts   = p1["idx_ts"]
        node_feats = p1["node_feats"]
        node_index = p1["node_index"]
        G        = p1["G"]

        in_dim = X_tr_s.shape[1]

        # ── 2a. VAE ───────────────────────────────────────────────────────────
        logger.info("[Phase 2a] Training VAE ...")
        vae = VAE(
            input_dim=in_dim,
            hidden_dims=cfg_v["hidden_dims"],
            latent_dim=cfg_v["latent_dim"],
            dropout=cfg_v["dropout"],
            beta_kl=cfg_v["beta_kl"],
        )
        t0  = time.perf_counter()
        vae_history = train_vae(
            vae, X_tr_s, X_vl_s,
            epochs=cfg_v["epochs"],
            batch_size=cfg_v["batch_size"],
            lr=cfg_v["lr"],
            device=self.device,
            seed=self.seed,
        )
        vae_train_time = time.perf_counter() - t0

        Z_all    = get_latent_vectors(vae, X_all_s, device=self.device)
        Z_ts     = Z_all[idx_ts]
        recon_all = get_reconstruction_error(vae, X_all_s, device=self.device)
        recon_ts  = recon_all[idx_ts]
        anom_thresh = anomaly_threshold(recon_all[idx_tr], cfg_v["anomaly_percentile"])

        logger.info("[Phase 2a] VAE done — anomaly threshold: %.4f", anom_thresh)

        # ── 2b. HDBSCAN ───────────────────────────────────────────────────────
        logger.info("[Phase 2b] HDBSCAN clustering on latent space ...")
        t0 = time.perf_counter()
        if cfg_h.get("use_approximate", False):
            from spectra.cluster import run_hdbscan_approximate
            logger.info("[Phase 2b] Using approximate HDBSCAN (hdbscan.use_approximate=true) "
                        "- exact HDBSCAN's O(n^2) mutual-reachability construction is "
                        "infeasible at this row count (see Limitations).")
            cluster_labels, soft_probs, hdb = run_hdbscan_approximate(
                Z_all,
                min_cluster_size=cfg_h["min_cluster_size"],
                min_samples=cfg_h["min_samples"],
                cluster_selection_epsilon=cfg_h["cluster_selection_epsilon"],
                metric=cfg_h["metric"],
                subsample_frac=cfg_h.get("approximate_subsample_frac", 0.1),
                seed=self.seed,
            )
        else:
            cluster_labels, soft_probs, hdb = run_hdbscan(
                Z_all,
                min_cluster_size=cfg_h["min_cluster_size"],
                min_samples=cfg_h["min_samples"],
                cluster_selection_epsilon=cfg_h["cluster_selection_epsilon"],
                metric=cfg_h["metric"],
                seed=self.seed,
            )
        hdbscan_train_time = time.perf_counter() - t0
        # IMPORTANT: evaluate on idx_ts only, matching the population every
        # clustering baseline (PlainHDBSCAN, KMeans, DBSCAN, etc.) is scored
        # on in phase3_baselines (X_ts/y_ts, 45,001 rows) - NOT all 300,000
        # rows the HDBSCAN model was fit on. Fitting on the full set is fine
        # (unsupervised), but comparing quality metrics computed over 300,000
        # rows against baselines' metrics computed over a 45,001-row subset
        # is not a fair like-for-like comparison. The full-population version
        # is kept under clust_metrics_full_population for transparency.
        clust_metrics = clustering_metrics(Z_all[idx_ts], cluster_labels[idx_ts], y_true=y[idx_ts])
        clust_metrics_full_population = clustering_metrics(Z_all, cluster_labels, y_true=y)
        logger.info("[Phase 2b] Cluster metrics (test-only): %s", clust_metrics)

        # ── 2c. Wasserstein drift ─────────────────────────────────────────────
        logger.info("[Phase 2c] Wasserstein drift detection ...")
        win_keys, drift_scores, alerts = wasserstein_drift(
            cluster_labels, df["block_time"].values,
            window=cfg_d["window"],
            alert_threshold=cfg_d["alert_threshold"],
        )

        # ── 2d. Bayesian trust scoring ────────────────────────────────────────
        logger.info("[Phase 2d] Bayesian trust scoring ...")
        bt_scorer = BayesianTrustScorer(
            alpha_prior=cfg_b["alpha_prior"],
            beta_prior=cfg_b["beta_prior"],
            likelihood_scale=cfg_b["likelihood_scale"],
        )
        from spectra.graph import _parse_address_list
        input_addrs_all = [
            (_parse_address_list(df.iloc[i]["input_addresses"]) + ["__unknown__"])[0]
            for i in range(len(df))
        ]
        trust_all = bt_scorer.update_batch(input_addrs_all, recon_all)

        # ── 2e. Markov chain profiler ─────────────────────────────────────────
        logger.info("[Phase 2e] Markov chain profiler ...")
        n_states = max(cluster_labels.max() + 2, 2)  # +2 for noise state
        mc = MarkovChainProfiler(n_states=n_states, smoothing=cfg_m["smoothing"])
        mc.fit_from_dataframe(df, cluster_labels)
        P_matrix = mc.transition_matrix()
        pi_dist  = mc.stationary_dist(tol=float(cfg_m["stationarity_tol"]))
        markov_scores = mc.batch_anomaly_scores(df, cluster_labels)

        # ── 2f. Build GNN node features ───────────────────────────────────────
        logger.info("[Phase 2f] Building GNN input features ...")
        n_nodes    = node_feats.shape[0]
        n_clusters = max(cluster_labels.max() + 1, 1)

        # Map transactions to graph nodes via input addresses
        node_trust  = np.full(n_nodes, 0.5, dtype=np.float32)
        node_markov = np.zeros(n_nodes, dtype=np.float32)
        node_clust  = np.zeros(n_nodes, dtype=np.int32)

        for i in range(len(df)):
            addrs = _parse_address_list(df.iloc[i]["input_addresses"])
            for addr in addrs:
                if addr in node_index:
                    nid = node_index[addr]
                    node_trust[nid]  = float(trust_all[i])
                    node_markov[nid] = float(markov_scores[i])
                    node_clust[nid]  = int(cluster_labels[i])

        # Cluster ID one-hot (cap at 20 clusters to avoid explosion)
        max_cls_onehot = min(n_clusters, 20)
        clust_oh       = np.eye(max_cls_onehot, dtype=np.float32)[
            np.clip(node_clust, 0, max_cls_onehot - 1)
        ]

        gnn_x = np.hstack([
            node_feats,
            trust_all[:n_nodes].reshape(-1, 1) if len(trust_all) >= n_nodes
                else node_trust.reshape(-1, 1),
            node_markov.reshape(-1, 1),
            clust_oh,
        ]).astype(np.float32)

        # Node labels = majority transaction type for each address
        node_labels = build_majority_node_labels(df, node_index, y, n_nodes)

        # Edge index from NetworkX graph
        edges   = list(G.edges())
        edge_src = np.array([node_index.get(u, 0) for u, v in edges], dtype=np.int64)
        edge_dst = np.array([node_index.get(v, 0) for u, v in edges], dtype=np.int64)
        valid    = (edge_src < n_nodes) & (edge_dst < n_nodes)
        edge_src = edge_src[valid]
        edge_dst = edge_dst[valid]

        # Node-level train/val/test masks
        rng_n    = np.random.default_rng(self.seed)
        all_nids = np.arange(n_nodes)
        rng_n.shuffle(all_nids)
        n_tr_n   = int(0.70 * n_nodes)
        n_vl_n   = int(0.15 * n_nodes)
        tr_mask  = np.zeros(n_nodes, bool); tr_mask[all_nids[:n_tr_n]] = True
        vl_mask  = np.zeros(n_nodes, bool); vl_mask[all_nids[n_tr_n:n_tr_n + n_vl_n]] = True
        ts_mask  = np.zeros(n_nodes, bool); ts_mask[all_nids[n_tr_n + n_vl_n:]] = True

        from spectra.classifier import build_pyg_data
        data_pyg = build_pyg_data(
            gnn_x, edge_src, edge_dst, node_labels, tr_mask, vl_mask, ts_mask
        )

        # ── 2g. Train GNN ─────────────────────────────────────────────────────
        logger.info("[Phase 2g] Training GNN classifier ...")
        gnn = SPECTRAClassifier(
            in_dim=gnn_x.shape[1],
            hidden_dim=cfg_c["hidden_dim"],
            embed_dim=cfg_c["embed_dim"],
            n_classes=6,
            dropout=cfg_c["dropout"],
            use_graph=True,
        )
        t0 = time.perf_counter()
        gnn_history = train_classifier(
            gnn, data_pyg,
            epochs=cfg_c["epochs"],
            lr=cfg_c["lr"],
            weight_decay=cfg_c["weight_decay"],
            device=self.device,
            patience=cfg_c["patience"],
            seed=self.seed,
        )
        gnn_train_time = time.perf_counter() - t0

        gnn_metrics = full_evaluation(gnn, data_pyg, self.device)
        logger.info("[Phase 2g] SPECTRA-full test metrics: %s", gnn_metrics["test"])

        # ── 2h. Transaction-level predictions (for baseline comparison) ───────
        gnn.eval().to(self.device)
        gnn_x_t  = torch.tensor(gnn_x, dtype=torch.float32).to(self.device)
        data_pyg = data_pyg.to(self.device)
        with torch.no_grad():
            gnn_probs_all = gnn.predict_proba(gnn_x_t, data_pyg.edge_index).cpu().numpy()

        # Map node predictions back to transaction indices (input addresses
        # first, else output addresses, else uniform prior for transactions
        # with no address in the pruned hub graph).
        tx_proba = map_node_probs_to_tx(df, node_index, gnn_probs_all, np.arange(len(df)))

        out = {
            "vae": vae,
            "vae_history": vae_history,
            "vae_train_time": vae_train_time,
            "Z_all": Z_all,
            "Z_ts": Z_ts,
            "recon_all": recon_all,
            "recon_ts": recon_ts,
            "anom_thresh": anom_thresh,
            "cluster_labels": cluster_labels,
            "soft_probs": soft_probs,
            "clust_metrics": clust_metrics,
            "clust_metrics_full_population": clust_metrics_full_population,
            "hdbscan_train_time": hdbscan_train_time,
            "win_keys": win_keys,
            "drift_scores": drift_scores,
            "alerts": alerts,
            "bt_scorer": bt_scorer,
            "trust_all": trust_all,
            "mc": mc,
            "P_matrix": P_matrix,
            "pi_dist": pi_dist,
            "markov_scores": markov_scores,
            "gnn": gnn,
            "data_pyg": data_pyg,
            "gnn_history": gnn_history,
            "gnn_train_time": gnn_train_time,
            "gnn_metrics": gnn_metrics,
            "gnn_probs_ts": tx_proba[p1["idx_ts"]],
            "n_states": n_states,
            "n_clusters": n_clusters,
        }
        self.ckpt.put("phase2", out)
        logger.info("[Phase 2] DONE")
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 — BASELINES
    # ─────────────────────────────────────────────────────────────────────────

    def phase3_baselines(self, p1: Dict, p2: Dict) -> Dict[str, Any]:
        """Train and evaluate all baseline models."""
        if self.ckpt.has("phase3"):
            logger.info("[Phase 3] Loaded from checkpoint.")
            return self.ckpt.get("phase3")

        logger.info("=" * 60)
        logger.info("[Phase 3] BASELINE COMPARISON")
        logger.info("=" * 60)

        from spectra.baselines import (
            run_aagnn, run_autoencoder_kmeans, run_bert4eth, run_dbscan,
            run_evolvegcn, run_graph_autoencoder, run_isolation_forest,
            run_kmeans, run_plain_hdbscan, run_xgboost,
        )

        cfg_bl = self.cfg["baselines"]
        X_tr   = p1["X_tr_s"]; y_tr = p1["y_tr"]
        X_vl   = p1["X_vl_s"]; y_vl = p1["y_vl"]
        X_ts   = p1["X_ts_s"]; y_ts = p1["y_ts"]
        Z_ts   = p2["Z_ts"]
        y_all  = p1["y"]

        results: Dict[str, Dict] = {}

        logger.info("[Phase 3] B1 KMeans ...")
        results["KMeans"] = run_kmeans(
            X_tr, X_ts, y_ts, n_clusters=cfg_bl["kmeans_k"], seed=self.seed
        )

        logger.info("[Phase 3] B2 DBSCAN ...")
        results["DBSCAN"] = run_dbscan(
            X_ts, y_ts, eps=cfg_bl["dbscan_eps"],
            min_samples=cfg_bl["dbscan_min_samples"]
        )

        logger.info("[Phase 3] B3 Isolation Forest ...")
        results["IsolationForest"] = run_isolation_forest(
            X_tr, X_ts, y_ts,
            contamination=cfg_bl["isolation_forest_contamination"],
            seed=self.seed,
        )

        logger.info("[Phase 3] B4 Autoencoder + KMeans ...")
        results["AE+KMeans"] = run_autoencoder_kmeans(
            X_tr, X_ts, y_ts,
            hidden_dims=cfg_bl["autoencoder_hidden"],
            epochs=cfg_bl["autoencoder_epochs"],
            n_clusters=cfg_bl["kmeans_k"],
            seed=self.seed,
        )

        logger.info("[Phase 3] B5 Plain HDBSCAN ...")
        results["PlainHDBSCAN"] = run_plain_hdbscan(
            X_ts, y_ts,
            min_cluster_size=self.cfg["hdbscan"]["min_cluster_size"],
            min_samples=self.cfg["hdbscan"]["min_samples"],
        )

        logger.info("[Phase 3] B6 XGBoost ...")
        results["XGBoost"] = run_xgboost(
            X_tr, y_tr, X_vl, y_vl, X_ts, y_ts,
            n_estimators=cfg_bl["xgboost_n_estimators"],
            max_depth=cfg_bl["xgboost_max_depth"],
            lr=cfg_bl["xgboost_lr"],
            seed=self.seed,
        )

        logger.info("[Phase 3] B7 EvolveGCN ...")
        results["EvolveGCN"] = run_evolvegcn(
            X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, seed=self.seed
        )

        if cfg_bl.get("skip_bert4eth", False):
            logger.info("[Phase 3] B8 BERT4ETH ... SKIPPED (baselines.skip_bert4eth=true; "
                        "its transformer-over-tabular-features training time scales worse "
                        "than linearly with row count and dominates full-dataset runtime)")
        else:
            logger.info("[Phase 3] B8 BERT4ETH ...")
            results["BERT4ETH"] = run_bert4eth(
                X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, seed=self.seed
            )

        logger.info("[Phase 3] B9 Graph Autoencoder ...")
        results["GraphAE"] = run_graph_autoencoder(
            X_tr, y_tr, X_vl, y_vl, X_ts, y_ts,
            n_clusters=cfg_bl["kmeans_k"], seed=self.seed
        )

        logger.info("[Phase 3] B10 AA-GNN ...")
        results["AA-GNN"] = run_aagnn(
            X_tr, y_tr, X_vl, y_vl, X_ts, y_ts, seed=self.seed
        )

        # Add SPECTRA-full to results for joint comparison.
        # IMPORTANT: metrics are computed transaction-level (from y_pred/y_proba
        # below, honest 1/6-uniform fallback for the ~78.5% of test transactions
        # with no address in the pruned hub graph included), NOT copied from
        # gnn_metrics["test"] (a node-level evaluation restricted to the hub
        # graph's own ~4,500-node test split). Every other model in `results`
        # above is scored on the full transaction-level test set (X_ts/y_ts),
        # so this keeps SPECTRA comparable to them on the same population.
        # The node-level number is kept under metrics_node_level for reference.
        from spectra.classifier import classification_metrics_from_arrays
        y_pred_tx  = p2["gnn_probs_ts"].argmax(axis=1)
        tx_metrics = classification_metrics_from_arrays(y_ts, p2["gnn_probs_ts"])
        tx_metrics["train_time_s"] = p2["gnn_train_time"]
        tx_metrics["inference_ms"] = 0.0
        results["SPECTRA"] = {
            "y_pred":  y_pred_tx,
            "y_proba": p2["gnn_probs_ts"],
            "metrics": tx_metrics,
            "metrics_node_level": p2["gnn_metrics"]["test"],
        }

        self.ckpt.put("phase3", results)
        logger.info("[Phase 3] DONE — %d models evaluated", len(results))
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4 — ABLATION STUDY
    # ─────────────────────────────────────────────────────────────────────────

    def phase4_ablation(self, p1: Dict, p2: Dict) -> Dict[str, Dict]:
        """Six SPECTRA variants for the ablation study."""
        if self.ckpt.has("phase4"):
            logger.info("[Phase 4] Loaded from checkpoint.")
            return self.ckpt.get("phase4")

        logger.info("=" * 60)
        logger.info("[Phase 4] ABLATION STUDY")
        logger.info("=" * 60)

        from spectra.classifier import (
            SPECTRAClassifier, build_pyg_data,
            full_evaluation, train_classifier,
        )
        from spectra.vae import VAE, get_latent_vectors, get_reconstruction_error, train_vae

        cfg_c  = self.cfg["gnn"]
        cfg_v  = self.cfg["vae"]
        X_tr_s = p1["X_tr_s"]; X_vl_s = p1["X_vl_s"]; X_ts_s = p1["X_ts_s"]
        X_all_s = p1["scaler"].transform(p1["X_sel"]).astype(np.float32)
        y       = p1["y"]
        idx_tr  = p1["idx_tr"]; idx_vl = p1["idx_vl"]; idx_ts = p1["idx_ts"]
        node_feats = p1["node_feats"]
        node_index = p1["node_index"]
        G          = p1["G"]
        cluster_labels = p2["cluster_labels"]
        trust_all      = p2["trust_all"]
        markov_scores  = p2["markov_scores"]
        n_clusters     = p2["n_clusters"]

        # Pre-build edge arrays
        from spectra.graph import _parse_address_list
        df = p1["df"]
        edges    = list(G.edges())
        n_nodes  = node_feats.shape[0]
        edge_src = np.array([node_index.get(u, 0) for u, v in edges], dtype=np.int64)
        edge_dst = np.array([node_index.get(v, 0) for u, v in edges], dtype=np.int64)
        valid    = (edge_src < n_nodes) & (edge_dst < n_nodes)
        edge_src = edge_src[valid]; edge_dst = edge_dst[valid]

        node_labels = build_majority_node_labels(df, node_index, y, n_nodes)

        rng_n = np.random.default_rng(self.seed)
        all_nids = np.arange(n_nodes); rng_n.shuffle(all_nids)
        n_tr_n = int(0.70 * n_nodes); n_vl_n = int(0.15 * n_nodes)
        tr_mask = np.zeros(n_nodes, bool); tr_mask[all_nids[:n_tr_n]] = True
        vl_mask = np.zeros(n_nodes, bool); vl_mask[all_nids[n_tr_n:n_tr_n + n_vl_n]] = True
        ts_mask = np.zeros(n_nodes, bool); ts_mask[all_nids[n_tr_n + n_vl_n:]] = True

        max_cls_oh = min(n_clusters, 20)
        node_trust  = np.full(n_nodes, 0.5, dtype=np.float32)
        node_markov = np.zeros(n_nodes, dtype=np.float32)
        node_clust  = np.zeros(n_nodes, dtype=np.int32)
        for i in range(len(df)):
            for addr in _parse_address_list(df.iloc[i]["input_addresses"]):
                if addr in node_index:
                    nid = node_index[addr]
                    node_trust[nid]  = float(trust_all[i]) if i < len(trust_all) else 0.5
                    node_markov[nid] = float(markov_scores[i]) if i < len(markov_scores) else 0.0
                    node_clust[nid]  = int(cluster_labels[i])

        clust_oh = np.eye(max_cls_oh, dtype=np.float32)[np.clip(node_clust, 0, max_cls_oh - 1)]

        def _build_gnn_x(use_spectral, use_vae_clust, use_bayesian, use_markov):
            parts = []
            if use_spectral:
                parts.append(node_feats)
            else:
                # Replace spectral embedding with raw per-node aggregated features
                parts.append(node_feats[:, -6:])  # address features only
            if use_bayesian:
                parts.append(node_trust.reshape(-1, 1))
            if use_markov:
                parts.append(node_markov.reshape(-1, 1))
            if use_vae_clust:
                parts.append(clust_oh)
            return np.hstack(parts).astype(np.float32) if parts else node_feats[:, :1]

        def _train_variant(name, gnn_x, use_graph=True):
            logger.info("[Phase 4] Variant: %s ...", name)
            data_v = build_pyg_data(
                gnn_x, edge_src, edge_dst, node_labels, tr_mask, vl_mask, ts_mask
            )
            gnn_v = SPECTRAClassifier(
                in_dim=gnn_x.shape[1],
                hidden_dim=cfg_c["hidden_dim"],
                embed_dim=cfg_c["embed_dim"],
                n_classes=6, dropout=cfg_c["dropout"], use_graph=use_graph,
            )
            train_classifier(
                gnn_v, data_v, epochs=min(cfg_c["epochs"], 60),
                lr=cfg_c["lr"], weight_decay=cfg_c["weight_decay"],
                device=self.device, patience=10, seed=self.seed,
            )
            node_metrics = full_evaluation(gnn_v, data_v, self.device)["test"]

            # Transaction-level re-scoring (same address-matching + uniform-
            # fallback logic as phase2/phase3's "SPECTRA" entry) so ablation
            # variants are comparable to the full-transaction-level baselines
            # rather than only to each other on the easier hub-node subset.
            from spectra.classifier import classification_metrics_from_arrays
            gnn_v.eval().to(self.device)
            data_v_dev = data_v.to(self.device)
            with torch.no_grad():
                all_probs = gnn_v.predict_proba(
                    data_v_dev.x,
                    data_v_dev.edge_index if use_graph else None,
                ).cpu().numpy()
            tx_proba  = map_node_probs_to_tx(df, node_index, all_probs, idx_ts)
            tx_metrics = classification_metrics_from_arrays(p1["y_ts"], tx_proba)

            # Top-level keys stay transaction-level (comparable to baselines);
            # node-level kept alongside for transparency/diagnostics.
            return {**tx_metrics, "metrics_node_level": node_metrics}

        ablation: Dict[str, Dict] = {}

        # SPECTRA-noS: no spectral embedding, raw address features only
        x_noS  = _build_gnn_x(False, True, True, True)
        ablation["SPECTRA-noS"]   = _train_variant("SPECTRA-noS",   x_noS)

        # SPECTRA-noVAE: replace VAE clusters with KMeans on raw features
        from sklearn.cluster import KMeans as KM
        km_tmp = KM(n_clusters=max_cls_oh, random_state=self.seed, n_init=5).fit(X_all_s)
        km_labels = km_tmp.predict(X_all_s)
        km_oh  = np.eye(max_cls_oh, dtype=np.float32)[np.clip(km_labels, 0, max_cls_oh - 1)]
        km_clust_oh = np.zeros_like(clust_oh)
        for i in range(len(df)):
            for addr in _parse_address_list(df.iloc[i]["input_addresses"]):
                if addr in node_index:
                    nid = node_index[addr]
                    km_clust_oh[nid] = km_oh[i]
        gnn_x_noVAE = np.hstack([node_feats, node_trust.reshape(-1,1), node_markov.reshape(-1,1), km_clust_oh]).astype(np.float32)
        ablation["SPECTRA-noVAE"] = _train_variant("SPECTRA-noVAE", gnn_x_noVAE)

        # SPECTRA-noB: no Bayesian trust
        x_noB  = _build_gnn_x(True, True, False, True)
        ablation["SPECTRA-noB"]   = _train_variant("SPECTRA-noB",   x_noB)

        # SPECTRA-noM: no Markov features
        x_noM  = _build_gnn_x(True, True, True, False)
        ablation["SPECTRA-noM"]   = _train_variant("SPECTRA-noM",   x_noM)

        # SPECTRA-noW: no Wasserstein drift (model unchanged; we note W1 alerts not used)
        # For the GNN ablation, noW = same as full (drift is a monitoring signal, not a feature)
        # We therefore ablate by replacing HDBSCAN labels with KMeans labels (no density info)
        ablation["SPECTRA-noW"]   = _train_variant("SPECTRA-noW",   gnn_x_noVAE)  # KMeans clusters

        # SPECTRA-full: complete model
        gnn_x_full = _build_gnn_x(True, True, True, True)
        ablation["SPECTRA-full"]  = _train_variant("SPECTRA-full",  gnn_x_full)

        self.ckpt.put("phase4", ablation)
        logger.info("[Phase 4] DONE")
        return ablation

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5 — VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def phase5_figures(self, p1: Dict, p2: Dict, p3: Dict, p4: Dict):
        """Generate all publication figures."""
        logger.info("=" * 60)
        logger.info("[Phase 5] FIGURES")
        logger.info("=" * 60)

        from spectra import visualize as viz

        out_dir = self.cfg["output"]["figures_dir"]
        y_ts    = p1["y_ts"]
        Z_all   = p2["Z_all"]
        y_all   = p1["y"]

        viz.fig1_latent_space_overview(
            p1["X_sel"], Z_all, y_all, p2["recon_all"], out_dir=out_dir, seed=self.seed
        )
        viz.fig3_roc_curves(
            {k: v for k, v in p3.items() if v.get("y_proba") is not None},
            y_ts, out_dir=out_dir,
        )
        viz.fig4_pr_curves(
            {k: v for k, v in p3.items() if v.get("y_proba") is not None},
            y_ts, out_dir=out_dir,
        )
        viz.fig5_confusion_matrices(
            {k: v for k, v in p3.items() if v.get("y_pred") is not None},
            y_ts, class_names=self.CLASS_NAMES, out_dir=out_dir,
        )
        viz.fig6_wasserstein_drift(
            p2["win_keys"], p2["drift_scores"], p2["alerts"],
            threshold=self.cfg["drift"]["alert_threshold"], out_dir=out_dir,
        )
        viz.fig7_markov_heatmap(
            p2["P_matrix"], p2["pi_dist"],
            state_names=[f"C{i}" for i in range(p2["P_matrix"].shape[0])],
            out_dir=out_dir,
        )
        viz.fig8_nmf_patterns(
            p1["H_nmf"], p1["sel_names"], out_dir=out_dir
        )
        viz.fig9_feature_importance(
            p1["mi_scores"][:len(p1["sel_names"])],
            p1["mi_names"][:len(p1["sel_names"])],
            p1["entropy_sorted"], out_dir=out_dir,
        )
        viz.fig10_ablation(p4, out_dir=out_dir)
        viz.fig_training_curves(p2["vae_history"], p2["gnn_history"], out_dir=out_dir)
        viz.fig11_persistence_diagram(Z_all, out_dir=out_dir, seed=self.seed)

        logger.info("[Phase 5] All figures saved to %s", out_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 6 — RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────

    def phase6_results(self, p3: Dict, p4: Dict, p1: Dict, p2: Dict):
        """Write results_table.csv and results_table.tex."""
        logger.info("=" * 60)
        logger.info("[Phase 6] RESULTS TABLE")
        logger.info("=" * 60)

        cls_metrics  = ["accuracy", "f1_macro", "f1_weighted", "precision_macro",
                         "recall_macro", "auc_roc", "auc_pr", "mcc",
                         "train_time_s", "inference_ms"]
        clust_metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "nmi",
                         "train_time_s", "inference_ms"]

        rows = []
        for name, res in p3.items():
            m    = res.get("metrics", {})
            kind = "classification" if "accuracy" in m else "clustering"
            cols = cls_metrics if kind == "classification" else clust_metrics
            row  = {"Model": name, "Type": kind}
            for c in cls_metrics + clust_metrics:
                row[c] = round(m.get(c, float("nan")), 4)
            rows.append(row)

        # Add ablation rows
        for name, m in p4.items():
            row = {"Model": name, "Type": "ablation"}
            for c in cls_metrics:
                row[c] = round(m.get(c, float("nan")), 4)
            for c in clust_metrics:
                row.setdefault(c, float("nan"))
            rows.append(row)

        # Add clustering metrics for SPECTRA HDBSCAN
        rows.append({
            "Model": "SPECTRA-HDBSCAN", "Type": "clustering",
            **{c: round(p2["clust_metrics"].get(c, float("nan")), 4)
               for c in cls_metrics + clust_metrics},
        })

        df_res = pd.DataFrame(rows)
        csv_path = self.cfg["output"]["results_csv"]
        df_res.to_csv(csv_path, index=False)
        logger.info("Results CSV: %s", csv_path)

        # LaTeX table
        tex_path = self.cfg["output"]["results_tex"]
        self._write_latex_table(df_res, tex_path)
        logger.info("Results LaTeX: %s", tex_path)
        return df_res

    @staticmethod
    def _write_latex_table(df: pd.DataFrame, path: str):
        """Write IEEE-style booktabs LaTeX table."""
        cls_cols = ["accuracy", "f1_macro", "auc_roc", "mcc", "train_time_s"]
        keep = ["Model"] + [c for c in cls_cols if c in df.columns]
        sub  = df[keep].copy()
        sub.columns = ["Model", "Accuracy", "F1 (Macro)", "AUC-ROC", "MCC", "Train(s)"]

        lines = [
            r"\begin{table}[!t]",
            r"\caption{SPECTRA vs. Baselines — Classification Performance on Bitcoin Tx Dataset}",
            r"\label{tab:results}",
            r"\centering",
            r"\begin{tabular}{l" + "c" * (len(sub.columns) - 1) + r"}",
            r"\toprule",
            " & ".join(sub.columns) + r" \\",
            r"\midrule",
        ]
        spectra_models = {"SPECTRA", "SPECTRA-full"}
        for _, row in sub.iterrows():
            vals = []
            for c, v in row.items():
                if c == "Model":
                    vals.append(str(v))
                elif pd.isna(v):
                    vals.append("---")
                else:
                    vals.append(f"{float(v):.4f}")
            line = " & ".join(vals) + r" \\"
            if row["Model"] in spectra_models:
                line = r"\textbf{" + line + "}"
            lines.append(line)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

        with open(path, "w") as f:
            f.write("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────────────
    # FULL RUN
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute all six phases in order."""
        t_start = time.perf_counter()
        logger.info(">> SPECTRA pipeline starting ...")

        p1 = self.phase1_data()
        p2 = self.phase2_spectra(p1)
        p3 = self.phase3_baselines(p1, p2)
        p4 = self.phase4_ablation(p1, p2)
        self.phase5_figures(p1, p2, p3, p4)
        df_results = self.phase6_results(p3, p4, p1, p2)

        elapsed = time.perf_counter() - t_start
        logger.info(">> SPECTRA pipeline complete in %.1f s (%.1f min)", elapsed, elapsed / 60)

        return {
            "phase1": p1, "phase2": p2, "phase3": p3,
            "phase4": p4, "results_df": df_results,
        }
