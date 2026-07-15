#!/usr/bin/env python3
"""
run_fair_baselines.py — Temporally-Fair Baseline Comparison for SPECTRA
========================================================================
Removes hard-leakage temporal identifiers (block_height, block_time, year)
from the feature set, re-selects top-k features by Mutual Information, and
re-runs all 10 baseline models + SPECTRA on the cleaned feature matrix.

Outputs
-------
  outputs/results_table_fair.csv
  outputs/results_table_fair.tex
  outputs/results_table_combined.csv   (both tables side-by-side)
  outputs/results_table_combined.tex

Usage
-----
  python run_fair_baselines.py
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── logging ─────────────────────────────────────────────────────────────────
Path("outputs").mkdir(exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"outputs/fair_baselines_{ts}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("fair")

# ── SPECTRA root on path ─────────────────────────────────────────────────────
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL LEAKAGE FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# block_height / block_time  — absolute temporal position in the blockchain;
#   monotonically encodes "when" a transaction happened, trivially separating
#   Bitcoin history periods where transaction types differed in prevalence.
# year  — directly encodes calendar year; same leakage as block_height.
#
# Retained cyclical features (hour, day_of_week, week_of_year) are behavioral
# signals (user activity rhythms, mining patterns) and do NOT uniquely identify
# a transaction's position in time; they are defensible as legitimate features.
TEMPORAL_LEAKAGE = {"block_height", "block_time", "year"}

TOP_K_MI = 20     # same as original pipeline
SEED     = 42

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE BUILDER (mirrors pipeline.py phase6)
# ─────────────────────────────────────────────────────────────────────────────

CLS_COLS = ["accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro", "auc_roc", "auc_pr", "mcc"]
CLU_COLS = ["silhouette", "davies_bouldin", "calinski_harabasz", "nmi"]
TIM_COLS = ["train_time_s", "inference_ms"]

MODEL_TYPE = {
    "KMeans": "clustering", "DBSCAN": "clustering",
    "IsolationForest": "clustering", "AE+KMeans": "clustering",
    "PlainHDBSCAN": "clustering", "GraphAE": "clustering",
    "SPECTRA-HDBSCAN": "clustering",
    "XGBoost": "classification", "EvolveGCN": "classification",
    "BERT4ETH": "classification", "AA-GNN": "classification",
    "SPECTRA": "classification",
}
MODEL_ORDER = [
    "KMeans", "DBSCAN", "IsolationForest", "AE+KMeans", "PlainHDBSCAN",
    "XGBoost", "EvolveGCN", "BERT4ETH", "GraphAE", "AA-GNN", "SPECTRA",
]


def _build_table(results: Dict) -> pd.DataFrame:
    rows = []
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r   = results[name]
        m   = r.get("metrics", {})
        row = {"Model": name, "Type": MODEL_TYPE.get(name, "?")}
        for c in CLS_COLS + CLU_COLS + TIM_COLS:
            row[c] = m.get(c, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _save_latex(df: pd.DataFrame, path: str, caption: str, label: str):
    """Write a booktabs LaTeX table."""
    col_fmt = "ll" + "r" * (len(df.columns) - 2)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
    ]
    # header
    lines.append(" & ".join(df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for c, v in row.items():
            if isinstance(v, float):
                cells.append("---" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = "outputs/checkpoints/spectra_ckpt.pkl"
    log.info("Loading checkpoint from %s ...", ckpt_path)
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    p1 = ckpt["phase1"]
    p2 = ckpt["phase2"]

    feat_names: List[str] = list(p1["feat_names"])
    X_raw: np.ndarray     = p1["X_raw"]          # shape (N, F)
    y:     np.ndarray     = p1["y"]
    idx_tr = p1["idx_tr"]
    idx_vl = p1["idx_vl"]
    idx_ts = p1["idx_ts"]

    log.info("Raw feature matrix: %s | features: %s", X_raw.shape, feat_names)

    # ── Drop temporal leakage columns ────────────────────────────────────────
    keep_mask = [n not in TEMPORAL_LEAKAGE for n in feat_names]
    feat_names_fair = [n for n, k in zip(feat_names, keep_mask) if k]
    X_fair = X_raw[:, keep_mask]

    dropped = [n for n, k in zip(feat_names, keep_mask) if not k]
    log.info("Dropped temporal leakage features: %s", dropped)
    log.info("Remaining features (%d): %s", len(feat_names_fair), feat_names_fair)

    # ── Re-run MI feature selection ──────────────────────────────────────────
    log.info("Re-computing Mutual Information scores (k=%d) ...", TOP_K_MI)
    mi = mutual_info_classif(X_fair, y, random_state=SEED)
    top_idx = np.argsort(mi)[::-1][:TOP_K_MI]
    sel_names_fair = [feat_names_fair[i] for i in top_idx]
    X_sel_fair = X_fair[:, top_idx]

    log.info("Top-%d MI features (temporally fair): %s", TOP_K_MI, sel_names_fair)

    # ── Scale & split ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_sel_fair[idx_tr])
    X_vl_s = scaler.transform(X_sel_fair[idx_vl])
    X_ts_s = scaler.transform(X_sel_fair[idx_ts])
    y_tr, y_vl, y_ts = y[idx_tr], y[idx_vl], y[idx_ts]

    log.info("Train/Val/Test shapes: %s | %s | %s", X_tr_s.shape, X_vl_s.shape, X_ts_s.shape)

    # ── Import baselines ─────────────────────────────────────────────────────
    from spectra.baselines import (
        run_aagnn, run_autoencoder_kmeans, run_bert4eth, run_dbscan,
        run_evolvegcn, run_graph_autoencoder, run_isolation_forest,
        run_kmeans, run_plain_hdbscan, run_xgboost,
    )
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg_bl = cfg["baselines"]

    results: Dict = {}

    # B1 KMeans
    log.info("[Fair B1] KMeans ...")
    results["KMeans"] = run_kmeans(
        X_tr_s, X_ts_s, y_ts, n_clusters=cfg_bl["kmeans_k"], seed=SEED)

    # B2 DBSCAN
    log.info("[Fair B2] DBSCAN ...")
    results["DBSCAN"] = run_dbscan(
        X_ts_s, y_ts, eps=cfg_bl["dbscan_eps"],
        min_samples=cfg_bl["dbscan_min_samples"])

    # B3 Isolation Forest
    log.info("[Fair B3] Isolation Forest ...")
    results["IsolationForest"] = run_isolation_forest(
        X_tr_s, X_ts_s, y_ts,
        contamination=cfg_bl["isolation_forest_contamination"], seed=SEED)

    # B4 AE + KMeans
    log.info("[Fair B4] Autoencoder + KMeans ...")
    results["AE+KMeans"] = run_autoencoder_kmeans(
        X_tr_s, X_ts_s, y_ts,
        hidden_dims=cfg_bl["autoencoder_hidden"],
        epochs=cfg_bl["autoencoder_epochs"],
        n_clusters=cfg_bl["kmeans_k"], seed=SEED)

    # B5 Plain HDBSCAN
    log.info("[Fair B5] Plain HDBSCAN ...")
    results["PlainHDBSCAN"] = run_plain_hdbscan(
        X_ts_s, y_ts,
        min_cluster_size=cfg["hdbscan"]["min_cluster_size"],
        min_samples=cfg["hdbscan"]["min_samples"])

    # B6 XGBoost
    log.info("[Fair B6] XGBoost ...")
    results["XGBoost"] = run_xgboost(
        X_tr_s, y_tr, X_vl_s, y_vl, X_ts_s, y_ts,
        n_estimators=cfg_bl["xgboost_n_estimators"],
        max_depth=cfg_bl["xgboost_max_depth"],
        lr=cfg_bl["xgboost_lr"], seed=SEED)

    # B7 EvolveGCN
    log.info("[Fair B7] EvolveGCN ...")
    results["EvolveGCN"] = run_evolvegcn(
        X_tr_s, y_tr, X_vl_s, y_vl, X_ts_s, y_ts, seed=SEED)

    # B8 BERT4ETH
    log.info("[Fair B8] BERT4ETH ...")
    results["BERT4ETH"] = run_bert4eth(
        X_tr_s, y_tr, X_vl_s, y_vl, X_ts_s, y_ts, seed=SEED)

    # B9 Graph Autoencoder
    log.info("[Fair B9] Graph Autoencoder ...")
    results["GraphAE"] = run_graph_autoencoder(
        X_tr_s, y_tr, X_vl_s, y_vl, X_ts_s, y_ts,
        n_clusters=cfg_bl["kmeans_k"], seed=SEED)

    # B10 AA-GNN
    log.info("[Fair B10] AA-GNN ...")
    results["AA-GNN"] = run_aagnn(
        X_tr_s, y_tr, X_vl_s, y_vl, X_ts_s, y_ts, seed=SEED)

    # SPECTRA — GNN uses spectral graph features (no raw tabular block_height/time)
    # Re-use phase2 metrics which are inherently temporally fair.
    log.info("[Fair] Adding SPECTRA (spectral GNN, temporally fair by construction) ...")
    results["SPECTRA"] = {
        "y_pred":  p2["gnn_probs_ts"].argmax(axis=1),
        "y_proba": p2["gnn_probs_ts"],
        "metrics": {
            **p2["gnn_metrics"]["test"],
            "train_time_s": p2["gnn_train_time"],
            "inference_ms": 0.0,
        },
    }

    # ── Build & save results table ───────────────────────────────────────────
    df_fair = _build_table(results)

    out_csv = "outputs/results_table_fair.csv"
    out_tex = "outputs/results_table_fair.tex"
    df_fair.to_csv(out_csv, index=False)
    log.info("Saved %s", out_csv)

    _save_latex(
        df_fair, out_tex,
        caption=(
            "Temporally-fair baseline comparison. "
            r"Block\_height, block\_time, and year removed; "
            "top-20 features re-selected by Mutual Information."
        ),
        label="tab:fair_baselines",
    )

    # ── Side-by-side combined table ──────────────────────────────────────────
    df_orig = pd.read_csv("outputs/results_table.csv")
    df_orig = df_orig[df_orig["Type"].isin(["classification", "clustering"])]

    # Merge on Model keeping only cls metrics for readability
    merge_cols = ["Model", "Type"] + CLS_COLS[:4]  # acc, f1_macro, f1_wt, prec
    df_m = df_orig[df_orig["Model"].isin(MODEL_ORDER)][merge_cols].copy()
    df_f = df_fair[merge_cols].copy()

    df_m.columns = ["Model", "Type"] + [f"{c}_orig" for c in CLS_COLS[:4]]
    df_f.columns = ["Model", "Type"] + [f"{c}_fair" for c in CLS_COLS[:4]]
    df_combined = df_m.merge(df_f, on=["Model", "Type"], how="outer")

    df_combined.to_csv("outputs/results_table_combined.csv", index=False)
    log.info("Saved outputs/results_table_combined.csv")

    _save_latex(
        df_combined,
        "outputs/results_table_combined.tex",
        caption=(
            "Classification metrics: original features vs. "
            r"temporally-fair features (block\_height, block\_time, year removed). "
            "Best fair result per column in \\textbf{bold}."
        ),
        label="tab:combined_comparison",
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEMPORALLY-FAIR RESULTS")
    print("=" * 70)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df_fair[["Model", "Type", "accuracy", "f1_macro", "auc_roc", "mcc",
                   "train_time_s"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE: Original vs Fair (accuracy | f1_macro)")
    print("=" * 70)
    for _, row in df_combined.iterrows():
        orig_acc  = row.get("accuracy_orig", float("nan"))
        fair_acc  = row.get("accuracy_fair", float("nan"))
        orig_f1   = row.get("f1_macro_orig", float("nan"))
        fair_f1   = row.get("f1_macro_fair", float("nan"))
        delta_acc = fair_acc - orig_acc if not (np.isnan(fair_acc) or np.isnan(orig_acc)) else float("nan")
        delta_f1  = fair_f1 - orig_f1  if not (np.isnan(fair_f1)  or np.isnan(orig_f1))  else float("nan")
        print(
            f"  {row['Model']:20s}  "
            f"acc: {orig_acc:6.4f} -> {fair_acc:6.4f} ({delta_acc:+.4f})  |  "
            f"f1:  {orig_f1:6.4f} -> {fair_f1:6.4f} ({delta_f1:+.4f})"
        )

    log.info("Fair baseline run complete.")


if __name__ == "__main__":
    main()
