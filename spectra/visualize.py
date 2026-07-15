"""
SPECTRA — Publication-Ready Figure Generation (IEEE TIFS format)
================================================================
All figures: 300 DPI, Times New Roman font, IEEE column widths.
Saved as both PDF and PNG in outputs/figures/.

Fig 1  — t-SNE / UMAP clusters (raw features vs SPECTRA side-by-side)
Fig 2  — VAE latent space scatter (cluster + authenticity)
Fig 3  — ROC curves: all models
Fig 4  — PR curves: all models
Fig 5  — Confusion matrix grid
Fig 6  — Wasserstein drift score over time
Fig 7  — Markov stationary distribution heatmap
Fig 8  — NMF basis patterns W (top-5)
Fig 9  — Feature importance (MI ranking bar chart)
Fig 10 — Ablation study F1-score bar chart
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# IEEE RCPARAMS
# ─────────────────────────────────────────────────────────────────────────────

IEEE_RCPARAMS = {
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "text.usetex":      False,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  1.2,
}

W_SINGLE = 3.5   # IEEE single-column width (inches)
W_DOUBLE = 7.0   # IEEE double-column width (inches)

TX_TYPE_NAMES = {
    0: "Standard", 1: "P2P", 2: "Consolidation",
    3: "Distribution", 4: "BatchPayment", 5: "CoinJoin",
}
PALETTE = sns.color_palette("tab10", 6)


def _setup():
    plt.rcParams.update(IEEE_RCPARAMS)


def _save(fig: plt.Figure, name: str, out_dir: str = "outputs/figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        logger.info("Saved %s", path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — t-SNE / UMAP COMPARISON (raw vs SPECTRA)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_tsne_comparison(
    X_raw:     np.ndarray,
    Z_spectra: np.ndarray,
    labels:    np.ndarray,
    out_dir:   str = "outputs/figures",
    method:    str = "umap",
    n_sample:  int = 5000,
    seed:      int = 42,
):
    """Side-by-side embedding: raw features vs SPECTRA latent space."""
    _setup()
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_raw), size=min(n_sample, len(X_raw)), replace=False)
    X_r, Z_s, lbl = X_raw[idx], Z_spectra[idx], labels[idx]

    scaler = StandardScaler()
    X_r_sc = scaler.fit_transform(X_r)

    if method.lower() == "umap":
        from umap import UMAP
        emb_raw     = UMAP(n_components=2, random_state=seed).fit_transform(X_r_sc)
        emb_spectra = UMAP(n_components=2, random_state=seed).fit_transform(Z_s)
        method_name = "UMAP"
    else:
        from sklearn.manifold import TSNE
        emb_raw     = TSNE(n_components=2, random_state=seed, perplexity=40).fit_transform(X_r_sc)
        emb_spectra = TSNE(n_components=2, random_state=seed, perplexity=40).fit_transform(Z_s)
        method_name = "t-SNE"

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.0))
    for ax, emb, title in [
        (axes[0], emb_raw,     f"{method_name} — Raw Features"),
        (axes[1], emb_spectra, f"{method_name} — SPECTRA Embedding"),
    ]:
        for cls, name in TX_TYPE_NAMES.items():
            m = lbl == cls
            if m.sum() == 0:
                continue
            ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5,
                       color=PALETTE[cls], label=name, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel(f"{method_name}-1")
        ax.set_ylabel(f"{method_name}-2")

    handles = [mpatches.Patch(color=PALETTE[c], label=n) for c, n in TX_TYPE_NAMES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.tight_layout()
    _save(fig, "fig1_tsne_comparison", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — VAE LATENT SPACE
# ─────────────────────────────────────────────────────────────────────────────

def fig2_vae_latent(
    Z:       np.ndarray,
    labels:  np.ndarray,
    recon_errors: np.ndarray,
    out_dir: str = "outputs/figures",
    n_sample: int = 5000,
    seed: int = 42,
):
    """VAE latent space (2-D UMAP projection), colored by cluster and anomaly score."""
    _setup()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(n_sample, len(Z)), replace=False)
    Z_s, lbl_s, err_s = Z[idx], labels[idx], recon_errors[idx]

    from umap import UMAP
    emb = UMAP(n_components=2, random_state=seed).fit_transform(Z_s)

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.0))

    # Left: color by cluster type
    for cls, name in TX_TYPE_NAMES.items():
        m = lbl_s == cls
        if m.sum() == 0: continue
        axes[0].scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5,
                        color=PALETTE[cls], label=name, rasterized=True)
    axes[0].set_title("VAE Latent — Transaction Type")
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")
    handles = [mpatches.Patch(color=PALETTE[c], label=n) for c, n in TX_TYPE_NAMES.items()]
    axes[0].legend(handles=handles, fontsize=7, markerscale=2)

    # Right: color by reconstruction error (anomaly score)
    sc = axes[1].scatter(emb[:, 0], emb[:, 1], c=err_s, s=4, alpha=0.6,
                         cmap="hot_r", vmin=np.percentile(err_s, 5),
                         vmax=np.percentile(err_s, 95), rasterized=True)
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("Reconstruction Error", fontsize=8)
    axes[1].set_title("VAE Latent — Anomaly Score")
    axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")

    fig.tight_layout()
    _save(fig, "fig2_vae_latent", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────

def fig3_roc_curves(
    results: Dict[str, Dict],
    y_true:  np.ndarray,
    out_dir: str = "outputs/figures",
):
    """Multi-model ROC curves (macro-average OVR) on one plot."""
    _setup()
    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.0))
    cmap = plt.cm.get_cmap("tab10")

    for i, (name, res) in enumerate(results.items()):
        proba = res.get("y_proba")
        if proba is None:
            continue
        n_cls = proba.shape[1] if proba.ndim == 2 else 2
        fprs, tprs, aucs = [], [], []
        for c in range(n_cls):
            y_bin = (y_true == c).astype(int)
            p     = proba[:, c] if proba.ndim == 2 else proba
            fpr, tpr, _ = roc_curve(y_bin, p)
            fprs.append(fpr); tprs.append(tpr)
            aucs.append(auc(fpr, tpr))
        mean_auc = np.mean(aucs)
        # Plot macro-average ROC
        all_fpr = np.unique(np.concatenate(fprs))
        mean_tpr = np.zeros_like(all_fpr)
        for j in range(n_cls):
            mean_tpr += np.interp(all_fpr, fprs[j], tprs[j])
        mean_tpr /= n_cls
        ax.plot(all_fpr, mean_tpr, color=cmap(i), lw=1.2,
                label=f"{name} (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Macro-Average)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    _save(fig, "fig3_roc_curves", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — PR CURVES
# ─────────────────────────────────────────────────────────────────────────────

def fig4_pr_curves(
    results: Dict[str, Dict],
    y_true:  np.ndarray,
    out_dir: str = "outputs/figures",
):
    """Precision-Recall curves for all models."""
    _setup()
    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.0))
    cmap = plt.cm.get_cmap("tab10")

    for i, (name, res) in enumerate(results.items()):
        proba = res.get("y_proba")
        if proba is None:
            continue
        n_cls = proba.shape[1] if proba.ndim == 2 else 2
        aps   = []
        for c in range(n_cls):
            y_bin = (y_true == c).astype(int)
            p     = proba[:, c] if proba.ndim == 2 else proba
            prec, rec, _ = precision_recall_curve(y_bin, p)
            ap = average_precision_score(y_bin, p)
            aps.append(ap)
        mean_ap = np.mean(aps)
        # Macro PR: interpolate
        all_rec  = np.linspace(0, 1, 100)
        mean_pre = np.zeros(100)
        for c in range(n_cls):
            y_bin = (y_true == c).astype(int)
            p     = proba[:, c] if proba.ndim == 2 else proba
            prec_c, rec_c, _ = precision_recall_curve(y_bin, p)
            mean_pre += np.interp(all_rec, rec_c[::-1], prec_c[::-1])
        mean_pre /= n_cls
        ax.plot(all_rec, mean_pre, color=cmap(i), lw=1.2,
                label=f"{name} (AP={mean_ap:.3f})")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Macro-Average)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    _save(fig, "fig4_pr_curves", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — CONFUSION MATRICES GRID
# ─────────────────────────────────────────────────────────────────────────────

def fig5_confusion_matrices(
    results:     Dict[str, Dict],
    y_true:      np.ndarray,
    class_names: List[str],
    out_dir:     str = "outputs/figures",
):
    """Heatmap grid of normalized confusion matrices."""
    _setup()
    cls_models = {n: r for n, r in results.items() if "y_pred" in r}
    n = len(cls_models)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(W_DOUBLE, 2.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (name, res) in enumerate(cls_models.items()):
        ax  = axes_flat[i]
        cm  = confusion_matrix(y_true, res["y_pred"], normalize="true")
        sns.heatmap(
            cm, ax=ax, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, linewidths=0.3,
        )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0,  labelsize=6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig5_confusion_matrices", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — WASSERSTEIN DRIFT OVER TIME
# ─────────────────────────────────────────────────────────────────────────────

def fig6_wasserstein_drift(
    window_keys:  List[str],
    drift_scores: List[float],
    alerts:       List[bool],
    threshold:    float,
    out_dir:      str = "outputs/figures",
):
    """Line plot of Wasserstein drift score per consecutive time window."""
    _setup()
    if len(drift_scores) == 0:
        logger.warning("No drift scores to plot — skipping Fig 6")
        return

    x = np.arange(len(drift_scores))
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.8))
    ax.plot(x, drift_scores, marker="o", ms=4, color="#1f77b4", lw=1.2)
    ax.axhline(threshold, color="red", ls="--", lw=1.0, label=f"Threshold δ={threshold:.3f}")

    alert_idx = [i for i, a in enumerate(alerts) if a]
    if alert_idx:
        ax.scatter(alert_idx, [drift_scores[i] for i in alert_idx],
                   color="red", zorder=5, s=20, label="Alert")

    tick_labels = window_keys[1:] if len(window_keys) > len(drift_scores) else window_keys
    step = max(1, len(x) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        [tick_labels[i] if i < len(tick_labels) else "" for i in x[::step]],
        rotation=45, ha="right",
    )
    ax.set_xlabel("Time Window"); ax.set_ylabel("Wasserstein Distance W₁")
    ax.set_title("Cluster Distribution Drift Over Time (SPECTRA — Wasserstein)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig6_wasserstein_drift", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — MARKOV STATIONARY DISTRIBUTION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def fig7_markov_heatmap(
    P:           np.ndarray,
    pi:          np.ndarray,
    state_names: Optional[List[str]] = None,
    out_dir:     str = "outputs/figures",
):
    """Heatmap of cluster-to-cluster Markov transition matrix + stationary π."""
    _setup()
    n = P.shape[0]
    if state_names is None:
        state_names = [f"S{i}" for i in range(n)]

    fig = plt.figure(figsize=(W_DOUBLE, 3.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.3)
    ax_P  = fig.add_subplot(gs[0])
    ax_pi = fig.add_subplot(gs[1])

    sns.heatmap(
        P, ax=ax_P, cmap="YlOrRd", annot=(n <= 10),
        fmt=".2f" if n <= 10 else "",
        xticklabels=state_names, yticklabels=state_names,
        cbar_kws={"fraction": 0.046, "pad": 0.04},
        linewidths=0.2,
    )
    ax_P.set_title("Markov Transition Matrix P")
    ax_P.set_xlabel("Next State"); ax_P.set_ylabel("Current State")

    sns.heatmap(
        pi.reshape(-1, 1), ax=ax_pi, cmap="Blues",
        yticklabels=state_names, xticklabels=["π"],
        annot=True, fmt=".3f", cbar=False,
    )
    ax_pi.set_title("Stationary\nDistribution π")
    fig.tight_layout()
    _save(fig, "fig7_markov_heatmap", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — NMF BASIS PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def fig8_nmf_patterns(
    H:           np.ndarray,
    feat_names:  List[str],
    n_patterns:  int = 5,
    out_dir:     str = "outputs/figures",
):
    """Bar chart of top-5 NMF basis patterns (rows of H)."""
    _setup()
    n_show = min(n_patterns, H.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(W_DOUBLE, 1.6 * n_show))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        h     = H[i]
        order = np.argsort(h)[::-1][:15]  # top-15 features per pattern
        ax = axes[i]
        ax.bar(range(len(order)), h[order],
               color=PALETTE[i % len(PALETTE)], edgecolor="none")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([feat_names[j] for j in order], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Weight", fontsize=7)
        ax.set_title(f"NMF Basis Pattern {i + 1}", fontsize=8)

    fig.suptitle("NMF Basis Patterns H (top features per component)", fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, "fig8_nmf_patterns", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — FEATURE IMPORTANCE (MI RANKING)
# ─────────────────────────────────────────────────────────────────────────────

def fig9_feature_importance(
    mi_scores:  np.ndarray,
    feat_names: List[str],
    entropy:    np.ndarray,
    top_k:      int = 20,
    out_dir:    str = "outputs/figures",
):
    """Horizontal bar chart: mutual information + Shannon entropy per feature."""
    _setup()
    k = min(top_k, len(mi_scores))
    order = np.argsort(mi_scores)[::-1][:k]

    mi_top  = mi_scores[order]
    ent_top = entropy[order] if len(entropy) == len(mi_scores) else np.zeros(k)
    names   = [feat_names[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 0.3 * k + 1.5), sharey=True)

    axes[0].barh(range(k), mi_top[::-1], color="#1f77b4", edgecolor="none")
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels(names[::-1], fontsize=7)
    axes[0].set_xlabel("Mutual Information (bits)")
    axes[0].set_title("MI Ranking")
    axes[0].invert_xaxis()

    axes[1].barh(range(k), ent_top[::-1], color="#ff7f0e", edgecolor="none")
    axes[1].set_xlabel("Shannon Entropy (bits)")
    axes[1].set_title("Entropy")

    fig.suptitle("Feature Importance — Top-20 by Mutual Information", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig9_feature_importance", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def fig10_ablation(
    ablation_results: Dict[str, Dict[str, float]],
    metrics:          List[str] = None,
    out_dir:          str = "outputs/figures",
):
    """Grouped bar chart comparing SPECTRA variants across key metrics."""
    _setup()
    if metrics is None:
        metrics = ["f1_macro", "accuracy", "auc_roc", "mcc"]

    variants = list(ablation_results.keys())
    n_vars   = len(variants)
    n_metrics= len(metrics)

    x      = np.arange(n_vars)
    width  = 0.8 / n_metrics
    cmap   = plt.cm.get_cmap("Set2")

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.5))

    for j, metric in enumerate(metrics):
        vals = [
            ablation_results[v].get(metric, 0.0) for v in variants
        ]
        offset = (j - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric.replace("_", " ").upper(),
                      color=cmap(j / n_metrics), edgecolor="none")
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("SPECTRA Ablation Study")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim([0, 1.12])
    fig.tight_layout()
    _save(fig, "fig10_ablation", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL FIG 11 — PERSISTENCE DIAGRAM (TDA)
# ─────────────────────────────────────────────────────────────────────────────

def fig11_persistence_diagram(
    Z: np.ndarray,
    out_dir: str = "outputs/figures",
    n_sample: int = 2000,
    seed: int = 42,
):
    """Persistence diagram using giotto-tda (optional, skipped if not installed)."""
    try:
        from gtda.homology import VietorisRipsPersistence
        from gtda.plotting import plot_diagram
    except ImportError:
        logger.warning("giotto-tda not installed — skipping Fig 11")
        return

    _setup()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(n_sample, len(Z)), replace=False)
    Z_s = Z[idx]

    vrp  = VietorisRipsPersistence(homology_dimensions=[0, 1])
    diag = vrp.fit_transform(Z_s[np.newaxis, :, :])[0]

    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.0))
    for dim, color in [(0, "#1f77b4"), (1, "#ff7f0e")]:
        d_mask = diag[:, 2] == dim
        birth, death = diag[d_mask, 0], diag[d_mask, 1]
        ax.scatter(birth, death, s=10, alpha=0.6, color=color, label=f"H{dim}", rasterized=True)
    max_val = diag[np.isfinite(diag[:, 1]), 1].max() if np.isfinite(diag[:, 1]).any() else 1
    ax.plot([0, max_val], [0, max_val], "k--", lw=0.8)
    ax.set_xlabel("Birth"); ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram — SPECTRA Latent Space")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig11_persistence_diagram", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES (supplementary)
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curves(
    vae_history: dict,
    gnn_history: dict,
    out_dir:     str = "outputs/figures",
):
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.5))

    axes[0].plot(vae_history["train_loss"], label="Train")
    axes[0].plot(vae_history["val_loss"],   label="Val")
    axes[0].set_title("VAE Total Loss (ELBO)")
    axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=7)

    axes[1].plot(vae_history["recon_loss"], label="Recon")
    axes[1].plot(vae_history["kl_loss"],    label="KL")
    axes[1].set_title("VAE Loss Components")
    axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=7)

    axes[2].plot(gnn_history["train_acc"], label="Train Acc")
    axes[2].plot(gnn_history["val_acc"],   label="Val Acc")
    axes[2].set_title("GNN Accuracy")
    axes[2].set_xlabel("Epoch"); axes[2].legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "fig_training_curves", out_dir)
