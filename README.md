# SPECTRA
## Spectral-Probabilistic Ensemble Clustering for Transaction Recognition and Authentication

**PhD Research | IEEE Transactions on Information Forensics & Security**
Author: Sagar Korde
Dataset: Bitcoin transaction parquet (~5.88M rows, 53 features, 2022–2025)

---

## Architecture

```
Module S   spectra/graph.py       Spectral graph + Laplacian eigenvectors
Module E   spectra/features.py    Entropy / MI ranking / NMF decomposition
Module C   spectra/vae.py         VAE latent representation + anomaly score
           spectra/cluster.py     HDBSCAN + Wasserstein drift detection
Module P   spectra/profiler.py    Bayesian trust scoring (Beta posterior)
Module T   spectra/profiler.py    Markov chain temporal profiling
Module R   spectra/classifier.py  GraphSAGE transaction authenticator
Module A   spectra/pipeline.py    Authentication output + uncertainty
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install -r requirements.txt

# 2. Edit data path in config.yaml
#    data.path: "C:/path/to/Dataset.parquet"

# 3. Run full pipeline
python run_spectra.py

# 4. Run individual phases (each checkpoints so you can resume)
python run_spectra.py --phase data       # Phase 1: load, EDA, spectral graph
python run_spectra.py --phase spectra    # Phase 2: VAE + HDBSCAN + GNN
python run_spectra.py --phase baselines  # Phase 3: all 10 baselines
python run_spectra.py --phase ablation   # Phase 4: 6 ablation variants
python run_spectra.py --phase figures    # Phase 5: all 10+ figures
python run_spectra.py --phase results    # Phase 6: CSV + LaTeX tables

# 5. Reset and restart
python run_spectra.py --reset
```

---

## Outputs

```
outputs/
  figures/
    fig1_tsne_comparison.{pdf,png}
    fig2_vae_latent.{pdf,png}
    fig3_roc_curves.{pdf,png}
    fig4_pr_curves.{pdf,png}
    fig5_confusion_matrices.{pdf,png}
    fig6_wasserstein_drift.{pdf,png}
    fig7_markov_heatmap.{pdf,png}
    fig8_nmf_patterns.{pdf,png}
    fig9_feature_importance.{pdf,png}
    fig10_ablation.{pdf,png}
    fig11_persistence_diagram.{pdf,png}   # optional (requires giotto-tda)
    fig_training_curves.{pdf,png}
  checkpoints/
    spectra_ckpt.pkl                      # resume checkpoint
  results_table.csv                       # all models × all metrics
  results_table.tex                       # LaTeX booktabs table (copy-paste)
  spectra_*.log                           # timestamped run log
```

---

## Classification Target (6 classes)

| ID | Class        | Count     | % of dataset |
|----|--------------|-----------|--------------|
| 0  | Standard     | ~2,869,307| ~48.8%       |
| 1  | P2P          | 1,842,344 | 31.3%        |
| 2  | Consolidation| 369,919   | 6.3%         |
| 3  | Distribution | 554,722   | 9.4%         |
| 4  | BatchPayment | 137,743   | 2.3%         |
| 5  | CoinJoin     | 110,352   | 1.9%         |

---

## Baselines (IEEE-Mandatory)

| ID  | Model             | Type           | Reference                          |
|-----|-------------------|----------------|------------------------------------|
| B1  | K-Means           | Clustering     | Lloyd 1957                         |
| B2  | DBSCAN            | Clustering     | Ester et al. KDD 1996              |
| B3  | Isolation Forest  | Anomaly        | Liu et al. ICDM 2008               |
| B4  | AE + K-Means      | Clustering     | —                                  |
| B5  | Plain HDBSCAN     | Clustering     | Campello et al. TKDD 2015          |
| B6  | XGBoost           | Classification | Chen & Guestrin KDD 2016           |
| B7  | EvolveGCN         | Classification | Pareja et al. AAAI 2020            |
| B8  | BERT4ETH          | Classification | He et al. WWW 2023                 |
| B9  | Graph Autoencoder | Anomaly        | Liu et al. IEEE TDSC 2024          |
| B10 | AA-GNN            | Classification | Fan et al. IEEE TII 2024           |

---

## Ablation Variants

| Variant        | What is removed                              |
|----------------|----------------------------------------------|
| SPECTRA-noS    | Spectral graph embedding (raw features only) |
| SPECTRA-noVAE  | VAE → plain autoencoder + KMeans clusters    |
| SPECTRA-noB    | Bayesian trust scoring                       |
| SPECTRA-noM    | Markov temporal features                     |
| SPECTRA-noW    | Wasserstein drift (density-free clustering)  |
| SPECTRA-full   | Complete proposed model                      |

---

## Evaluation Metrics

**Clustering:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, NMI
**Classification:** Accuracy, Precision, Recall, F1 (Macro + Weighted), AUC-ROC, AUC-PR, MCC
**Efficiency:** Training time (s), Inference time per sample (ms)

---

## Notes on Dataset Schema

- `has_taproot` is always `False` despite `input_script_types` / `output_script_types`
  containing `witness_v1_taproot`. This appears to be a feature engineering error in
  the source pipeline. The boolean column is dropped; script type strings are used directly
  for address parsing in Module S.
- `input_address_count`, `output_address_count` are binary (0/1), not actual counts.
- `is_self_transfer`, `has_p2pk`, `has_p2pkh`, `has_p2sh`, `has_p2wpkh`, `has_p2wsh`,
  `address_reuse`, `output_script_count` are always 0/False — dropped before ML.
- Spectral graph is built on a 150K-row subsample for computational tractability;
  the full 5.88M-row dataset is used for all tabular ML/DL models.

---

## Hardware & Runtime (Reference)

Tested on: HP Omen i9-13th Gen | 64 GB DDR5 | RTX 4060 8 GB | Windows 11
- Phase 1 (data + graph): ~10–20 min
- Phase 2 (VAE + GNN): ~15–30 min
- Phase 3 (baselines): ~30–60 min
- Phase 4 (ablation): ~20–40 min
- Total: ~1.5–2.5 hours (GPU recommended for Phases 2–4)
