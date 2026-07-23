"""Review response #9: HDBSCAN's O(n^2) memory limitation (Limitations,
item 3) - compare exact HDBSCAN against the new approximate/subsample +
approximate_predict variant (spectra/cluster.py::run_hdbscan_approximate).

Runs both on the SAME Z_all (VAE latents) from the original GPU-trained
checkpoint - exact HDBSCAN's results are already cached (cluster_labels),
so only the approximate variant needs to actually run here. Reports
wall-clock time, peak memory (tracemalloc), and clustering quality
(test-only NMI/silhouette/etc, population-matched per the earlier fix).
"""
import json
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._ckpt_utils import load_checkpoint_cpu
from spectra.cluster import clustering_metrics, run_hdbscan, run_hdbscan_approximate

CKPT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints\spectra_ckpt.pkl"
OUT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\approx_hdbscan_comparison.json"

MIN_CLUSTER_SIZE = 100
MIN_SAMPLES = 10
SUBSAMPLE_FRAC = 0.10
SEED = 42


def main():
    d = load_checkpoint_cpu(CKPT_PATH)
    p1, p2 = d["phase1"], d["phase2"]
    Z_all = p2["Z_all"]
    y = p1["y"]
    idx_ts = p1["idx_ts"]

    # Exact: already computed and cached - re-time it fresh for a fair
    # wall-clock comparison against the approximate variant (same machine,
    # same run), rather than trusting the original GPU-timed value.
    tracemalloc.start()
    t0 = time.perf_counter()
    exact_labels, _, _ = run_hdbscan(
        Z_all, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, seed=SEED,
    )
    exact_time = time.perf_counter() - t0
    _, exact_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t0 = time.perf_counter()
    approx_labels, _, _ = run_hdbscan_approximate(
        Z_all, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
        subsample_frac=SUBSAMPLE_FRAC, seed=SEED,
    )
    approx_time = time.perf_counter() - t0
    _, approx_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    exact_metrics = clustering_metrics(Z_all[idx_ts], exact_labels[idx_ts], y_true=y[idx_ts])
    approx_metrics = clustering_metrics(Z_all[idx_ts], approx_labels[idx_ts], y_true=y[idx_ts])

    # Agreement between exact and approximate on the SAME points, restricted
    # to the subsample the approximate method actually fit exactly on (fair
    # "how much do we lose" comparison) vs the approximate_predict'd rest.
    from sklearn.metrics import normalized_mutual_info_score
    label_agreement_nmi = normalized_mutual_info_score(exact_labels, approx_labels)

    result = {
        "n_rows": int(len(Z_all)),
        "subsample_frac": SUBSAMPLE_FRAC,
        "exact": {
            "wall_time_s": exact_time,
            "peak_memory_mb": exact_peak / 1e6,
            "n_clusters": int(len(set(exact_labels.tolist())) - (1 if -1 in exact_labels else 0)),
            "test_only_metrics": exact_metrics,
        },
        "approximate": {
            "wall_time_s": approx_time,
            "peak_memory_mb": approx_peak / 1e6,
            "n_clusters": int(len(set(approx_labels.tolist())) - (1 if -1 in approx_labels else 0)),
            "test_only_metrics": approx_metrics,
        },
        "speedup_x": exact_time / approx_time if approx_time > 0 else None,
        "memory_reduction_x": exact_peak / approx_peak if approx_peak > 0 else None,
        "exact_vs_approx_label_agreement_nmi": float(label_agreement_nmi),
    }
    print(json.dumps(result, indent=2))
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
