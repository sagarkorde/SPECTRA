"""Review response #6: is the NMI gap between SPECTRA-HDBSCAN (VAE latent
space) and PlainHDBSCAN (raw features) statistically meaningful?

Uses the population-matched clust_metrics from the fixed_eval/fixed_labels
checkpoint (both now computed on the same idx_ts test subset as
PlainHDBSCAN, fixing the earlier all-300k-rows vs 45,001-test-rows
population mismatch). Bootstraps NMI over resamples of the test set for
both cluster-label assignments (with the same resample indices applied to
both, i.e. a paired bootstrap) to get a confidence interval on the gap.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._ckpt_utils import load_checkpoint_cpu

ORIG_CKPT = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints\spectra_ckpt.pkl"
OUT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\vae_hdbscan_significance_test.json"

N_BOOTSTRAP = 2000
SEED = 42


def main():
    # Use the ORIGINAL GPU-trained checkpoint directly - no retrain needed,
    # since the population fix is purely a different metric-computation
    # scope over already-existing cluster_labels/Z_all, not a code path
    # that depends on the (CPU-only) node-label-fix retrain.
    d_orig = load_checkpoint_cpu(ORIG_CKPT)

    p1 = d_orig["phase1"]
    p2 = d_orig["phase2"]
    idx_ts = p1["idx_ts"]
    y_ts = p1["y_ts"]

    # SPECTRA-HDBSCAN: cluster_labels over ALL rows, restrict to idx_ts
    spectra_labels_ts = p2["cluster_labels"][idx_ts]
    spectra_nmi_test_only = normalized_mutual_info_score(y_ts, spectra_labels_ts)
    print(f"SPECTRA-HDBSCAN NMI (test-only, {len(idx_ts)} rows): {spectra_nmi_test_only:.4f}")
    print(f"  (stored clust_metrics['nmi'] for cross-check): {p2['clust_metrics']['nmi']:.4f}")

    # PlainHDBSCAN: already computed on X_ts only (45,001 rows) - from the
    # ORIGINAL checkpoint since PlainHDBSCAN itself is unaffected by the
    # node-label / eval-population fixes (it's an independent baseline).
    plain_labels = d_orig["phase3"]["PlainHDBSCAN"]["labels"]
    plain_nmi = normalized_mutual_info_score(y_ts, plain_labels)
    print(f"PlainHDBSCAN NMI ({len(plain_labels)} rows): {plain_nmi:.4f}")

    assert len(spectra_labels_ts) == len(plain_labels) == len(y_ts), (
        "Population size mismatch - cannot do a paired bootstrap"
    )

    rng = np.random.default_rng(SEED)
    n = len(y_ts)
    gaps = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        nmi_spectra = normalized_mutual_info_score(y_ts[idx], spectra_labels_ts[idx])
        nmi_plain = normalized_mutual_info_score(y_ts[idx], plain_labels[idx])
        gaps.append(nmi_spectra - nmi_plain)
    gaps = np.array(gaps)

    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])
    observed_gap = spectra_nmi_test_only - plain_nmi
    p_value_two_sided = 2 * min((gaps > 0).mean(), (gaps < 0).mean())

    result = {
        "n_test": int(n),
        "spectra_hdbscan_nmi": float(spectra_nmi_test_only),
        "plain_hdbscan_nmi": float(plain_nmi),
        "observed_gap_spectra_minus_plain": float(observed_gap),
        "bootstrap_n": N_BOOTSTRAP,
        "bootstrap_gap_95ci": [float(ci_lo), float(ci_hi)],
        "bootstrap_gap_mean": float(gaps.mean()),
        "approx_two_sided_p_value": float(p_value_two_sided),
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
    }
    print(json.dumps(result, indent=2))
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
