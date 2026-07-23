"""Review response #7: justify (or revise) the Wasserstein drift alert
threshold delta = configs/config.yaml's drift.alert_threshold (0.1).

spectra/cluster.py's wasserstein_drift() already returns the raw per-window
W1 distances BEFORE thresholding (cached in the checkpoint as
phase2["drift_scores"]); alerts are just `w1 > alert_threshold`. This
re-thresholds those cached floats across a delta grid - no rerun needed.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._ckpt_utils import load_checkpoint_cpu

CKPT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints\spectra_ckpt.pkl"
OUT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\wasserstein_threshold_sweep.json"


def main():
    d = load_checkpoint_cpu(CKPT_PATH)
    p2 = d["phase2"]
    win_keys = p2["win_keys"]
    drift_scores = np.asarray(p2["drift_scores"], dtype=float)
    current_alerts = np.asarray(p2["alerts"])

    print(f"n windows (consecutive pairs): {len(drift_scores)}")
    print(f"raw W1 distances: min={drift_scores.min():.4f}, "
          f"max={drift_scores.max():.4f}, mean={drift_scores.mean():.4f}, "
          f"median={np.median(drift_scores):.4f}")
    print(f"current threshold=0.1 alert rate: {current_alerts.mean()*100:.1f}% "
          f"({current_alerts.sum()}/{len(current_alerts)})")

    grid = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    sweep = []
    for delta in grid:
        alerts = drift_scores > delta
        sweep.append({
            "delta": delta,
            "n_alerts": int(alerts.sum()),
            "alert_rate_pct": float(alerts.mean() * 100),
        })
        print(f"  delta={delta:5.2f} -> {alerts.sum():2d}/{len(alerts)} alerts "
              f"({alerts.mean()*100:5.1f}%)")

    # Percentile-based candidate thresholds as an alternative justification
    percentiles = {p: float(np.percentile(drift_scores, p)) for p in (50, 75, 90, 95)}

    result = {
        "n_windows": int(len(drift_scores)),
        "raw_distances": drift_scores.tolist(),
        "window_keys": [str(k) for k in win_keys],
        "current_threshold": 0.1,
        "current_alert_rate_pct": float(current_alerts.mean() * 100),
        "sweep": sweep,
        "percentiles_of_raw_distances": percentiles,
    }
    print("\nPercentiles of raw W1 distances:", percentiles)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
