"""Review response #2: per-class precision/recall/F1 breakdown supporting
the macro-F1 discrepancy discussion (SPECTRA-full f1_macro=0.3806 vs
EvolveGCN's 0.5561, from outputs/results_table.csv).

Reads cached test-set predictions from the (already eval-mismatch-fixed)
outputs/checkpoints_fixed_eval/spectra_ckpt.pkl checkpoint - falls back to
the original checkpoint for models whose predictions didn't change (XGBoost,
EvolveGCN, AA-GNN weren't affected by the node/transaction-level fix).
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._ckpt_utils import load_checkpoint_cpu

FIXED_CKPT = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints_fixed_eval\spectra_ckpt.pkl"
ORIG_CKPT = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints\spectra_ckpt.pkl"
OUT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\per_class_metrics.json"

CLASS_NAMES = ["Standard", "P2P", "Consolidation", "Distribution", "BatchPayment", "CoinJoin"]


def main():
    d_fixed = load_checkpoint_cpu(FIXED_CKPT)
    d_orig = load_checkpoint_cpu(ORIG_CKPT)
    p1 = d_fixed["phase1"]
    y_ts = p1["y_ts"]

    p3_fixed = d_fixed["phase3"]
    p3_orig = d_orig["phase3"]

    models_of_interest = ["SPECTRA", "XGBoost", "EvolveGCN", "AA-GNN", "BERT4ETH"]
    result = {}
    for name in models_of_interest:
        # Use the fixed checkpoint's entry if present there (it re-ran all
        # baselines too), else fall back to the original.
        entry = p3_fixed.get(name) or p3_orig.get(name)
        y_pred = np.asarray(entry["y_pred"])
        report = classification_report(
            y_ts, y_pred, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        result[name] = report
        print(f"\n=== {name} ===")
        print(classification_report(y_ts, y_pred, target_names=CLASS_NAMES, zero_division=0))

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
