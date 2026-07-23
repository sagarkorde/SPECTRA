"""Review response #3: hub-node coverage breakdown for the SPECTRA classifier.

The paper already states (SPECTRA_final.tex ~2260-2263) that 78.5% of test
transactions have no address in the 30,000-node hub graph and therefore
fall back to a uniform 1/6 prior in Module R (spectra/pipeline.py:461,
tx_proba[:] = 1.0/6). This script reconstructs the exact hub-covered mask
using the same address-matching logic as spectra/pipeline.py:462-473
(input addresses in node_index, else output addresses), then splits the
already-computed test-set predictions (checkpoint's phase2["gnn_probs_ts"])
into hub-covered vs. hub-uncovered groups and reports accuracy/F1-macro/MCC
for each. No retraining; everything here reads cached arrays.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._ckpt_utils import load_checkpoint_cpu
from spectra.graph import _parse_address_list

CKPT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints_fixed_labels\spectra_ckpt.pkl"
OUT_PATH = r"C:\Users\sagar\Desktop\SPECTRA\outputs\hub_coverage_breakdown.json"


def main():
    d = load_checkpoint_cpu(CKPT_PATH)
    p1, p2 = d["phase1"], d["phase2"]

    df = p1["df"]
    node_index = p1["node_index"]
    idx_ts = p1["idx_ts"]
    y_ts = p1["y_ts"]
    gnn_probs_ts = p2["gnn_probs_ts"]  # already sliced to idx_ts, shape (n_test, 6)

    assert len(idx_ts) == len(y_ts) == gnn_probs_ts.shape[0], (
        "idx_ts/y_ts/gnn_probs_ts length mismatch — checkpoint layout changed?"
    )

    covered = np.zeros(len(idx_ts), dtype=bool)
    for j, i in enumerate(idx_ts):
        row = df.iloc[i]
        matched = any(a in node_index for a in _parse_address_list(row["input_addresses"]))
        if not matched:
            matched = any(a in node_index for a in _parse_address_list(row["output_addresses"]))
        covered[j] = matched

    y_pred_ts = gnn_probs_ts.argmax(axis=1)

    def metrics(mask):
        if mask.sum() == 0:
            return None
        yt, yp = y_ts[mask], y_pred_ts[mask]
        return {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(yt, yp)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 else None,
        }

    result = {
        "n_test_total": int(len(idx_ts)),
        "n_hub_covered": int(covered.sum()),
        "pct_hub_covered": float(covered.mean() * 100),
        "pct_hub_uncovered": float((~covered).mean() * 100),
        "hub_covered": metrics(covered),
        "hub_uncovered": metrics(~covered),
        "overall": metrics(np.ones_like(covered)),
    }

    print(json.dumps(result, indent=2))
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
