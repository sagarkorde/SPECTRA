"""One-off setup: copy the existing checkpoint (CPU-safe), keep phase1+phase2
(expensive: VAE/GraphSAGE training, spectral embeddings), strip phase3+phase4
so the pipeline recomputes just those two phases with the eval-mismatch fix.
Saves to a NEW checkpoints directory - never touches the original checkpoint.
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._ckpt_utils import load_checkpoint_cpu

SRC = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints\spectra_ckpt.pkl"
DST_DIR = r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints_fixed_eval"
DST = os.path.join(DST_DIR, "spectra_ckpt.pkl")

os.makedirs(DST_DIR, exist_ok=True)

d = load_checkpoint_cpu(SRC)
print("Original keys:", list(d.keys()))
d.pop("phase3", None)
d.pop("phase4", None)
print("Keys kept:", list(d.keys()))

with open(DST, "wb") as f:
    pickle.dump(d, f)
print(f"Wrote {DST} ({os.path.getsize(DST) / 1e6:.1f} MB)")
