"""Shared CPU-safe checkpoint loader.

outputs/checkpoints/spectra_ckpt.pkl was pickled on a CUDA machine, so plain
pickle.load() fails on a CPU-only machine with:
    RuntimeError: Attempting to deserialize object on a CUDA device...
This redirects torch's CUDA storage deserialization through map_location='cpu'.
"""
import io
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_checkpoint_cpu(path: str) -> dict:
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\sagar\Desktop\SPECTRA\outputs\checkpoints\spectra_ckpt.pkl"
    d = load_checkpoint_cpu(path)
    print("Top-level keys:", list(d.keys()))
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {k}: {list(v.keys())}")
