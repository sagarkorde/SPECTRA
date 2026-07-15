"""
SPECTRA — Spectral-Probabilistic Ensemble Clustering for Transaction
Recognition and Authentication.

Package structure
-----------------
spectra/graph.py       Module S  — spectral graph construction + embeddings
spectra/features.py    Module E  — entropy / MI ranking / NMF decomposition
spectra/vae.py         Module C  — VAE latent space + reconstruction anomaly
spectra/cluster.py     Module C  — HDBSCAN clustering + Wasserstein drift
spectra/profiler.py    Module P/T— Bayesian trust scoring + Markov chain
spectra/classifier.py  Module R  — GraphSAGE transaction authenticator
spectra/baselines.py              All baseline models (IEEE-mandatory)
spectra/visualize.py              Publication-ready figure generation
spectra/pipeline.py               End-to-end SPECTRA runner
"""

__version__ = "1.0.0"
__author__  = "Sagar Korde"
