"""
SPECTRA — Module P/T: Dynamic Transaction Profiling
====================================================
Two complementary profiling mechanisms:

Module P — Bayesian Trust Scoring
  Each address is assigned a Beta-distributed trust prior Beta(alpha, beta).
  After each observed transaction, the posterior is updated using the
  VAE reconstruction probability as the likelihood signal.

    Prior:     P(authentic) ~ Beta(alpha₀, beta₀)
    Likelihood: P(tx | authentic) ∝ exp(−λ · recon_error)
    Posterior:  P(authentic | tx) ∝ likelihood · prior   (Beta conjugate update)
    Trust:      T(addr) = E[posterior] = alpha_post / (alpha_post + beta_post)

Module T — Markov Chain Temporal Modeling
  Cluster IDs (from HDBSCAN) form behavioral states.  For each address we
  observe a sequence of states s_1, s_2, ..., s_n ordered by block_height.
  The transition matrix P_ij = P(state j | state i) captures switching patterns.
  The stationary distribution pi is solved via power iteration.
  Anomaly signal: deviation of the observed state-visit frequency from pi.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE P — BAYESIAN TRUST SCORER
# ─────────────────────────────────────────────────────────────────────────────

class BayesianTrustScorer:
    """Maintains per-address Beta posteriors updated by transaction evidence.

    Usage
    -----
    scorer = BayesianTrustScorer(alpha_prior=2.0, beta_prior=2.0)
    scorer.update_batch(address_list, recon_errors)
    trust_scores = scorer.trust_scores(address_list)
    """

    def __init__(
        self,
        alpha_prior: float = 2.0,
        beta_prior:  float = 2.0,
        likelihood_scale: float = 1.0,
    ):
        self.alpha0 = float(alpha_prior)
        self.beta0  = float(beta_prior)
        self.scale  = float(likelihood_scale)
        # Per-address accumulated posteriors (use dict, not defaultdict, for pickling)
        self._alpha: Dict[str, float] = {}
        self._beta:  Dict[str, float] = {}

    def _get_alpha(self, address: str) -> float:
        return self._alpha.get(address, self.alpha0)

    def _get_beta(self, address: str) -> float:
        return self._beta.get(address, self.beta0)

    def update(self, address: str, recon_error: float) -> float:
        """Update posterior for one address given a reconstruction error.

        Likelihood: p(tx | authentic) = exp(−scale · error)
        Conjugate Beta update for Bernoulli likelihood (Laplace-approx):
          alpha_post = alpha_prev + p_authentic
          beta_post = beta_prev + (1 − p_authentic)
        """
        p_auth = float(np.exp(-self.scale * recon_error))
        p_auth = float(np.clip(p_auth, 1e-6, 1 - 1e-6))
        self._alpha[address] = self._get_alpha(address) + p_auth
        self._beta[address]  = self._get_beta(address)  + (1.0 - p_auth)
        return self.trust_score(address)

    def update_batch(
        self,
        addresses: List[str],
        recon_errors: np.ndarray,
    ) -> np.ndarray:
        """Batch update — returns updated trust scores for each address."""
        scores = np.zeros(len(addresses), dtype=np.float32)
        for i, (addr, err) in enumerate(zip(addresses, recon_errors)):
            scores[i] = self.update(addr, float(err))
        return scores

    def trust_score(self, address: str) -> float:
        """E[Beta(alpha, beta)] = alpha / (alpha + beta) in (0, 1)."""
        a = self._get_alpha(address)
        b = self._get_beta(address)
        return float(a / (a + b))

    def trust_scores(self, addresses: List[str]) -> np.ndarray:
        return np.array([self.trust_score(a) for a in addresses], dtype=np.float32)

    def uncertainty(self, address: str) -> float:
        """Posterior variance = alphabeta / [(alpha+beta)²(alpha+beta+1)]."""
        a = self._get_alpha(address)
        b = self._get_beta(address)
        s = a + b
        return float(a * b / (s * s * (s + 1)))

    def uncertainties(self, addresses: List[str]) -> np.ndarray:
        return np.array([self.uncertainty(a) for a in addresses], dtype=np.float32)

    def authentication_decision(
        self,
        address: str,
        thresh_authentic: float = 0.65,
        thresh_suspicious: float = 0.40,
    ) -> str:
        """Map trust score to one of: Authentic / Suspicious / Fraudulent."""
        t = self.trust_score(address)
        if t >= thresh_authentic:
            return "Authentic"
        elif t >= thresh_suspicious:
            return "Suspicious"
        return "Fraudulent"

    def build_address_trust_map(
        self,
        df: pd.DataFrame,
        recon_errors: np.ndarray,
        input_col: str = "input_addresses",
    ) -> Dict[str, float]:
        """Populate scorer from a DataFrame + reconstruction errors array.

        Processes transactions in block_height order so the posterior
        accumulates chronologically.
        """
        order = df["block_height"].argsort().values
        from spectra.graph import _parse_address_list

        for idx in order:
            row   = df.iloc[idx]
            addrs = _parse_address_list(row[input_col])
            err   = float(recon_errors[idx])
            for addr in addrs:
                if addr:
                    self.update(addr, err)

        return {addr: self.trust_score(addr) for addr in self._alpha}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE T — MARKOV CHAIN TEMPORAL MODELER
# ─────────────────────────────────────────────────────────────────────────────

class MarkovChainProfiler:
    """Builds a cluster-transition Markov chain from address state sequences.

    States = HDBSCAN cluster IDs (noise -1 mapped to a dedicated state index).

    Usage
    -----
    mc = MarkovChainProfiler(n_states)
    mc.fit(address_sequences)      # list of (addr, [s1, s2, ...]) pairs
    P   = mc.transition_matrix()   # n_states x n_states
    pi  = mc.stationary_dist()     # n_states stationary distribution
    dev = mc.anomaly_score(seq)    # deviation of observed freq from pi
    """

    def __init__(self, n_states: int, smoothing: float = 1e-6):
        self.n_states  = int(n_states)
        self.smoothing = float(smoothing)
        self._counts   = np.full((self.n_states, self.n_states), self.smoothing, dtype=np.float64)
        self._P: Optional[np.ndarray] = None
        self._pi: Optional[np.ndarray] = None

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit_from_sequences(self, sequences: List[List[int]]) -> None:
        """Accumulate transition counts from multiple state sequences."""
        for seq in sequences:
            for a, b in zip(seq[:-1], seq[1:]):
                ai = self._clamp(a)
                bi = self._clamp(b)
                self._counts[ai, bi] += 1.0
        self._P  = None  # invalidate cache
        self._pi = None

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        address_col: str = "input_addresses",
    ) -> None:
        """Build state sequences per address from df + HDBSCAN labels.

        Transactions are ordered by block_height within each address.
        """
        from spectra.graph import _parse_address_list

        addr_rows: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, (_, row) in enumerate(df.iterrows()):
            for addr in _parse_address_list(row[address_col]):
                if addr:
                    addr_rows[addr].append((int(row["block_height"]), int(labels[i])))

        sequences: List[List[int]] = []
        for addr, pairs in addr_rows.items():
            pairs.sort(key=lambda x: x[0])
            seq = [p[1] for p in pairs]
            if len(seq) >= 2:
                sequences.append(seq)

        self.fit_from_sequences(sequences)
        logger.info(
            "[Module T] Markov chain fitted from %d address sequences.",
            len(sequences),
        )

    def _clamp(self, state: int) -> int:
        """Map any state (including -1 noise) into [0, n_states)."""
        if state < 0:
            return self.n_states - 1  # last state reserved for noise
        return min(state, self.n_states - 2)

    # ── transition matrix ─────────────────────────────────────────────────────
    def transition_matrix(self) -> np.ndarray:
        """Row-normalized transition probability matrix P."""
        if self._P is not None:
            return self._P
        row_sums = self._counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self._P  = (self._counts / row_sums).astype(np.float32)
        return self._P

    # ── stationary distribution ───────────────────────────────────────────────
    def stationary_dist(self, tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
        """Solve pi = piP via power iteration.

        Returns pi as a probability vector over states.
        """
        if self._pi is not None:
            return self._pi

        P  = self.transition_matrix()
        pi = np.ones(self.n_states, dtype=np.float64) / self.n_states
        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < tol:
                break
            pi = pi_new
        self._pi = (pi / pi.sum()).astype(np.float32)
        return self._pi

    # ── per-sequence anomaly score ────────────────────────────────────────────
    def anomaly_score(self, seq: List[int]) -> float:
        """L2 deviation between observed state-visit frequency and stationary pi.

        High deviation → address behaves differently from typical long-run pattern.
        """
        pi = self.stationary_dist()
        obs = np.zeros(self.n_states, dtype=np.float64)
        for s in seq:
            obs[self._clamp(s)] += 1
        if obs.sum() > 0:
            obs /= obs.sum()
        return float(np.linalg.norm(obs - pi))

    def batch_anomaly_scores(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        address_col: str = "input_addresses",
    ) -> np.ndarray:
        """Return per-transaction Markov anomaly score.

        For each transaction, compute the anomaly score of the sending address's
        state sequence up to and including that transaction.
        """
        from spectra.graph import _parse_address_list

        # Build per-address state sequences (chronological)
        addr_states: Dict[str, List[int]] = defaultdict(list)
        row_addrs:   List[List[str]]      = []

        for i, (_, row) in enumerate(df.iterrows()):
            addrs = _parse_address_list(row[address_col])
            row_addrs.append(addrs)
            for addr in addrs:
                if addr:
                    addr_states[addr].append(int(labels[i]))

        scores = np.zeros(len(df), dtype=np.float32)
        for i, addrs in enumerate(row_addrs):
            if addrs:
                seq   = addr_states.get(addrs[0], [0])
                scores[i] = self.anomaly_score(seq)

        return scores
