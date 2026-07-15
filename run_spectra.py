#!/usr/bin/env python3
"""
run_spectra.py — SPECTRA Main Entry Point
==========================================
Usage
-----
  # Full pipeline
  python run_spectra.py

  # Individual phases (resume-friendly)
  python run_spectra.py --phase data
  python run_spectra.py --phase spectra
  python run_spectra.py --phase baselines
  python run_spectra.py --phase ablation
  python run_spectra.py --phase figures
  python run_spectra.py --phase results

  # Clear checkpoints and restart
  python run_spectra.py --reset

  # Use a different config
  python run_spectra.py --config configs/my_config.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "outputs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    datefmt = "%H:%M:%S"

    # Force UTF-8 on Windows so special chars in messages don't crash the stream
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(log_dir, f"spectra_{ts}.log"),
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    logger = logging.getLogger("spectra")
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPECTRA Bitcoin Transaction Analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "data", "spectra", "baselines", "ablation",
                                 "figures", "results"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear all checkpoints and restart from scratch")
    args = parser.parse_args()

    # ── Validate config ───────────────────────────────────────────────────────
    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Logging ───────────────────────────────────────────────────────────────
    logger = setup_logging(cfg["output"]["dir"])
    logger.info("=" * 70)
    logger.info("  SPECTRA — Spectral-Probabilistic Ensemble Clustering")
    logger.info("  for Transaction Recognition and Authentication")
    logger.info("  Author: Sagar Korde")
    logger.info("=" * 70)
    logger.info("Config: %s | Phase: %s", args.config, args.phase)
    logger.info("Device: %s | CUDA: %s",
                "cuda" if torch.cuda.is_available() else "cpu",
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    # ── Import pipeline ───────────────────────────────────────────────────────
    # Add SPECTRA root to path so `spectra.*` imports work regardless of CWD
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from spectra.pipeline import SPECTRAPipeline

    pipeline = SPECTRAPipeline(args.config)

    if args.reset:
        logger.info("Clearing checkpoints ...")
        pipeline.ckpt.clear()

    # ── Run requested phase(s) ────────────────────────────────────────────────
    t0 = time.perf_counter()

    if args.phase == "all":
        results = pipeline.run()

    elif args.phase == "data":
        p1 = pipeline.phase1_data()
        logger.info("Phase 1 done — dataset shape: %s", p1["X_sel"].shape)

    elif args.phase == "spectra":
        p1 = pipeline.phase1_data()
        p2 = pipeline.phase2_spectra(p1)
        logger.info("Phase 2 done — SPECTRA test F1: %.4f",
                    p2["gnn_metrics"]["test"].get("f1_macro", 0))

    elif args.phase == "baselines":
        p1 = pipeline.phase1_data()
        p2 = pipeline.phase2_spectra(p1)
        p3 = pipeline.phase3_baselines(p1, p2)
        logger.info("Phase 3 done — %d baselines evaluated", len(p3))

    elif args.phase == "ablation":
        p1 = pipeline.phase1_data()
        p2 = pipeline.phase2_spectra(p1)
        p4 = pipeline.phase4_ablation(p1, p2)
        logger.info("Phase 4 done — %d ablation variants", len(p4))

    elif args.phase == "figures":
        p1 = pipeline.phase1_data()
        p2 = pipeline.phase2_spectra(p1)
        p3 = pipeline.phase3_baselines(p1, p2)
        p4 = pipeline.phase4_ablation(p1, p2)
        pipeline.phase5_figures(p1, p2, p3, p4)

    elif args.phase == "results":
        p1 = pipeline.phase1_data()
        p2 = pipeline.phase2_spectra(p1)
        p3 = pipeline.phase3_baselines(p1, p2)
        p4 = pipeline.phase4_ablation(p1, p2)
        df = pipeline.phase6_results(p3, p4, p1, p2)
        print("\n" + df.to_string(index=False))

    elapsed = time.perf_counter() - t0
    logger.info("Total wall-clock time: %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("Outputs: %s", cfg["output"]["dir"])


if __name__ == "__main__":
    main()
