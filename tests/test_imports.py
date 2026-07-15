"""
Smoke tests: verify the package and its modules import cleanly and the
shipped config parses. These are import/sanity checks, not a full test
suite — they catch broken imports and config drift, not correctness of
the underlying algorithms.

Requires the dependencies in requirements.txt to be installed.
"""
import yaml


def test_package_imports():
    import spectra
    assert spectra.__version__


def test_submodules_import():
    from spectra import (
        baselines,
        classifier,
        cluster,
        features,
        graph,
        pipeline,
        profiler,
        vae,
        visualize,
    )


def test_config_parses():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["seed"] == 42
    assert "data" in cfg and "path" in cfg["data"]
    assert "gnn" in cfg and cfg["gnn"]["model"] == "GraphSAGE"


def test_pipeline_class_importable():
    from spectra.pipeline import SPECTRAPipeline
    assert callable(SPECTRAPipeline)
