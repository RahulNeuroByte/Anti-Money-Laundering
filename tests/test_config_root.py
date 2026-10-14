import importlib

import numpy as np

from aml import evaluation as ev


def test_project_root_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("AML_PROJECT_ROOT", str(tmp_path))
    import aml.config as cfg

    try:
        importlib.reload(cfg)
        assert cfg.PROJECT_ROOT == tmp_path
    finally:
        monkeypatch.delenv("AML_PROJECT_ROOT")
        importlib.reload(cfg)


def test_threshold_sweep_matches_threshold_metrics():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.9])
    sw = ev.threshold_sweep(y, p, np.array([0.3, 0.5, 0.7]))
    assert sw["threshold"] == [0.3, 0.5, 0.7]
    for i, t in enumerate(sw["threshold"]):
        m = ev.threshold_metrics(y, p, t)
        assert (sw["precision"][i], sw["recall"][i], sw["f1"][i]) == (
            m["precision"],
            m["recall"],
            m["f1"],
        )
    assert sw["recall"][0] >= sw["recall"][1] >= sw["recall"][2]
