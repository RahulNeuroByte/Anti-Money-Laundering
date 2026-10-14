import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from mlops.azure.build_package import build

from aml.inference import load_artifact, prepare_frame, score

ROOT = Path(__file__).resolve().parents[1]
TX = {
    "Timestamp": "2023-01-20T12:30:00",
    "Sender_account": 12,
    "Receiver_account": 34,
    "Amount": 18000.5,
    "Payment_type": "Cross-border",
    "Payment_currency": "UK pounds",
    "Received_currency": "Euro",
    "Sender_bank_location": "UK",
    "Receiver_bank_location": "Spain",
}


@pytest.fixture(scope="module")
def package(tiny_model, tmp_path_factory):
    return build(model_dir=tiny_model.parent, out=tmp_path_factory.mktemp("azure"))


@pytest.fixture()
def scorer(package, monkeypatch):
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(package / "model"))
    spec = importlib.util.spec_from_file_location("score_mod", package / "code" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init()
    return mod


def test_single_batch_and_bare_payloads(scorer):
    one = scorer.run(json.dumps({"data": [TX]}))
    assert len(one) == 1 and set(one[0]) == {"prediction", "probability", "threshold", "model"}
    assert len(scorer.run(json.dumps({"data": [TX, TX]}))) == 2
    assert len(scorer.run({"data": TX})) == 1
    assert len(scorer.run(TX)) == 1
    assert len(scorer.run([TX])) == 1


def test_matches_local_inference(scorer, tiny_model):
    got = scorer.run({"data": [TX]})[0]
    artifact = load_artifact(tiny_model)
    proba, label = score(artifact, prepare_frame([TX]))
    assert got["probability"] == pytest.approx(float(proba[0]), abs=1e-12)
    assert got["prediction"] == int(label[0])


@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": "x"},
        {"data": [{**TX, "Payment_type": "Atlantis"}]},
        {"data": [{**TX, "Amount": -1}]},
        {"data": [{k: v for k, v in TX.items() if k != "Amount"}]},
        {"data": [TX, {**TX, "Sender_bank_location": "Mars"}]},
    ],
)
def test_invalid_payloads_raise_without_echoing_values(scorer, payload):
    with pytest.raises(ValueError) as err:
        scorer.run(payload)
    assert "Atlantis" not in str(err.value) and "Mars" not in str(err.value)


def test_not_json_and_too_many_rows(scorer):
    with pytest.raises(ValueError):
        scorer.run("{not json")
    with pytest.raises(ValueError):
        scorer.run({"data": [TX] * (scorer.MAX_ROWS + 1)})


def test_run_before_init_fails(package):
    spec = importlib.util.spec_from_file_location("score_cold", package / "code" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(RuntimeError):
        mod.run(TX)


def test_package_is_self_contained(package):
    """Run from the built code folder with no project on the path."""
    code = f"import os, json, score; score.init(); print(json.dumps(score.run({json.dumps(TX)})))"
    env = {"AZUREML_MODEL_DIR": str(package / "model"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=package / "code", env=env,
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr[-800:]
    assert 0 <= json.loads(out.stdout.strip().splitlines()[-1])[0]["probability"] <= 1


def test_build_fails_without_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(model_dir=tmp_path, out=tmp_path / "out")


@pytest.mark.skipif(not (ROOT / "models" / "aml_pipeline.pkl").exists(), reason="no real model")
def test_real_model_scores_like_local(tmp_path, monkeypatch):
    pkg = build(out=tmp_path / "azure")
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(pkg / "model"))
    spec = importlib.util.spec_from_file_location("score_real", pkg / "code" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init()
    tx = {**TX, "Timestamp": "2023-08-01T12:30:00"}
    got = mod.run({"data": [tx]})[0]
    proba, label = score(load_artifact(ROOT / "models" / "aml_pipeline.pkl"), prepare_frame([tx]))
    assert got["probability"] == pytest.approx(float(proba[0]), abs=1e-12)
    assert got["prediction"] == int(label[0])
    assert isinstance(pd.Timestamp(tx["Timestamp"]), pd.Timestamp)
