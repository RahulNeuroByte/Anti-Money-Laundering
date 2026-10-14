import copy
from pathlib import Path

import pandas as pd
import pytest
from api.main import create_app
from api.schemas import CURRENCIES, LOCATIONS, PAYMENT_TYPES
from fastapi.testclient import TestClient

from aml.features import REQUIRED_INPUT
from aml.inference import load_artifact, score

VALID = {
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


@pytest.fixture()
def client(tiny_model):
    with TestClient(create_app(tiny_model)) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model": "tiny", "detail": None}


def test_predict_valid(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "probability", "threshold", "model"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] == int(body["probability"] >= body["threshold"])


@pytest.mark.parametrize("field", list(VALID))
def test_missing_field_is_422(client, field):
    payload = copy.deepcopy(VALID)
    del payload[field]
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Amount", "abc"),
        ("Amount", None),
        ("Sender_account", "x1"),
        ("Sender_account", 1.5),
        ("Timestamp", "not-a-date"),
        ("Payment_type", 5),
    ],
)
def test_wrong_type_is_422(client, field, value):
    payload = {**VALID, field: value}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Amount", 0),
        ("Amount", -10),
        ("Amount", 1e12),
        ("Sender_account", -1),
        ("Receiver_account", 10**12),
    ],
)
def test_out_of_range_is_422(client, field, value):
    assert client.post("/predict", json={**VALID, **{field: value}}).status_code == 422


@pytest.mark.parametrize(
    "field",
    ["Payment_type", "Payment_currency", "Received_currency", "Sender_bank_location"],
)
def test_unseen_category_is_422(client, field):
    assert client.post("/predict", json={**VALID, field: "Atlantis"}).status_code == 422


def test_extra_field_is_422(client):
    assert client.post("/predict", json={**VALID, "Laundering_type": "Smurfing"}).status_code == 422


def test_non_finite_amount_rejected(client):
    r = client.post(
        "/predict", content='{"Amount": NaN}', headers={"content-type": "application/json"}
    )
    assert r.status_code == 422


def test_timezone_aware_timestamp_accepted(client):
    r = client.post("/predict", json={**VALID, "Timestamp": "2023-01-20T12:30:00+05:30"})
    assert r.status_code == 200


def test_model_load_failure_keeps_service_up(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pkl")) as c:
        h = c.get("/health")
        assert h.status_code == 503
        assert h.json()["model_loaded"] is False and "FileNotFoundError" in h.json()["detail"]
        assert c.post("/predict", json=VALID).status_code == 503


def test_corrupt_model_file_reported(tmp_path):
    (tmp_path / "aml_pipeline.pkl").write_bytes(b"not a pickle")
    (tmp_path / "model_meta.json").write_text('{"model": "x", "threshold": 0.5}')
    with TestClient(create_app(tmp_path / "aml_pipeline.pkl")) as c:
        assert c.get("/health").status_code == 503


def test_bad_threshold_in_metadata_rejected(tiny_model, tmp_path):
    import shutil

    shutil.copy(tiny_model, tmp_path / "aml_pipeline.pkl")
    (tmp_path / "model_meta.json").write_text('{"model": "x", "threshold": 1.5}')
    with pytest.raises(ValueError):
        load_artifact(tmp_path / "aml_pipeline.pkl")


def test_api_matches_local_inference(client, tiny_model):
    artifact = load_artifact(tiny_model)
    payload = {**VALID, "Amount": 4321.0}
    body = client.post("/predict", json=payload).json()
    frame = pd.DataFrame([{**payload, "Timestamp": pd.Timestamp(payload["Timestamp"])}])
    proba, label = score(artifact, frame[REQUIRED_INPUT])
    assert body["probability"] == pytest.approx(float(proba[0]), abs=1e-12)
    assert body["prediction"] == int(label[0])
    assert body["threshold"] == artifact.threshold


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "models" / "aml_pipeline.pkl").exists(), reason="no real model")
def test_real_artifact_categories_match_schema_and_parity():
    artifact = load_artifact(ROOT / "models" / "aml_pipeline.pkl")
    cats = artifact.pipeline.named_steps["features"].categories_
    strip = {k: [v for v in vals if v != "__unseen__"] for k, vals in cats.items()}
    assert tuple(strip["Payment_type"]) == PAYMENT_TYPES
    assert tuple(strip["Payment_currency"]) == CURRENCIES
    assert tuple(strip["Received_currency"]) == CURRENCIES
    assert tuple(strip["Sender_bank_location"]) == LOCATIONS
    assert tuple(strip["Receiver_bank_location"]) == LOCATIONS
    payload = {**VALID, "Timestamp": "2023-08-01T12:30:00"}
    with TestClient(create_app(ROOT / "models" / "aml_pipeline.pkl")) as c:
        body = c.post("/predict", json=payload).json()
    frame = pd.DataFrame([{**payload, "Timestamp": pd.Timestamp(payload["Timestamp"])}])
    proba, label = score(artifact, frame[REQUIRED_INPUT])
    assert body["probability"] == pytest.approx(float(proba[0]), abs=1e-12)
    assert body["prediction"] == int(label[0])


def test_validation_errors_do_not_echo_submitted_values(client):
    r = client.post("/predict", json={**VALID, "Payment_type": "Atlantis"})
    assert r.status_code == 422
    assert "Atlantis" not in r.text
    assert set(r.json()["detail"][0]) == {"loc", "msg", "type"}
