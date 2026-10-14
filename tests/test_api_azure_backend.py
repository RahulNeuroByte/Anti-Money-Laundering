"""FastAPI with the Azure backend, against a local stub that runs the real scoring script."""

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from api.main import create_app
from fastapi.testclient import TestClient
from mlops.azure.build_package import build

KEY = "test-secret-key-123"
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
def stub(tiny_model, tmp_path_factory):
    pkg = build(model_dir=tiny_model.parent, out=tmp_path_factory.mktemp("azure"))
    import os

    os.environ["AZUREML_MODEL_DIR"] = str(pkg / "model")
    spec = importlib.util.spec_from_file_location("score_stub", pkg / "code" / "score.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.init()
    state = {"mode": "ok", "seen_auth": None, "seen_body": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["seen_auth"], state["seen_body"] = self.headers.get("Authorization"), body
            if self.headers.get("Authorization") != f"Bearer {KEY}":
                return self._send(401, b'{"message": "unauthorized"}')
            if state["mode"] == "http500":
                return self._send(500, b'{"message": "boom"}')
            if state["mode"] == "garbage":
                return self._send(200, b"not json")
            if state["mode"] == "shape":
                return self._send(200, b'{"unexpected": 1}')
            if state["mode"] == "slow":
                time.sleep(1.0)
            out = scorer.run(body.decode())
            self._send(200, json.dumps(out).encode())

        def _send(self, code, data):
            try:  # the client may already have timed out and closed the connection
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/score", state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _client(monkeypatch, url, key=KEY, timeout="5"):
    monkeypatch.setenv("AZURE_ENDPOINT_URL", url)
    monkeypatch.setenv("AZURE_ENDPOINT_KEY", key)
    monkeypatch.setenv("AZURE_ENDPOINT_TIMEOUT", timeout)
    return TestClient(create_app(backend="azure"))


def test_azure_backend_matches_local_backend(stub, tiny_model, monkeypatch):
    url, state = stub
    state["mode"] = "ok"
    with _client(monkeypatch, url) as remote, TestClient(create_app(tiny_model)) as local:
        r, loc = remote.post("/predict", json=TX), local.post("/predict", json=TX)
    assert r.status_code == 200
    assert r.json()["probability"] == pytest.approx(loc.json()["probability"], abs=1e-12)
    assert r.json()["prediction"] == loc.json()["prediction"]
    assert state["seen_auth"] == f"Bearer {KEY}"
    sent = json.loads(state["seen_body"])
    assert list(sent) == ["data"] and sent["data"][0]["Timestamp"] == TX["Timestamp"]


def test_health_reports_azure_backend(stub, monkeypatch):
    with _client(monkeypatch, stub[0]) as c:
        assert c.get("/health").json()["model"] == "azure-endpoint"


@pytest.mark.parametrize(("mode", "expected"), [("http500", 502), ("garbage", 502), ("shape", 502)])
def test_upstream_failures_become_502(stub, monkeypatch, mode, expected):
    url, state = stub
    state["mode"] = mode
    with _client(monkeypatch, url) as c:
        r = c.post("/predict", json=TX)
    state["mode"] = "ok"
    assert r.status_code == expected and KEY not in r.text


def test_wrong_key_is_502_and_key_not_leaked(stub, monkeypatch, caplog):
    url, state = stub
    state["mode"] = "ok"
    with _client(monkeypatch, url, key="wrong-key-999") as c:
        r = c.post("/predict", json=TX)
    assert r.status_code == 502 and "wrong-key-999" not in r.text + caplog.text


def test_timeout_is_504(stub, monkeypatch):
    url, state = stub
    state["mode"] = "slow"
    with _client(monkeypatch, url, timeout="0.3") as c:
        r = c.post("/predict", json=TX)
    state["mode"] = "ok"
    assert r.status_code == 504


def test_unreachable_endpoint_is_504(monkeypatch):
    with _client(monkeypatch, "http://127.0.0.1:9/score", timeout="1") as c:
        assert c.post("/predict", json=TX).status_code == 504


def test_invalid_input_never_reaches_endpoint(stub, monkeypatch):
    url, state = stub
    state["seen_body"] = None
    with _client(monkeypatch, url) as c:
        assert c.post("/predict", json={**TX, "Amount": -1}).status_code == 422
    assert state["seen_body"] is None


def test_missing_config_keeps_service_up(monkeypatch):
    monkeypatch.delenv("AZURE_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AZURE_ENDPOINT_KEY", raising=False)
    with TestClient(create_app(backend="azure")) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=TX).status_code == 503


def test_bad_backend_name_rejected():
    with pytest.raises(ValueError):
        create_app(backend="gcp")
