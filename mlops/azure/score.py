"""Azure ML managed online endpoint scoring script (init/run contract).

Accepted request bodies (same fields and allowed values as the FastAPI service):
  {"data": [ {transaction}, ... ]}   or   {"data": {transaction}}   or   a bare transaction/list.
Response: a list with one {prediction, probability, threshold, model} object per transaction.
Invalid input raises ValueError (HTTP 500 from the inference server) with a message that names
the row and field but never echoes submitted values.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from api.schemas import TransactionRequest
from pydantic import ValidationError

from aml.inference import load_artifact, prepare_frame, score

log = logging.getLogger("aml.score")
MAX_ROWS = 1000
_artifact = None


def init() -> None:
    """Called once when the container starts."""
    global _artifact
    root = Path(os.environ["AZUREML_MODEL_DIR"])
    matches = sorted(root.rglob("aml_pipeline.pkl"))
    if not matches:
        raise FileNotFoundError(f"aml_pipeline.pkl not found under {root}")
    _artifact = load_artifact(matches[0])
    log.info("Model loaded: %s", _artifact.meta.get("model"))


def _records(payload: Any) -> list[dict[str, Any]]:
    body = payload.get("data", payload) if isinstance(payload, dict) else payload
    records = [body] if isinstance(body, dict) else body
    if not isinstance(records, list) or not records:
        raise ValueError("Request must contain a transaction or a non-empty list of transactions")
    if len(records) > MAX_ROWS:
        raise ValueError(f"Too many transactions: limit is {MAX_ROWS}")
    return records


def run(raw_data: str | bytes | dict | list) -> list[dict[str, Any]]:
    """Called for every request."""
    if _artifact is None:
        raise RuntimeError("Model is not loaded")
    try:
        payload = json.loads(raw_data) if isinstance(raw_data, str | bytes) else raw_data
    except json.JSONDecodeError as exc:
        raise ValueError("Request body is not valid JSON") from exc
    validated = []
    for i, rec in enumerate(_records(payload)):
        try:
            validated.append(TransactionRequest.model_validate(rec).model_dump())
        except ValidationError as exc:
            fields = ", ".join(
                ".".join(str(p) for p in e["loc"]) + f": {e['msg']}" for e in exc.errors()
            )
            raise ValueError(f"Row {i} invalid ({fields})") from None
    proba, label = score(_artifact, prepare_frame(validated))
    model = str(_artifact.meta.get("model"))
    return [
        {
            "prediction": int(lab),
            "probability": float(p),
            "threshold": _artifact.threshold,
            "model": model,
        }
        for p, lab in zip(proba, label, strict=True)
    ]
