"""Single inference code path shared by offline evaluation and the API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aml.features import REQUIRED_INPUT


@dataclass(frozen=True)
class Artifact:
    pipeline: Any
    threshold: float
    meta: dict[str, Any]


def load_artifact(model_path: str | Path, meta_path: str | Path | None = None) -> Artifact:
    """Load the serialised pipeline and its metadata. Only load files you built yourself."""
    model_path = Path(model_path)
    meta_path = Path(meta_path) if meta_path else model_path.with_name("model_meta.json")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    threshold = float(meta["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid decision threshold in metadata: {threshold}")
    return Artifact(joblib.load(model_path), threshold, meta)


def score(artifact: Artifact, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels) for raw transaction rows."""
    missing = [c for c in REQUIRED_INPUT if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    proba = artifact.pipeline.predict_proba(frame[REQUIRED_INPUT])[:, 1]
    return proba, (proba >= artifact.threshold).astype(int)


def prepare_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Raw transaction dicts -> frame in model column order.

    Timestamps may be ISO strings or datetimes; timezone-aware values are converted to naive UTC.
    Shared by the FastAPI service and the Azure scoring script so both preprocess identically.
    """
    frame = pd.DataFrame(records)
    missing = [c for c in REQUIRED_INPUT if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    ts = pd.to_datetime(frame["Timestamp"], utc=False)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    frame["Timestamp"] = ts
    return frame[REQUIRED_INPUT]
