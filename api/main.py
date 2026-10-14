"""FastAPI service for AML transaction scoring.

Two backends, selected by AML_BACKEND:
  local (default)  loads models/aml_pipeline.pkl (override with AML_MODEL_PATH)
  azure            forwards each request to the Azure ML managed online endpoint
                   (AZURE_ENDPOINT_URL, AZURE_ENDPOINT_KEY, optional AZURE_ENDPOINT_TIMEOUT)

Run from the project root:  $env:PYTHONPATH="src"; uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from aml.config import PROJECT_ROOT
from aml.inference import load_artifact, prepare_frame, score
from api.azure_client import AzureEndpointClient, UpstreamError
from api.schemas import HealthResponse, PredictionResponse, TransactionRequest

log = logging.getLogger("aml.api")


def default_model_path() -> Path:
    raw = os.environ.get("AML_MODEL_PATH", "models/aml_pipeline.pkl")
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_app(
    model_path: str | Path | None = None,
    backend: str | None = None,
) -> FastAPI:
    backend = (backend or os.environ.get("AML_BACKEND", "local")).lower()
    if backend not in ("local", "azure"):
        raise ValueError("AML_BACKEND must be 'local' or 'azure'")
    path = Path(model_path) if model_path else default_model_path()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artifact = None
        app.state.client = None
        app.state.load_error = None
        try:
            if backend == "local":
                app.state.artifact = load_artifact(path)
                log.info("Model loaded: %s", app.state.artifact.meta.get("model"))
            else:
                app.state.client = AzureEndpointClient.from_env()
                log.info("Azure endpoint backend configured")
        except Exception as exc:  # keep the service up and report the failure via /health
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            log.error("Backend failed to start: %s", app.state.load_error)
        yield

    app = FastAPI(title="AML Detection API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Location and message only: never echo submitted values (privacy; NaN is not JSON).
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": detail}
        )

    def _ready(request: Request) -> bool:
        return request.app.state.artifact is not None or request.app.state.client is not None

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        if not _ready(request):
            body = HealthResponse(
                status="unavailable", model_loaded=False, detail=request.app.state.load_error
            )
            return JSONResponse(status_code=503, content=body.model_dump())
        name = (
            str(request.app.state.artifact.meta.get("model"))
            if request.app.state.artifact
            else "azure-endpoint"
        )
        return HealthResponse(status="ok", model_loaded=True, model=name)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(tx: TransactionRequest, request: Request) -> PredictionResponse:
        if not _ready(request):
            raise HTTPException(status_code=503, detail="Model is not loaded")
        record = tx.model_dump(mode="json")
        client = request.app.state.client
        if client is not None:
            try:
                out = client.predict([record])[0]
                response = PredictionResponse(
                    prediction=int(out["prediction"]),
                    probability=float(out["probability"]),
                    threshold=float(out["threshold"]),
                    model=str(out["model"]),
                )
            except UpstreamError as exc:
                log.error("Endpoint call failed: %s", exc)
                raise HTTPException(status_code=exc.status, detail=str(exc)) from None
            except (KeyError, TypeError, ValueError):
                raise HTTPException(
                    status_code=502, detail="Endpoint response is invalid"
                ) from None
            return response
        artifact = request.app.state.artifact
        try:
            proba, label = score(artifact, prepare_frame([record]))
        except Exception as exc:
            log.exception("Scoring failed")  # no transaction data is logged
            raise HTTPException(status_code=500, detail="Scoring failed") from exc
        return PredictionResponse(
            prediction=int(label[0]),
            probability=float(proba[0]),
            threshold=artifact.threshold,
            model=str(artifact.meta.get("model")),
        )

    return app


app = create_app()
