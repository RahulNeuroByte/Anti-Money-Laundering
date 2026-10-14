"""Minimal client for the Azure ML managed online endpoint (key authentication)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class UpstreamError(Exception):
    """The remote endpoint failed. `status` is the HTTP status the API should return."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class AzureEndpointClient:
    def __init__(self, url: str, key: str, timeout: float = 15.0):
        if not url.startswith(("https://", "http://")):
            raise ValueError("AZURE_ENDPOINT_URL must start with https:// (http:// only for tests)")
        if not key:
            raise ValueError("AZURE_ENDPOINT_KEY is empty")
        self._url, self._key, self._timeout = url, key, timeout

    @classmethod
    def from_env(cls) -> AzureEndpointClient:
        url = os.environ.get("AZURE_ENDPOINT_URL", "")
        key = os.environ.get("AZURE_ENDPOINT_KEY", "")
        if not url or not key:
            raise ValueError("AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY must both be set")
        return cls(url, key, float(os.environ.get("AZURE_ENDPOINT_TIMEOUT", "15")))

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        body = json.dumps({"data": records}).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                raw = response.read()
        except urllib.error.HTTPError as exc:
            exc.close()
            raise UpstreamError(f"Endpoint returned HTTP {exc.code}", 502) from None
        except (urllib.error.URLError, TimeoutError):
            raise UpstreamError("Endpoint unreachable or timed out", 504) from None
        try:
            payload = json.loads(raw)
            if isinstance(payload, str):  # some deployments double-encode the response
                payload = json.loads(payload)
        except json.JSONDecodeError:
            raise UpstreamError("Endpoint returned a non-JSON response", 502) from None
        if not isinstance(payload, list) or len(payload) != len(records):
            raise UpstreamError("Endpoint returned an unexpected response shape", 502)
        return payload
