"""Call the deployed Azure ML endpoint from Python and compare it with local predictions.

    $env:PYTHONPATH="src"
    python mlops/azure/call_endpoint.py --url <scoring_uri> --key <primary key> [--compare-local 20]

URL and key can also come from AZURE_ENDPOINT_URL and AZURE_ENDPOINT_KEY. The key is never printed.
Exit code is 1 if cloud and local predictions disagree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.azure_client import AzureEndpointClient, UpstreamError  # noqa: E402

from aml.config import load_config  # noqa: E402
from aml.data import load_dataset  # noqa: E402
from aml.features import REQUIRED_INPUT  # noqa: E402
from aml.inference import load_artifact, prepare_frame, score  # noqa: E402

PROB_TOLERANCE = 1e-6


def _records(frame) -> list[dict]:
    out = frame[REQUIRED_INPUT].copy()
    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    for col in out.columns:
        if str(out[col].dtype) == "category":
            out[col] = out[col].astype(str)
    out["Amount"] = out["Amount"].astype("float64")
    return json.loads(out.to_json(orient="records"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("AZURE_ENDPOINT_URL"))
    ap.add_argument("--key", default=os.environ.get("AZURE_ENDPOINT_KEY"))
    ap.add_argument("--sample-file", default=str(ROOT / "mlops" / "azure" / "sample_request.json"))
    ap.add_argument("--compare-local", type=int, default=0, metavar="N")
    args = ap.parse_args()
    if not args.url or not args.key:
        print("Provide --url and --key (or AZURE_ENDPOINT_URL / AZURE_ENDPOINT_KEY).")
        return 2
    client = AzureEndpointClient(args.url, args.key, timeout=60.0)
    try:
        sample = json.loads(Path(args.sample_file).read_text(encoding="utf-8"))["data"]
        print("Sample request ->", json.dumps(client.predict(sample), indent=1))
        if args.compare_local <= 0:
            return 0
        cfg, split = load_config()["data"], load_config()["split"]
        df = load_dataset()
        start = int(len(df) * (split["train_frac"] + split["valid_frac"]))
        test = df.iloc[start:]
        pos = test[test[cfg["target"]] == 1]
        neg = test[test[cfg["target"]] == 0]
        half = max(args.compare_local // 2, 1)
        rows = test.loc[list(pos.index[:half]) + list(neg.index[: args.compare_local - half])]
        records = _records(rows)
        cloud = client.predict(records)
    except UpstreamError as exc:
        print(f"Endpoint call failed: {exc}")
        return 1
    artifact = load_artifact(ROOT / "models" / "aml_pipeline.pkl")
    proba, label = score(artifact, prepare_frame(records))
    cloud_p = np.array([c["probability"] for c in cloud])
    cloud_l = np.array([c["prediction"] for c in cloud])
    diff = float(np.max(np.abs(cloud_p - proba)))
    mismatches = int(np.sum(cloud_l != label))
    print(
        f"Compared {len(records)} test rows: max |cloud - local| probability = {diff:.2e}, "
        f"label mismatches = {mismatches}"
    )
    return 0 if diff <= PROB_TOLERANCE and mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
