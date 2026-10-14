"""Create the data asset, environment, components and pipeline job in Azure ML, run it, register the model.

Python >= 3.10 with azure-ai-ml and azure-identity (the Azure ML 'Python 3.10 - SDK v2' kernel works).

  python mlops/azure/submit_pipeline.py --dry-run
  python mlops/azure/submit_pipeline.py --wait --register --download azure_outputs

Workspace: --subscription-id/--resource-group/--workspace, or AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP /
AZURE_ML_WORKSPACE, or a config.json in the working folder (Azure ML studio > Download config.json).
Running the pipeline consumes compute-cluster time (billable). --dry-run touches nothing in Azure.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mlops.azure import build_job_code, pipeline_lib  # noqa: E402


def dry_run(args):
    job_code = build_job_code.build()
    comps = pipeline_lib.load_components()
    job = pipeline_lib.build_pipeline(
        comps, args.compute, f"azureml:{pipeline_lib.DATA_NAME}@latest", args.neg_pool, args.cv_neg
    )
    summary = {
        "job_code": str(job_code),
        "components": {n: c.name for n, c in comps.items()},
        "steps": list(job.jobs),
        "compute": args.compute,
        "outputs": list(job.outputs),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subscription-id")
    ap.add_argument("--resource-group")
    ap.add_argument("--workspace")
    ap.add_argument("--compute", default="aml-mlops-compute")
    ap.add_argument(
        "--create-cluster", metavar="NAME", help="create a scale-to-zero cluster and use it"
    )
    ap.add_argument("--raw-csv", default=str(ROOT / "data" / "raw" / "SAML-D.csv"))
    ap.add_argument(
        "--skip-data-upload", action="store_true", help="use the latest saml-d-raw asset"
    )
    ap.add_argument("--neg-pool", type=int, default=200_000)
    ap.add_argument("--cv-neg", type=int, default=120_000)
    ap.add_argument("--wait", action="store_true", help="stream logs until the pipeline finishes")
    ap.add_argument("--register", action="store_true", help="register the model after success")
    ap.add_argument(
        "--download", metavar="DIR", help="download all step outputs (PNG, JSON, model)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="build everything offline; no Azure calls"
    )
    args = ap.parse_args(argv)

    if args.dry_run:
        return dry_run(args)

    client = pipeline_lib.get_client(args.subscription_id, args.resource_group, args.workspace)
    compute = args.create_cluster or args.compute
    if args.create_cluster:
        print(
            "Creating cluster",
            compute,
            pipeline_lib.create_cluster(client, compute).provisioning_state,
        )
    print("Compute:", json.dumps(pipeline_lib.check_compute(client, compute), indent=1))

    if args.skip_data_upload:
        data = client.data.get(pipeline_lib.DATA_NAME, label="latest")
    else:
        print("Uploading raw CSV as a data asset (about 1 GB)...")
        data = pipeline_lib.register_data(client, args.raw_csv)
    print(f"Data asset: {data.name}:{data.version}")

    build_job_code.build()
    env = pipeline_lib.register_environment(client)
    print(f"Environment: {env.name}:{env.version}")
    comps = pipeline_lib.register_components(client)
    print("Components:", {n: f"{c.name}:{c.version}" for n, c in comps.items()})

    job = pipeline_lib.build_pipeline(
        comps, compute, f"azureml:{data.name}:{data.version}", args.neg_pool, args.cv_neg
    )
    job = pipeline_lib.submit(client, job)
    print("Submitted pipeline job:", job.name)
    print("Studio URL:", getattr(job, "studio_url", "(see Azure ML studio > Jobs)"))
    if not (args.wait or args.register or args.download):
        return 0

    status = pipeline_lib.wait(client, job)
    print("Final status:", status)
    if status != "Completed":
        return 1
    if args.register:
        model = pipeline_lib.register_model(client, job.name, args.download or "azure_outputs")
        print(f"Registered model: {model.name}:{model.version}")
    if args.download:
        print(
            "Downloaded:",
            json.dumps(pipeline_lib.download_outputs(client, job.name, args.download), indent=1),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
