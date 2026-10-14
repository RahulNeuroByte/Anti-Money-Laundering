"""Azure ML SDK v2 helpers: data asset, environment, components, pipeline job, model registration.

Runs on the Azure ML 'Python 3.10 - SDK v2' kernel (or any Python >= 3.10 with azure-ai-ml and
azure-identity). It deliberately imports nothing from the aml package: the pipeline steps themselves run
in their own Python 3.11 job environment (job_conda.yml), independent of the kernel that submits them.
No `from __future__ import annotations` here: dsl.pipeline inspects the real annotations.
"""

import hashlib
import json
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, dsl, load_component
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import AmlCompute, Data, Environment, Model
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

ROOT = Path(__file__).resolve().parents[2]
AZURE_DIR = ROOT / "mlops" / "azure"
COMPONENT_ORDER = ["prepare", "features", "imbalance", "tune", "final"]
DATA_NAME = "saml-d-raw"
ENV_NAME = "aml-train-env"
MODEL_NAME = "aml-detector-scoring"
EXPERIMENT = "aml-detection-pipeline"
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"


# ------------------------------------------------------------------------------- connection
def get_credential():
    """DefaultAzureCredential, falling back to an interactive browser login."""
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        return credential
    except ClientAuthenticationError:
        return InteractiveBrowserCredential()


def get_client(subscription_id=None, resource_group=None, workspace=None):
    """MLClient from explicit ids, from AZURE_* environment variables, or from config.json."""
    subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = resource_group or os.environ.get("AZURE_RESOURCE_GROUP")
    workspace = workspace or os.environ.get("AZURE_ML_WORKSPACE")
    credential = get_credential()
    if subscription_id and resource_group and workspace:
        return MLClient(credential, subscription_id, resource_group, workspace)
    return MLClient.from_config(credential)


# ---------------------------------------------------------------------------------- compute
def check_compute(ml_client, name):
    """Return a summary of the compute target; raise if it does not exist."""
    compute = ml_client.compute.get(name)
    info = {
        "name": name,
        "type": getattr(compute, "type", None),
        "size": getattr(compute, "size", None),
        "state": getattr(compute, "provisioning_state", None),
    }
    if str(info["type"]).lower() != "amlcompute":
        info["warning"] = (
            "Pipeline jobs normally run on a compute cluster (amlcompute). If the job is rejected "
            "on this compute, create a cluster with --create-cluster."
        )
    return info


def create_cluster(ml_client, name, size="Standard_DS3_v2", max_instances=1):
    """Create a scale-to-zero compute cluster (billed only while nodes are running)."""
    cluster = AmlCompute(name=name, size=size, min_instances=0, max_instances=max_instances,
                         idle_time_before_scale_down=300)  # fmt: skip
    return ml_client.compute.begin_create_or_update(cluster).result()


# ------------------------------------------------------------------------------------ assets
def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def register_data(ml_client, csv_path):
    """Upload the raw CSV as a versioned uri_file data asset."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
    data = Data(
        name=DATA_NAME,
        path=str(csv_path),
        type=AssetTypes.URI_FILE,
        description="SAML-D synthetic AML transactions (raw CSV)",
        tags={"sha256": sha256_of(csv_path), "source": "SAML-D"},
    )
    return ml_client.data.create_or_update(data)


def register_environment(ml_client):
    env = Environment(
        name=ENV_NAME,
        image=BASE_IMAGE,
        conda_file=str(AZURE_DIR / "environment" / "job_conda.yml"),
        description="Python 3.11.9 with pinned scikit-learn stack and MLflow for Azure ML",
    )
    return ml_client.environments.create_or_update(env)


def _content_version():
    """Short hash of the component YAMLs and job code: same content gives the same version."""
    h = hashlib.sha256()
    files = sorted((AZURE_DIR / "components").glob("*.yml"))
    files += sorted(p for p in (ROOT / "build" / "azure" / "job_code").rglob("*") if p.is_file())
    files += [AZURE_DIR / "environment" / "job_conda.yml"]
    for f in files:
        if "__pycache__" not in f.parts:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def load_components():
    """Components loaded from YAML (offline; not yet registered)."""
    return {n: load_component(source=str(AZURE_DIR / "components" / f"{n}.yml"))
            for n in COMPONENT_ORDER}  # fmt: skip


def register_components(ml_client):
    version = _content_version()
    registered = {}
    for name, comp in load_components().items():
        comp.version = version
        registered[name] = ml_client.components.create_or_update(comp)
    return registered


# ----------------------------------------------------------------------------------- pipeline
def build_pipeline(components, compute, raw_data_path, neg_pool=200_000, cv_neg=120_000):
    """Connect the five components into one pipeline job (prepare > features > imbalance > tune > final)."""

    @dsl.pipeline(
        name="aml_detection_pipeline",
        display_name="AML detection pipeline",
        description="Profile, EDA, causal features, imbalance experiments, tuning, evaluation and packaging",
        compute=compute,
    )
    def aml_pipeline(raw_data: Input, neg_pool_size: int, cv_negatives: int):
        prep = components["prepare"](raw=raw_data)
        feat = components["features"](prepared=prep.outputs.prepared, neg_pool=neg_pool_size)
        imb = components["imbalance"](features=feat.outputs.features)
        tune = components["tune"](
            features=feat.outputs.features, imbalance=imb.outputs.imbalance, cv_neg=cv_negatives
        )
        fin = components["final"](
            features=feat.outputs.features,
            prepared=prep.outputs.prepared,
            tuning=tune.outputs.tuning,
        )
        return {
            "prepared": prep.outputs.prepared,
            "imbalance": imb.outputs.imbalance,
            "tuning": tune.outputs.tuning,
            "model": fin.outputs.model,
            "evaluation": fin.outputs.evaluation,
        }

    raw = Input(type=AssetTypes.URI_FILE, path=raw_data_path, mode=InputOutputModes.DOWNLOAD)
    return aml_pipeline(raw_data=raw, neg_pool_size=neg_pool, cv_negatives=cv_neg)


def submit(ml_client, pipeline_job, experiment=EXPERIMENT):
    return ml_client.jobs.create_or_update(pipeline_job, experiment_name=experiment)


def wait(ml_client, job):
    """Stream logs until the pipeline ends and return its final status."""
    ml_client.jobs.stream(job.name)
    return ml_client.jobs.get(job.name).status


# ----------------------------------------------------------------------- outputs and models
def download_outputs(ml_client, job_name, dest):
    """Download every step's outputs (reports, PNG figures, model files) to dest/<step>/."""
    dest = Path(dest)
    done = []
    for child in ml_client.jobs.list(parent_job_name=job_name):
        target = dest / (child.display_name or child.name).replace(" ", "_")
        ml_client.jobs.download(name=child.name, download_path=str(target), all=True)
        done.append(str(target))
    return done


def register_model(ml_client, job_name, work_dir="azure_outputs"):
    """Register the pipeline's `model` output as a versioned custom_model with metadata tags."""
    ml_client.jobs.download(name=job_name, output_name="model", download_path=str(work_dir))
    metas = sorted(Path(work_dir).rglob("model_meta.json"))
    if not metas:
        raise FileNotFoundError("model_meta.json not found in the downloaded model output")
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    model = Model(
        name=MODEL_NAME,
        path=f"azureml://jobs/{job_name}/outputs/model",
        type=AssetTypes.CUSTOM_MODEL,
        description="AML detection pipeline (feature builder, preprocessing, classifier) and threshold",
        tags={
            "stage": "Production",
            "algorithm": str(meta.get("model")),
            "imbalance_strategy": str(meta.get("strategy")),
            "threshold": str(meta.get("threshold")),
            "threshold_rule": str(meta.get("threshold_rule")),
            "pipeline_job": job_name,
            "dataset": "SAML-D",
        },
    )
    return ml_client.models.create_or_update(model)
