import json
import re
import sys
from pathlib import Path

import pytest
from mlops.azure import build_job_code

ROOT = Path(__file__).resolve().parents[1]
AZ = ROOT / "mlops" / "azure"


def _pins(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*(?:- )?([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def test_job_code_snapshot_is_self_contained(tmp_path):
    out = build_job_code.build(tmp_path / "job_code")
    assert (out / "run_step.py").is_file() and (out / "configs" / "config.yaml").is_file()
    assert (out / "aml" / "train.py").is_file() and (out / "aml" / "features.py").is_file()
    assert not list(out.rglob("__pycache__"))


def test_job_environment_pins_match_project_pins():
    job, proj = _pins(AZ / "environment" / "job_conda.yml"), _pins(ROOT / "requirements.txt")
    # mlflow-skinny differs on purpose: azureml-mlflow supports <=3.9 in the job environment
    shared = [p for p in job if p in proj and p != "mlflow-skinny"]
    assert {"pandas", "numpy", "scikit-learn", "imbalanced-learn", "pyarrow", "joblib"} <= set(
        shared
    )
    assert all(job[p] == proj[p] for p in shared), {p: (job[p], proj[p]) for p in shared}
    assert "python=3.11.9" in (AZ / "environment" / "job_conda.yml").read_text()


@pytest.mark.parametrize("step", ["prepare", "features", "imbalance", "tune", "final"])
def test_component_yaml_points_at_the_step_runner(step):
    import yaml

    spec = yaml.safe_load((AZ / "components" / f"{step}.yml").read_text())
    assert spec["name"] == f"aml_{step}" and spec["type"] == "command"
    assert f"run_step.py {step}" in spec["command"]
    assert spec["environment"].startswith("azureml:aml-train-env")
    for name in list(spec["inputs"]) + list(spec["outputs"]):
        assert (
            "${{" + f"inputs.{name}}}" in spec["command"]
            or "${{" + f"outputs.{name}}}" in spec["command"]
        )


def test_run_step_rejects_unknown_step():
    sys.path.insert(0, str(AZ))
    import run_step

    with pytest.raises(SystemExit):
        run_step.main(["nonsense"])


def test_notebook_is_valid_and_dry_run_cells_execute(tmp_path, monkeypatch):
    pytest.importorskip("azure.ai.ml")
    nb = json.loads((ROOT / "notebooks" / "azureml_sdk2_pipeline.ipynb").read_text())
    assert nb["nbformat"] == 4
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = src.replace("DRY_RUN = False", "DRY_RUN = True")
    monkeypatch.chdir(ROOT)
    namespace: dict = {}
    exec(compile(src, "notebook", "exec"), namespace)  # noqa: S102
    assert list(namespace["pipeline_job"].jobs) == ["prep", "feat", "imb", "tune", "fin"]


def test_pipeline_dry_run_builds_offline():
    pytest.importorskip("azure.ai.ml")
    from mlops.azure import pipeline_lib

    comps = pipeline_lib.load_components()
    job = pipeline_lib.build_pipeline(comps, "aml-mlops-compute", "azureml:saml-d-raw@latest")
    assert list(job.jobs) == ["prep", "feat", "imb", "tune", "fin"]
    assert set(job.outputs) == {"prepared", "imbalance", "tuning", "model", "evaluation"}
