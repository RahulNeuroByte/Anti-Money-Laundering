"""Assemble the code snapshot uploaded with every pipeline component: build/azure/job_code.

job_code/run_step.py      step runner
job_code/aml/             the project package (copy of src/aml)
job_code/configs/         config.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build(out: Path | None = None) -> Path:
    out = out or ROOT / "build" / "azure" / "job_code"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(Path(__file__).with_name("run_step.py"), out / "run_step.py")
    shutil.copytree(ROOT / "src" / "aml", out / "aml", ignore=shutil.ignore_patterns("__pycache__"))
    (out / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "config.yaml", out / "configs" / "config.yaml")
    return out


if __name__ == "__main__":
    print(f"Built {build()}")
