"""Assemble the deployment folders under build/azure (run before any `az ml` command).

code/   score.py + aml/ + api/schemas.py   (scoring code)
model/  aml_pipeline.pkl + model_meta.json (the Phase 4 artifact)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build(model_dir: Path | None = None, out: Path | None = None) -> Path:
    out = out or ROOT / "build" / "azure"
    model_dir = model_dir or ROOT / "models"
    for needed in ("aml_pipeline.pkl", "model_meta.json"):
        if not (model_dir / needed).is_file():
            raise FileNotFoundError(f"{model_dir / needed} is missing; train the model first")
    if out.exists():
        shutil.rmtree(out)
    code, model = out / "code", out / "model"
    code.mkdir(parents=True)
    model.mkdir()
    shutil.copy(Path(__file__).with_name("score.py"), code / "score.py")
    shutil.copytree(
        ROOT / "src" / "aml", code / "aml", ignore=shutil.ignore_patterns("__pycache__")
    )
    (code / "api").mkdir()
    (code / "api" / "__init__.py").write_text("")
    shutil.copy(ROOT / "api" / "schemas.py", code / "api" / "schemas.py")
    for needed in ("aml_pipeline.pkl", "model_meta.json"):
        shutil.copy(model_dir / needed, model / needed)
    return out


if __name__ == "__main__":
    print(f"Built {build()}", file=sys.stderr)
