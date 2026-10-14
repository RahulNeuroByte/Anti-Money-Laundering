# End-to-End Anti-Money Laundering (AML) Detection

Machine learning, FastAPI, GitHub Actions, Azure Machine Learning and MLflow on the SAML-D
transaction dataset. Track chosen: **Anti-Money Laundering detection** (target `Is_laundering`).

## What is in the project
| Part | Where | What it does |
|---|---|---|
| Data profile, cleaning, EDA | `src/aml/profile_data.py`, `cleaning.py`, `eda.py` | Shape, dtypes, missing values, imbalance, outliers, leakage audit, 8 EDA figures |
| Features and split | `src/aml/features.py`, `splitting.py`, `train.py features` | Causal account-profile features (28-day window, 7-day blocks), chronological 70/15/15 split |
| Models | `src/aml/train.py` | Logistic Regression, Decision Tree, Random Forest; 5 imbalance strategies; tuning; evaluation; `.pkl` |
| API | `api/` | FastAPI `GET /health`, `POST /predict`; local or Azure-endpoint backend |
| CI | `.github/workflows/ci.yml` | Install, `pip check`, `ruff`, `pytest` on Python 3.11.9 |
| MLflow | `mlops/track_models.py` | Logs 3 runs, compares, registers the best as `aml-detector` |
| Azure ML pipeline | `mlops/azure/submit_pipeline.py`, `pipeline_lib.py`, `components/`, `run_step.py`, `notebooks/azureml_sdk2_pipeline.ipynb` | SDK v2 (Python 3.10 kernel): data asset, environment, 5 components, pipeline job, model registration; steps log metrics, PNG charts and artifacts with MLflow |
| Azure endpoint | `mlops/azure/` | Scoring script, YAML, deploy / invoke / teardown scripts, Python client |
| Report | `src/aml/report.py` | Builds `reports/AML_Detection_Capstone_Report.docx` |

Documents: `PROJECT_EXECUTION_GUIDE.md` (ordered commands and troubleshooting), `INSTRUCTION.md`
(VS Code, Git, CI, Azure ML, MLflow), `PROJECT_STATE.md` (execution record).

## Results (reference run, test window, threshold chosen on validation)
| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.216 | 0.921 | 0.302 | 0.236 | 0.265 |
| Decision Tree | 0.617 | 0.956 | 0.770 | 0.714 | 0.741 |
| **Random Forest (selected)** | **0.797** | **0.983** | **0.847** | **0.700** | **0.767** |

The selected model flags 1,402 of 1,425,728 test transactions (1,188 true, 214 false alarms, 509 missed).
Selection used validation PR-AUC; accuracy (about 99.95%) is reported but is not a selection criterion
because predicting "normal" for everything already scores above 99.8%. Exact values are in
`reports/model_results.json`; your machine may differ slightly if library versions differ.
The data is synthetic, so these numbers do not show real-world AML performance.

## Design choices that matter
- `Laundering_type` determines the target, so it is **never a feature** (leakage).
- Account profiles are **causal**: a transaction only sees the previous 28 days of activity.
  Whole-window profiles looked stronger on training than on validation (see `reports/train_vs_valid_auc.json`).
- The test window is scored once, with thresholds fixed on validation.
- The deployed artifact scores a single transaction with a **frozen** profile table (ends 2023-07-05);
  this costs about 0.03 PR-AUC versus refreshed profiles (`reports/artifact_validation.json`).

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q                                   # needs no dataset
uvicorn api.main:app --port 8000            # uses models/aml_pipeline.pkl, then open http://127.0.0.1:8000/docs
```
Full pipeline from the raw CSV: see `PROJECT_EXECUTION_GUIDE.md`. The dataset is not in the repository:
place `SAML-D.csv` at `data\raw\SAML-D.csv`.

## Demo request
```powershell
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@api/sample_transaction.json"
```
Response fields: `prediction` (1 = flagged), `probability`, `threshold`, `model`.

## Verification status
| Item | Status |
|---|---|
| Pipeline, tests, API, MLflow (local), Azure scoring package with Azure's own inference server | Run and passing in a Linux sandbox (Python 3.11.9) |
| Dependency wheels for Windows, Python 3.11 | Resolution checked (binary wheels exist); not installed on Windows by the author |
| GitHub Actions run | Workflow written; first run not observed |
| Azure pipeline step scripts (same Python 3.11 libraries as the job environment, MLflow local store) | Run end to end on the real data; metrics identical to the reference run |
| Azure SDK pipeline objects (Python 3.10 + azure-ai-ml 1.35.1) | Built offline (dry run) and tested; notebook dry-run cells execute |
| Azure ML pipeline job, data upload, model registration, endpoint deployment, MLflow logging to Azure, Application Insights | **Not run by the author.** Scripts are provided; you run them and record the outcome |
