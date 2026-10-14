# Azure ML deployment runbook

STATUS: prepared and tested locally; NOTHING HAS BEEN DEPLOYED. No Azure resource exists.

## What was verified (local, Linux sandbox, Python 3.11.9)
- `mlops/azure/build_package.py` builds `build/azure/{code,model}`; the built `code` folder runs on
  its own (subprocess test with no project on the path).
- `mlops/azure/score.py` passes unit tests (single, batch, bare payload, invalid payloads, size limit,
  not-JSON, run-before-init) and matches local inference to 1e-12 with the real model.
- Azure's own inference server (`azmlinfsrv`, azureml-inference-server-http 1.2.2) served the package
  locally: a valid request returned a prediction (HTTP 200); an invalid one returned HTTP 500.
- `endpoint.yml` and `deployment.yml` load with the Azure ML SDK (`azure-ai-ml` 1.35.1) offline.
- Environment pip dependencies resolve together on Python 3.11.

## NOT verified
`az` CLI syntax and PowerShell scripts (not run), Azure login/quota, base image
`mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04` with `python=3.11.9` from conda, image build,
deployment, key auth, latency, logging in Azure, cost, MLflow-to-Azure tracking.

## Findings from local testing
- Unpickling the model needs `pyarrow` (pandas 3 string columns). It is pinned in the environment
  file; without it the container fails at start.
- The inference server returns a generic HTTP 500 for any exception in `run()`. Validation details
  (row and field, never values) appear only in the logs. The FastAPI service returns 422 with details.

## Your Azure facts (from your portal screenshot, not independently checked)
Workspace `aml-mlops`, resource group `AML-mlops-capstone-project`, Central India, "Azure subscription 1",
provisioning state Succeeded. The deploy scripts default to these names and never create or delete the
workspace. Your compute `aml-mlops-compute` is NOT used by the endpoint: a managed online endpoint
provisions its own VM. That compute is for jobs (e.g. remote training); stop it if it is a running instance.

## Before you run anything
1. Quota: managed online endpoints need VM quota in Central India for the VM family in `deployment.yml`
   (Standard_DS2_v2, 2 vCPU). New or trial subscriptions often have zero quota for this: check
   Azure ML studio > Quota, and request an increase or pick another VM size if needed.
2. The deployment bills per hour while it exists, even with no traffic. Delete it right after testing.
3. The model is a pickle. Deploy only artifacts you built.

## Run (PowerShell, project root, venv active, `az login` done)
```powershell
az login
.\mlops\azure\deploy.ps1 -EndpointName aml-detector-XXXX            # read-only checks + plan
.\mlops\azure\deploy.ps1 -EndpointName aml-detector-XXXX -ApproveCost   # creates endpoint + deployment
.\mlops\azure\invoke.ps1 -EndpointName aml-detector-XXXX            # curl valid + invalid request
.\mlops\azure\teardown.ps1 -EndpointName aml-detector-XXXX          # stop charges
```
Replace XXXX so the name is unique in the region (3-32 chars, lowercase, digits, hyphens).
If deployment fails, run `az ml online-deployment get-logs --name blue --endpoint-name NAME` and paste the
output: it usually names the cause (quota, provider registration, image build, import error).
