# Deploys the AML scoring endpoint into the EXISTING Azure ML workspace.
# NOT RUN OR SYNTAX-CHECKED IN POWERSHELL BY THE AUTHOR: run step by step and read each result.
# Without -ApproveCost it only runs read-only checks and prints the plan.
param(
  [string]$SubscriptionId,
  [string]$ResourceGroup = "AML-mlops-capstone-project",
  [string]$Workspace = "aml-mlops",
  [Parameter(Mandatory)] [string]$EndpointName,
  [string]$ModelVersion = "1",     # version of model asset aml-detector-scoring to deploy
  [switch]$SkipModelCreate,        # use a version already registered (e.g. by the SDK pipeline)
  [switch]$ApproveCost
)
$ErrorActionPreference = "Stop"
function Run { & $args[0] @($args | Select-Object -Skip 1); if ($LASTEXITCODE -ne 0) { throw "Command failed: $($args -join ' ')" } }

if ($EndpointName -notmatch '^[a-z][a-z0-9-]{1,30}[a-z0-9]$') {
  throw "Endpoint name must be 3-32 chars: lowercase letters, digits, hyphens; start with a letter. It must be unique in the region."
}

# --- read-only checks (no cost) ---
Run az extension add --name ml --yes
if ($SubscriptionId) { Run az account set --subscription $SubscriptionId }
Run az account show --query "{subscription:name, id:id, user:user.name}" -o table
Run az ml workspace show --name $Workspace --resource-group $ResourceGroup --query "{name:name, location:location, state:provisioning_state}" -o table
Run python mlops/azure/build_package.py

Write-Host ""
Write-Host "PLAN: model aml-detector-scoring version $ModelVersion"
Write-Host "PLAN: endpoint '$EndpointName' (key auth) with deployment 'blue' on ONE Standard_DS2_v2 VM in workspace '$Workspace'."
Write-Host "      The VM is billed per hour while the deployment exists, even with no traffic."
Write-Host "      The first deployment builds an environment image: expect 10-25 minutes."
if (-not $ApproveCost) { Write-Host "DRY RUN: nothing created. Re-run with -ApproveCost to deploy."; exit 0 }

# --- billable steps ---
if (-not $SkipModelCreate) {
  Run az ml model create --file mlops/azure/model.yml --resource-group $ResourceGroup --workspace-name $Workspace
}
# Deployment YAML with the chosen model version (written without a BOM)
$yaml = (Get-Content mlops/azure/deployment.yml -Raw) -replace 'azureml:aml-detector-scoring:\d+', "azureml:aml-detector-scoring:$ModelVersion"
$generated = Join-Path (Get-Location) "mlops/azure/deployment.generated.yml"
[System.IO.File]::WriteAllText($generated, $yaml, (New-Object System.Text.UTF8Encoding $false))
Run az ml online-endpoint create --name $EndpointName --file mlops/azure/endpoint.yml --resource-group $ResourceGroup --workspace-name $Workspace
Run az ml online-deployment create --name blue --endpoint-name $EndpointName --file mlops/azure/deployment.generated.yml --all-traffic --resource-group $ResourceGroup --workspace-name $Workspace
Run az ml online-deployment get-logs --name blue --endpoint-name $EndpointName --lines 60 --resource-group $ResourceGroup --workspace-name $Workspace
Write-Host "Deployed. Test with mlops/azure/invoke.ps1, and run teardown.ps1 when finished."
