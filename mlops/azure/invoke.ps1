# Calls the deployed endpoint with curl. NOT RUN OR SYNTAX-CHECKED IN POWERSHELL BY THE AUTHOR.
param(
  [string]$ResourceGroup = "AML-mlops-capstone-project",
  [string]$Workspace = "aml-mlops",
  [Parameter(Mandatory)] [string]$EndpointName,
  [string]$RequestFile = "mlops/azure/sample_request.json"
)
$ErrorActionPreference = "Stop"
$scoring = az ml online-endpoint show --name $EndpointName --resource-group $ResourceGroup --workspace-name $Workspace --query scoring_uri -o tsv
$key = az ml online-endpoint get-credentials --name $EndpointName --resource-group $ResourceGroup --workspace-name $Workspace --query primaryKey -o tsv
Write-Host "Scoring URI: $scoring"
curl.exe -s -w "`nHTTP %{http_code}`n" -X POST $scoring -H "Authorization: Bearer $key" -H "Content-Type: application/json" -d "@$RequestFile"
# Invalid request (expect HTTP 500 with a generic message; details are in the deployment logs):
curl.exe -s -w "`nHTTP %{http_code}`n" -X POST $scoring -H "Authorization: Bearer $key" -H "Content-Type: application/json" -d "@mlops/azure/invalid_request.json"

# Python client with a cloud-vs-local comparison on 20 real test rows (needs data/raw/SAML-D.csv):
$env:PYTHONPATH = "src"
python mlops/azure/call_endpoint.py --url $scoring --key $key --compare-local 20
