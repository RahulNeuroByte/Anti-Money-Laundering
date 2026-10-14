# Deletes the endpoint and its deployment (stops the VM charge). The workspace, its storage,
# key vault and your compute are NOT touched. NOT RUN OR SYNTAX-CHECKED IN POWERSHELL BY THE AUTHOR.
param(
  [string]$ResourceGroup = "AML-mlops-capstone-project",
  [string]$Workspace = "aml-mlops",
  [Parameter(Mandatory)] [string]$EndpointName
)
$ErrorActionPreference = "Stop"
az ml online-endpoint delete --name $EndpointName --resource-group $ResourceGroup --workspace-name $Workspace --yes
az ml online-endpoint list --resource-group $ResourceGroup --workspace-name $Workspace -o table
Write-Host "The registered model asset 'aml-detector-scoring' (small storage cost) is kept; delete it in the studio if not needed."
Write-Host "Confirm the list above no longer contains '$EndpointName'. Stop aml-mlops-compute too if it is a running compute instance."
