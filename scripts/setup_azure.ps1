param(
    [Parameter(Mandatory = $true)] [string] $SubscriptionId,
    [Parameter(Mandatory = $true)] [string] $GitHubRepository,
    [string] $ResourceGroup = "prem-league-predictor-rg",
    [string] $NamePrefix = "plpredictor",
    [string] $Location = "australiaeast",
    [ValidateSet("F1", "B1")] [string] $AppServiceSku = "F1"
)

$ErrorActionPreference = "Stop"

az account set --subscription $SubscriptionId
az group create --name $ResourceGroup --location $Location --output none

$deployment = az deployment group create `
    --resource-group $ResourceGroup `
    --template-file infra/main.bicep `
    --parameters namePrefix=$NamePrefix appServiceSku=$AppServiceSku `
    --query properties.outputs `
    --output json | ConvertFrom-Json

$appDisplayName = "$NamePrefix-github-actions"
$clientId = az ad app list --display-name $appDisplayName --query "[0].appId" --output tsv
if (-not $clientId) {
    $clientId = az ad app create --display-name $appDisplayName --query appId --output tsv
}

$servicePrincipalId = az ad sp show --id $clientId --query id --output tsv 2>$null
if (-not $servicePrincipalId) {
    $servicePrincipalId = az ad sp create --id $clientId --query id --output tsv
}

$credentialName = "github-main"
$existingCredential = az ad app federated-credential list --id $clientId --query "[?name=='$credentialName'].name | [0]" --output tsv
if (-not $existingCredential) {
    $credential = @{
        name = $credentialName
        issuer = "https://token.actions.githubusercontent.com"
        subject = "repo:${GitHubRepository}:ref:refs/heads/main"
        audiences = @("api://AzureADTokenExchange")
    } | ConvertTo-Json -Compress
    az ad app federated-credential create --id $clientId --parameters $credential --output none
}

az role assignment create --assignee-object-id $servicePrincipalId --assignee-principal-type ServicePrincipal `
    --role "Website Contributor" --scope $deployment.webAppId.value --output none
az role assignment create --assignee-object-id $servicePrincipalId --assignee-principal-type ServicePrincipal `
    --role "Storage Blob Data Contributor" --scope $deployment.storageAccountId.value --output none

$tenantId = az account show --query tenantId --output tsv

Write-Host "Azure resources are ready. Add these GitHub Actions secrets:"
Write-Host "AZURE_CLIENT_ID=$clientId"
Write-Host "AZURE_TENANT_ID=$tenantId"
Write-Host "AZURE_SUBSCRIPTION_ID=$SubscriptionId"
Write-Host ""
Write-Host "Add these GitHub Actions variables:"
Write-Host "AZURE_WEBAPP_NAME=$($deployment.webAppName.value)"
Write-Host "AZURE_STORAGE_ACCOUNT=$($deployment.storageAccountName.value)"
Write-Host ""
Write-Host "Public dashboard: $($deployment.webAppUrl.value)"
