@description('Short lowercase prefix used for globally unique Azure resource names.')
param namePrefix string

@description('Azure region for the deployment.')
param location string = resourceGroup().location

@allowed([
  'F1'
  'B1'
])
@description('App Service plan SKU. F1 is suitable for a demo; B1 avoids free-tier sleeping and quotas.')
param appServiceSku string = 'F1'

var suffix = uniqueString(subscription().subscriptionId, resourceGroup().id, namePrefix)
var storageName = take(toLower(replace('${namePrefix}${suffix}', '-', '')), 24)
var appName = take(toLower('${namePrefix}-${suffix}'), 60)
var planName = '${namePrefix}-plan'
var insightsName = '${namePrefix}-insights'
var workspaceName = '${namePrefix}-logs'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource forecasts 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'forecasts'
  properties: { publicAccess: 'None' }
}

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: workspaceName
  location: location
  properties: { retentionInDays: 30 }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: insightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: workspace.id
  }
}

resource plan 'Microsoft.Web/serverfarms@2023-12-01' = {
  name: planName
  location: location
  sku: {
    name: appServiceSku
    tier: appServiceSku == 'F1' ? 'Free' : 'Basic'
  }
  kind: 'linux'
  properties: { reserved: true }
}

resource app 'Microsoft.Web/sites@2023-12-01' = {
  name: appName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    serverFarmId: plan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.12'
      appCommandLine: 'python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.headless true'
      alwaysOn: appServiceSku != 'F1'
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      appSettings: [
        { name: 'SCM_DO_BUILD_DURING_DEPLOYMENT', value: 'true' }
        { name: 'STORAGE_ACCOUNT_URL', value: 'https://${storage.name}.blob.${environment().suffixes.storage}' }
        { name: 'FORECAST_CONTAINER', value: forecasts.name }
        { name: 'FORECAST_BLOB', value: 'latest.json' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: insights.properties.ConnectionString }
        { name: 'ApplicationInsightsAgent_EXTENSION_VERSION', value: '~3' }
      ]
    }
  }
}

var blobReaderRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1')

resource appBlobReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, app.id, blobReaderRoleId)
  scope: storage
  properties: {
    roleDefinitionId: blobReaderRoleId
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output webAppName string = app.name
output webAppUrl string = 'https://${app.properties.defaultHostName}'
output storageAccountName string = storage.name
output storageAccountId string = storage.id
output webAppId string = app.id
output forecastContainer string = forecasts.name
