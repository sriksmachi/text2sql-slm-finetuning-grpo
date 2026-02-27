// ============================================================
// main.bicep – Azure ML workspace, compute, endpoints, private link
// ============================================================

@description('Location for all resources')
param location string = resourceGroup().location

@description('Base name used to derive resource names')
param baseName string = 'text2sql'

@description('Environment tag (dev, staging, prod)')
param environment string = 'dev'

@description('SKU for the GPU compute cluster')
param computeVmSize string = 'Standard_NC16as_T4_v3'

@description('Maximum number of compute nodes')
param computeMaxNodes int = 4

@description('Object ID of the principal that should be Owner of the AML workspace')
param ownerObjectId string

// ── Derived names ──────────────────────────────────────────
var suffix = '${baseName}-${environment}'
var storageAccountName = replace('sa${suffix}', '-', '')
var keyVaultName = 'kv-${suffix}'
var appInsightsName = 'ai-${suffix}'
var logAnalyticsName = 'la-${suffix}'
var containerRegistryName = replace('cr${suffix}', '-', '')
var workspaceName = 'aml-${suffix}'
var computeClusterName = 'gpu-cluster-${environment}'
var onlineEndpointName = 'ep-text2sql-${environment}'
var vnetName = 'vnet-${suffix}'
var subnetName = 'snet-aml'

// ── Virtual Network ────────────────────────────────────────
resource vnet 'Microsoft.Network/virtualNetworks@2023-09-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: ['10.0.0.0/16']
    }
    subnets: [
      {
        name: subnetName
        properties: {
          addressPrefix: '10.0.0.0/24'
          privateEndpointNetworkPolicies: 'Disabled'
        }
      }
    ]
  }
}

// ── Log Analytics ──────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Application Insights ───────────────────────────────────
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ── Key Vault ──────────────────────────────────────────────
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
  }
}

// ── Storage Account ────────────────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
  }
}

// ── Container Registry ─────────────────────────────────────
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  sku: { name: 'Premium' }
  properties: {
    adminUserEnabled: false
    networkRuleSet: {
      defaultAction: 'Deny'
    }
  }
}

// ── Azure ML Workspace ─────────────────────────────────────
resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-01-01-preview' = {
  name: workspaceName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: containerRegistry.id
    publicNetworkAccess: 'Disabled'
  }
}

// ── GPU Compute Cluster ────────────────────────────────────
resource computeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-01-01-preview' = {
  parent: amlWorkspace
  name: computeClusterName
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: computeVmSize
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: computeMaxNodes
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      subnet: {
        id: '${vnet.id}/subnets/${subnetName}'
      }
    }
  }
}

// ── Online Endpoint ────────────────────────────────────────
resource onlineEndpoint 'Microsoft.MachineLearningServices/workspaces/onlineEndpoints@2024-01-01-preview' = {
  parent: amlWorkspace
  name: onlineEndpointName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    authMode: 'Key'
    publicNetworkAccess: 'Disabled'
  }
}

// ── Private Endpoints ──────────────────────────────────────
resource peAml 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: 'pe-aml-${environment}'
  location: location
  properties: {
    subnet: {
      id: '${vnet.id}/subnets/${subnetName}'
    }
    privateLinkServiceConnections: [
      {
        name: 'plsc-aml'
        properties: {
          privateLinkServiceId: amlWorkspace.id
          groupIds: ['amlworkspace']
        }
      }
    ]
  }
}

// ── Role assignments ───────────────────────────────────────
var contributorRoleId = 'b24988ac-6180-42a0-ab88-20f7382dd24c'
resource ownerRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(amlWorkspace.id, ownerObjectId, contributorRoleId)
  scope: amlWorkspace
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRoleId)
    principalId: ownerObjectId
  }
}

// ── Outputs ────────────────────────────────────────────────
output workspaceName string = amlWorkspace.name
output workspaceId string = amlWorkspace.id
output computeClusterName string = computeCluster.name
output onlineEndpointName string = onlineEndpoint.name
output storageAccountName string = storageAccount.name
