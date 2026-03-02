// ============================================================
// main.bicep – subscription-scoped entry point
//
// Creates the resource group, then deploys all resources via
// the resources.bicep module (resource-group scoped).
//
// Deploy with:
//   az deployment sub create \
//     --location eastus \
//     --template-file main.bicep \
//     --parameters envName=text2sql-grpo
// ============================================================
targetScope = 'subscription'

@description('Short environment name – used to derive every resource name')
param envName string = 'text2sql-grpo'

@description('Azure region for all resources')
param location string = 'eastus'

@description('VM size for the GPU compute instance')
param computeVmSize string = 'Standard_NC16as_T4_v3'

// ── Resource Group ─────────────────────────────────────────
resource rg 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: '${envName}-rg'
  location: location
  tags: {
    environment: envName
    managedBy: 'bicep'
  }
}

// ── All resources (resource-group scoped module) ──────────
module resources 'resources.bicep' = {
  name: 'resources-${envName}'
  scope: rg
  params: {
    location: location
    envName: envName
    computeVmSize: computeVmSize
  }
}

// ── Outputs ────────────────────────────────────────────────
output resourceGroupName string = rg.name
output workspaceName string = resources.outputs.workspaceName
output storageAccountName string = resources.outputs.storageAccountName
output keyVaultName string = resources.outputs.keyVaultName
output containerRegistryName string = resources.outputs.containerRegistryName
output computeName string = resources.outputs.computeName
output studioUrl string = resources.outputs.studioUrl

