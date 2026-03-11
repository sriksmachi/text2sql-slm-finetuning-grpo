#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Build and push a prebuilt Docker image for the text2sql AML environment.

.DESCRIPTION
    Builds the Docker image from azure/environments/Dockerfile, pushes it to
    the workspace container registry, and optionally registers the AML
    environment from that pushed image.

.PARAMETER ResourceGroup
    Azure resource group that contains the AML workspace.
    Defaults to: txt2sql-grpo-rg

.PARAMETER Workspace
    Azure ML workspace name.
    Defaults to: txt2sql-grpo-ws

.PARAMETER ImageName
    Repository name in ACR.
    Defaults to: text2sql-grpo

.PARAMETER ImageTag
    Image tag to build and push.
    Defaults to: env-v1

.PARAMETER BuildMode
    Where to build the image:
      acr   - build remotely in Azure Container Registry
      local - build locally with docker and then push
    Defaults to: acr

.PARAMETER RegisterEnvironment
    If set, register text2sql-grpo-env in AML from the pushed image.

.EXAMPLE
    .\build_image.ps1 -RegisterEnvironment

    # Build remotely in ACR (default)
    .\build_image.ps1 -BuildMode acr -RegisterEnvironment
#>

[CmdletBinding()]
param (
    [string] $ResourceGroup = "txt2sql-grpo-rg",
    [string] $Workspace = "txt2sql-grpo-ws",
    [string] $ImageName = "text2sql-grpo",
    [string] $ImageTag = "env-v1",
    [ValidateSet("acr", "local")]
    [string] $BuildMode = "acr",
    [switch] $RegisterEnvironment
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$DockerContext = Join-Path $ScriptDir "environments"
$DockerfilePath = Join-Path $DockerContext "Dockerfile"
$CreateEnvPs1 = Join-Path $ScriptDir "create_env.ps1"

if (-not (Test-Path $DockerfilePath)) {
    Write-Error "Dockerfile not found: $DockerfilePath"
    exit 1
}

az account show --output none
if ($LASTEXITCODE -ne 0) {
    Write-Error "Not logged in. Run: az login"
    exit 1
}

$containerRegistryResourceId = az ml workspace show --name $Workspace --resource-group $ResourceGroup --only-show-errors --query "container_registry" -o tsv
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($containerRegistryResourceId)) {
    Write-Error "Could not resolve the workspace container registry."
    exit 1
}

$acrName = Split-Path $containerRegistryResourceId -Leaf
$loginServer = az acr show --name $acrName --resource-group $ResourceGroup --only-show-errors --query "loginServer" -o tsv
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($loginServer)) {
    Write-Error "Could not resolve ACR login server for '$acrName'."
    exit 1
}

$imageUri = "$loginServer/${ImageName}:$ImageTag"

if ($BuildMode -eq "acr") {
    Write-Host "`n=== Building prebuilt runtime image in ACR ===" -ForegroundColor Magenta
    Write-Host "► az acr build --registry $acrName --image ${ImageName}:$ImageTag $DockerContext" -ForegroundColor Cyan
    az acr build --registry $acrName --image "${ImageName}:$ImageTag" $DockerContext --file $DockerfilePath --only-show-errors
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Remote ACR build failed."
        exit 1
    }
}
else {
    Write-Host "`n=== Building prebuilt runtime image locally ===" -ForegroundColor Magenta
    Write-Host "► docker build -f $DockerfilePath -t $imageUri $DockerContext" -ForegroundColor Cyan
    docker build -f $DockerfilePath -t $imageUri $DockerContext
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed."
        exit 1
    }

    Write-Host "`n=== Logging into ACR ===" -ForegroundColor Magenta
    Write-Host "► az acr login --name $acrName" -ForegroundColor Cyan
    az acr login --name $acrName --only-show-errors
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ACR login failed."
        exit 1
    }

    Write-Host "`n=== Pushing image ===" -ForegroundColor Magenta
    Write-Host "► docker push $imageUri" -ForegroundColor Cyan
    docker push $imageUri
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker push failed."
        exit 1
    }
}

Write-Host "`n✅ Image pushed: $imageUri" -ForegroundColor Green

if ($RegisterEnvironment) {
    if (-not (Test-Path $CreateEnvPs1)) {
        Write-Error "Environment registration script not found: $CreateEnvPs1"
        exit 1
    }

    Write-Host "`n=== Registering AML environment from prebuilt image ===" -ForegroundColor Magenta
    & $CreateEnvPs1 -ResourceGroup $ResourceGroup -Workspace $Workspace -Image $imageUri -ImageName $ImageName -ImageTag $ImageTag
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Environment registration from prebuilt image failed."
        exit 1
    }
}
