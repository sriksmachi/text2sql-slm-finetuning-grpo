#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Register the Azure ML environment from the image pushed by build_image.ps1.

.PARAMETER ResourceGroup
    Azure resource group that contains the AML workspace.

.PARAMETER Workspace
    Azure ML workspace name.

.PARAMETER EnvironmentName
    Azure ML environment name to create or update.

.PARAMETER Image
    Optional fully qualified container image URI. If omitted, the script
    resolves the image from the workspace ACR using ImageName and ImageTag.

.PARAMETER ImageName
    Repository name in the workspace ACR.

.PARAMETER ImageTag
    Image tag in the workspace ACR.

.PARAMETER Version
    Optional environment version override.

.PARAMETER Stream
    If set, show the full Azure CLI output instead of the compact table view.

.EXAMPLE
    .\create_env.ps1
#>

[CmdletBinding()]
param (
    # Azure resource group that contains the AML workspace
    [string] $ResourceGroup = "sriks-aml-rg",

    # Azure ML workspace name
    [string] $Workspace = "sriks-aml-ws",

    # Name of the Azure ML environment to create or update in the workspace
    [string] $EnvironmentName = "text2sql-grpo-env",

    # Fully qualified image URI (e.g. myacr.azurecr.io/repo:tag).
    # When omitted the script auto-resolves the URI from the workspace ACR
    # using $ImageName and $ImageTag.
    [string] $Image,

    # Repository name inside the workspace ACR (used when $Image is not set)
    [string] $ImageName = "text2sql-grpo",

    # Image tag inside the workspace ACR (used when $Image is not set)
    [string] $ImageTag = "env-v1",

    # Environment version registered in Azure ML (default: "latest")
    [string] $Version = "latest",

    # Stream full Azure CLI output instead of the compact table view
    [switch] $Stream
)

Set-StrictMode -Version Latest

$ErrorActionPreference = "Stop"

az account show --output none

if ($LASTEXITCODE -ne 0) {
    Write-Error "Not logged in. Run: az login"
    exit 1
}

$mlExt = az extension list --query "[?name=='ml'].name" -o tsv

if (-not $mlExt) {
    Write-Error "Azure ML CLI extension is required. Run: az extension add --name ml"
    exit 1
}

if (-not $Image) {
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

    $imageRef = "{0}:{1}" -f $ImageName, $ImageTag
    az acr repository show --name $acrName --image $imageRef --only-show-errors --output none
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Image '$imageRef' was not found in ACR '$acrName'. Push it first with .\build_image.ps1."
        exit 1
    }

    $Image = "$loginServer/$imageRef"
}

$temporarySpecPath = Join-Path $env:TEMP "text2sql-prebuilt-env-$Version.yml"

# If version is "latest" (the default), auto-increment by querying the highest
# existing version number — AML treats "latest" as an immutable label once set.
if ($Version -eq "latest") {
    $existingVersions = az ml environment list `
        --name $EnvironmentName `
        --resource-group $ResourceGroup `
        --workspace-name $Workspace `
        --only-show-errors `
        --query "[].version" -o tsv 2>$null
    $maxVer = 0
    if ($existingVersions) {
        foreach ($v in ($existingVersions -split "`n" | Where-Object { $_ -match '^\d+$' })) {
            if ([int]$v -gt $maxVer) { $maxVer = [int]$v }
        }
    }
    $Version = [string]($maxVer + 1)
    Write-Host "Auto-incrementing environment version → $Version" -ForegroundColor Yellow
}

@(
    "`$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json",
    "",
    "name: $EnvironmentName",
    "version: `"$Version`"",
    "description: Prebuilt Docker image for Text-to-SQL GRPO jobs.",
    "image: $Image",
    "tags:",
    "  framework: unsloth",
    "  build: prebuilt",
    "  source: acr",
    "  image_name: $ImageName",
    "  image_tag: $ImageTag"
) | Set-Content -Path $temporarySpecPath -Encoding ascii

Write-Host "Using prebuilt image: $Image" -ForegroundColor Cyan

$cmd = @(
    "ml", "environment", "create",
    "--file", $temporarySpecPath,
    "--resource-group", $ResourceGroup,
    "--workspace-name", $Workspace
)

if (-not $Stream) {
    $cmd += @("--only-show-errors", "-o", "table")
}

Write-Host "`n=== Registering Azure ML environment ===" -ForegroundColor Magenta
Write-Host "► az $($cmd -join ' ')" -ForegroundColor Cyan

try {
    az @cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Environment registration failed."
        exit 1
    }
}
finally {
    if (Test-Path $temporarySpecPath) {
        Remove-Item $temporarySpecPath -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "`n✅ Environment registered." -ForegroundColor Green