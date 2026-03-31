#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Submit the data-preparation job, wait for it to finish, and register the
    output CSV splits as a versioned Azure ML dataset.

.DESCRIPTION
    This script decouples the expensive data-preparation step from the
    training/evaluation pipeline.  Run it once (or whenever you need a fresh
    dataset) and take note of the printed dataset ID (name:version).  Pass
    that ID to run_jobs.ps1 via -DatasetId to skip data prep on every run.

.PARAMETER ResourceGroup
    Azure resource group that contains the AML workspace.
    Defaults to: sriks-aml-rg

.PARAMETER Workspace
    Azure ML workspace name.
    Defaults to: sriks-aml-ws

.PARAMETER CpuCluster
    Name of the CPU compute cluster used for data preparation.
    Defaults to: cpu-cluster

.PARAMETER SampleSize
    Number of databases to sample (-1 = all available, 31 = full set).
    Default: 16

.PARAMETER DatasetName
    Name to register the output dataset under in the workspace.
    Default: text2sql-grpo-splits

.PARAMETER DatasetVersion
    Optional explicit version string.  When omitted Azure ML auto-increments
    the version (1, 2, 3 …).

.EXAMPLE
    # Run with defaults and let Azure ML choose the version
    .\prep_data.ps1

    # Run with full dataset and a specific version label
    .\prep_data.ps1 -SampleSize -1 -DatasetVersion "full-v1"
#>

[CmdletBinding(PositionalBinding = $false)]
param (
    # Azure resource group containing the AML workspace
    [string] $ResourceGroup   = "sriks-aml-rg",

    # Azure ML workspace name
    [string] $Workspace       = "sriks-aml-ws",

    # CPU compute cluster used for data preparation (no GPU needed)
    [string] $CpuCluster      = "cpu-cluster",

    # Number of databases to sample; -1 uses all 31 available
    [int]    $SampleSize      = 16,

    # Dataset name to register under in the workspace
    [string] $DatasetName     = "text2sql-grpo-splits",

    # Explicit version; leave empty to let Azure ML auto-increment
    [string] $DatasetVersion  = "",

    # Raw-data asset name (Spider + BIRD .sqlite files used by exec_reward)
    [string] $RawDataName     = "text2sql-grpo-rawdata"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir    = $PSScriptRoot
$JobsDir      = Join-Path $ScriptDir "ml_jobs"
$DataPrepYaml = Join-Path $JobsDir "data_prep_job.yaml"

# Shared workspace args reused across all az ml calls
$BaseArgs = @("--resource-group", $ResourceGroup, "--workspace-name", $Workspace)

# ── Verify az CLI login ──────────────────────────────────────────────────────
az account show --output none
if ($LASTEXITCODE -ne 0) { Write-Error "Not logged in. Run: az login"; exit 1 }

$mlExt = az extension list --query "[?name=='ml'].name" -o tsv
if (-not $mlExt) {
    Write-Error "Azure ML CLI extension is required. Run: az extension add --name ml"
    exit 1
}

# ── Submit data preparation job ──────────────────────────────────────────────
Write-Host "`n=== Submitting data preparation job ===" -ForegroundColor Magenta

$submitArgs = @(
    "ml", "job", "create",
    "--file", $DataPrepYaml
) + $BaseArgs + @(
    "--set", "compute=azureml:$CpuCluster",
    "--set", "inputs.sample_size=$SampleSize",
    "--only-show-errors",
    "--query", "name",
    "-o", "tsv"
)

Write-Host "`n► az $($submitArgs -join ' ')" -ForegroundColor Cyan
$jobName = (az @submitArgs)
$jobName = if ($jobName) { $jobName.Trim() } else { "" }

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($jobName)) {
    Write-Error "Data prep job submission failed."
    exit 1
}

Write-Host "Job submitted: $jobName" -ForegroundColor Green

# ── Stream logs and wait for completion ─────────────────────────────────────
Write-Host "`nStreaming logs for '$jobName' (blocks until job finishes)..." -ForegroundColor Yellow

$streamArgs = @("ml", "job", "stream", "--name", $jobName) + $BaseArgs + @("--only-show-errors")
Write-Host "`n► az $($streamArgs -join ' ')" -ForegroundColor Cyan
az @streamArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Streaming failed — job '$jobName' may have errored. Check the studio for details."
    exit 1
}

# ── Confirm successful terminal status ──────────────────────────────────────
$statusArgs = @("ml", "job", "show", "--name", $jobName) + $BaseArgs + @(
    "--only-show-errors", "--query", "status", "-o", "tsv"
)
$jobStatus = (az @statusArgs)
$jobStatus = if ($jobStatus) { $jobStatus.Trim() } else { "" }

if ($jobStatus -ne "Completed") {
    Write-Error "Job '$jobName' ended with status '$jobStatus' (expected Completed)."
    exit 1
}

Write-Host "`nJob '$jobName' completed successfully." -ForegroundColor Green

# ── Resolve the csv_splits output URI ────────────────────────────────────────
Write-Host "`nResolving csv_splits output URI..." -ForegroundColor Yellow

$outputUriArgs = @("ml", "job", "show", "--name", $jobName) + $BaseArgs + @(
    "--only-show-errors",
    "--query", "outputs.csv_splits.uri",
    "-o", "tsv"
)
$csvSplitsUri = (az @outputUriArgs)
$csvSplitsUri = if ($csvSplitsUri) { $csvSplitsUri.Trim() } else { "" }

if ([string]::IsNullOrWhiteSpace($csvSplitsUri) -or $csvSplitsUri -notmatch '^azureml://datastores/') {
    # az ml job show does not always return a fully-qualified datastore URI.
    # Job outputs land in the default datastore under azureml/<job-id>/<output-name>/.
    $csvSplitsUri = "azureml://datastores/workspaceblobstore/paths/azureml/$jobName/csv_splits/"
    Write-Host "Resolved output URI not a valid datastore path; using constructed path:" -ForegroundColor Yellow
}

Write-Host "csv_splits URI: $csvSplitsUri" -ForegroundColor Cyan

# ── Register the folder as a versioned Azure ML dataset ─────────────────────
Write-Host "`n=== Registering dataset '$DatasetName' ===" -ForegroundColor Magenta

$registerArgs = @(
    "ml", "data", "create",
    "--name",        $DatasetName,
    "--type",        "uri_folder",
    "--path",        $csvSplitsUri,
    "--description", "CSV prompt-record splits — data_prep job $jobName (sample_size=$SampleSize)"
) + $BaseArgs + @(
    "--only-show-errors",
    "--query", "version",
    "-o", "tsv"
)

if ($DatasetVersion) {
    # Insert --version before the output flags
    $registerArgs = $registerArgs[0..($registerArgs.Count - 4)] +
                    @("--version", $DatasetVersion) +
                    $registerArgs[($registerArgs.Count - 3)..($registerArgs.Count - 1)]
}

Write-Host "`n► az $($registerArgs -join ' ')" -ForegroundColor Cyan
$registeredVersion = (az @registerArgs)
$registeredVersion = if ($registeredVersion) { $registeredVersion.Trim() } else { "" }

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($registeredVersion)) {
    Write-Error "Dataset registration failed."
    exit 1
}

$datasetId = "${DatasetName}:${registeredVersion}"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  Dataset registered: $datasetId" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Run training + evaluation with:" -ForegroundColor Cyan
Write-Host "  .\run_jobs.ps1 -DatasetId '$datasetId'" -ForegroundColor White
Write-Host ""

# Also write the ID to a local file so callers / CI pipelines can read it back
$idFile = Join-Path $ScriptDir "last_dataset_id.txt"
Set-Content -Path $idFile -Value $datasetId -Encoding UTF8
Write-Host "Dataset ID written to: $idFile" -ForegroundColor DarkGray

# ── Register rawdata_dir (Spider + BIRD .sqlite files) ───────────────────────
Write-Host "`n=== Registering raw-data asset '$RawDataName' ==" -ForegroundColor Magenta

$rawDataUri = "azureml://datastores/workspaceblobstore/paths/azureml/$jobName/rawdata_dir/"
$rawOutputUriArgs = @("ml", "job", "show", "--name", $jobName) + $BaseArgs + @(
    "--only-show-errors",
    "--query", "outputs.rawdata_dir.uri",
    "-o", "tsv"
)
$resolvedRawUri = (az @rawOutputUriArgs)
$resolvedRawUri = if ($resolvedRawUri) { $resolvedRawUri.Trim() } else { "" }
if ($resolvedRawUri -match '^azureml://datastores/') { $rawDataUri = $resolvedRawUri }
Write-Host "rawdata_dir URI: $rawDataUri" -ForegroundColor Cyan

$registerRawArgs = @(
    "ml", "data", "create",
    "--name",        $RawDataName,
    "--type",        "uri_folder",
    "--path",        $rawDataUri,
    "--description", "Raw Spider + BIRD databases (.sqlite) — data_prep job $jobName"
) + $BaseArgs + @(
    "--only-show-errors",
    "--query", "version",
    "-o", "tsv"
)
if ($DatasetVersion) {
    $registerRawArgs = $registerRawArgs[0..($registerRawArgs.Count - 4)] +
                       @("--version", $DatasetVersion) +
                       $registerRawArgs[($registerRawArgs.Count - 3)..($registerRawArgs.Count - 1)]
}
Write-Host "`n► az $($registerRawArgs -join ' ')" -ForegroundColor Cyan
$registeredRawVersion = (az @registerRawArgs)
$registeredRawVersion = if ($registeredRawVersion) { $registeredRawVersion.Trim() } else { "" }

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($registeredRawVersion)) {
    Write-Warning "Raw-data asset registration failed — exec_reward will not have SQLite files at training time."
} else {
    $rawDataId = "${RawDataName}:${registeredRawVersion}"
    $rawIdFile = Join-Path $ScriptDir "last_rawdata_id.txt"
    Set-Content -Path $rawIdFile -Value $rawDataId -Encoding UTF8
    Write-Host "Raw-data asset registered: $rawDataId" -ForegroundColor Green
    Write-Host "Raw-data ID written to: $rawIdFile" -ForegroundColor DarkGray
    Write-Host "`nRun training + evaluation with both assets:" -ForegroundColor Cyan
    Write-Host "  .\run_jobs.ps1 -DatasetId '$datasetId' -RawDataId '$rawDataId'" -ForegroundColor White
}
