#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Submit text2sql GRPO jobs to an existing Azure ML cluster.

.DESCRIPTION
    Supports two modes:
      pipeline  – submit the full end-to-end pipeline (data prep → train → eval)
      job       – submit a single named job (data_prep | train | eval)

.PARAMETER ResourceGroup
    Azure resource group that contains the AML workspace.
    Defaults to: txt2sql-grpo-rg

.PARAMETER Workspace
    Azure ML workspace name.
    Defaults to: txt2sql-grpo-ws

.PARAMETER Mode
    Execution mode: pipeline | job  (default: pipeline)

.PARAMETER Job
    When Mode=job, which job to submit: data_prep | train | eval

.PARAMETER CpuCluster
    Name of the CPU compute cluster used for data preparation.
    Defaults to: cpu-cluster

.PARAMETER GpuCluster
    Name of the GPU compute cluster used for training and evaluation.
    Defaults to: gpu-cluster-2

.PARAMETER SampleSize
    Number of Q/SQL pairs to sample (0 = all). Default: 400

.PARAMETER Stream
    If set, stream job logs to the terminal (blocks until completion).

.NOTES
    Requires the Azure ML environment `text2sql-grpo-env@latest` to exist in
    the target workspace. Register it first with .\create_env.ps1.

.EXAMPLE
    # Full pipeline with defaults
    .\run_jobs.ps1

    # Full pipeline, stream logs
    .\run_jobs.ps1 -Stream

    # Single job
    .\run_jobs.ps1 -Mode job -Job data_prep
#>

[CmdletBinding(PositionalBinding = $false)]
param (
    [string] $ResourceGroup = "txt2sql-grpo-rg",
    [string] $Workspace = "txt2sql-grpo-ws",

    [ValidateSet("pipeline", "job")]
    [string] $Mode = "pipeline",

    [ValidateSet("data_prep", "train", "eval")]
    [string] $Job = "data_prep",

    [string] $CpuCluster  = "cpu-cluster",
    [string] $GpuCluster  = "gpu-cluster-2",

    [int]    $SampleSize  = 6,

    [switch] $Stream

)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Resolve paths ────────────────────────────────────────────────────────────
$ScriptDir    = $PSScriptRoot
$JobsDir      = Join-Path $ScriptDir "ml_jobs"
$PipelineYaml = Join-Path $JobsDir "pipeline.yaml"
$DataPrepYaml = Join-Path $JobsDir "data_prep_job.yaml"
$TrainYaml    = Join-Path $JobsDir "grpo_train_job.yaml"
$EvalYaml     = Join-Path $JobsDir "eval_job.yaml"
$CreateEnvPs1 = Join-Path $ScriptDir "create_env.ps1"
$EnvironmentName = "text2sql-grpo-env"

# ── Shared CLI args ──────────────────────────────────────────────────────────
$BaseArgs   = @("--resource-group", $ResourceGroup, "--workspace-name", $Workspace)
$StreamFlag = if ($Stream) { @("--stream") } else { @() }

# ── Helper: submit a job ─────────────────────────────────────────────────────
function Submit-Job {
    param(
        [string]   $YamlFile,
        [string[]] $SetArgs = @()
    )
    $cmd = @("ml", "job", "create", "--file", $YamlFile) + $BaseArgs + $SetArgs + $StreamFlag 
    Write-Host "`n► az $($cmd -join ' ')" -ForegroundColor Cyan
    az @cmd
    if ($LASTEXITCODE -ne 0) { Write-Error "Job submission failed."; exit 1 }
}

function Assert-EnvironmentExists {
    $cmd = @(
        "ml", "environment", "show",
        "--name", $EnvironmentName,
        "--label", "latest"
    ) + $BaseArgs + @("--only-show-errors", "--query", "name", "-o", "tsv") 

    Write-Host "Checking env $cmd" -ForegroundColor Magenta
    
    $environmentNameResult = az @cmd 2>$null

    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($environmentNameResult)) {
        return
    }

    Write-Host "`nAzure ML environment '$EnvironmentName@latest' was not found in workspace '$Workspace'." -ForegroundColor Yellow
    if (Test-Path $CreateEnvPs1) {
        Write-Host "Create it first with .\create_env.ps1" -ForegroundColor Yellow
    }
    else {
        Write-Host "Register it first with az ml environment create --file azure/environments/environment.yml ..." -ForegroundColor Yellow
    }

    Write-Error "Missing Azure ML environment '$EnvironmentName@latest'."
    exit 1
}

# ── Verify az CLI login ──────────────────────────────────────────────────────
az account show --output none 

if ($LASTEXITCODE -ne 0) { Write-Error "Not logged in. Run: az login"; exit 1 }

$mlExt = az extension list --query "[?name=='ml'].name" -o tsv 

if (-not $mlExt) {
    Write-Error "Azure ML CLI extension is required. Run: az extension add --name ml"
    exit 1
}

Assert-EnvironmentExists

# ─────────────────────────────────────────────────────────────────────────────
# MODE: pipeline
# ─────────────────────────────────────────────────────────────────────────────
if ($Mode -eq "pipeline") {
    Write-Host "`n=== Submitting full pipeline ===" -ForegroundColor Magenta
    # Log input args for visibility; the actual submission command is in Submit-Job
    Write-Host "► az ml job create --file $PipelineYaml --set inputs.sample_size=$SampleSize jobs.data_prep.compute=azureml:$CpuCluster jobs.train.compute=azureml:$GpuCluster jobs.eval.compute=azureml:$GpuCluster" -ForegroundColor Cyan
    Submit-Job -YamlFile $PipelineYaml -SetArgs @(
        "--set", "inputs.sample_size=$SampleSize",
        "--set", "jobs.data_prep.compute=azureml:$CpuCluster",
        "--set", "jobs.train.compute=azureml:$GpuCluster",
        "--set", "jobs.eval.compute=azureml:$GpuCluster"
    )
    Write-Host "`n✅ Pipeline submitted." -ForegroundColor Green
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE: job  (single job)
# ─────────────────────────────────────────────────────────────────────────────
$jobYaml = switch ($Job) {
    "data_prep" { $DataPrepYaml }
    "train"     { $TrainYaml    }
    "eval"      { $EvalYaml     }
}

$jobCompute = if ($Job -eq "data_prep") { $CpuCluster } else { $GpuCluster }

Write-Host "`n=== Submitting job: $Job ===" -ForegroundColor Magenta
$jobSetArgs = @(
    "--set", "compute=azureml:$jobCompute"
)

if ($Job -eq "data_prep") {
    $jobSetArgs += @("--set", "inputs.sample_size=$SampleSize")
}
elseif ($Job -in @("train", "eval")) {
    Write-Host "Standalone '$Job' jobs require their data inputs to be supplied explicitly via the job YAML or additional --set arguments." -ForegroundColor Yellow
}

Submit-Job -YamlFile $jobYaml -SetArgs $jobSetArgs
Write-Host "`n✅ Job submitted." -ForegroundColor Green
