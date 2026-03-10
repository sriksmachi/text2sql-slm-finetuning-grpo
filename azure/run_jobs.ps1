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

.PARAMETER Workspace
    Azure ML workspace name.

.PARAMETER Mode
    Execution mode: pipeline | job  (default: pipeline)

.PARAMETER Job
    When Mode=job, which job to submit: data_prep | train | eval

.PARAMETER CpuCluster
    Name of the CPU compute cluster used for data preparation.
    Defaults to: cpu-cluster

.PARAMETER GpuCluster
    Name of the GPU compute cluster used for training and evaluation.
    Defaults to: gpu-cluster

.PARAMETER SampleSize
    Number of Q/SQL pairs to sample (0 = all). Default: 400

.PARAMETER Stream
    If set, stream job logs to the terminal (blocks until completion).

.EXAMPLE
    # Full pipeline with defaults
    .\run_jobs.ps1 -ResourceGroup rg-text2sql -Workspace aml-text2sql

    # Full pipeline, stream logs
    .\run_jobs.ps1 -ResourceGroup rg-text2sql -Workspace aml-text2sql -Stream

    # Single job
    .\run_jobs.ps1 -ResourceGroup rg-text2sql -Workspace aml-text2sql -Mode job -Job train
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory)][string] $ResourceGroup,
    [Parameter(Mandatory)][string] $Workspace,

    [ValidateSet("pipeline", "job")]
    [string] $Mode = "pipeline",

    [ValidateSet("data_prep", "train", "eval")]
    [string] $Job = "data_prep",

    [string] $CpuCluster  = "cpu-cluster",
    [string] $GpuCluster  = "gpu-cluster",

    [int]    $SampleSize  = 400,

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

# ── Verify az CLI login ──────────────────────────────────────────────────────
az account show --output none
if ($LASTEXITCODE -ne 0) { Write-Error "Not logged in. Run: az login"; exit 1 }

$mlExt = az extension list --query "[?name=='ml'].name" -o tsv
if (-not $mlExt) {
    Write-Host "Installing Azure ML CLI extension..." -ForegroundColor Yellow
    az extension add --name ml --yes
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE: pipeline
# ─────────────────────────────────────────────────────────────────────────────
if ($Mode -eq "pipeline") {
    Write-Host "`n=== Submitting full pipeline ===" -ForegroundColor Magenta
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
Submit-Job -YamlFile $jobYaml -SetArgs @(
    "--set", "inputs.sample_size=$SampleSize",
    "--set", "compute=azureml:$jobCompute"
)
Write-Host "`n✅ Job submitted." -ForegroundColor Green

<#
.SYNOPSIS
    Submit text2sql GRPO jobs to an existing Azure ML cluster.

.DESCRIPTION
    Supports three modes:
      pipeline  – submit the full end-to-end pipeline (data prep → train → eval)
      steps     – submit each job individually and chain outputs
      job       – submit a single named job (data_prep | train | eval)

.PARAMETER ResourceGroup
    Azure resource group that contains the AML workspace.

.PARAMETER Workspace
    Azure ML workspace name.

.PARAMETER Mode
    Execution mode: pipeline | steps | job  (default: pipeline)

.PARAMETER Job
    When Mode=job, which job to submit: data_prep | train | eval

.PARAMETER CpuCluster
    Name of the CPU compute cluster used for data preparation.
    Defaults to: cpu-cluster

.PARAMETER GpuCluster
    Name of the GPU compute cluster used for training and evaluation.
    Defaults to: gpu-cluster

.PARAMETER SampleSize
    Number of Q/SQL pairs to sample (0 = all). Default: 400

.PARAMETER Epochs
    Override num_train_epochs in the training job. Default: 2

.PARAMETER Stream
    If set, stream job logs to the terminal (blocks until completion).

.EXAMPLE
    # Full pipeline with defaults
    .\run_jobs.ps1 -ResourceGroup rg-text2sql -Workspace aml-text2sql

    # Full pipeline, stream logs
    .\run_jobs.ps1 -ResourceGroup rg-text2sql -Workspace aml-text2sql -Stream

    # Individual steps, custom clusters
    .\run_jobs.ps1 -ResourceGroup rg-text2sql -Workspace aml-text2sql `
        -Mode steps -CpuCluster my-cpu -GpuCluster my-a100

    # Single job
    .\run_jobs.ps1 -ResourceGroup rg-text2sql -Workspace aml-text2sql `
        -Mode job -Job train
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory)][string] $ResourceGroup,
    [Parameter(Mandatory)][string] $Workspace,

    [ValidateSet("pipeline", "steps", "job")]
    [string] $Mode = "pipeline",

    [ValidateSet("data_prep", "train", "eval")]
    [string] $Job = "data_prep",

    [string] $CpuCluster  = "cpu-cluster",
    [string] $GpuCluster  = "gpu-cluster",

    [int]    $SampleSize  = 400,
    [int]    $Epochs      = 2,

    [switch] $Stream
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Resolve paths ────────────────────────────────────────────────────────────
$ScriptDir  = $PSScriptRoot
$JobsDir    = Join-Path $ScriptDir "ml_jobs"
$PipelineYaml  = Join-Path $JobsDir "pipeline.yaml"
$DataPrepYaml  = Join-Path $JobsDir "data_prep_job.yaml"
$TrainYaml     = Join-Path $JobsDir "grpo_train_job.yaml"
$EvalYaml      = Join-Path $JobsDir "eval_job.yaml"

# ── Shared CLI args ──────────────────────────────────────────────────────────
$BaseArgs = @(
    "--resource-group", $ResourceGroup,
    "--workspace-name", $Workspace
)
$StreamFlag = if ($Stream) { "--stream" } else { $null }

# ── Helper: submit a job and return its name ─────────────────────────────────
function Submit-Job {
    param(
        [string]   $YamlFile,
        [string[]] $SetArgs   = @()
    )

    $cmd = @("ml", "job", "create", "--file", $YamlFile) + $BaseArgs + $SetArgs
    if ($StreamFlag) { $cmd += $StreamFlag }

    Write-Host "`n► az $($cmd -join ' ')" -ForegroundColor Cyan
    $output = az @cmd 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Job submission failed:`n$output"
        exit 1
    }

    # Parse the job name from the JSON output
    $json     = $output | ConvertFrom-Json -ErrorAction SilentlyContinue
    $jobName  = $json.name
    Write-Host "  Submitted: $jobName" -ForegroundColor Green
    return $jobName
}

# ── Helper: wait for a job and get its output paths ─────────────────────────
function Wait-JobAndGetOutput {
    param([string] $JobName, [string] $OutputName)

    Write-Host "`n⏳ Waiting for job '$JobName' to complete..." -ForegroundColor Yellow
    az ml job stream --name $JobName @BaseArgs | Out-Null

    $json = az ml job show --name $JobName @BaseArgs | ConvertFrom-Json
    if ($json.status -ne "Completed") {
        Write-Error "Job '$JobName' finished with status: $($json.status)"
        exit 1
    }

    $outputPath = $json.outputs.$OutputName.uri
    Write-Host "  Output '$OutputName': $outputPath" -ForegroundColor Green
    return $outputPath
}

# ── Verify az CLI and ml extension ──────────────────────────────────────────
Write-Host "Checking Azure CLI..." -ForegroundColor Cyan
az account show --output none
if ($LASTEXITCODE -ne 0) {
    Write-Error "Not logged in. Run: az login"
    exit 1
}

$mlExt = az extension list --query "[?name=='ml'].name" -o tsv
if (-not $mlExt) {
    Write-Host "Installing Azure ML CLI extension..." -ForegroundColor Yellow
    az extension add --name ml --yes
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE: pipeline
# ─────────────────────────────────────────────────────────────────────────────
if ($Mode -eq "pipeline") {
    Write-Host "`n=== Submitting full pipeline ===" -ForegroundColor Magenta

    $setArgs = @(
        "--set", "inputs.sample_size=$SampleSize",
        "--set", "jobs.data_prep.compute=azureml:$CpuCluster",
        "--set", "jobs.train.compute=azureml:$GpuCluster",
        "--set", "jobs.eval.compute=azureml:$GpuCluster"
    )

    Submit-Job -YamlFile $PipelineYaml -SetArgs $setArgs | Out-Null
    Write-Host "`n✅ Pipeline submitted." -ForegroundColor Green
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE: steps  (sequential with output chaining)
# ─────────────────────────────────────────────────────────────────────────────
if ($Mode -eq "steps") {
    Write-Host "`n=== Step 1/3 : Data Preparation ===" -ForegroundColor Magenta

    $dataPrepName = Submit-Job -YamlFile $DataPrepYaml -SetArgs @(
        "--set", "inputs.sample_size=$SampleSize",
        "--set", "compute=azureml:$CpuCluster"
    )

    $rawdataPath   = Wait-JobAndGetOutput -JobName $dataPrepName -OutputName "rawdata_dir"
    $hfSplitsPath  = Wait-JobAndGetOutput -JobName $dataPrepName -OutputName "hf_splits"
    $csvSplitsPath = Wait-JobAndGetOutput -JobName $dataPrepName -OutputName "csv_splits"

    # ── Step 2: Training ──────────────────────────────────────────────────────
    Write-Host "`n=== Step 2/3 : GRPO Training ===" -ForegroundColor Magenta

    $trainName = Submit-Job -YamlFile $TrainYaml -SetArgs @(
        "--set", "inputs.hf_splits.path=$hfSplitsPath",
        "--set", "inputs.rawdata_dir.path=$rawdataPath",
        "--set", "compute=azureml:$GpuCluster"
    )

    $modelDirPath = Wait-JobAndGetOutput -JobName $trainName -OutputName "model_dir"

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    Write-Host "`n=== Step 3/3 : Evaluation ===" -ForegroundColor Magenta

    Submit-Job -YamlFile $EvalYaml -SetArgs @(
        "--set", "inputs.model_dir.path=$modelDirPath",
        "--set", "inputs.csv_splits.path=$csvSplitsPath",
        "--set", "inputs.rawdata_dir.path=$rawdataPath",
        "--set", "compute=azureml:$GpuCluster"
    ) | Out-Null

    Write-Host "`n✅ All steps submitted." -ForegroundColor Green
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE: job  (single job)
# ─────────────────────────────────────────────────────────────────────────────
if ($Mode -eq "job") {
    switch ($Job) {
        "data_prep" {
            Write-Host "`n=== Submitting: Data Preparation ===" -ForegroundColor Magenta
            Submit-Job -YamlFile $DataPrepYaml -SetArgs @(
                "--set", "inputs.sample_size=$SampleSize",
                "--set", "compute=azureml:$CpuCluster"
            ) | Out-Null
        }
        "train" {
            Write-Host "`n=== Submitting: GRPO Training ===" -ForegroundColor Magenta
            Submit-Job -YamlFile $TrainYaml -SetArgs @(
                "--set", "compute=azureml:$GpuCluster"
            ) | Out-Null
        }
        "eval" {
            Write-Host "`n=== Submitting: Evaluation ===" -ForegroundColor Magenta
            Submit-Job -YamlFile $EvalYaml -SetArgs @(
                "--set", "compute=azureml:$GpuCluster"
            ) | Out-Null
        }
    }

    Write-Host "`n✅ Job submitted." -ForegroundColor Green
    exit 0
}
