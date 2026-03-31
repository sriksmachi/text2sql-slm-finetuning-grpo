#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Submit text2sql GRPO train+eval jobs to an existing Azure ML cluster.

.DESCRIPTION
    Data preparation is decoupled from training/evaluation.  Run
    .\prep_data.ps1 once to prepare and register the dataset, then supply
    the printed dataset ID here via -DatasetId.

    Supports two modes:
      pipeline  – submit the train+eval pipeline using a registered dataset
      job       – submit a single named job (data_prep | train | eval)

.PARAMETER ResourceGroup
    Azure resource group that contains the AML workspace.
    Defaults to: sriks-aml-rg

.PARAMETER Workspace
    Azure ML workspace name.
    Defaults to: sriks-aml-ws

.PARAMETER Mode
    Execution mode: pipeline | job  (default: pipeline)

.PARAMETER Job
    When Mode=job, which job to submit: data_prep | train | eval

.PARAMETER DatasetId
    Registered dataset ID in name:version format produced by prep_data.ps1
    (e.g. "text2sql-grpo-splits:1").  Required when Mode=pipeline.
    The last registered ID is also saved to azure/last_dataset_id.txt by
    prep_data.ps1 for convenience.

.PARAMETER RawDataId
    Registered raw-data asset ID in name:version format produced by prep_data.ps1
    (e.g. "text2sql-grpo-rawdata:1").  Required when Mode=pipeline so that
    exec_reward can locate the SQLite database files during training.
    The last registered ID is saved to azure/last_rawdata_id.txt by prep_data.ps1.

.PARAMETER CpuCluster
    Name of the CPU compute cluster used for data preparation.
    Defaults to: cpu-cluster

.PARAMETER GpuCluster
    Name of the GPU compute cluster used for training and evaluation.
    Defaults to: gpu-cluster-2

.PARAMETER SampleSize
    Number of databases to sample when Mode=job -Job data_prep.
    Default: 16

.PARAMETER Stream
    If set, stream job logs to the terminal (blocks until completion).

.NOTES
    Requires the Azure ML environment `text2sql-grpo-env@latest` to exist in
    the target workspace. Register it first with .\create_env.ps1.

.EXAMPLE
    # Step 1 – prepare data and register dataset (run once)
    .\prep_data.ps1

    # Step 2 – train + eval using the registered dataset ID printed above
    .\run_jobs.ps1 -DatasetId 'text2sql-grpo-splits:1'

    # Re-use the last registered dataset ID saved by prep_data.ps1
    .\run_jobs.ps1 -DatasetId (Get-Content .\last_dataset_id.txt)

    # Submit only the training job manually
    .\run_jobs.ps1 -Mode job -Job train
#>

[CmdletBinding(PositionalBinding = $false)]
param (
    # Azure resource group containing the AML workspace
    [string] $ResourceGroup = "sriks-aml-rg",

    # Azure ML workspace name
    [string] $Workspace     = "sriks-aml-ws",

    # Execution mode: pipeline (train+eval) or job (single step)
    [ValidateSet("pipeline", "job")]
    [string] $Mode          = "pipeline",

    # Which single job to submit when Mode=job
    [ValidateSet("data_prep", "train", "eval")]
    [string] $Job           = "data_prep",

    # Registered dataset ID (name:version) produced by prep_data.ps1.
    # Required for Mode=pipeline.  Example: "text2sql-grpo-splits:1"
    [string] $DatasetId     = "",

    # Registered raw-data asset ID (name:version) produced by prep_data.ps1.
    # Required for Mode=pipeline so exec_reward can locate SQLite files.
    # Example: "text2sql-grpo-rawdata:1"
    [string] $RawDataId     = "",

    # CPU compute cluster (data preparation)
    [string] $CpuCluster    = "cpu-cluster",

    # GPU compute cluster (training and evaluation)
    [string] $GpuCluster    = "gpu-cluster-2",

    # Number of databases to sample when Mode=job -Job data_prep (-1 = all 31)
    [int]    $SampleSize    = 16,

    # Stream job logs to the terminal (blocks until the job finishes)
    [switch] $Stream
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Resolve paths ────────────────────────────────────────────────────────────
$ScriptDir            = $PSScriptRoot
$JobsDir              = Join-Path $ScriptDir "ml_jobs"
# Train+eval pipeline — consumes a pre-registered dataset (no data_prep step)
$TrainEvalPipelineYaml = Join-Path $JobsDir "train_eval_pipeline.yaml"
$DataPrepYaml         = Join-Path $JobsDir "data_prep_job.yaml"
$TrainYaml            = Join-Path $JobsDir "grpo_train_job.yaml"
$EvalYaml             = Join-Path $JobsDir "eval_job.yaml"
$CreateEnvPs1         = Join-Path $ScriptDir "create_env.ps1"
$EnvironmentName      = "text2sql-grpo-env"

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
# MODE: pipeline  (train + eval — data prep is decoupled)
# ─────────────────────────────────────────────────────────────────────────────
if ($Mode -eq "pipeline") {

    # Require a registered dataset ID.  Check the flat file written by
    # prep_data.ps1 as a fallback so callers don't have to type it manually.
    if ([string]::IsNullOrWhiteSpace($DatasetId)) {
        $idFile = Join-Path $ScriptDir "last_dataset_id.txt"
        if (Test-Path $idFile) {
            $DatasetId = (Get-Content $idFile -Raw).Trim()
            Write-Host "Using dataset ID from last_dataset_id.txt: $DatasetId" -ForegroundColor Yellow
        }
    }

    if ([string]::IsNullOrWhiteSpace($RawDataId)) {
        $rawIdFile = Join-Path $ScriptDir "last_rawdata_id.txt"
        if (Test-Path $rawIdFile) {
            $RawDataId = (Get-Content $rawIdFile -Raw).Trim()
            Write-Host "Using raw-data ID from last_rawdata_id.txt: $RawDataId" -ForegroundColor Yellow
        }
    }

    if ([string]::IsNullOrWhiteSpace($DatasetId)) {
        Write-Error @"
-DatasetId is required for pipeline mode.
Run .\prep_data.ps1 first to prepare and register the dataset, then pass
the printed ID here:
  .\run_jobs.ps1 -DatasetId 'text2sql-grpo-splits:1' -RawDataId 'text2sql-grpo-rawdata:1'
"@
        exit 1
    }

    if ([string]::IsNullOrWhiteSpace($RawDataId)) {
        Write-Error @"
-RawDataId is required for pipeline mode (needed by exec_reward for SQLite lookups).
Run .\prep_data.ps1 first — it registers both assets and saves their IDs to
last_dataset_id.txt and last_rawdata_id.txt.
  .\run_jobs.ps1 -DatasetId 'text2sql-grpo-splits:1' -RawDataId 'text2sql-grpo-rawdata:1'
"@
        exit 1
    }

    Write-Host "`n=== Submitting train+eval pipeline (dataset: $DatasetId, rawdata: $RawDataId) ===" -ForegroundColor Magenta
    $gitSha = git rev-parse --short HEAD 2>$null
    if (-not $gitSha) { $gitSha = "unknown" }
    $pipelineArgs = @(
        "--set", "inputs.csv_splits.path=azureml:$DatasetId",
        "--set", "inputs.rawdata_dir.path=azureml:$RawDataId",
        "--set", "jobs.grpo_train.compute=azureml:$GpuCluster",
        "--set", "jobs.eval.compute=azureml:$GpuCluster",
        "--set", "tags.git_sha=$gitSha",
        "--set", "tags.dataset_id=$DatasetId",
        "--set", "tags.rawdata_id=$RawDataId"
    )

    # Capture the pipeline job name so we can register the model after completion
    $pipelineJobName = az ml job create --file $TrainEvalPipelineYaml @BaseArgs @pipelineArgs `
        --only-show-errors --query "name" -o tsv
    $pipelineJobName = if ($pipelineJobName) { $pipelineJobName.Trim() } else { "" }
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pipelineJobName)) {
        Write-Error "Pipeline submission failed."; exit 1
    }

    Write-Host "`n✅ Pipeline submitted: $pipelineJobName" -ForegroundColor Green

    if ($Stream) {
        Write-Host "`nStreaming pipeline logs..." -ForegroundColor Yellow
        az ml job stream --name $pipelineJobName @BaseArgs --only-show-errors

        # ── Register the trained LoRA adapter as a versioned AML Model ───────
        $jobStatus = az ml job show --name $pipelineJobName @BaseArgs `
            --only-show-errors --query "status" -o tsv
        $jobStatus = if ($jobStatus) { $jobStatus.Trim() } else { "Unknown" }

        if ($jobStatus -eq "Completed") {
            Write-Host "`nRegistering trained LoRA adapter as AML Model..." -ForegroundColor Yellow
            $modelPath = "azureml://jobs/$pipelineJobName/outputs/model_dir"
            az ml model create `
                --name "text2sql-grpo-lora" `
                --path $modelPath `
                --type "custom_model" `
                @BaseArgs --only-show-errors
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Model registered: text2sql-grpo-lora" -ForegroundColor Green
            } else {
                Write-Warning "Model registration failed — register manually with:"
                Write-Host "  az ml model create --name text2sql-grpo-lora --path $modelPath --type custom_model --resource-group $ResourceGroup --workspace-name $Workspace" -ForegroundColor White
            }
        } else {
            Write-Host "Pipeline status: $jobStatus — skipping model registration." -ForegroundColor Yellow
        }
    }
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
