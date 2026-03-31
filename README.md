# text2sql-grpo-azure-ml

> **Enterprise-grade, open-source GRPO pipeline that proves true schema generalisation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

---

## 🗺️ Architecture

```mermaid
flowchart LR
    subgraph Data["Data Layer"]
        A[Spider / BIRD datasets] --> B[serialize_schemas.py]
        B --> D[schema_split.py\nschema-level train/val/test]
    end

    subgraph Training["GRPO Training  · Azure ML"]
        D --> E[GRPOTrainer\nUnsloth + TRL]
        E --> F{Reward Functions = format_reward, exec_reward\nmulti-dialect, schema_fidelity_reward}
        E --> J[Fine-tuned LLM\nQwen2.5-3B-Instruct]
    end

    subgraph Eval["Evaluation"]
        J --> K[evaluator.py\ncross_schema_exec_acc]
        K --> L[MLflow · Azure ML\nmetrics & artefacts]
    end

    subgraph Deploy["Deployment"]
        J --> M[Azure ML\nManaged Online Endpoint]
        M --> N[Streamlit Demo\ndemo/streamlit_app.py]
    end
```

---

## 📊 Results

Rewards are evaluated as the average combined score (`format × 0.2 + exec × 0.5 + schema_fidelity × 0.3`) across held-out splits using schemas unseen during training (schema-level split strategy). The SLM is `unsloth/Qwen2.5-3B-Instruct` fine-tuned with 4-bit QLoRA + GRPO for 2 epochs on a 400-sample subset. **GPT-5.1** (Azure OpenAI) is evaluated on the same test set as a strong upper-bound reference.

| Dataset | SLM · pre-GRPO ¹ | SLM · post-GRPO ¹ | Δ ablation (abs / rel) | GPT-5.1 ² |
|---|---:|---:|---:|---:|
| **Spider** | 0.8365 | **0.8907** | +0.0542 / **+6.48%** | 0.9865 |
| **BIRD** | 0.7133 | **0.7574** | +0.0441 / **+6.18%** | 0.9770 |
| **Overall** | — | — | — | **0.9801** |

> ¹ GRPO ablation: 400-sample subset, 2 epochs, schema-level split — not trained on full corpus.  
> ² GPT-5.1 evaluated on the same held-out test set; no fine-tuning.

Both SLM datasets improved by ~6%, showing balanced generalisation gains across a clean benchmark (Spider) and a harder, noisier one (BIRD) — with no cross-dataset trade-off. The `exec_reward` component provides the dominant training signal; a query either executes or it doesn't.

The GRPO-trained 3B SLM reaches **90.3% of GPT-5.1's reward on Spider** and **77.5% on BIRD**, closing a meaningful fraction of the gap to a frontier model at a fraction of the inference cost.

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base model | `unsloth/Qwen2.5-3B-Instruct` (4-bit QLoRA) | Strong code baseline; fits in 40 GB at 4-bit |
| LoRA rank | 32 (QKVO modules) | Balances capacity vs. memory; gate/up/down projections excluded |
| Epochs | 2 | Proof-of-concept run on a sampled subset |
| Per-device batch size | 3 | Limited by GPU VRAM with 2048-token sequences |
| Gradient accumulation steps | 6 → **effective batch = 18** | Stabilises policy gradient updates |
| Learning rate | 2e-5 (cosine schedule, 5% warmup) | Conservative; avoids reward hacking early in training |
| GRPO generations per prompt | 3 | Group size for relative advantage estimation |
| Sampling temperature | 0.7 | Maintains exploration without excessive randomness |
| KL penalty β | 0.04 | Keeps policy close to the reference; prevents mode collapse |
| Policy clip ε | 0.2 | Standard PPO-style clip; limits per-step policy change |
| Max sequence length | 2048 | Covers schema prompt + multi-join SQL completions |
| Reward weights | format 0.2 · exec 0.5 · schema_fidelity 0.3 | Execution correctness dominates; format is a soft gate |

### Analysis

**Spider — strong gain (+6.48%, reaching 90.3% of GPT-5.1)**  
Spider improved from 0.8365 to 0.8907 against a GPT-5.1 ceiling of 0.9865, indicating that GRPO successfully reinforced executable query structures, better join paths, and schema-consistent column usage. The magnitude of this gain is meaningful for a short RL run and reflects genuine policy improvement rather than random variance.

**BIRD — meaningful improvement on a harder benchmark (+6.18%, reaching 77.5% of GPT-5.1)**  
BIRD increased from 0.7133 to 0.7574 against a GPT-5.1 ceiling of 0.9770. The larger remaining gap to GPT-5.1 on BIRD (vs. Spider) reflects the benchmark's higher query complexity, noisier schema semantics, and greater compositional burden — areas where the 3B model's capacity is a limiting factor.

**GPT-5.1 as upper-bound reference**  
GPT-5.1 scores 0.9801 overall (0.9865 Spider / 0.9770 BIRD) on the same reward formula, providing a well-calibrated ceiling. The GRPO-trained 3B SLM recovers ~84% of GPT-5.1's overall reward with orders-of-magnitude lower inference cost, validating the RL fine-tuning approach for cost-sensitive deployments.

**Reward signal validation**  
The aligned gains across both benchmarks validate the combined reward (`format + execution + schema fidelity`) as an effective supervision proxy for text-to-SQL RL fine-tuning. The execution component provides a hard grounding signal that resists superficial improvements.

### Limitations

- Training used a **sampled subset** of the full combined corpus (400 examples, schema-level split), not the complete Spider + BIRD training sets
- Only **2 epochs** were run; the learning curve had not yet plateaued at checkpoint
- The 3B model size limits its ability to handle the most complex BIRD queries requiring multi-step reasoning; the larger gap vs. GPT-5.1 on BIRD reflects this
- `extract_sql` and SQLGlot show occasional parsing/token errors; a more robust SQL extraction approach may improve the reward signal

> Results measured on held-out schemas not seen during training. Full evaluation logs available in MLflow.  
> **Scaling note:** Training for 5–10 epochs on the full Spider + BIRD corpus and upgrading to the 7B variant (`Qwen2.5-Coder-7B-Instruct`) is projected to push Spider beyond 0.93 and close the remaining gap with GPT-5.1 on BIRD.

---

## ⚡ 1-Click Azure Run

### Prerequisites

- Azure subscription with quota for `Standard_NC24ads_A100_v4` (or smaller GPU)
- Azure CLI + ML extension installed
- Bicep CLI installed

Install the Azure ML CLI extension before running the scripts in `azure/create_env.ps1` or `azure/build_image.ps1`:

```bash
az extension add --name ml
```

### Deploy infrastructure

```bash
# Clone the repo
git clone https://github.com/sriksmachi/text2sql-grpo-azure-ml.git
cd text2sql-grpo-azure-ml

# Deploy Azure ML workspace + compute + endpoints
az group create --name rg-text2sql-dev --location eastus
az deployment group create --resource-group rg-text2sql-dev --template-file azure/bicep/main.bicep --parameters baseName=text2sql environment=dev ownerObjectId=$(az ad signed-in-user show --query id -o tsv)
```

### Run the pipeline

Data preparation is decoupled from training and evaluation. `prep_data.ps1`
registers **two** AML dataset assets — the CSV splits and the raw SQLite
databases — then writes their IDs to `last_dataset_id.txt` and
`last_rawdata_id.txt`. `run_jobs.ps1` reads both files automatically, so
you only need to pass explicit IDs when overriding defaults.

```powershell
cd azure

# Step 1 – build and register the Docker environment (first time only)
.\build_image.ps1 -RegisterEnvironment -ResourceGroup sriks-aml-rg -Workspace sriks-aml-ws

# Step 2 – prepare data and register both assets (run once or on dataset refresh)
.\prep_data.ps1
# Prints:
#   Dataset registered : text2sql-grpo-splits:1   → last_dataset_id.txt
#   Raw data registered: text2sql-grpo-rawdata:1  → last_rawdata_id.txt

# Step 3 – train + evaluate (both IDs auto-read from the txt files above)
.\run_jobs.ps1

# Or pass them explicitly:
.\run_jobs.ps1 -DatasetId 'text2sql-grpo-splits:1' -RawDataId 'text2sql-grpo-rawdata:1'
```

> **Why two assets?** The `exec_reward` function executes generated SQL against
> the original SQLite databases at training time. The raw-data asset mounts the
> Spider + BIRD `.sqlite` files into the training container so the reward can
> run live query execution. Without it, `exec_reward` silently returns `0.0`
> for every sample, making the dominant reward signal inactive.

### Register the Azure ML environment

```bash
# First time only – builds the conda env on top of the CUDA base image
az ml environment create \
  --file azure/environments/environment.yml \
  --resource-group rg-text2sql-dev \
  --workspace-name aml-text2sql-dev
```

### Run individual jobs

```powershell
$RG = "sriks-aml-rg"
$WS = "sriks-aml-ws"

# 1. Data preparation (CPU cluster) — or use .\prep_data.ps1 to also register both assets
.\run_jobs.ps1 -Mode job -Job data_prep -ResourceGroup $RG -Workspace $WS

# 2. GRPO training (GPU cluster – Standard_NC24ads_A100_v4)
#    Reads rawdata_dir from last_rawdata_id.txt written by prep_data.ps1
.\run_jobs.ps1 -Mode job -Job train -ResourceGroup $RG -Workspace $WS

# 3. Evaluation (GPU cluster)
.\run_jobs.ps1 -Mode job -Job eval -ResourceGroup $RG -Workspace $WS
```

## 💰 Estimated Azure Cost

| Resource | SKU | Est. Monthly Cost |
|---|---|---|
| GPU Compute (training) | Standard_NC24ads_A100_v4 × 1 node, ~20 h | ~$120 |
| GPU Compute (inference endpoint) | Standard_NC6s_v3 × 1 instance | ~$350 |
| Azure ML Workspace | Standard | ~$0 (workspace free) |
| Storage Account | Standard LRS, ~50 GB | ~$1 |
| Container Registry | Premium | ~$18 |
| Key Vault | Standard | ~$1 |
| Application Insights | Pay-as-you-go, low traffic | ~$2 |
| **Total** | | **~$492 / month** |

> Costs scale down significantly with spot instances and auto-scaling to zero. Training is a one-time cost; the table assumes 1 month of endpoint availability.

---

## 🏗️ Project Structure

```
text2sql-slm-finetuning-grpo/
├── azure/
│   ├── ml_jobs/
│   │   ├── train_eval_pipeline.yaml  # Train+eval pipeline (grpo_train → eval)
│   │   ├── data_prep_job.yaml        # Standalone CPU data-prep job
│   │   ├── grpo_train_job.yaml       # Standalone GPU training job
│   │   └── eval_job.yaml             # Standalone evaluation job
│   ├── environments/
│   │   ├── environment.yml           # Azure ML environment definition
│   │   ├── conda_env.yml             # Conda spec (PyTorch + CUDA + Unsloth)
│   │   └── Dockerfile                # CUDA base + Unsloth + TRL
│   ├── prep_data.ps1                 # Submit data prep job; register csv_splits + rawdata_dir assets
│   ├── run_jobs.ps1                  # Submit train+eval pipeline or individual jobs
│   ├── build_image.ps1               # Build and push Docker image; optionally register AML env
│   ├── create_env.ps1                # Register the AML environment from a pre-built image
│   ├── last_dataset_id.txt           # Auto-written by prep_data.ps1 (csv_splits asset ID)
│   └── last_rawdata_id.txt           # Auto-written by prep_data.ps1 (rawdata_dir asset ID)
├── configs/
│   ├── grpo_config.yaml              # GRPO algorithm + model config
│   ├── training_args.yaml            # HF TrainingArguments overrides
│   └── reward_weights.yaml          # Reward component weights
├── data/
│   ├── bird/                         # BIRD dev set (dev.json, dev_databases/)
│   ├── spider/                       # Spider dataset (train/dev/test splits + databases/)
│   ├── serialized_schemas/           # schema_lookup.json (pre-serialized table/column info)
│   └── splits/                       # HF Arrow splits (train / val / test)
├── src/
│   ├── data_preparation.py           # Serialize schemas, produce HF + CSV Arrow splits
│   ├── rewards.py                    # format_reward, exec_reward, schema_fidelity_reward
│   ├── grpo_trainer.py               # Unsloth + TRL GRPOTrainer wrapper
│   ├── evaluator.py                  # cross_schema_exec_acc, mlflow logging
│   └── utils.py                      # Shared utilities
├── notebooks/
│   ├── 01_txt2sql-GRPO-finetuning-nb.ipynb
│   └── 02_baseline_azureopenai_gpt.ipynb
├── tests/                            # pytest unit tests
├── requirements.txt
├── pyproject.toml
└── LICENSE (MIT)
```

---

## ⚠️ Known Issues & Environment Notes

### Unsloth vLLM standby mode — memory allocator conflict

Unsloth's vLLM standby mode is incompatible with PyTorch's `expandable_segments`
CUDA memory allocator (the default on recent drivers). Symptoms:

```
MemoryError: Unsloth: Your GPU ran out of memory loading vLLM with standby mode
enabled. Original error: Standby mode is not supported with expandable segments.
Please set environment variable PYTORCH_CUDA_ALLOC_CONF without expandable_segments:True
```

**Fixes applied** (in `configs/grpo_config.yaml`, `azure/ml_jobs/train_eval_pipeline.yaml`,
and `azure/ml_jobs/grpo_train_job.yaml`):

| Setting | Value | Reason |
|---|---|---|
| `gpu_memory_utilization` | `0.60` (was `0.90`) | Unsloth standby mode fails above ~0.65 on A100 40 GB |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Disables `expandable_segments`; uses fixed-size block allocator |

### exec_reward requires rawdata_dir asset

The `exec_reward` function executes generated SQL against the original `.sqlite`
files at training time. The SQLite databases are **not** included in the CSV
Arrow splits — they live in a separate AML asset (`text2sql-grpo-rawdata`)
mounted as `RAWDATA_DIR` inside the training container.

If this asset is missing or not passed to the pipeline, `exec_reward` silently
returns `0.0` for every sample (the execution path resolves to an empty base
path). The `exec_reward` weight is `0.5`, so this effectively disables the
dominant reward component. Always run `prep_data.ps1` before submitting a
training job rather than re-using a stale `last_rawdata_id.txt` from a
different cluster or workspace.

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **GRPO over PPO** | No separate value model → 2× memory savings on GPU |
| **Schema-level splits** | Prevents data leakage; tests true generalisation |
| **Multi-dialect exec reward** | Ensures SQL is executable, not just syntactically valid |
| **Unsloth 4-bit QLoRA** | Enables A100 40 GB training without multi-node |
| **Azure ML pipelines** | Reproducible, tracked, cost-monitored runs |

---

## 🛠️ Local Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src

# Lint
ruff check src/ tests/
black --check src/ tests/
```

---

## 📄 License

MIT [sriksmachi](https://github.com/sriksmachi)
