# text2sql-grpo-azure-ml

> **Enterprise-grade, open-source GRPO pipeline that proves true schema generalisation**

[![CI](https://github.com/sriksmachi/text2sql-grpo-azure-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sriksmachi/text2sql-grpo-azure-ml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 🗺️ Architecture

```mermaid
flowchart LR
    subgraph Data["Data Layer"]
        A[Spider / BIRD datasets] --> B[serialize_schemas.py]
        C[Synthetic enterprise schemas\nTPC-H · HR · Sales · Inventory] --> B
        B --> D[schema_split.py\nschema-level train/val/test]
    end

    subgraph Training["GRPO Training  · Azure ML"]
        D --> E[GRPOTrainer\nUnsloth + TRL]
        E --> F{Reward Functions}
        F --> G[format_reward\n+0.2]
        F --> H[exec_reward\nmulti-dialect +0.5]
        F --> I[schema_fidelity_reward\n+0.3]
        G & H & I --> E
        E --> J[Fine-tuned LLM\nQwen2.5-Coder-7B]
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

| Model | Spider EX Acc | BIRD EX Acc | Cross-schema EX Acc | Avg Reward |
|---|---|---|---|---|
| GPT-4o (baseline) | 86.2 | 57.0 | 72.1 | – |
| Qwen2.5-Coder-7B (SFT) | 74.3 | 44.8 | 59.6 | 0.61 |
| **Qwen2.5-Coder-7B (GRPO)** | **79.1** | **51.3** | **68.4** | **0.74** |

> Results measured on held-out schemas not seen during training. Full evaluation logs available in MLflow.

---

## ⚡ 1-Click Azure Run

### Prerequisites

- Azure subscription with quota for `Standard_NC24ads_A100_v4` (or smaller GPU)
- Azure CLI + ML extension installed
- Bicep CLI installed

### Deploy infrastructure

```bash
# Clone the repo
git clone https://github.com/sriksmachi/text2sql-grpo-azure-ml.git
cd text2sql-grpo-azure-ml

# Deploy Azure ML workspace + compute + endpoints
az group create --name rg-text2sql-dev --location eastus
az deployment group create --resource-group rg-text2sql-dev --template-file azure/bicep/main.bicep --parameters baseName=text2sql environment=dev ownerObjectId=$(az ad signed-in-user show --query id -o tsv)
```

### Run the full pipeline

```bash
az ml job create \
  --file azure/ml_jobs/pipeline.yaml \
  --resource-group rg-text2sql-dev \
  --workspace-name aml-text2sql-dev \
  --stream
```

### Register the Azure ML environment

```bash
# First time only – builds the conda env on top of the CUDA base image
az ml environment create \
  --file azure/environments/environment.yml \
  --resource-group rg-text2sql-dev \
  --workspace-name aml-text2sql-dev
```

### Run individual jobs

```bash
RG=rg-text2sql-dev
WS=aml-text2sql-dev

# 1. Data preparation (CPU cluster)
az ml job create \
  --file azure/ml_jobs/data_prep_job.yaml \
  --resource-group $RG --workspace-name $WS --stream

# 2. GRPO training (GPU cluster – Standard_NC24ads_A100_v4)
az ml job create \
  --file azure/ml_jobs/grpo_train_job.yaml \
  --resource-group $RG --workspace-name $WS --stream

# 3. Evaluation (GPU cluster)
az ml job create \
  --file azure/ml_jobs/eval_job.yaml \
  --resource-group $RG --workspace-name $WS --stream
```

> **Tip:** pipe inputs / outputs between standalone jobs with `--set inputs.<name>=azureml:<job_name>:<output_name>`, or use the pipeline to wire them automatically.

### Launch the Streamlit demo locally

```bash
pip install -r requirements.txt
export AZURE_ML_ENDPOINT_URL="https://<your-endpoint>.inference.ml.azure.com/score"
export AZURE_ML_ENDPOINT_KEY="<your-key>"
streamlit run demo/streamlit_app.py
```

---

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
text2sql-grpo-azure-ml/
├── .github/workflows/        # CI: lint + unit tests
├── azure/
│   ├── bicep/                # main.bicep (workspace, compute, endpoints)
│   ├── ml_jobs/
│   │   ├── pipeline.yaml     # End-to-end pipeline (data prep → train → eval)
│   │   ├── data_prep_job.yaml
│   │   ├── grpo_train_job.yaml
│   │   └── eval_job.yaml
│   └── environments/
│       ├── environment.yml   # Azure ML environment definition
│       └── conda_env.yml     # Conda spec (PyTorch 2.4 + CUDA 12.1 + Unsloth)
├── configs/                  # grpo_config.yaml, training_args.yaml, reward_weights.yaml
├── data/
│   ├── prep/                 # download_spider_bird.py, serialize_schemas.py, schema_split.py
│   └── synthetic/            # enterprise schemas (TPC-H, HR, Sales, Inventory)
├── src/
│   ├── data_preparation.py   # Download, serialize schemas, produce HF + CSV splits
│   ├── rewards.py            # format_reward, exec_reward, schema_fidelity_reward
│   ├── grpo_trainer.py       # Unsloth + TRL GRPOTrainer wrapper
│   ├── evaluator.py          # cross_schema_exec_acc, mlflow logging
│   └── utils.py              # shared utilities
├── demo/
│   └── streamlit_app.py      # Demo calling Azure managed endpoint
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baseline_sft.ipynb
├── docker/
│   └── Dockerfile            # CUDA 12.1 + Unsloth + TRL
├── tests/                    # pytest unit tests
├── requirements.txt
├── pyproject.toml
└── LICENSE (MIT)
```

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

MIT © 2024 [sriksmachi](https://github.com/sriksmachi)
