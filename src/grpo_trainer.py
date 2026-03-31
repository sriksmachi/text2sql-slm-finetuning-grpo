
"""
grpo_trainer.py – Unsloth + TRL GRPOTrainer wrapper for Text-to-SQL.

Training pipeline
-----------------
1. Load GRPO, training, and reward-weight configs from YAML files.
2. Load the base SLM via Unsloth (4-bit quantised) and attach a LoRA adapter.
3. Build chat-format prompts for every training and validation example.
4. Define a reward wrapper that calls ``combined_reward`` with per-sample
   ``source`` and ``db_id`` metadata so execution is routed to the correct
   SQLite file.
5. Run GRPOTrainer and save the LoRA adapter.

Usage (CLI)
-----------
    python grpo_trainer.py \\
        --config configs/grpo_config.yaml \\
        --training-args configs/training_args.yaml \\
        --reward-weights configs/reward_weights.yaml \\
        --train-data data/splits/train.csv \
        --val-data data/splits/val.csv \
        --output-dir outputs/checkpoint \\
        --mlflow-tracking-uri azureml://...

Config keys used from grpo_config.yaml
---------------------------------------
model.name_or_path            HuggingFace model id or local path
model.load_in_4bit            Whether to load in 4-bit (bool)
model.fast_inference          true/false/auto for Unsloth vLLM path
model.lora_rank               LoRA rank r (also used for lora_alpha)
model.use_gradient_checkpointing  "unsloth" or True/False
grpo.num_generations          Rollouts per prompt
grpo.max_completion_length    Max generated tokens per rollout
grpo.temperature              Sampling temperature
grpo.beta                     KL penalty coefficient
grpo.epsilon                  Policy-ratio clip range
grpo.num_iterations           PPO update steps per batch
tokenizer.max_length          Max sequence length
"""
from __future__ import annotations
import os

# === STRONGER DISABLE FOR vLLM v1 + FULL CUDA GRAPH (critical for your error) ===
os.environ["VLLM_USE_V1"] = "0"                          # Primary: force old vLLM engine
os.environ["VLLM_ENFORCE_EAGER"] = "1"                   # ← NEW: Disable all CUDA graphs entirely (most reliable workaround)
os.environ["UNSLOTH_VLLM_DISABLE_FULL_CUDAGRAPH"] = "1"
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"
os.environ["VLLM_FLASH_ATTN"] = "0"
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Extra safety to reduce graph-related compilation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_CUDAGRAPH_MODE"] = "0"


import argparse
import ast
import importlib.metadata
import json as _json
from contextlib import nullcontext
from numbers import Number
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from loguru import logger

# Import unsloth before transformers to apply all kernel optimisations.
try:
    import unsloth as _unsloth  # noqa: F401
except ImportError:
    pass

from rewards import combined_reward
from utils import (
    configure_mlflow_tracking,
    get_gpu_runtime_profile,
    make_prompt_record,
    resolve_fast_inference,
    resolve_model_dtype,
    setup_logging,
)
from transformers import TrainerCallback


def _force_disable_flashinfer_sampler() -> None:
    """Force-disable FlashInfer sampler before Unsloth/vLLM import."""
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.putenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Reduce CUDA allocator fragmentation so the allocator can reuse reserved
    # but un-allocated segments instead of requesting new blocks from the driver.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _assert_unsloth_runtime_compatibility() -> None:
    """Fail early with an actionable error if the Torch build is incompatible."""
    import torch

    torch_version = torch.__version__.split("+")[0]
    inductor_config = getattr(getattr(torch, "_inductor", None), "config", None)
    if inductor_config is None:
        raise RuntimeError(
            "Incompatible Torch runtime for Unsloth. Expected a Torch build with "
            "torch._inductor.config available. Rebuild the Azure ML image with "
            "a torch 2.4.x build and unsloth[cu118-ampere-torch240]. "
            f"Detected torch=={torch_version}."
        )


def _configure_vllm_runtime() -> None:
    """Adjust vLLM environment flags for the current GPU before import."""
    import torch

    _force_disable_flashinfer_sampler()
    logger.info(
        "Configured VLLM_USE_FLASHINFER_SAMPLER="
        f"{os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')} before model load."
    )

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()

    # FlashInfer sampler requires newer GPUs than V100/T4 class devices.
    if major < 8:
        _force_disable_flashinfer_sampler()
        logger.info(
            "Disabled FlashInfer sampler for GPU compute capability "
            f"{major}.{minor}."
        )

    # Older or newer vLLM builds may warn on this env var; drop it unless a
    # caller explicitly depends on it.
    if "VLLM_ATTENTION_BACKEND" in os.environ:
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)


def _log_runtime_versions() -> None:
    """Log the resolved package/runtime versions for the current job."""
    package_names = [
        "torch",
        "transformers",
        "trl",
        "datasets",
        "unsloth",
        "vllm",
        "triton",
        "xformers",
        "bitsandbytes",
        "mlflow",
        "azureml-mlflow",
        "azure-ai-ml",
    ]

    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"

    try:
        import torch

        versions["cuda_available"] = str(torch.cuda.is_available())
        versions["torch_cuda"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            versions["gpu_name"] = torch.cuda.get_device_name(0)
            versions["gpu_compute_capability"] = f"{major}.{minor}"
    except Exception as exc:
        versions["torch_runtime_probe"] = f"failed: {exc}"

    logger.info(f"Resolved runtime versions: {versions}")

def _load_prompt_records(path: str | Path) -> list[dict[str, Any]]:
    """Load notebook-style prompt records from a CSV split."""
    records_df = pd.read_csv(path)
    required_columns = {"prompt", "solution", "schema", "source", "db_id"}
    missing_columns = required_columns.difference(records_df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {sorted(missing_columns)}"
        )

    records: list[dict[str, Any]] = []
    for row in records_df.to_dict(orient="records"):
        prompt = row["prompt"]
        schema = row["schema"]
        if isinstance(prompt, str):
            prompt = ast.literal_eval(prompt)
        if isinstance(schema, str):
            schema = ast.literal_eval(schema)
        records.append(
            {
                "prompt": prompt,
                "solution": row["solution"],
                "schema": schema,
                "source": row["source"],
                "db_id": row["db_id"],
            }
        )

    return records

class MLflowLoggingCallback(TrainerCallback):
    """Log trainer metrics to the active MLflow run."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled or not logs:
            return control

        metrics: dict[str, float] = {}
        logger.debug(f"Trainer logs at step {state.global_step}: {logs}")
        for key, value in logs.items():
            if isinstance(value, bool):
                metrics[key] = float(value)
            elif isinstance(value, Number):
                metrics[key] = float(value)

        if not metrics:
            return control

        mlflow.log_metrics(metrics, step=state.global_step)
        return control

def _validate_split(path: str | Path, label: str = "") -> None:
    """Fail fast if a CSV split is missing required columns or contains nulls."""
    required_cols = {"prompt", "solution", "schema", "source", "db_id"}
    valid_sources = {"spider", "bird"}
    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[{label}] Dataset at {path} is missing columns: {sorted(missing)}"
        )
    null_cols = [c for c in required_cols if df[c].isna().any()]
    if null_cols:
        raise ValueError(
            f"[{label}] Dataset at {path} has null values in columns: {sorted(null_cols)}"
        )
    bad_sources = set(df["source"].unique()) - valid_sources
    if bad_sources:
        raise ValueError(
            f"[{label}] Dataset at {path} contains unknown source values: {bad_sources}"
        )
    logger.info(f"[{label}] Preflight OK – {len(df)} rows, path={path}")


def train(
    grpo_config_path: str,
    training_args_path: str,
    reward_weights_path: str,
    train_data_dir: str,
    val_data_dir: str,
    output_dir: str,
    lora_dir: str | None = None,
) -> None:
    """Run the GRPO fine-tuning loop.

    Parameters
    ----------
    grpo_config_path:
        Path to ``configs/grpo_config.yaml``.
    training_args_path:
        Path to ``configs/training_args.yaml``.
    reward_weights_path:
        Path to ``configs/reward_weights.yaml``.
    train_data_dir:
        CSV file containing the training split prompt records.
    val_data_dir:
        CSV file containing the validation split prompt records.
    output_dir:
        Directory where HF checkpoints will be saved.
    lora_dir:
        Directory where the final LoRA adapter weights are saved.
        Defaults to ``<output_dir>/grpo_saved_lora`` when not supplied.
    """
    # ── Load configs ───────────────────────────────────────
    grpo_cfg = _load_yaml(grpo_config_path)
    train_cfg = _load_yaml(training_args_path)
    reward_cfg = _load_yaml(reward_weights_path)

    train_cfg["output_dir"] = output_dir

    # Map YAML keys to the weight dict expected by combined_reward
    reward_weights = {
        "format": reward_cfg.get("format_reward", 0.15),
        "exec": reward_cfg.get("exec_reward", 0.5),
        "schema_fidelity": reward_cfg.get("schema_fidelity_reward", 0.25),
        "sql_fence": reward_cfg.get("sql_fence_reward", 0.1),
        "no_sql_penalty": reward_cfg.get("no_sql_penalty", -2.0),
        "unknown_schema_item_penalty": reward_cfg.get("unknown_schema_item_penalty", 0.0),
    }

    lora_rank: int = grpo_cfg["model"]["lora_rank"]

    _log_runtime_versions()

    # ── MLflow ─────────────────────────────────────────────
    mlflow_enabled, mlflow_message = configure_mlflow_tracking(
        train_cfg.get("run_name", "text2sql-grpo"),
    )
    if not mlflow_enabled and mlflow_message:
        logger.warning(f"MLflow disabled: {mlflow_message}")

    # ── Model + tokeniser (Unsloth) ────────────────────────
    # Unsloth's FastLanguageModel wraps HuggingFace and adds kernel
    # optimisations, 4-bit quant, and vLLM fast-inference support.
    try:
        _assert_unsloth_runtime_compatibility()
        _configure_vllm_runtime()
        from unsloth import FastLanguageModel  # type: ignore

        gpu_profile = get_gpu_runtime_profile()
        model_dtype = resolve_model_dtype(
            grpo_cfg["model"].get("torch_dtype"),
            gpu_profile,
        )
        fast_inference = resolve_fast_inference(
            grpo_cfg["model"].get("fast_inference", "auto"),
            gpu_profile,
        )

        logger.info(
            "Using Unsloth runtime settings: dtype={}, fast_inference={}, gpu={}.",
            model_dtype or "default",
            fast_inference,
            gpu_profile.get("device_name") or "cpu",
        )
        logger.info(f"Loading model {grpo_cfg['model']['name_or_path']} with Unsloth...")
        
        # log the final resolved model dtype and fast inference settings
        logger.info(f"Resolved model dtype: {model_dtype}, fast_inference: {fast_inference}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=grpo_cfg["model"]["name_or_path"],
            max_seq_length=grpo_cfg["tokenizer"]["max_length"],
            load_in_4bit=grpo_cfg["model"].get("load_in_4bit", True),
            dtype=model_dtype,
            fast_inference=fast_inference,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=grpo_cfg["model"].get(
                "gpu_memory_utilization", 0.6
            )
        )
        # Attach a LoRA adapter.  Only QKVO projections are trained to stay
        # within GPU memory on a single T4 / A100-40 GB.
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=lora_rank,
            use_gradient_checkpointing=grpo_cfg["model"].get(
                "use_gradient_checkpointing", "unsloth"
            ),
            random_state=grpo_cfg["model"].get("random_state", 42),
        )
    except ImportError:
        logger.warning("unsloth not installed – falling back to plain transformers.")
        raise RuntimeError("Unsloth is required for this training script.")

    ablation = grpo_cfg["grpo"]["ablation"] 
    # safely parse ablation to true or false
    ablation = str(ablation).strip().lower() in {"true", "1", "yes", "on"}
    train_dataset: list[dict[str, Any]] = []
    val_dataset: list[dict[str, Any]] = []
    logger.debug(f"Ablation config: {ablation}")

    # ── Preflight validation ───────────────────────────────
    _validate_split(train_data_dir, label="train")
    _validate_split(val_data_dir, label="val")

    if ablation:
        logger.warning(f"Ablation mode enabled: {ablation}. This will affect rewards.")
        train_dataset = _load_prompt_records(train_data_dir)[:10]
        val_dataset = _load_prompt_records(val_data_dir)[:5]
        # override configuration
        grpo_cfg["grpo"]["num_iterations"] = 1
        train_cfg["num_train_epochs"] = 1
        grpo_cfg["grpo"]["num_generations"] = 2 # > 1 needed for averaging in reward_fn, but keep small for speed
        train_cfg["per_device_train_batch_size"] = 20
        train_cfg["gradient_accumulation_steps"] = 1
        train_cfg["logging_steps"] = 5
    else:
        logger.info(f"Running full experiment. Loading training data from {train_data_dir}...")
        train_dataset = _load_prompt_records(train_data_dir)
        logger.info(f"Sample: {train_dataset[0]}")
        val_dataset = _load_prompt_records(val_data_dir)
    
    # ── Reward wrapper ─────────────────────────────────────
    # GRPOTrainer passes extra dataset columns as keyword arguments to the
    # reward function.  We forward schema, source, and db_id so that
    # combined_reward can route execution to the correct SQLite file.
    def reward_fn(
        completions: list[list[dict[str, str]]],
        prompts: list[list[dict[str, str]]] | None = None,
        **kw: Any,
    ) -> list[float]:
        schemas = kw.get("schema")
        source = kw.get("source")
        db_ids = kw.get("db_id")
        return combined_reward(
            completions,
            prompts=prompts,
            schemas=schemas,
            source=source,
            db_paths=db_ids,
            weights=reward_weights,
        )

    # ── GRPOTrainer ────────────────────────────────────────
    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore

        grpo_train_cfg = GRPOConfig(
            output_dir=output_dir,
            num_generations=grpo_cfg["grpo"]["num_generations"],
            max_completion_length=grpo_cfg["grpo"]["max_completion_length"],
            temperature=grpo_cfg["grpo"]["temperature"],
            beta=grpo_cfg["grpo"]["beta"],
            epsilon=grpo_cfg["grpo"]["epsilon"],
            num_iterations=grpo_cfg["grpo"]["num_iterations"],
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 3),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 6),
            learning_rate=train_cfg.get("learning_rate", 2e-5),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
            weight_decay=train_cfg.get("weight_decay", 0.0),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            num_train_epochs=train_cfg.get("num_train_epochs", 3),
            bf16=train_cfg.get("bf16", False),
            fp16=train_cfg.get("fp16", False),
            logging_steps=train_cfg.get("logging_steps", 5),
            save_steps=train_cfg.get("save_steps", 25),
            save_total_limit=train_cfg.get("save_total_limit", 10),
            eval_steps=train_cfg.get("eval_steps", 25),
            seed=train_cfg.get("seed", 42),
            report_to=train_cfg.get("report_to", "none"),
        )

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            args=grpo_train_cfg,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        trainer.add_callback(MLflowLoggingCallback(enabled=mlflow_enabled))
    except ImportError as exc:
        raise RuntimeError("trl>=0.8.6 is required for GRPOTrainer.") from exc

    # ── Run ────────────────────────────────────────────────
    lora_save_path = str(Path(lora_dir) if lora_dir else Path(output_dir) / "grpo_saved_lora")
    run_context = mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")) if mlflow_enabled else nullcontext()
    with run_context:
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "model": grpo_cfg["model"]["name_or_path"],
                    "lora_rank": lora_rank,
                    "num_generations": grpo_cfg["grpo"]["num_generations"],
                    "beta": grpo_cfg["grpo"]["beta"],
                    "epsilon": grpo_cfg["grpo"]["epsilon"],
                    **{f"reward_weight_{k}": v for k, v in reward_weights.items()},
                    "learning_rate": grpo_train_cfg.learning_rate,
                    "per_device_train_batch_size": grpo_train_cfg.per_device_train_batch_size,
                    "gradient_accumulation_steps": grpo_train_cfg.gradient_accumulation_steps,
                    "num_train_epochs": grpo_train_cfg.num_train_epochs,
                    "bf16": grpo_train_cfg.bf16,
                    "model_dtype": model_dtype or "default",
                        "fast_inference": fast_inference,
                }
                    )
        trainer.train()
        # Save only the LoRA delta weights.
        # save_lora() is Unsloth-specific (fast_inference=True path).
        # Fall back to PEFT's save_pretrained() which also saves adapter-only
        # weights and works on both plain PEFT and Unsloth models.
        if hasattr(model, "save_lora"):
            model.save_lora(lora_save_path)
        else:
            trainer.model.save_pretrained(lora_save_path)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"LoRA adapter saved to {lora_save_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO Text-to-SQL trainer")
    p.add_argument("--config", required=True, help="Path to grpo_config.yaml")
    p.add_argument("--training-args", required=True, help="Path to training_args.yaml")
    p.add_argument("--reward-weights", required=True, help="Path to reward_weights.yaml")
    p.add_argument("--train-data", required=True, help="Training split directory")
    p.add_argument("--val-data", required=True, help="Validation split directory")
    p.add_argument("--output-dir", required=True, help="Directory for checkpoints")
    p.add_argument(
        "--lora-dir",
        default=None,
        help="Directory for final LoRA adapter weights (default: <output-dir>/grpo_saved_lora)",
    )
    p.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Loguru + stdlib logging verbosity (default: DEBUG)",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    setup_logging(args.log_level)
    logger.info(f"VLLM_USE_V1 = {os.environ.get('VLLM_USE_V1')}")
    logger.info(f"VLLM_ENFORCE_EAGER = {os.environ.get('VLLM_ENFORCE_EAGER')}")
    train(
        grpo_config_path=args.config,
        training_args_path=args.training_args,
        reward_weights_path=args.reward_weights,
        train_data_dir=args.train_data,
        val_data_dir=args.val_data,
        output_dir=args.output_dir,
        lora_dir=args.lora_dir,
    )
