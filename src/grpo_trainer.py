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
        --train-data data/splits/train \\
        --val-data data/splits/val \\
        --output-dir outputs/checkpoint \\
        --mlflow-tracking-uri azureml://...

Config keys used from grpo_config.yaml
---------------------------------------
model.name_or_path            HuggingFace model id or local path
model.load_in_4bit            Whether to load in 4-bit (bool)
model.lora_rank               LoRA rank r (also used for lora_alpha)
model.use_gradient_checkpointing  "unsloth" or True/False
grpo.num_generations          Rollouts per prompt
grpo.temperature              Sampling temperature
grpo.beta                     KL penalty coefficient
grpo.epsilon                  Policy-ratio clip range
grpo.num_iterations           PPO update steps per batch
tokenizer.max_length          Max sequence length
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import mlflow
import yaml
from datasets import load_from_disk
from loguru import logger

from rewards import combined_reward
from utils import make_prompt_record, load_schema_lookup


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def train(
    grpo_config_path: str,
    training_args_path: str,
    reward_weights_path: str,
    train_data_dir: str,
    val_data_dir: str,
    output_dir: str,
    mlflow_tracking_uri: str | None = None,
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
        Directory containing the training split (HuggingFace ``datasets``
        format or CSV; must have columns question/SQL/schema/source/db_id).
    val_data_dir:
        Directory containing the validation split.
    output_dir:
        Directory where the LoRA adapter will be saved.
    mlflow_tracking_uri:
        Optional MLflow tracking URI (e.g. an Azure ML workspace URI).
    """
    # ── Load configs ───────────────────────────────────────
    grpo_cfg = _load_yaml(grpo_config_path)
    train_cfg = _load_yaml(training_args_path)
    reward_cfg = _load_yaml(reward_weights_path)

    train_cfg["output_dir"] = output_dir

    # Map YAML keys to the weight dict expected by combined_reward
    reward_weights = {
        "format": reward_cfg.get("format_reward", 0.2),
        "exec": reward_cfg.get("exec_reward", 0.5),
        "schema_fidelity": reward_cfg.get("schema_fidelity_reward", 0.3),
    }

    lora_rank: int = grpo_cfg["model"]["lora_rank"]

    # ── MLflow ─────────────────────────────────────────────
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(train_cfg.get("run_name", "text2sql-grpo"))

    # ── Model + tokeniser (Unsloth) ────────────────────────
    # Unsloth's FastLanguageModel wraps HuggingFace and adds kernel
    # optimisations, 4-bit quant, and vLLM fast-inference support.
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=grpo_cfg["model"]["name_or_path"],
            max_seq_length=grpo_cfg["tokenizer"]["max_length"],
            load_in_4bit=grpo_cfg["model"].get("load_in_4bit", True),
            fast_inference=True,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.9,
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
            random_state=3407,
        )
    except ImportError:
        logger.warning("unsloth not installed – falling back to plain transformers.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(grpo_cfg["model"]["name_or_path"])
        tokenizer = AutoTokenizer.from_pretrained(grpo_cfg["model"]["name_or_path"])

    # ── Dataset ────────────────────────────────────────────
    train_dataset = load_from_disk(train_data_dir)
    val_dataset = load_from_disk(val_data_dir)

    # Convert each row to a training record with a chat-format prompt.
    # The record keys (prompt/solution/schema/source/db_id) are used by both
    # GRPOTrainer (prompt) and the reward wrapper (schema/source/db_id).
    def _to_record(examples: dict[str, list]) -> dict[str, list]:
        records = [
            make_prompt_record(
                question=examples["question"][i],
                schema=examples.get("schema", [{}] * len(examples["question"]))[i],
                answer=examples["SQL"][i],
                source=examples["source"][i],
                db_id=examples["db_id"][i],
            )
            for i in range(len(examples["question"]))
        ]
        # Transpose list-of-dicts → dict-of-lists for HF datasets
        return {k: [r[k] for r in records] for k in records[0]}

    train_dataset = train_dataset.map(_to_record, batched=True)
    val_dataset = val_dataset.map(_to_record, batched=True)

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
            max_new_tokens=grpo_cfg["grpo"]["max_new_tokens"],
            temperature=grpo_cfg["grpo"]["temperature"],
            beta=grpo_cfg["grpo"]["beta"],
            epsilon=grpo_cfg["grpo"]["epsilon"],
            num_iterations=grpo_cfg["grpo"]["num_iterations"],
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 3),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 6),
            learning_rate=train_cfg.get("learning_rate", 2e-5),
            num_train_epochs=train_cfg.get("num_train_epochs", 2),
            bf16=train_cfg.get("bf16", False),
            logging_steps=train_cfg.get("logging_steps", 10),
            save_steps=train_cfg.get("save_steps", 10),
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_fn],
            args=grpo_train_cfg,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
    except ImportError as exc:
        raise RuntimeError("trl>=0.8.6 is required for GRPOTrainer.") from exc

    # ── Run ────────────────────────────────────────────────
    lora_save_path = str(Path(output_dir) / "grpo_saved_lora")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": grpo_cfg["model"]["name_or_path"],
                "lora_rank": lora_rank,
                "num_generations": grpo_cfg["grpo"]["num_generations"],
                "beta": grpo_cfg["grpo"]["beta"],
            }
        )
        trainer.train()
        # Save only the LoRA delta weights (much smaller than the full model)
        model.save_lora(lora_save_path)
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
    p.add_argument("--mlflow-tracking-uri", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        grpo_config_path=args.config,
        training_args_path=args.training_args,
        reward_weights_path=args.reward_weights,
        train_data_dir=args.train_data,
        val_data_dir=args.val_data,
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
