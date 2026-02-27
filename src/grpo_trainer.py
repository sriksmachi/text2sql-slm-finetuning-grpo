"""
grpo_trainer.py – Unsloth + TRL GRPOTrainer wrapper for Text-to-SQL.

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
from transformers import TrainingArguments

from rewards import combined_reward
from utils import build_prompt, load_schema_lookup


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _build_training_args(cfg: dict[str, Any]) -> TrainingArguments:
    return TrainingArguments(**cfg)


def train(
    grpo_config_path: str,
    training_args_path: str,
    reward_weights_path: str,
    train_data_dir: str,
    val_data_dir: str,
    output_dir: str,
    mlflow_tracking_uri: str | None = None,
) -> None:
    """Run the GRPO training loop."""
    # ── Load configs ───────────────────────────────────────
    grpo_cfg = _load_yaml(grpo_config_path)
    train_cfg = _load_yaml(training_args_path)
    reward_cfg = _load_yaml(reward_weights_path)

    train_cfg["output_dir"] = output_dir

    reward_weights = {
        "format": reward_cfg.get("format_reward", 0.2),
        "exec": reward_cfg.get("exec_reward", 0.5),
        "schema_fidelity": reward_cfg.get("schema_fidelity_reward", 0.3),
    }

    # ── MLflow ─────────────────────────────────────────────
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(train_cfg.get("run_name", "text2sql-grpo"))

    # ── Model + tokeniser (Unsloth) ────────────────────────
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=grpo_cfg["model"]["name_or_path"],
            max_seq_length=grpo_cfg["tokenizer"]["max_length"],
            dtype=None,
            load_in_4bit=grpo_cfg["model"].get("load_in_4bit", True),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=grpo_cfg["model"].get(
                "use_gradient_checkpointing", "unsloth"
            ),
        )
    except ImportError:
        logger.warning("unsloth not installed – falling back to plain transformers.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            grpo_cfg["model"]["name_or_path"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            grpo_cfg["model"]["name_or_path"],
        )

    # ── Dataset ────────────────────────────────────────────
    train_dataset = load_from_disk(train_data_dir)
    val_dataset = load_from_disk(val_data_dir)

    schema_lookup = load_schema_lookup(train_data_dir)

    def _make_prompt(examples: dict[str, list]) -> dict[str, list]:
        prompts = []
        for i, question in enumerate(examples["question"]):
            schema = examples.get("schema", [{}] * len(examples["question"]))[i]
            prompts.append(build_prompt(question, schema))
        return {"prompt": prompts}

    train_dataset = train_dataset.map(_make_prompt, batched=True)
    val_dataset = val_dataset.map(_make_prompt, batched=True)

    # ── Reward wrapper ─────────────────────────────────────
    def reward_fn(
        completions: list[list[dict[str, str]]],
        prompts: list[list[dict[str, str]]] | None = None,
        **kw: Any,
    ) -> list[float]:
        schemas = kw.get("schemas")
        return combined_reward(
            completions,
            prompts=prompts,
            schemas=schemas,
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
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=train_cfg.get("learning_rate", 2e-5),
            num_train_epochs=train_cfg.get("num_train_epochs", 3),
            bf16=train_cfg.get("bf16", True),
            logging_steps=train_cfg.get("logging_steps", 10),
            save_steps=train_cfg.get("save_steps", 100),
            report_to=train_cfg.get("report_to", "mlflow"),
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
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": grpo_cfg["model"]["name_or_path"],
                "num_generations": grpo_cfg["grpo"]["num_generations"],
                "beta": grpo_cfg["grpo"]["beta"],
            }
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO Text-to-SQL trainer")
    p.add_argument("--config", required=True)
    p.add_argument("--training-args", required=True)
    p.add_argument("--reward-weights", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--output-dir", required=True)
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
