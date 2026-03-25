"""
evaluator.py – Baseline and post-training evaluation for Text-to-SQL.

Evaluation pipeline
-------------------
The notebook runs two evaluation passes on the test split:

1. **Baseline** – zero-shot inference with the base model (no LoRA).
   Establishes the pre-training performance floor.

2. **Fine-tuned** – inference with the GRPO-trained LoRA adapter loaded.
   Measures the improvement from GRPO fine-tuning.

Both passes call ``combined_reward`` (format + exec + schema_fidelity) on
every prediction and report per-source mean scores so Spider and BIRD
results can be compared independently.

Key functions
-------------
run_prompt          Generate a SQL completion for a single chat-format prompt
                    using Unsloth's ``model.fast_generate``.
compute_rewards     Batch-score a DataFrame of completions with
                    ``combined_reward``.
evaluate            End-to-end pipeline: load model → generate → score →
                    save CSVs → log to MLflow.

Usage (CLI)
-----------
    python evaluator.py \\
        --model-dir outputs/checkpoint \\
        --lora-path outputs/checkpoint/grpo_saved_lora \\
        --test-data data/splits/test \\
        --output-dir outputs/eval \\
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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
    extract_sql_from_text,
    get_gpu_runtime_profile,
    resolve_fast_inference,
    resolve_model_dtype,
    setup_logging,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def run_prompt(
    prompt: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    fast_inference: bool,
    lora_request: Any | None = None,
) -> str:
    """Generate a SQL completion for a single chat-format prompt.

    Supports both vLLM (``fast_inference=True``) and HuggingFace
    (``fast_inference=False``) backends.  Pass ``lora_request`` for the vLLM
    path to activate a LoRA adapter; for the HF path the adapter must already
    be loaded into the model weights.

    Parameters
    ----------
    prompt:
        Chat-format prompt as a list of ``{"role": …, "content": …}`` dicts.
    model:
        Unsloth ``FastLanguageModel`` instance.
    tokenizer:
        Corresponding tokenizer.
    temperature / top_p / max_new_tokens:
        Sampling parameters.
    fast_inference:
        ``True`` → vLLM backend (``SamplingParams``);  ``False`` → HF backend.
    lora_request:
        vLLM-only: LoRA adapter request from ``model.load_lora()``.  Ignored
        when ``fast_inference=False``.

    Returns
    -------
    The raw generated text (assistant turn content).
    """
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    if fast_inference:
        from vllm import SamplingParams  # type: ignore

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens
        )
        kwargs: dict[str, Any] = {"sampling_params": sampling_params}
        if lora_request is not None:
            kwargs["lora_request"] = lora_request
        return model.fast_generate(text, **kwargs)[0].outputs[0].text
    else:
        tokens = tokenizer(text, return_tensors="pt")
        outputs = model.fast_generate(
            **tokens.to("cuda"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        input_len = tokens["input_ids"].shape[-1]
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def compute_rewards(
    dataset: pd.DataFrame,
    completion_col: str = "completion",
) -> list[float]:
    """Score all rows in *dataset* using ``combined_reward``.

    Wraps each completion in the expected TRL message format and passes
    per-row ``schema``, ``source``, and ``db_id`` columns to the reward
    functions so execution is routed to the correct SQLite file.

    Parameters
    ----------
    dataset:
        DataFrame with at minimum columns:
        ``completion``, ``prompt``, ``schema``, ``source``, ``db_id``.
    completion_col:
        Name of the column containing the model's generated SQL text.

    Returns
    -------
    List of combined reward floats, one per row.
    """
    # Wrap each completion string in the TRL message-list format expected
    # by the reward functions.  SQL is enclosed in a code fence so
    # ``extract_sql`` can locate it.
    completions = [
        [{"role": "assistant", "content": f"```sql\n{row}\n```"}]
        for row in dataset[completion_col]
    ]
    # ensure prompts, schemas, sources, and db_ids are all lists of the same length as completions
    return combined_reward(
        completions,
        prompts=dataset["prompt"].tolist(),
        schemas=dataset["schema"].tolist(),
        source=dataset["source"].tolist(),
        db_paths=dataset["db_id"].tolist(),
    )


# ---------------------------------------------------------------------------
# End-to-end evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate(
    grpo_config_path: str,
    test_data_path: str,
    eval_results: str,
    lora_path: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> dict[str, float]:
    """Run baseline and (optionally) fine-tuned evaluation on the test split.

    Steps
    -----
    1. Load the Unsloth model and tokeniser.
    2. Run zero-shot inference on every test example (baseline).
    3. If ``lora_path`` is provided, re-run inference with the LoRA adapter
       loaded (fine-tuned evaluation).
    4. Score both passes with ``combined_reward``.
    5. Save per-row CSVs and aggregate metrics JSON to ``eval_results``.
    6. Log metrics to MLflow.

    Parameters
    ----------
    grpo_config_path:
        Path to ``configs/grpo_config.yaml``.  The base model is read from
        ``model.name_or_path`` in this file.
    test_data_path:
        Path to the test CSV produced by the data-prep notebook.
    eval_results:
        Directory for saving results.
    lora_path:
        Path to the saved LoRA adapter (``grpo_saved_lora/``).
        If ``None``, only baseline evaluation is performed.
    temperature / top_p / max_tokens:
        Sampling parameters for generation.

    Returns
    -------
    Dict of metric names → values (baseline and fine-tuned combined rewards
    per source, and overall averages).
    """
    from tqdm.auto import tqdm

    logger.debug(
        "evaluate() called with: grpo_config_path={}, "
        "test_data_path={}, eval_results={}, lora_path={}, "
        "temperature={}, top_p={}, max_tokens={}",
        grpo_config_path, test_data_path, eval_results,
        lora_path, temperature, top_p, max_tokens,
    )

    output_path = Path(eval_results)
    output_path.mkdir(parents=True, exist_ok=True)
    grpo_cfg = _load_yaml(grpo_config_path)
    max_seq_length = grpo_cfg["tokenizer"]["max_length"]
    model_name = grpo_cfg["model"]["name_or_path"]
    logger.debug("Loaded grpo_config: model={}, tokenizer.max_length={}", model_name, max_seq_length)

    # ── MLflow ─────────────────────────────────────────────
    import mlflow

    mlflow_enabled, mlflow_message = configure_mlflow_tracking(
        "text2sql-evaluation",
    )
    if not mlflow_enabled and mlflow_message:
        logger.warning(f"MLflow disabled: {mlflow_message}")

    # ── Load model (Unsloth) ───────────────────────────────
    gpu_profile = get_gpu_runtime_profile()
    logger.info("GPU runtime profile: {}", gpu_profile)
    fast_inference = resolve_fast_inference(
        grpo_cfg["model"].get("fast_inference", "auto"),
        gpu_profile,
    )
    model_dtype = resolve_model_dtype(
        grpo_cfg["model"].get("torch_dtype"),
        gpu_profile,
    )
    logger.info(
        "Loading model {} with Unsloth (max_seq_length={}, load_in_4bit=True, "
        "fast_inference={}, dtype={}, gpu={})…",
        model_name, max_seq_length, fast_inference,
        model_dtype or "default",
        gpu_profile.get("device_name") or "cpu",
    )
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=grpo_cfg["model"].get("load_in_4bit", True),
            dtype=model_dtype,
            fast_inference=fast_inference,
        )
        logger.info("Model loaded successfully.")
    except ImportError:
        raise ImportError(
            "Unsloth is required for evaluation. Install with `pip install unsloth`."
        )

    logger.debug("Generation params: fast_inference={}, temperature={}, top_p={}, max_tokens={}",
                 fast_inference, temperature, top_p, max_tokens)

    # ── Load test data ─────────────────────────────────────
    test_df = pd.read_csv(test_data_path)
    
    # prompt was serialized as a Python list of dicts by build_prompt(); read_csv
    # loads it back as a plain string — deserialize it so apply_chat_template works.
    test_df["prompt"] = test_df["prompt"].apply(ast.literal_eval)
    test_df["schema"] = test_df["schema"].apply(ast.literal_eval)
    test_df["db_id"] = test_df["db_id"].astype(str)  # ensure db_id is string for reward routing
    test_df["source"] = test_df["source"].astype(str)  # ensure source is string for grouping
    
    logger.info("Test set loaded: {} rows, columns={}", len(test_df), list(test_df.columns))
    logger.debug("Test set source distribution:\n{}", test_df["source"].value_counts().to_string())
    
    tqdm.pandas(desc="Generating SQL")

    metrics: dict[str, float] = {}

    logger.debug("lora_path={} — {} evaluation pass(es) will run.",
                 lora_path, "2" if lora_path else "1 (baseline only)")

    run_context = mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")) if mlflow_enabled else nullcontext()
    with run_context:
        if mlflow_enabled:
            import mlflow as _mlflow
            _mlflow.log_params({
                "model_name": model_name,
                "max_seq_length": max_seq_length,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "lora_path": lora_path or "none",
                "test_rows": len(test_df),
            })
        # ── Baseline (no LoRA) ─────────────────────────────
        logger.info("Running baseline inference (no LoRA)…")
        baseline_df = test_df.copy()
        baseline_df["completion"] = baseline_df["prompt"].progress_apply(
            lambda p: run_prompt(p, model, tokenizer, temperature, top_p, max_tokens, fast_inference, lora_request=None)
        )
        logger.debug("Baseline inference complete. Scoring {} rows…", len(baseline_df))
        baseline_df["reward"] = compute_rewards(baseline_df)

        baseline_csv = output_path / "baseline_results.csv"
        baseline_df.to_csv(baseline_csv, index=False)
        logger.info(f"Baseline results saved to {baseline_csv}")

        baseline_scores = baseline_df.groupby("source")["reward"].mean().to_dict()
        baseline_avg = baseline_df["reward"].mean()
        for src, score in baseline_scores.items():
            metrics[f"baseline_reward_{src}"] = round(score, 4)
        metrics["baseline_reward_avg"] = round(baseline_avg, 4)

        logger.info(f"Baseline scores per source: {baseline_scores}")
        logger.info(f"Baseline average reward: {baseline_avg:.4f}")

        # ── Fine-tuned (with LoRA) ─────────────────────────
        if lora_path is not None:
            logger.info(f"Running fine-tuned inference with LoRA from {lora_path}…")
            if fast_inference:
                # vLLM path: load_lora() returns a lora_request object used per-call.
                lora_request = model.load_lora(lora_path)
            else:
                # HF path: model is a plain HF model; load the PEFT adapter in-place.
                from peft import PeftModel  # type: ignore

                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
                lora_request = None

            finetuned_df = test_df.copy()
            finetuned_df["completion"] = finetuned_df["prompt"].progress_apply(
                lambda p: run_prompt(p, model, tokenizer, temperature, top_p, max_tokens, fast_inference, lora_request)
            )
            logger.debug("Fine-tuned inference complete. Scoring {} rows…", len(finetuned_df))
            finetuned_df["reward"] = compute_rewards(finetuned_df)

            finetuned_csv = output_path / "finetuned_results.csv"
            finetuned_df.to_csv(finetuned_csv, index=False)
            logger.info(f"Fine-tuned results saved to {finetuned_csv}")

            finetuned_scores = finetuned_df.groupby("source")["reward"].mean().to_dict()
            finetuned_avg = finetuned_df["reward"].mean()
            for src, score in finetuned_scores.items():
                metrics[f"finetuned_reward_{src}"] = round(score, 4)
            metrics["finetuned_reward_avg"] = round(finetuned_avg, 4)

            logger.info(f"Fine-tuned scores per source: {finetuned_scores}")
            logger.info(f"Fine-tuned average reward: {finetuned_avg:.4f}")

        # ── Log & save metrics ──────────────────────────────
        if mlflow_enabled:
            mlflow.log_metrics(metrics)

        logger.debug("Final metrics: {}", metrics)

    logger.info("Evaluation complete. Results written to {}.", eval_results)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-to-SQL evaluator (baseline + fine-tuned)")
    p.add_argument("--config", required=True, help="Path to grpo_config.yaml")
    p.add_argument("--test-data", required=True, help="Path to test CSV")
    p.add_argument("--eval-results", required=True, help="Directory for results")
    p.add_argument(
        "--lora-path",
        default=None,
        help="Path to saved LoRA adapter for fine-tuned evaluation",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=1024)
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
    logger.info("Starting evaluation with args: {}", args)
    evaluate(
        grpo_config_path=args.config,
        test_data_path=args.test_data,
        eval_results=args.eval_results,
        lora_path=args.lora_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
