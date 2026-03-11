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
        --mlflow-tracking-uri azureml://...
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from rewards import combined_reward
from utils import configure_mlflow_tracking, extract_sql_from_text


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def run_prompt(
    prompt: list[dict[str, str]],
    model: Any,
    tokenizer: Any,
    sampling_params: Any,
    lora_request: Any | None = None,
) -> str:
    """Generate a SQL completion for a single chat-format prompt.

    Uses Unsloth's ``model.fast_generate`` (vLLM-backed) for efficient
    batched inference.  Pass ``lora_request`` to enable a LoRA adapter;
    omit it (or pass ``None``) for zero-shot baseline inference.

    Parameters
    ----------
    prompt:
        Chat-format prompt as a list of ``{"role": …, "content": …}`` dicts
        (the ``"prompt"`` field from a training record).
    model:
        Unsloth ``FastLanguageModel`` instance (with ``fast_inference=True``).
    tokenizer:
        Corresponding tokenizer.
    sampling_params:
        vLLM ``SamplingParams`` instance (temperature, top_p, max_tokens).
    lora_request:
        If provided, the LoRA adapter is activated during generation.
        Obtain via ``model.load_lora(lora_path)``.

    Returns
    -------
    The raw generated text (assistant turn content).
    """
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    kwargs: dict[str, Any] = {"sampling_params": sampling_params}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    output = model.fast_generate(text, **kwargs)[0].outputs[0].text
    return output


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
    model_dir: str,
    test_data_path: str,
    output_dir: str,
    lora_path: str | None = None,
    mlflow_tracking_uri: str | None = None,
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
    5. Save per-row CSVs and aggregate metrics JSON to ``output_dir``.
    6. Log metrics to MLflow.

    Parameters
    ----------
    model_dir:
        HuggingFace model id or local checkpoint directory.
    test_data_path:
        Path to the test CSV produced by the data-prep notebook.
    output_dir:
        Directory for saving results.
    lora_path:
        Path to the saved LoRA adapter (``grpo_saved_lora/``).
        If ``None``, only baseline evaluation is performed.
    mlflow_tracking_uri:
        Optional MLflow tracking URI.
    temperature / top_p / max_tokens:
        Sampling parameters for generation.

    Returns
    -------
    Dict of metric names → values (baseline and fine-tuned combined rewards
    per source, and overall averages).
    """
    from tqdm.auto import tqdm
    from vllm import SamplingParams  # type: ignore

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── MLflow ─────────────────────────────────────────────
    import mlflow

    mlflow_enabled, mlflow_message = configure_mlflow_tracking(
        mlflow_tracking_uri,
        "text2sql-evaluation",
    )
    if not mlflow_enabled and mlflow_message:
        logger.warning(f"MLflow disabled: {mlflow_message}")

    # ── Load model (Unsloth) ───────────────────────────────
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            load_in_4bit=True,
            fast_inference=True,
        )
    except ImportError:
        logger.warning("unsloth not installed – using plain transformers (no fast_generate).")
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # ── Load test data ─────────────────────────────────────
    test_df = pd.read_csv(test_data_path)
    tqdm.pandas(desc="Generating SQL")

    metrics: dict[str, float] = {}

    run_context = mlflow.start_run() if mlflow_enabled else nullcontext()
    with run_context:
        # ── Baseline (no LoRA) ─────────────────────────────
        logger.info("Running baseline inference (no LoRA)…")
        baseline_df = test_df.copy()
        baseline_df["completion"] = baseline_df["prompt"].progress_apply(
            lambda p: run_prompt(p, model, tokenizer, sampling_params, lora_request=None)
        )
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
            lora_request = model.load_lora(lora_path)

            finetuned_df = test_df.copy()
            finetuned_df["completion"] = finetuned_df["prompt"].progress_apply(
                lambda p: run_prompt(p, model, tokenizer, sampling_params, lora_request)
            )
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
        metrics_json = output_path / "metrics.json"
        with open(metrics_json, "w") as fh:
            json.dump(metrics, fh, indent=2)
        if mlflow_enabled:
            mlflow.log_artifact(str(metrics_json))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-to-SQL evaluator (baseline + fine-tuned)")
    p.add_argument("--model-dir", required=True, help="HuggingFace model id or local path")
    p.add_argument("--test-data", required=True, help="Path to test CSV")
    p.add_argument("--output-dir", required=True, help="Directory for results")
    p.add_argument(
        "--lora-path",
        default=None,
        help="Path to saved LoRA adapter for fine-tuned evaluation",
    )
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=1024)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        model_dir=args.model_dir,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        lora_path=args.lora_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
