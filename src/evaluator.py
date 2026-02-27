"""
evaluator.py – Cross-schema execution accuracy evaluation with MLflow logging.

Usage (CLI)
-----------
    python evaluator.py \\
        --model-dir outputs/checkpoint \\
        --test-data data/splits/test \\
        --output-dir outputs/eval \\
        --mlflow-tracking-uri azureml://... \\
        --dialects sqlite postgresql
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import sqlglot
from loguru import logger

from utils import build_prompt, extract_sql_from_text


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------


def _execute_sql(sql: str, db_path: str | None = None, dialect: str = "sqlite") -> bool:
    """Return True if *sql* (transpiled to SQLite) executes without error."""
    try:
        transpiled = sqlglot.transpile(sql, read=dialect, write="sqlite")[0]
    except sqlglot.errors.ParseError:
        return False

    conn = sqlite3.connect(db_path or ":memory:")
    try:
        conn.execute(transpiled)
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cross-schema execution accuracy
# ---------------------------------------------------------------------------


def cross_schema_exec_acc(
    predictions: list[str],
    references: list[str],
    db_paths: list[str | None] | None = None,
    dialects: list[str] | None = None,
) -> dict[str, float]:
    """Compute execution accuracy across schemas and dialects.

    Parameters
    ----------
    predictions:
        Predicted SQL strings.
    references:
        Gold SQL strings.
    db_paths:
        Per-sample SQLite DB paths (or ``None`` for in-memory).
    dialects:
        List of dialects to evaluate against. Results are reported per dialect
        and as an average.

    Returns
    -------
    Dict mapping ``"exec_acc_<dialect>"`` and ``"exec_acc_avg"`` to floats.
    """
    dialects = dialects or ["sqlite"]
    n = len(predictions)
    paths = db_paths if db_paths is not None else [None] * n

    results: dict[str, float] = {}
    all_accs: list[float] = []

    for dialect in dialects:
        correct = 0
        for pred, ref, db_path in zip(predictions, references, paths):
            pred_ok = _execute_sql(pred, db_path, dialect)
            ref_ok = _execute_sql(ref, db_path, dialect)
            # Credit the prediction only if the gold also executes
            if ref_ok and pred_ok:
                correct += 1
            elif not ref_ok:
                # Gold doesn't execute either (schema issue) – skip
                n -= 1

        acc = correct / n if n > 0 else 0.0
        results[f"exec_acc_{dialect}"] = acc
        all_accs.append(acc)

    results["exec_acc_avg"] = sum(all_accs) / len(all_accs) if all_accs else 0.0
    return results


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate(
    model_dir: str,
    test_data_dir: str,
    output_dir: str,
    mlflow_tracking_uri: str | None = None,
    dialects: list[str] | None = None,
) -> dict[str, float]:
    """End-to-end evaluation: generate SQL → measure exec accuracy → log to MLflow."""
    import mlflow
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dialects = dialects or ["sqlite"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── MLflow ─────────────────────────────────────────────
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("text2sql-evaluation")

    # ── Load model ─────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        logger.warning("unsloth not installed – falling back to plain transformers.")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    model.eval()

    # ── Load test data ─────────────────────────────────────
    dataset = load_from_disk(test_data_dir)
    questions = dataset["question"]
    gold_sqls = dataset["sql"]
    schemas = dataset.get("schema", [{}] * len(questions))
    db_paths = dataset.get("db_path", [None] * len(questions))

    # ── Generate predictions ───────────────────────────────
    predictions: list[str] = []
    for question, schema in zip(questions, schemas):
        prompt = build_prompt(question, schema)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = extract_sql_from_text(decoded) or ""
        predictions.append(sql)

    # ── Compute metrics ────────────────────────────────────
    metrics = cross_schema_exec_acc(
        predictions=predictions,
        references=gold_sqls,
        db_paths=db_paths,
        dialects=dialects,
    )

    logger.info(f"Evaluation results: {metrics}")

    # ── Save predictions ───────────────────────────────────
    records = [
        {"question": q, "prediction": p, "gold": g}
        for q, p, g in zip(questions, predictions, gold_sqls)
    ]
    with open(output_path / "predictions.json", "w") as fh:
        json.dump(records, fh, indent=2)

    with open(output_path / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ── Log to MLflow ──────────────────────────────────────
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_path / "predictions.json"))
        mlflow.log_artifact(str(output_path / "metrics.json"))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-to-SQL evaluator")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--test-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--dialects", nargs="+", default=["sqlite"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        model_dir=args.model_dir,
        test_data_dir=args.test_data,
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        dialects=args.dialects,
    )
