"""
baseline_nltosql.py – Azure OpenAI GPT-4.1 NL→SQL baseline on synthetic data.

This script generates SQL from natural language questions using Azure OpenAI and
evaluates execution accuracy with the same metric function used by
``src/evaluator.py`` (``cross_schema_exec_acc``).

Usage
-----
	python data/prep/baseline_nltosql.py \
		--qa-path data/synthetic/synthetic_qa_pairs.json \
		--schema-path data/synthetic/enterprise_schemas.json \
		--output-dir outputs/eval_openai_baseline \
		--azure-openai-endpoint https://<resource>.openai.azure.com/ \
		--azure-openai-deployment gpt-4.1



Authentication
--------------
Uses Microsoft Entra ID (``DefaultAzureCredential``). Ensure your identity has
access to the Azure OpenAI resource.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from loguru import logger
from openai import AzureOpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluator import cross_schema_exec_acc
from utils import build_prompt, extract_sql_from_text

def _quote_ident(name: str) -> str:
    """Quote an SQL identifier (table or column name) for SQLite."""
    escaped = name.replace('"', '""')
    return escaped if escaped.isidentifier() else f'"{escaped}"'

def _create_schema_only_sqlite(schema: dict[str, list[str]], db_path: Path) -> None:
	conn = sqlite3.connect(str(db_path))
	try:
		for table, columns in schema.items():
			safe_table = _quote_ident(table)
			if columns:
				cols_sql = ", ".join(f"{_quote_ident(col)} TEXT" for col in columns)
			else:
				cols_sql = '"_dummy" TEXT'
			conn.execute(f"CREATE TABLE IF NOT EXISTS {safe_table} ({cols_sql})")
		conn.commit()
	finally:
		conn.close()


def _build_db_paths_for_eval(
	examples: list[dict[str, str]],
	schema_lookup: dict[str, dict[str, list[str]]],
) -> tuple[list[str | None], tempfile.TemporaryDirectory[str]]:
	temp_dir = tempfile.TemporaryDirectory(prefix="text2sql_baseline_")
	tmp_path = Path(temp_dir.name)

	db_by_id: dict[str, Path] = {}
	for db_id, schema in schema_lookup.items():
		sqlite_path = tmp_path / f"{db_id}.sqlite"
		_create_schema_only_sqlite(schema, sqlite_path)
		db_by_id[db_id] = sqlite_path

	db_paths = [str(db_by_id.get(ex["db_id"])) if ex["db_id"] in db_by_id else None for ex in examples]
	return db_paths, temp_dir


def _build_client(endpoint: str, api_version: str) -> AzureOpenAI:
	credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
	token_provider = get_bearer_token_provider(
		credential,
		"https://cognitiveservices.azure.com/.default",
	)
	return AzureOpenAI(
		azure_endpoint=endpoint,
		azure_ad_token_provider=token_provider,
		api_version=api_version,
	)


def run_baseline(
	qa_path: str,
	schema_path: str,
	output_dir: str,
	azure_openai_endpoint: str,
	azure_openai_deployment: str,
	api_version: str,
	max_samples: int | None,
	dialects: list[str],
	temperature: float,
	max_output_tokens: int,
	mlflow_tracking_uri: str | None = None,
) -> dict[str, float]:
	with open(qa_path, encoding="utf-8") as fh:
		qa_pairs: list[dict[str, str]] = json.load(fh)
	with open(schema_path, encoding="utf-8") as fh:
		schema_lookup: dict[str, dict[str, list[str]]] = json.load(fh)

	if max_samples is not None:
		qa_pairs = qa_pairs[:max_samples]

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	client = _build_client(endpoint=azure_openai_endpoint, api_version=api_version)

	predictions: list[str] = []
	references: list[str] = []
	records: list[dict[str, str]] = []

	for ex in tqdm(qa_pairs, desc="Azure OpenAI NL→SQL"):
		question = ex["question"]
		db_id = ex["db_id"]
		gold_sql = ex["sql"]
		schema = schema_lookup.get(db_id, {})

		prompt = build_prompt(question, schema)

		try:
			response = client.chat.completions.create(
				model=azure_openai_deployment,
				messages=[{"role": "user", "content": prompt}],
				temperature=temperature,
				max_tokens=max_output_tokens,
			)
			content = response.choices[0].message.content or ""
		except Exception as exc:  # noqa: BLE001
			logger.warning(f"Generation failed for db_id={db_id!r}, question={question!r}: {exc}")
			content = ""

		pred_sql = extract_sql_from_text(content) or content.strip()

		predictions.append(pred_sql)
		references.append(gold_sql)
		records.append(
			{
				"db_id": db_id,
				"question": question,
				"prediction": pred_sql,
				"gold": gold_sql,
			}
		)

	db_paths, temp_dir = _build_db_paths_for_eval(qa_pairs, schema_lookup)
	try:
		metrics = cross_schema_exec_acc(
			predictions=predictions,
			references=references,
			db_paths=db_paths,
			dialects=dialects,
		)
	finally:
		temp_dir.cleanup()

	metrics["n_samples"] = float(len(qa_pairs))

	with open(output_path / "predictions.json", "w", encoding="utf-8") as fh:
		json.dump(records, fh, indent=2, ensure_ascii=False)

	with open(output_path / "metrics.json", "w", encoding="utf-8") as fh:
		json.dump(metrics, fh, indent=2, ensure_ascii=False)

	logger.info(f"Baseline metrics: {metrics}")
	logger.success(f"Saved outputs to {output_path}")

	if mlflow_tracking_uri:
		import mlflow

		mlflow.set_tracking_uri(mlflow_tracking_uri)
		mlflow.set_experiment("text2sql-evaluation")
		with mlflow.start_run(run_name="azure-openai-gpt4.1-baseline"):
			mlflow.log_params(
				{
					"baseline_model": "azure-openai-gpt-4.1",
					"deployment": azure_openai_deployment,
					"api_version": api_version,
					"temperature": temperature,
					"max_output_tokens": max_output_tokens,
				}
			)
			mlflow.log_metrics(metrics)
			mlflow.log_artifact(str(output_path / "predictions.json"))
			mlflow.log_artifact(str(output_path / "metrics.json"))

	return metrics


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Azure OpenAI GPT-4.1 NL→SQL baseline evaluator")
	parser.add_argument(
		"--qa-path",
		default=str(REPO_ROOT / "data" / "synthetic" / "synthetic_qa_pairs.json"),
		help="Path to synthetic QA JSON.",
	)
	parser.add_argument(
		"--schema-path",
		default=str(REPO_ROOT / "data" / "synthetic" / "enterprise_schemas.json"),
		help="Path to schema lookup JSON.",
	)
	parser.add_argument(
		"--output-dir",
		default=str(REPO_ROOT / "outputs" / "eval_openai_baseline"),
		help="Directory to write predictions and metrics.",
	)
	parser.add_argument(
		"--azure-openai-endpoint",
		default=None,
		help="Azure OpenAI endpoint, e.g. https://<resource>.openai.azure.com/.",
	)
	parser.add_argument(
		"--azure-openai-deployment",
		default=None,
		help="Azure OpenAI deployment name for GPT-4.1.",
	)
	parser.add_argument("--api-version", default="2024-10-21")
	parser.add_argument("--max-samples", type=int, default=None)
	parser.add_argument("--dialects", nargs="+", default=["sqlite"])
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--max-output-tokens", type=int, default=512)
	parser.add_argument("--mlflow-tracking-uri", default=None)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()

	endpoint = args.azure_openai_endpoint or ""
	deployment = args.azure_openai_deployment or ""

	if not endpoint:
		raise ValueError("Missing --azure-openai-endpoint")
	if not deployment:
		raise ValueError("Missing --azure-openai-deployment")

	run_baseline(
		qa_path=args.qa_path,
		schema_path=args.schema_path,
		output_dir=args.output_dir,
		azure_openai_endpoint=endpoint,
		azure_openai_deployment=deployment,
		api_version=args.api_version,
		max_samples=args.max_samples,
		dialects=args.dialects,
		temperature=args.temperature,
		max_output_tokens=args.max_output_tokens,
		mlflow_tracking_uri=args.mlflow_tracking_uri,
	)
