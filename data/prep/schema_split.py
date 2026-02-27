"""
schema_split.py – Split a dataset by schema (database) to prevent leakage.

The script reads a Hugging Face ``datasets``-compatible directory produced by
``serialize_schemas.py`` and partitions it into train / val / test splits such
that no database schema appears in more than one split.

Usage
-----
    python schema_split.py \\
        --input-dir /path/to/serialized_schemas \\
        --output-dir /path/to/split_data \\
        --train-ratio 0.8 \\
        --val-ratio 0.1
        
    Example:
    python schema_split.py --input-dir data/serialized_schemas --output-dir data/splits --train-ratio 0.7 --val-ratio 0.15"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict
from loguru import logger


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def schema_split(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Partition dataset by database ID (schema-level split)."""
    assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
    assert 0 < val_ratio < 1, "val_ratio must be in (0, 1)"
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1"

    in_path = Path(input_dir)
    out_path = Path(output_dir)

    # Try to load examples from common formats
    examples: list[dict] = []
    for pattern in ("*.jsonl", "train.json", "schema_lookup.json", "data.json"):
        for f in in_path.rglob(pattern):
            print(f"Found {f} matching pattern {pattern}")
            if f.suffix == ".jsonl":
                examples.extend(_load_jsonl(f))
            else:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    examples.extend(data)

    if not examples:
        logger.warning(f"No examples found in {in_path}. Creating empty splits.")
        for split in ("train", "val", "test"):
            Dataset.from_list([]).save_to_disk(str(out_path / split))
        return

    # Group by db_id
    by_db: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        db_id = ex.get("db_id") or ex.get("database_id") or "unknown"
        by_db[db_id].append(ex)

    db_ids = sorted(by_db.keys())
    rng = random.Random(seed)
    rng.shuffle(db_ids)

    n = len(db_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_dbs = db_ids[:n_train]
    val_dbs = db_ids[n_train : n_train + n_val]
    test_dbs = db_ids[n_train + n_val :]

    splits = {
        "train": [ex for db in train_dbs for ex in by_db[db]],
        "val": [ex for db in val_dbs for ex in by_db[db]],
        "test": [ex for db in test_dbs for ex in by_db[db]],
    }

    for split_name, split_examples in splits.items():
        split_path = out_path / split_name
        Dataset.from_list(split_examples).save_to_disk(str(split_path))
        logger.info(f"{split_name}: {len(split_examples)} examples ({len(set(ex.get('db_id','') for ex in split_examples))} DBs)")

    logger.success(f"Schema splits saved to {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Schema-level dataset split")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    schema_split(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
