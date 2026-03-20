"""
data_preparation.py – End-to-end data pipeline for the Text-to-SQL GRPO project.

Pipeline overview
-----------------
1. **Download** – Fetch the Spider and BIRD dataset archives from their
   canonical URLs and extract them under ``rawdata_dir``.
2. **Extract BIRD databases** – BIRD ships its ``.sqlite`` files inside a
   nested ``dev_databases.zip``; this step unpacks that inner archive so
   the reward functions can reach individual database files.
3. **Serialize schemas** – Walk every ``.db`` / ``.sqlite`` file plus
   ``dev_tables.json`` (BIRD) and write a unified ``schema_lookup.json``
   under ``serialized_dir``.
4. **Load examples** – Read Spider ``dev.json`` and BIRD ``dev.json`` +
   ``dev_tied_append.json``.  Normalise column names, drop noise columns,
   and tag each row with ``source`` (``"spider"`` or ``"bird"``).
5. **Merge** – Inner-join examples with the serialized schema DataFrame on
   ``db_id``.
6. **Stratified split** – Sample ``sample_size`` database IDs, split them
   within each source into train/val/test so both sources appear in every
   partition.
7. **Build prompt datasets** – Apply ``make_prompt_record`` to each row,
   producing lists of training-ready dicts.
8. **Save splits** – Persist each list as a ``.csv`` file.

Typical usage
-------------
From Python::

    from src.data_preparation import prepare

    train_ds, val_ds, test_ds = prepare(
        rawdata_dir="data/raw",
        serialized_dir="data/serialized_schemas",
        splits_dir="data/splits",
        sample_size=200,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )

From the command line::

    python src/data_preparation.py \\
        --rawdata-dir /data/rawdata \\
        --serialized-dir /data/serialized_schemas \\
        --splits-dir /data/splits \\
        --sample-size 400 \\
        --train-ratio 0.7 \\
        --val-ratio 0.15 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from utils import make_prompt_record

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

SPIDER_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
)
BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

# ---------------------------------------------------------------------------
# Step 1 – Download datasets
# ---------------------------------------------------------------------------


def _stream_download(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest*, showing a progress bar."""
    logger.info(f"Downloading {url} → {dest}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))


def _extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip archive to *dest*."""
    logger.info(f"Extracting {archive} → {dest}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


def download_datasets(rawdata_dir: str) -> None:
    """Download and extract Spider and BIRD datasets into *rawdata_dir*.

    Each dataset is placed in its own subdirectory:

    * ``<rawdata_dir>/spider/`` – Spider archive contents
    * ``<rawdata_dir>/bird/``   – BIRD archive contents

    Already-extracted directories are skipped so reruns are safe.

    Parameters
    ----------
    rawdata_dir:
        Destination root.  Will be created if it does not exist.
    """
    out = Path(rawdata_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_downloads"
    tmp.mkdir(exist_ok=True)

    for name, url in (("spider", SPIDER_URL), ("bird", BIRD_URL)):
        extract_dir = out / name
        if extract_dir.exists():
            logger.info(f"{name} already extracted at {extract_dir}, skipping.")
            continue
        archive = tmp / f"{name}.zip"
        _stream_download(url, archive)
        _extract_zip(archive, extract_dir)
        archive.unlink(missing_ok=True)

    shutil.rmtree(tmp, ignore_errors=True)
    logger.success(f"Datasets ready in {out}")


# ---------------------------------------------------------------------------
# Step 2 – Extract BIRD's nested database archive
# ---------------------------------------------------------------------------


def extract_bird_databases(rawdata_dir: str) -> None:
    """Unpack the nested ``dev_databases.zip`` shipped inside the BIRD archive.

    BIRD's top-level zip contains a ``dev_databases.zip`` inside
    ``dev_20240627/``.  This function extracts that inner archive so every
    ``*.sqlite`` file becomes directly accessible at::

        <rawdata_dir>/bird/dev_databases/<db_id>/<db_id>.sqlite

    Safe to call multiple times; skips extraction if the directory already
    exists.

    Parameters
    ----------
    rawdata_dir:
        Same root used by :func:`download_datasets`.
    """
    bird_dir = Path(rawdata_dir) / "bird"
    inner_zip = bird_dir / "dev_20240627" / "dev_databases.zip"
    extract_target = bird_dir / "dev_databases"

    if extract_target.exists():
        logger.info(f"BIRD databases already extracted at {extract_target}, skipping.")
        return

    if not inner_zip.exists():
        logger.warning(
            f"BIRD inner archive not found at {inner_zip}. "
            "Run download_datasets() first."
        )
        return

    logger.info(f"Extracting BIRD databases: {inner_zip} → {bird_dir}")
    _extract_zip(inner_zip, bird_dir)
    logger.success(f"BIRD databases ready at {extract_target}")


# ---------------------------------------------------------------------------
# Step 3 – Schema serialisation helpers
# ---------------------------------------------------------------------------


def _is_probably_sqlite_file(p: Path) -> bool:
    """Return ``True`` if *p* looks like a SQLite database file.

    Skips macOS resource-fork files (``._*``) and ``__MACOSX`` artifacts,
    then reads the first 16 bytes to verify the SQLite magic header.

    Parameters
    ----------
    p:
        Candidate file path.
    """
    if p.name.startswith("._") or "__MACOSX" in p.parts:
        return False
    if not p.is_file():
        return False
    try:
        with p.open("rb") as fh:
            return fh.read(16).startswith(b"SQLite format 3")
    except OSError:
        return False


def _schema_from_sqlite(db_path: Path) -> dict[str, list[str]]:
    """Read table → [columns] mapping from a SQLite file via PRAGMA.

    All table names and column names are lower-cased for consistency.

    Parameters
    ----------
    db_path:
        Path to the ``.sqlite`` / ``.db`` file.

    Returns
    -------
    dict mapping each table name to a list of its column names.
    """
    schema: dict[str, list[str]] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            col_cursor = conn.execute(f"PRAGMA table_info(`{table}`)")
            schema[table.lower()] = [row[1].lower() for row in col_cursor.fetchall()]
    finally:
        conn.close()
    return schema


def _schema_from_tables_json(tables_json_path: Path) -> dict[str, dict[str, list[str]]]:
    """Parse a Spider / BIRD ``tables.json`` (or ``dev_tables.json``) file.

    The BIRD format stores column names as ``column_names_original``, a list
    of ``[table_index, column_name]`` pairs where ``table_index == -1``
    represents the wildcard ``*`` column (skipped).

    Parameters
    ----------
    tables_json_path:
        Path to the JSON file.

    Returns
    -------
    dict mapping each ``db_id`` to its table→columns schema dict.
    """
    with open(tables_json_path, encoding="utf-8") as fh:
        tables_data = json.load(fh)

    result: dict[str, dict[str, list[str]]] = {}
    for db in tables_data:
        db_id = db["db_id"]
        col_names = db.get("column_names_original", db.get("column_names", []))
        table_names = db.get("table_names_original", db.get("table_names", []))

        schema: dict[str, list[str]] = {t.lower(): [] for t in table_names}
        for table_idx, col_name in col_names:
            if table_idx < 0:
                continue  # wildcard column (*), skip
            tname = table_names[table_idx].lower()
            schema[tname].append(col_name.lower())
        result[db_id] = schema
    return result


def serialize_schemas(input_dir: str, output_dir: str) -> None:
    """Extract schemas from SQLite files and ``dev_tables.json``, write ``schema_lookup.json``.

    Two passes:

    1. Any ``dev_tables.json`` found anywhere under *input_dir* is parsed
       first (BIRD uses this format).
    2. Every ``.db`` / ``.sqlite`` file whose ``db_id`` is not already covered
       by step 1 is read directly via PRAGMA queries (Spider uses this).

    The output JSON is a **list** of objects::

        [{"db_id": "academic", "schema": {"author": ["aid", "name"], …}}, …]

    This list format is what :func:`load_schemas` and
    :func:`~utils.load_schema_lookup` expect.

    Parameters
    ----------
    input_dir:
        Root of the raw data directory (e.g. ``rawdata/``).
    output_dir:
        Where to write ``schema_lookup.json``.  Created if it does not exist.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_schemas: dict[str, dict[str, list[str]]] = {}

    # Pass 1: dev_tables.json (BIRD format, preferred when available)
    logger.info(f"Parsing dev_tables.json files under {in_path}…")
    for tables_json in in_path.rglob("dev_tables.json"):
        logger.debug(f"  {tables_json}")
        all_schemas.update(_schema_from_tables_json(tables_json))

    # Pass 2: SQLite files for any db_id not already covered
    sqlite_exts = {".db", ".sqlite", ".sqlite3", ".sqlllite"}
    db_files = [
        p for p in in_path.rglob("*")
        if _is_probably_sqlite_file(p) and p.suffix.lower() in sqlite_exts
    ]
    logger.info(f"Found {len(db_files)} SQLite file(s); scanning new db_ids…")
    for db_file in tqdm(db_files, desc="SQLite schemas"):
        db_id = db_file.stem
        if db_id in all_schemas:
            continue  # already covered by tables.json
        try:
            all_schemas[db_id] = _schema_from_sqlite(db_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  Could not read {db_file}: {exc}")

    # Write list-of-objects, sorted by db_id for determinism
    out_file = out_path / "schema_lookup.json"
    payload = [
        {"db_id": db_id, "schema": schema}
        for db_id, schema in sorted(all_schemas.items())
    ]
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.success(f"Serialized {len(all_schemas)} schemas → {out_file}")


# ---------------------------------------------------------------------------
# Step 3b – Load serialized schemas
# ---------------------------------------------------------------------------


def load_schemas(serialized_dir: str) -> pd.DataFrame:
    """Load ``schema_lookup.json`` into a DataFrame with columns ``db_id``, ``schema``.

    Parameters
    ----------
    serialized_dir:
        Directory that contains ``schema_lookup.json`` (output of
        :func:`serialize_schemas`).

    Returns
    -------
    DataFrame with one row per database: ``db_id`` (str) and ``schema``
    (dict[str, list[str]]).
    """
    schema_file = Path(serialized_dir) / "schema_lookup.json"
    with open(schema_file, encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        # list-of-objects: [{"db_id": …, "schema": …}, …]
        schemas_df = pd.DataFrame(data)
    else:
        # plain dict: {"db_id": schema_dict, …}
        schemas_df = pd.DataFrame(
            [{"db_id": k, "schema": v} for k, v in data.items()]
        )

    logger.info(f"Number of schemas: {len(schemas_df)}")
    return schemas_df


# ---------------------------------------------------------------------------
# Step 4 – Load examples
# ---------------------------------------------------------------------------


def _ensure_question(q: str) -> str:
    """Append a ``?`` if *q* does not already end with one."""
    q = q.strip()
    return q if q.endswith("?") else q + "?"


def load_spider_examples(rawdata_dir: str) -> pd.DataFrame:
    """Load Spider dev examples and normalise column names.

    Reads ``<rawdata_dir>/spider/spider_data/dev.json``, tags rows with
    ``source="spider"``, drops noisy columns (tokenised fields, SQL parse
    tree), renames ``query`` → ``SQL``, and deduplicates.

    Parameters
    ----------
    rawdata_dir:
        Root of the downloaded data (e.g. ``/kaggle/working/rawdata``).

    Returns
    -------
    DataFrame with columns: ``db_id``, ``question``, ``SQL``, ``source``.
    """
    json_path = Path(rawdata_dir) / "spider" / "spider_data" / "dev.json"
    df = pd.read_json(str(json_path))
    df["source"] = "spider"

    # Drop noise columns that are present in Spider but not in BIRD
    drop_cols = [c for c in ("query_toks_no_value", "query_toks", "question_toks", "sql") if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    df.rename(columns={"query": "SQL"}, inplace=True)
    df["question"] = df["question"].apply(_ensure_question)

    n_dupes = df.duplicated().sum()
    logger.info(f"Duplicates check: {n_dupes}")
    if n_dupes:
        logger.info(f"Dropping {n_dupes} duplicate Spider rows.")
        df.drop_duplicates(inplace=True)

    df.reset_index(drop=True, inplace=True)
    logger.info(f"Length of samples: {df.count().to_dict()}")
    logger.info(f"Spider examples loaded: {len(df)} rows.")
    return df


def load_bird_examples(rawdata_dir: str) -> pd.DataFrame:
    """Load BIRD dev examples and normalise column names.

    Reads and concatenates:

    * ``<rawdata_dir>/bird/dev_20240627/dev.json``
    * ``<rawdata_dir>/bird/dev_20240627/dev_tied_append.json``

    Tags rows with ``source="bird"``, drops BIRD-specific columns
    (``difficulty``, ``evidence``, ``question_id``), and renames
    ``SQL`` if needed.

    Parameters
    ----------
    rawdata_dir:
        Root of the downloaded data.

    Returns
    -------
    DataFrame with columns: ``db_id``, ``question``, ``SQL``, ``source``.
    """
    bird_dev_dir = Path(rawdata_dir) / "bird" / "dev_20240627"

    parts: list[pd.DataFrame] = []
    for fname in ("dev.json", "dev_tied_append.json"):
        fpath = bird_dev_dir / fname
        if not fpath.exists():
            logger.warning(f"BIRD file not found, skipping: {fpath}")
            continue
        part = pd.read_json(str(fpath))
        parts.append(part)

    if not parts:
        raise FileNotFoundError(f"No BIRD dev JSON files found under {bird_dev_dir}")

    df = pd.concat(parts, axis=0, ignore_index=True)
    df["source"] = "bird"

    # Drop BIRD-specific columns not present in Spider
    drop_cols = [c for c in ("difficulty", "evidence", "question_id") if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # BIRD uses "SQL" already; rename if it came through differently
    if "query" in df.columns and "SQL" not in df.columns:
        df.rename(columns={"query": "SQL"}, inplace=True)

    df["question"] = df["question"].apply(_ensure_question)

    n_dupes = df.duplicated().sum()
    logger.info(f"Duplicates check: {n_dupes}")
    if n_dupes:
        logger.info(f"Dropping {n_dupes} duplicate BIRD rows.")
        df.drop_duplicates(inplace=True)

    df.reset_index(drop=True, inplace=True)
    logger.info(f"Length of samples: {df.count().to_dict()}")
    logger.info(f"BIRD examples loaded: {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Step 5 – Merge
# ---------------------------------------------------------------------------


def merge_examples_with_schemas(
    schemas_ds: pd.DataFrame,
    examples_ds: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join examples with schemas on ``db_id``.

    Rows whose ``db_id`` does not appear in the schema lookup are silently
    dropped (this can happen when a database file is missing or corrupt).

    Parameters
    ----------
    schemas_ds:
        DataFrame returned by :func:`load_schemas`.
    examples_ds:
        Concatenated Spider + BIRD DataFrame from
        :func:`load_spider_examples` / :func:`load_bird_examples`.

    Returns
    -------
    Merged DataFrame with all columns from both inputs.
    """
    logger.info(
        "Merging examples with schemas: examples_rows=%s, schema_rows=%s, example_db_ids=%s, schema_db_ids=%s",
        len(examples_ds),
        len(schemas_ds),
        examples_ds["db_id"].nunique(),
        schemas_ds["db_id"].nunique(),
    )
    merged = pd.merge(schemas_ds, examples_ds, on="db_id", how="inner")
    n_null = merged.isnull().any(axis=1).sum()
    if n_null:
        raise ValueError(f"{n_null} rows with nulls after merge.")
    merged.reset_index(drop=True, inplace=True)
    logger.info(f"Total Examples: {len(merged)}")
    logger.info(f"Merged dataset: {len(merged)} rows.")
    return merged


# ---------------------------------------------------------------------------
# Step 6 – Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    merged_samples: pd.DataFrame,
    sample_size: int = 400,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``merged_samples`` into train / val / test, preserving source balance.

    Strategy
    --------
    1. Build a ``db_id → source`` mapping.
    2. For each source independently, randomly sample up to
       ``sample_size // n_sources`` database IDs.
    3. Within each source's chosen IDs, slice off ``train_ratio`` for train,
       ``val_ratio`` for val, and the remainder for test.
    4. Merge per-source splits, then filter ``merged_samples`` by membership.

    This ensures both Spider and BIRD databases appear in every partition even
    after random sampling, preventing the silent data leakage that occurs when
    you shuffle globally before slicing.

    Parameters
    ----------
    merged_samples:
        DataFrame with at least ``db_id`` and ``source`` columns.
    sample_size:
        Total number of *database IDs* to sample before splitting (not rows).
        Use ``-1`` to include all available database IDs. Defaults to 400.
    train_ratio:
        Fraction of sampled db_ids to put in train. Defaults to 0.70.
    val_ratio:
        Fraction of sampled db_ids to put in val.  Defaults to 0.15.
        The test fraction is ``1 - train_ratio - val_ratio``.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple of three DataFrames: ``(train_ds, val_ds, test_ds)``.
    """
    # number of sources (e.g. "spider", "bird") → list of db_ids belonging to each source
    logger.info("Performing stratified split by source…")
    logger.info(f"Total unique db_ids before sampling: {merged_samples['db_id'].nunique()}")
    # Shape of data after this step, Example: {'spider': 20, 'bird': 30}
    db_source_map = (
        merged_samples.drop_duplicates("db_id")
        .set_index("db_id")["source"]
    )
    # Source groups: {'spider': [db_id1, db_id2, ...], 'bird': [db_idA, db_idB, ...]}
    source_groups: dict[str, list[str]] = {
        src: grp.index.tolist()
        for src, grp in db_source_map.groupby(db_source_map)
    }
    n_sources = len(source_groups)
    logger.info(
        f"Split plan: sample_size={sample_size}, train_ratio={train_ratio}, "
        f"val_ratio={val_ratio}, test_ratio={1 - train_ratio - val_ratio:.2f}, seed={seed}"
    )
    logger.info(
        f"Available db_ids by source: "
        f"{ {src: len(ids) for src, ids in source_groups.items()} }"
    )
    effective_sample_size = merged_samples["db_id"].nunique() if sample_size == -1 else sample_size
    logger.debug(
        f"Sampling {effective_sample_size} db_ids stratified by source ({n_sources} sources)…"
    )
    per_source = effective_sample_size // n_sources
    remainder = effective_sample_size % n_sources

    rng = np.random.default_rng(seed=seed)
    train_dbs: list[str] = []
    val_dbs:   list[str] = []
    test_dbs:  list[str] = []

    for idx, src in enumerate(sorted(source_groups)):
        ids = source_groups[src]
        n_to_sample = per_source + (1 if idx < remainder else 0)
        chosen = rng.choice(ids, size=min(n_to_sample, len(ids)), replace=False).tolist()
        n = len(chosen)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, min(int(n * val_ratio), n - n_train))
        logger.info(
            f"Source '{src}': available_db_ids={len(ids)}, sampled_db_ids={len(chosen)}, "
            f"train_db_ids={n_train}, val_db_ids={n_val}, test_db_ids={n - n_train - n_val}"
        )

        train_dbs.extend(chosen[:n_train])
        val_dbs.extend(chosen[n_train : n_train + n_val])
        test_dbs.extend(chosen[n_train + n_val :])

    train_ds = merged_samples[merged_samples["db_id"].isin(train_dbs)].copy()
    val_ds   = merged_samples[merged_samples["db_id"].isin(val_dbs)].copy()
    test_ds  = merged_samples[merged_samples["db_id"].isin(test_dbs)].copy()

    logger.info(
        f"DBs in each set: Train, Val, Test: {(len(train_dbs), len(val_dbs), len(test_dbs))}"
    )
    logger.info(
        f"Rows in each set: Train, Val, Test: {(len(train_ds), len(val_ds), len(test_ds))}"
    )

    logger.info("Train DS Value Counts")
    logger.info(f"{train_ds['source'].value_counts().to_dict()}")
    logger.info("Val DS Value Counts")
    logger.info(f"{val_ds['source'].value_counts().to_dict()}")
    logger.info("Test DS Value Counts")
    logger.info(f"{test_ds['source'].value_counts().to_dict()}")

    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        logger.info(
            f"{name}: {len(ds)} rows, "
            f"{ds['source'].value_counts().to_dict()}"
        )

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Step 7 – Build prompt datasets
# ---------------------------------------------------------------------------


def build_prompt_datasets(
    train_ds: pd.DataFrame,
    val_ds: pd.DataFrame,
    test_ds: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply ``make_prompt_record`` to every row in each split.

    Converts raw DataFrames into lists of training-ready dicts with keys::

        prompt   – list[dict[str,str]]   (chat-format, from build_prompt)
        solution – str                   (gold SQL)
        schema   – dict[str, list[str]]  (table → columns)
        source   – str                   ("spider" or "bird")
        db_id    – str

    Parameters
    ----------
    train_ds, val_ds, test_ds:
        DataFrames with columns ``question``, ``schema``, ``SQL``, ``source``,
        ``db_id``.

    Returns
    -------
    Tuple of three lists: ``(train_dataset, val_dataset, test_dataset)``.
    """
    def _apply(df: pd.DataFrame) -> list[dict[str, Any]]:
        logger.info(
            f"Building prompt records for {len(df)} rows across {df['db_id'].nunique()} db_ids "
            f"({df['source'].value_counts().to_dict()})"
        )
        return df.apply(
            lambda row: make_prompt_record(
                question=row["question"],
                schema=row["schema"],
                answer=row["SQL"],
                source=row["source"],
                db_id=row["db_id"],
            ),
            axis=1,
        ).tolist()

    train_dataset = _apply(train_ds)
    val_dataset   = _apply(val_ds)
    test_dataset  = _apply(test_ds)

    logger.info(
        f"Prompt datasets built – "
        f"train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, "
        f"test: {len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Step 8 – Save splits
# ---------------------------------------------------------------------------


def save_splits(
    train_dataset: list[dict[str, Any]],
    val_dataset:   list[dict[str, Any]],
    test_dataset:  list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Persist each split to a ``.csv`` file under *output_dir*.

    Files produced:

    * ``<output_dir>/train.csv``
    * ``<output_dir>/val.csv``
    * ``<output_dir>/test.csv``

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset:
        Lists of dicts, as returned by :func:`build_prompt_datasets`.
    output_dir:
        Destination directory.  Created if it does not exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, dataset in (
        ("train", train_dataset),
        ("val",   val_dataset),
        ("test",  test_dataset),
    ):
        dest = out / f"{name}.csv"
        pd.DataFrame.from_records(dataset).to_csv(str(dest), index=False)
        logger.info(f"Saved {len(dataset)} records → {dest}")

    logger.success(f"All splits saved to {out}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def prepare(
    rawdata_dir: str,
    serialized_dir: str,
    splits_dir: str,
    sample_size: int = 400,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    skip_download: bool = False,
    skip_serialize: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the full data preparation pipeline.

    Steps executed in order:

    1. ``download_datasets``    (skipped when *skip_download* is True)
    2. ``extract_bird_databases``
    3. ``serialize_schemas``    (skipped when *skip_serialize* is True)
    4. ``load_schemas``
    5. ``load_spider_examples`` + ``load_bird_examples``
    6. Concat + ``merge_examples_with_schemas``
    7. ``stratified_split``
    8. ``build_prompt_datasets``
    9. ``save_splits``

    Parameters
    ----------
    rawdata_dir:
        Root directory for raw downloaded data.
    serialized_dir:
        Directory to write / read ``schema_lookup.json``.
    splits_dir:
        Directory to write ``train.csv``, ``val.csv``, ``test.csv``.
    sample_size:
        Total number of database IDs to sample before splitting.
        Use ``-1`` to include all available database IDs.
    train_ratio:
        Fraction of sampled db_ids assigned to train.
    val_ratio:
        Fraction of sampled db_ids assigned to val.
    seed:
        Random seed for reproducibility.
    skip_download:
        If True, assume the raw data is already present and skip download +
        BIRD extraction.  Useful when running repeatedly on a Kaggle session
        that already has the data.
    skip_serialize:
        If True, assume ``schema_lookup.json`` already exists in
        *serialized_dir* and skip schema serialisation.  Useful to avoid
        re-reading hundreds of SQLite files on every run.
    Returns
    -------
    Tuple ``(train_dataset, val_dataset, test_dataset)`` – lists of
    prompt-record dicts ready for GRPOTrainer.
    """
    # --- Step 1+2: Download (optional) ---
    if not skip_download:
        download_datasets(rawdata_dir)
    extract_bird_databases(rawdata_dir)

    # --- Step 3: Serialize schemas (optional) ---
    if not skip_serialize:
        serialize_schemas(rawdata_dir, serialized_dir)

    # --- Step 4: Load schemas ---
    schemas_ds = load_schemas(serialized_dir)

    # --- Step 5: Load examples ---
    spider_ds = load_spider_examples(rawdata_dir)
    bird_ds   = load_bird_examples(rawdata_dir)
    logger.info(
        f"Loaded source datasets: spider_rows={len(spider_ds)}, bird_rows={len(bird_ds)}, "
        f"spider_db_ids={spider_ds['db_id'].nunique()}, bird_db_ids={bird_ds['db_id'].nunique()}"
    )
    examples_ds = pd.concat([spider_ds, bird_ds], axis=0, ignore_index=True)
    logger.info(f"Total Examples: {len(examples_ds)}")
    logger.info(f"Examples by Source: {examples_ds['source'].value_counts().to_dict()}")
    logger.info(
        f"Total examples: {len(examples_ds)} "
        f"({examples_ds['source'].value_counts().to_dict()})"
    )

    # --- Step 6: Merge ---
    merged = merge_examples_with_schemas(schemas_ds, examples_ds)

    # --- Step 7: Split ---
    train_ds, val_ds, test_ds = stratified_split(
        merged,
        sample_size=sample_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    # --- Step 8: Build prompt records ---
    train_dataset, val_dataset, test_dataset = build_prompt_datasets(
        train_ds, val_ds, test_ds
    )

    # --- Step 9: Save CSV (for evaluator / human inspection) ---
    save_splits(train_dataset, val_dataset, test_dataset, splits_dir)

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full text-to-SQL data preparation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rawdata-dir",
        default="/data/rawdata",
        help="Root directory for raw downloaded data.",
    )
    p.add_argument(
        "--serialized-dir",
        default="/data/serialized_schemas",
        help="Directory to write/read schema_lookup.json.",
    )
    p.add_argument(
        "--splits-dir",
        default="/data/splits",
        help="Directory to write train/val/test CSV files.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=400,
        help="Total number of database IDs to sample before splitting; use -1 to include all databases.",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of sampled db_ids for the training split.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of sampled db_ids for the validation split.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading and extracting datasets (assumes data already present).",
    )
    p.add_argument(
        "--skip-serialize",
        action="store_true",
        help="Skip schema serialization (assumes schema_lookup.json already exists).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args() 
    logger.info(f"Starting data preparation with args: {args}")
    prepare(
        rawdata_dir=args.rawdata_dir,
        serialized_dir=args.serialized_dir,
        splits_dir=args.splits_dir,
        sample_size=args.sample_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_download=args.skip_download,
        skip_serialize=args.skip_serialize,
    )
