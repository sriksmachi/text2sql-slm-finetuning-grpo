"""
serialize_schemas.py – Convert Spider/BIRD schemas to a compact JSON format.

For each database found under *input_dir*, this script reads the SQLite
``.db`` file (or ``tables.json``) and writes a ``schema_lookup.json`` at
*output_dir* mapping database IDs to ``{table: [columns]}`` dicts.

Usage
-----
    python serialize_schemas.py \\
        --input-dir /path/to/raw_data \\
        --output-dir /path/to/serialized_schemas
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from loguru import logger
from tqdm import tqdm


def _schema_from_sqlite(db_path: Path) -> dict[str, list[str]]:
    """Extract table→columns mapping from a SQLite file."""
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
    """Parse Spider-style ``tables.json`` into a dict of db_id → schema."""
    with open(tables_json_path) as fh:
        tables_data = json.load(fh)

    result: dict[str, dict[str, list[str]]] = {}
    for db in tables_data:
        db_id = db["db_id"]
        col_names = db.get("column_names_original", db.get("column_names", []))
        table_names = db.get("table_names_original", db.get("table_names", []))

        schema: dict[str, list[str]] = {t.lower(): [] for t in table_names}
        for table_idx, col_name in col_names:
            if table_idx < 0:
                continue  # wildcard column
            tname = table_names[table_idx].lower()
            schema[tname].append(col_name.lower())
        result[db_id] = schema
    return result


def _is_probably_sqlite_file(p: Path) -> bool:
    """Quick check to skip obvious non-database files."""
    # Skip macOS resource-fork files and __MACOSX folder artifacts
    if p.name.startswith("._") or "__MACOSX" in p.parts:
        return False
    if not p.is_file():
        return False
    # Optional header check: SQLite files usually start with this magic string
    try:
        with p.open("rb") as fh:
            header = fh.read(16)
        return header.startswith(b"SQLite format 3")
    except OSError:
        return False

def serialize_schemas(input_dir: str, output_dir: str) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_schemas: dict[str, dict[str, list[str]]] = {}

    # Spider / BIRD style: tables.json at root of dataset
    logger.info(f"Parsing Spider/BIRD-style tables.json files under {in_path}...")
    for tables_json in in_path.rglob("dev_tables.json"):
        logger.debug(f"Processing {tables_json}")
        schemas = _schema_from_tables_json(tables_json)
        all_schemas.update(schemas)

    # Also scan individual SQLite .db files
    sqlite_exts = {".db", ".sqlite", ".sqlite3", ".sqlllite"}  
    db_files = [p for p in in_path.rglob("*") if _is_probably_sqlite_file(p) and (p.suffix.lower() in sqlite_exts)]
    logger.info(f"Parsing {len(db_files)} SQLite files for schemas...")
    for db_file in tqdm(db_files, desc="SQLite schemas"):
        db_id = db_file.stem
        if db_id not in all_schemas:
            try:
                all_schemas[db_id] = _schema_from_sqlite(db_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not read {db_file}: {exc}")

    out_file = out_path / "schema_lookup.json"
     # Write as list instead of object
    payload = [
        {"db_id": db_id, "schema": schema}
        for db_id, schema in sorted(all_schemas.items())
    ]
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.success(f"Serialised {len(all_schemas)} schemas → {out_file}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serialise Spider/BIRD schemas")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    serialize_schemas(args.input_dir, args.output_dir)
