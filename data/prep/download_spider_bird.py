"""
download_spider_bird.py. Download and extract Spider and BIRD datasets.

Usage
-----
    python download_spider_bird.py --output-dir /path/to/raw_data
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

SPIDER_URL = "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t"
BIRD_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

DATASETS = {
    "spider": SPIDER_URL,
    "bird": BIRD_URL,
}


def _download(url: str, dest: Path) -> None:
    """Stream download *url* to *dest*."""
    logger.info(f"Downloading {url} → {dest}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))


def _extract(archive: Path, dest: Path) -> None:
    logger.info(f"Extracting {archive} → {dest}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


def download_datasets(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_downloads"
    tmp.mkdir(exist_ok=True)

    for name, url in DATASETS.items():
        archive = tmp / f"{name}.zip"
        extract_dir = out / name
        if extract_dir.exists():
            logger.info(f"{name} already extracted at {extract_dir}, skipping.")
            continue
        _download(url, archive)
        _extract(archive, extract_dir)
        archive.unlink(missing_ok=True)

    shutil.rmtree(tmp, ignore_errors=True)
    logger.success(f"All datasets ready in {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Spider and BIRD datasets")
    p.add_argument("--output-dir", required=True, help="Destination directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_datasets(args.output_dir)
