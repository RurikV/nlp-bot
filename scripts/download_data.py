"""Идемпотентно скачивает данные DaNetQA и образец сабмита в data/.

Запуск: .venv/bin/python scripts/download_data.py [--force]
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"

SOURCES = [
    # (маркер после распаковки, url, headers, куда распаковывать)
    # ВАЖНО: у sample_submission.zip внутри НЕТ префикса sample/ (корень — submission/),
    # поэтому распаковываем его в data/sample, а не в data/
    (
        DATA / "DaNetQA" / "train.jsonl",
        "https://russiansuperglue.com/tasks/download/DaNetQA",
        {},
        DATA,
    ),
    (
        DATA / "sample" / "submission" / "DaNetQA.jsonl",
        "https://api.github.com/repos/RussianNLP/RussianSuperGLUE/contents/sample_submission.zip",
        {"Accept": "application/vnd.github.raw+json"},
        DATA / "sample",
    ),
]


def download(url: str, headers: dict) -> bytes:
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content


def extract_zip(content: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX") or name.endswith(".DS_Store"):
                continue
            zf.extract(name, dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="скачать заново")
    args = parser.parse_args()

    for marker, url, headers, dest in SOURCES:
        if marker.exists() and not args.force:
            print(f"ок, уже есть: {marker}")
            continue
        print(f"скачиваю {url} ...")
        extract_zip(download(url, headers), dest)
        if not marker.exists():
            print(f"ОШИБКА: после распаковки не найден {marker}", file=sys.stderr)
            return 1
        print(f"готово: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
