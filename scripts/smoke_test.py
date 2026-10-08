"""Smoke-тест: путь инференса бота (inference.py) воспроизводит val accuracy ноутбука.

По умолчанию N — весь val (821 пример); опциональный аргумент ограничивает выборку.
Допуск масштабируется как 2σ биномиального шума (≥2пп): расхождение в пределах
допуска означает, что препроцессинг бота совпадает с обучающим.

Запуск: .venv/bin/python scripts/smoke_test.py [N]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")

from inference import MODEL_DIR, DanetqaModel, format_answer  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else None


def main() -> int:
    rows = [json.loads(l) for l in open("data/DaNetQA/val.jsonl", encoding="utf-8")]
    if N is not None:
        rows = rows[:N]
    model = DanetqaModel.load(MODEL_DIR)

    correct = 0
    for x in rows:
        prob, truncated = model.predict(x["passage"], x["question"])
        if (prob >= 0.5) == bool(x["label"]):
            correct += 1
    acc = correct / len(rows)

    metrics = json.loads(Path(MODEL_DIR, "metrics.json").read_text())
    tol = max(0.02, 1.0 / math.sqrt(len(rows)))
    print(f"smoke accuracy на {len(rows)} примерах: {acc:.4f}")
    print(f"ноутбучный val_accuracy:            {metrics['val_accuracy']:.4f}")
    print(f"допуск (2σ, ≥2пп):                  {tol:.4f}")
    demo_prob, _ = model.predict(rows[0]["passage"], rows[0]["question"])
    print("демо:", format_answer(demo_prob))

    if abs(acc - metrics["val_accuracy"]) > tol:
        print("ОШИБКА: расхождение выше допуска — препроцессинг бота отличается от обучения", file=sys.stderr)
        return 1
    print("OK: препроцессинг бота идентичен обучающему")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
