"""Проверяет наличие скриншотов для ноутбука.

Скриншоты кладёт пользователь: assets/leaderboard.png, assets/bot_intro.png,
assets/bot_dialog.png. Ячейки уже ссылаются на эти пути; скрипт лишь проверяет
их наличие (ноутбук перегенерации не требует — ссылки уже вшиты).

Запуск: .venv/bin/python scripts/fill_screenshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ASSETS = ["leaderboard.png", "bot_intro.png", "bot_dialog.png"]


def main() -> int:
    missing = [a for a in ASSETS if not Path("assets", a).exists()]
    if missing:
        print("нет скринов (положи в assets/ и перезапусти):", ", ".join(missing), file=sys.stderr)
        return 1
    print("все скрины на месте — markdown-ячейки ноутбука уже ссылаются на них")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
