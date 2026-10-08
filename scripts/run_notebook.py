"""Исполняет ноутбук, сохраняя его после каждой ячейки (устойчиво к долгому обучению).

Запуск: .venv/bin/python scripts/run_notebook.py [hw_danetqa_bot.ipynb]
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PATH = sys.argv[1] if len(sys.argv) > 1 else "hw_danetqa_bot.ipynb"


def main() -> int:
    nb_path = Path(PATH).resolve()  # путь аргумента — от cwd вызывающего, до chdir
    # kernel наследует cwd процесса: data/, outputs/, sys.path и subprocess с
    # относительными путями должны работать при запуске из любой директории
    os.chdir(Path(__file__).resolve().parent.parent)
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=7200, kernel_name="python3")
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            print(f"=== ячейка {i + 1}/{len(nb.cells)} ({cell.cell_type}) ===", flush=True)
            try:
                # как сам nbclient (client.py:710): без store_history ядро всегда
                # отвечает execution_count=1, поэтому нумеруем сами
                client.execute_cell(cell, i, store_history=False,
                                    execution_count=client.code_cells_executed + 1)
            except Exception:
                traceback.print_exc()
                nbformat.write(nb, nb_path)
                return 1
            nbformat.write(nb, nb_path)
    print("ноутбук выполнен полностью")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
