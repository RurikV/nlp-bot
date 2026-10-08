"""Генерирует hw_danetqa_bot.ipynb (ноутбук-артефакт ДЗ) из кода ячеек.

Запуск: .venv/bin/python scripts/build_notebook.py
"""
from __future__ import annotations

import nbformat

CELLS = [
    ("md", "# Дообучение ruBert-base на DaNetQA (Russian SuperGLUE) и Telegram-бот\n\n"
           "**Задание**: взять одну задачу Russian SuperGLUE, зафайнтюнить модель с HuggingFace, "
           "сделать сабмит на лидерборд и встроить модель в телеграм-бота.\n\n"
           "**Задача DaNetQA** — бинарный вопросно-ответный датасет (да/нет): дан отрывок "
           "(passage) и вопрос (question); ответ должен следовать только из отрывка. "
           "Формулировка близка к NLI: пара текстов классифицируется в 2 класса. "
           "Метрика — Accuracy. Человеческий бейзлайн — 0.915."),

    ("code", "# version banner\n"
             "import sys\n"
             "import torch, transformers, datasets, sklearn\n"
             "print('python', sys.version.split()[0])\n"
             "print('torch', torch.__version__, '| mps:', torch.backends.mps.is_available())\n"
             "print('transformers', transformers.__version__, '| datasets', datasets.__version__,\n"
             "      '| sklearn', sklearn.__version__)\n"
             "SEED = 42"),

    ("md", "## 1. Данные\n\nСкачиваем официальный архив DaNetQA с russiansuperglue.com "
           "(скрипт идемпотентен) и смотрим на размеры, баланс классов и пример."),

    ("code", "import json\n"
             "from pathlib import Path\n"
             "import pandas as pd\n"
             "\n"
             "if not Path('data/DaNetQA/train.jsonl').exists():\n"
             "    import subprocess; subprocess.run(['.venv/bin/python', 'scripts/download_data.py'], check=True)\n"
             "\n"
             "def read_jsonl(p):\n"
             "    return [json.loads(l) for l in open(p, encoding='utf-8')]\n"
             "\n"
             "rows = {s: read_jsonl(f'data/DaNetQA/{s}.jsonl') for s in ['train', 'val', 'test']}\n"
             "for s, r in rows.items():\n"
             "    labels = [x.get('label') for x in r]\n"
             "    print(s, len(r), '| поля:', sorted(r[0]), '| label true:', sum(1 for l in labels if l))\n"
             "\n"
             "ex = rows['train'][0]\n"
             "print('\\nПример:')\n"
             "print('passage:', ex['passage'][:200], '...')\n"
             "print('question:', ex['question'])\n"
             "print('label:', ex['label'])"),

    ("md", "## 2. Токенизация\n\nПара текстов подаётся как `(passage, question)` — ровно так же, "
           "как потом в боте (`inference.py`). Обрезка — `truncation='only_first'`: режется хвост "
           "passage, вопрос сохраняется целиком. `MAX_LEN=256` — проверим по статистике длин."),

    ("code", "from transformers import AutoTokenizer\n"
             "\n"
             "MODEL_NAME = 'ai-forever/ruBert-base'\n"
             "MAX_LEN = 256\n"
             "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
             "\n"
             "lens = [len(tokenizer(x['passage'], x['question'])['input_ids']) for x in rows['train'] + rows['val']]\n"
             "s = pd.Series(lens)\n"
             "print(s.describe().round(1))\n"
             "for n in [192, 256, 320, 384, 512]:\n"
             "    print(f'покрытие max_len={n}: {(s <= n).mean():.1%}')"),

    ("code", "import datasets as hfds\n"
             "\n"
             "# у test-сплита метки скрыты — заглушка 0 (accuracy по ним не осмыслен)\n"
             "def to_ds(split):\n"
             "    items = [{'label': int(bool(x['label'])) if 'label' in x else 0}\n"
             "             for x in rows[split]]\n"
             "    texts = [(x['passage'], x['question']) for x in rows[split]]\n"
             "    ds = hfds.Dataset.from_list(items)\n"
             "    return ds.map(lambda _, i: tokenizer(texts[i][0], texts[i][1],\n"
             "                    truncation='only_first', max_length=MAX_LEN),\n"
             "                   with_indices=True)\n"
             "\n"
             "ds = {s: to_ds(s) for s in ['train', 'val', 'test']}\n"
             "ds['train'][0]['label']"),

    ("md", "## 3. Дообучение ruBert-base\n\n`AutoModelForSequenceClassification` (num_labels=2), "
           "HF Trainer на устройстве mps (Apple M1 Max): lr 2e-5, batch 16, 3 эпохи, "
           "warmup 10%, eval по эпохам, лучший чекпойнт по accuracy на val."),

    ("code", "import numpy as np\n"
             "from sklearn.metrics import accuracy_score\n"
             "from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments\n"
             "\n"
             "model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)\n"
             "model.config.id2label = {0: 'false', 1: 'true'}\n"
             "model.config.label2id = {'false': 0, 'true': 1}\n"
             "\n"
             "def compute_metrics(p):\n"
             "    return {'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1))}\n"
             "\n"
             "args = TrainingArguments(\n"
             "    output_dir='outputs/train',\n"
             "    learning_rate=2e-5,\n"
             "    per_device_train_batch_size=16,\n"
             "    per_device_eval_batch_size=64,\n"
             "    num_train_epochs=3,\n"
             "    weight_decay=0.01,\n"
             "    warmup_ratio=0.1,\n"
             "    eval_strategy='epoch',\n"
             "    save_strategy='epoch',\n"
             "    load_best_model_at_end=True,\n"
             "    metric_for_best_model='accuracy',\n"
             "    save_total_limit=1,\n"
             "    seed=SEED,\n"
             "    report_to=[],\n"
             "    dataloader_pin_memory=False,  # без ворнингов на mps\n"
             ")\n"
             "trainer = Trainer(model=model, args=args,\n"
             "                  train_dataset=ds['train'], eval_dataset=ds['val'],\n"
             "                  processing_class=tokenizer, compute_metrics=compute_metrics)\n"
             "trainer.train()"),

    ("md", "## 4. Оценка на val\n\nСравниваем с тривиальными опорными точками: "
           "константный бейзлайн (всегда «да») и человеческий бейзлайн 0.915."),

    ("code", "from sklearn.metrics import classification_report\n"
             "\n"
             "eval_out = trainer.evaluate()\n"
             "val_acc = eval_out['eval_accuracy']\n"
             "majority = sum(bool(x['label']) for x in rows['val']) / len(rows['val'])\n"
             "print(f'val accuracy: {val_acc:.3f}')\n"
             "print(f'бейзлайн «всегда да»: {majority:.3f}')\n"
             "print('человеческий бейзлайн: 0.915')\n"
             "\n"
             "pred = np.argmax(trainer.predict(ds['val']).predictions, axis=-1)\n"
             "gold = [int(bool(x['label'])) for x in rows['val']]\n"
             "print(classification_report(gold, pred, target_names=['false', 'true'], digits=3))"),

    ("md", "## 5. Сохранение модели для бота\n\nЧекпойнт и метрика — в `outputs/danetqa-rubert/`; "
           "бот (`bot.py`) и smoke-тест (`scripts/smoke_test.py`) грузят их через `inference.py`."),

    ("code", "import json\n"
             "\n"
             "OUT = Path('outputs/danetqa-rubert')\n"
             "OUT.mkdir(parents=True, exist_ok=True)\n"
             "trainer.save_model(OUT)\n"
             "tokenizer.save_pretrained(OUT)\n"
             "json.dump({'val_accuracy': val_acc, 'max_len': MAX_LEN, 'seed': SEED},\n"
             "          open(OUT / 'metrics.json', 'w'), ensure_ascii=False, indent=2)\n"
             "print('сохранено:', OUT, '| val_accuracy =', round(val_acc, 4))"),

    ("md", '## 6. Предсказания на test и сабмит\n\nФормат — как в `sample_submission.zip` официального '
           'репозитория: JSONL `{"idx": <int>, "label": "true"|"false"}`. '
           'Файл загружается в личном кабинете https://russiansuperglue.com/leaderboard/2.'),

    ("code", "test_pred = np.argmax(trainer.predict(ds['test']).predictions, axis=-1)\n"
             "sub = Path('outputs/submission'); sub.mkdir(parents=True, exist_ok=True)\n"
             "with open(sub / 'DaNetQA.jsonl', 'w', encoding='utf-8') as f:\n"
             "    for x, p in zip(rows['test'], test_pred):\n"
             "        f.write(json.dumps({'idx': x['idx'], 'label': 'true' if p == 1 else 'false'},\n"
             "                            ensure_ascii=False) + '\\n')\n"
             "print('строк в сабмите:', sum(1 for _ in open(sub / 'DaNetQA.jsonl')))\n"
             "print('классы:', pd.Series(test_pred).map({0: 'false', 1: 'true'}).value_counts().to_dict())\n"
             "print(open(sub / 'DaNetQA.jsonl').readline().strip())"),

    ("md", "## 7. Демо: как отвечает модель (тот же путь, что в боте)\n\nТри примера из val — "
           "passage/question/предсказание/эталон."),

    ("code", "import sys; sys.path.insert(0, '.')\n"
             "from inference import DanetqaModel, format_answer\n"
             "\n"
             "m = DanetqaModel(trainer.model, tokenizer, max_len=MAX_LEN)\n"
             "for x in rows['val'][:3]:\n"
             "    p, _ = m.predict(x['passage'], x['question'])\n"
             "    print(f\"Q: {x['question']}\")\n"
             "    print(f\"  ответ модели: {format_answer(p)} | эталон: {'Да' if x['label'] else 'Нет'}\")"),

    ("md", "## 8. Результат сабмита на лидерборд\n\nСкриншот страницы сабмита с russiansuperglue.com/leaderboard/2 "
           "(скрины лежат в `assets/`, наличие проверяет `scripts/fill_screenshots.py`):\n\n"
           "![leaderboard](assets/leaderboard.png)"),

    ("md", "## 9. Telegram-бот\n\nБот на aiogram 3 (`bot.py`): `/start` печатает описание задачи DaNetQA, "
           "далее FSM-диалог «отрывок → вопрос → ответ Да/Нет с уверенностью». "
           "Запуск: `.venv/bin/python bot.py` (токен в `.env`). Скриншоты работы:\n\n"
           "![бот 1](assets/bot_intro.png)\n\n![бот 2](assets/bot_dialog.png)"),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for i, (kind, src) in enumerate(CELLS):
        cell = nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        cell.id = f"cell-{i:02d}"  # детерминированные id — стабильный git-дифф при пересборке
        nb.cells.append(cell)
    nbformat.write(nb, "hw_danetqa_bot.ipynb")
    print("hw_danetqa_bot.ipynb: ячеек:", len(nb.cells))


if __name__ == "__main__":
    main()
