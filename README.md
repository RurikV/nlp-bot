# Дообучение ruBert-base на DaNetQA (Russian SuperGLUE) + Telegram-бот

Основной артефакт — **`hw_danetqa_bot.ipynb`**: данные → токенизация → файнтюн
ruBert-base → оценка на val (accuracy 0.62 против бейзлайна 0.50) → сабмит-файл
для лидерборда → демо ответов модели.

Бот — **`bot.py`** (aiogram 3): /start печатает описание задачи DaNetQA, дальше
FSM-диалог «отрывок → вопрос → ответ Да/Нет с уверенностью». Модель — чекпойнт
из `outputs/danetqa-rubert/`, препроцессинг общий с обучением (`inference.py`).

## Воспроизведение

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python scripts/download_data.py                 # данные DaNetQA + образец сабмита
.venv/bin/python scripts/build_notebook.py                # собрать ноутбук
.venv/bin/python scripts/run_notebook.py                  # обучение + сабмит (~10 мин на M1 Max, mps)
.venv/bin/python scripts/smoke_test.py                    # инференс бота == ноутбуку (допуск 2σ)
.venv/bin/python -m pytest tests/ -v                      # 11 тестов
```

## Запуск бота

1. Создай бота у @BotFather, скопируй токен: `cp .env.example .env` и впиши его в `TELEGRAM_BOT_TOKEN`.
2. `.venv/bin/python bot.py`
3. В Telegram: /start → отрывок текста → вопрос → ответ «Да/Нет» с уверенностью; /example — разобрать готовый пример.

Скриншоты работы бота и сабмита на лидерборд — в `assets/`, в ноутбуке — разделы 8–9.

## Сабмит на лидерборд

1. Зарегистрируйся на https://russiansuperglue.com (Login в шапке).
2. Загрузи `outputs/submission/DaNetQA.jsonl` на https://russiansuperglue.com/leaderboard/2
   (формат — как в `sample_submission.zip` официального репозитория: {"idx": ..., "label": "true"/"false"}).
3. Скрин результата → `assets/leaderboard.png`, затем `.venv/bin/python scripts/fill_screenshots.py`.

## Структура

- `hw_danetqa_bot.ipynb` — основной артефакт (генерируется `scripts/build_notebook.py`)
- `bot.py` / `inference.py` — бот и общий инференс-модуль
- `tests/` — pytest-тесты (инференс, тексты бота)
- `scripts/` — download_data, build/run_notebook, smoke_test, fill_screenshots
- `data/`, `outputs/`, `logs/` — генерируемые артефакты (в .gitignore)
