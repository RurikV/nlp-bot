"""Тесты текстов и структуры бота; импорт bot.py не требует токена."""
import inspect

import bot


def test_intro_mentions_task_and_model():
    t = bot.intro_text()
    assert "DaNetQA" in t
    assert "ruBert" in t
    assert "Да" in t and "Нет" in t  # суть задачи
    assert "/example" in t


def test_help_text():
    assert "отрывок" in bot.HELP_TEXT.lower()


def test_examples_are_daNetQA_shaped():
    for ex in bot.EXAMPLES:
        assert set(ex) == {"passage", "question", "label"}
        assert isinstance(ex["label"], bool)
        assert len(ex["passage"]) > 50


def test_handlers_inject_model_param():
    # Dispatcher(model=...) передаёт модель хендлерам по имени параметра (DI)
    for fn in (bot.cmd_example, bot.got_question):
        assert "model" in inspect.signature(fn).parameters
