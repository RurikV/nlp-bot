"""Телеграм-бот с ruBert-base, дообученным на DaNetQA (Russian SuperGLUE).

Запуск: .venv/bin/python bot.py  (TELEGRAM_BOT_TOKEN в .env или окружении).
"""
from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F, Router, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from inference import DanetqaModel, format_answer

router = Router()

HELP_TEXT = (
    "Пришли отрывок текста (passage), затем вопрос (question) — я отвечу «Да» или «Нет» "
    "по модели, дообученной на DaNetQA. /example — готовый пример, /reset — начать заново."
)

# Примеры из val-выборки DaNetQA (эталонные ответы известны).
EXAMPLES = [
    {
        "passage": (
            "Гидросфера Марса — это совокупность водных запасов планеты Марс. "
            "Вода на Марсе присутствует в виде льда в полярных шапках, а также "
            "в приповерхностном слое грунта на больших широтах."
        ),
        "question": "Есть ли вода на марсе?",
        "label": True,
    },
    {
        "passage": (
            "Отравления ртутью — расстройства здоровья, связанные с избыточным "
            "поступлением паров или соединений ртути в организм. Токсические "
            "свойства ртути известны с глубокой древности."
        ),
        "question": "Полезна ли ртуть с градусника?",
        "label": False,
    },
]


class QA(StatesGroup):
    passage = State()
    question = State()


def intro_text() -> str:
    return (
        "Привет! Я бот с моделью <b>ruBert-base</b>, дообученной на задаче "
        "<b>DaNetQA</b> бенчмарка <a href=\"https://russiansuperglue.com/tasks\">Russian SuperGLUE</a>.\n\n"
        "<b>О задаче:</b> DaNetQA — вопросы с ответами «да/нет» по отрывку текста. "
        "Даны отрывок (passage) и вопрос (question); нужно ответить «Да» или «Нет», "
        "опираясь только на содержание отрывка.\n\n"
        "<b>Как пользоваться:</b>\n"
        "1) пришли <i>отрывок текста</i>;\n"
        "2) пришли <i>вопрос</i> к нему;\n"
        "3) я отвечу «Да»/«Нет» с уверенностью модели — и снова попрошу отрывок.\n\n"
        "Команды: /example — разобрать готовый пример, /reset — сброс, /help — справка."
    )


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.set_state(QA.passage)
    await message.answer(intro_text())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("reset"))
async def cmd_reset(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(QA.passage)
    await message.answer("Сброшено. Пришли новый отрывок текста.")


@router.message(Command("example"))
async def cmd_example(message: Message, state: FSMContext, model: DanetqaModel) -> None:
    ex = EXAMPLES[0]
    try:
        prob, truncated = await asyncio.to_thread(model.predict, ex["passage"], ex["question"])
    except Exception:
        logging.exception("ошибка предсказания в /example")
        await message.answer("Не удалось обработать пример. Попробуй ещё раз или /reset.")
        return
    gold = "Да" if ex["label"] else "Нет"
    note = "\n\n(отрывок был обрезан из-за лимита модели)" if truncated else ""
    await message.answer(
        f"Пример из val-выборки DaNetQA:\n\n"
        f"<b>Отрывок:</b> {html.quote(ex['passage'])}\n\n"
        f"<b>Вопрос:</b> {html.quote(ex['question'])}\n\n"
        f"<b>Ответ модели:</b> {format_answer(prob)}\n"
        f"<b>Эталон:</b> {gold}{note}\n\n"
        f"Теперь попробуй сам: пришли свой отрывок."
    )
    await state.set_state(QA.passage)


@router.message(QA.passage, F.text)
async def got_passage(message: Message, state: FSMContext) -> None:
    await state.update_data(passage=message.text.strip())
    await state.set_state(QA.question)
    await message.answer("Принял отрывок. Теперь задай вопрос (да/нет-вопрос по отрывку).")


@router.message(QA.question, F.text)
async def got_question(message: Message, state: FSMContext, model: DanetqaModel) -> None:
    data = await state.get_data()
    passage = data.get("passage", "")
    question = message.text.strip()
    try:
        prob, truncated = await asyncio.to_thread(model.predict, passage, question)
    except Exception:
        logging.exception("ошибка предсказания")
        await message.answer("Не удалось обработать пример. Попробуй ещё раз или /reset.")
        return
    note = "\n(отрывок длинноват — я обрезал его хвост, вопрос учтён полностью)" if truncated else ""
    await message.answer(
        f"<b>Вопрос:</b> {html.quote(question)}\n"
        f"<b>Ответ модели:</b> {format_answer(prob)}{note}\n\n"
        f"Пришли следующий отрывок или /reset."
    )
    await state.set_state(QA.passage)


@router.message()
async def fallback(message: Message) -> None:
    await message.answer("Пришли текст: сначала отрывок, потом вопрос. /start — описание задачи.")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv

    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN не задан: скопируй .env.example в .env и впиши токен от @BotFather"
        )

    logging.info("загружаю модель из outputs/danetqa-rubert ...")
    model = DanetqaModel.load()
    logging.info("модель загружена")

    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(model=model)  # workflow_data → DI по имени параметра
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
