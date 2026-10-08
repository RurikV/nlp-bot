"""Инференс модели, дообученной на DaNetQA (Russian SuperGLUE).

Общий путь препроцессинга для обучения в ноутбуке, бота (bot.py)
и smoke-теста (scripts/smoke_test.py): пара (passage, question),
truncation='only_first', MAX_LEN=256, метки false=0 / true=1.
"""
from __future__ import annotations

import torch

MODEL_DIR = "outputs/danetqa-rubert"
MAX_LEN = 256

LABEL_TO_ID = {"false": 0, "true": 1}
ID_TO_LABEL = {0: "false", 1: "true"}


def format_answer(prob_true: float) -> str:
    """Человекочитаемый ответ по вероятности класса 'true'."""
    if prob_true >= 0.5:
        return f"Да (уверенность {prob_true:.0%})"
    return f"Нет (уверенность {1 - prob_true:.0%})"


class DanetqaModel:
    """Обёртка над seq-classification чекпойнтом для ответов Да/Нет."""

    def __init__(self, model, tokenizer, max_len: int = MAX_LEN):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_len = max_len

    @classmethod
    def load(cls, path: str = MODEL_DIR) -> "DanetqaModel":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        return cls(model, tokenizer)

    def predict(self, passage: str, question: str) -> tuple[float, bool]:
        """Вероятность класса 'true' и флаг «вход обрезан по max_len»."""
        enc_full = self.tokenizer(passage, question)
        truncated = len(enc_full["input_ids"]) > self.max_len

        inputs = self.tokenizer(
            passage,
            question,
            truncation="only_first",
            max_length=self.max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # у модели логиты (batch, n_classes), у заглушки в тестах — плоский вектор
        probs = torch.softmax(logits, dim=-1).reshape(-1)
        return float(probs[1]), truncated
