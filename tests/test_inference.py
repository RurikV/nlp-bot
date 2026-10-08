"""Тесты инференс-модуля: чистые функции + predict на заглушках модели."""
import torch

from inference import ID_TO_LABEL, LABEL_TO_ID, DanetqaModel, format_answer


class StubTokenizer:
    """Записывает аргументы вызова; 1-D тензор — как плоский список реального токенизатора."""
    def __init__(self):
        self.calls = []

    def __call__(self, passage, question, **kwargs):
        self.calls.append({"passage": passage, "question": question, **kwargs})
        n = max(1, (len(passage) + len(question)) // 3) + 3  # «3 символа = токен» + спецтокены
        ids = torch.ones(n, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids.clone()}


class StubModel:
    device = "cpu"

    def __init__(self, logits):
        self.logits = torch.tensor(logits)
        self.eval_called = False

    def __call__(self, **kwargs):
        class Out:
            pass
        out = Out()
        out.logits = self.logits
        return out

    def eval(self):
        self.eval_called = True
        return self


def test_label_maps_are_consistent():
    assert LABEL_TO_ID == {"false": 0, "true": 1}
    assert ID_TO_LABEL == {0: "false", 1: "true"}


def test_format_answer_yes():
    assert format_answer(0.873) == "Да (уверенность 87%)"


def test_format_answer_no_confidence_is_of_predicted_class():
    # уверенность считается для предсказанного класса («Нет»), а не для true
    assert format_answer(0.124) == "Нет (уверенность 88%)"


def test_init_puts_model_in_eval_mode():
    stub = StubModel([0.0, 5.0])
    DanetqaModel(stub, StubTokenizer())
    assert stub.eval_called is True


def test_predict_passes_pair_in_canonical_order():
    tok, model = StubTokenizer(), StubModel([0.0, 5.0])
    m = DanetqaModel(model, tok)
    m.predict("отрывок текста", "вопрос")
    call = tok.calls[-1]
    assert call["passage"] == "отрывок текста"
    assert call["question"] == "вопрос"
    assert call["truncation"] == "only_first"
    assert call["max_length"] == 256


def test_predict_returns_probability_of_true():
    m = DanetqaModel(StubModel([0.0, 5.0]), StubTokenizer())
    prob, truncated = m.predict("p", "q")
    assert 0.98 < prob <= 1.0
    assert truncated is False


def test_predict_flags_truncation_of_long_passage():
    tok = StubTokenizer()
    m = DanetqaModel(StubModel([0.0, 5.0]), tok)
    _, truncated = m.predict("т" * 3000, "короткий вопрос")
    assert truncated is True
