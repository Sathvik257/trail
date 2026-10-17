from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from text_utils import iter_language_lines, normalize_language, tokenize

END_TOKEN = "</s>"


class NGramPredictor:
    def __init__(self, max_order: int = 3, language: str = "telugu") -> None:
        if max_order < 1:
            raise ValueError("max_order must be at least 1")
        self.max_order = max_order
        self.language = normalize_language(language)
        self.context_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_predictions: dict[tuple[str, ...], tuple[str, ...]] = {}
        self.unigram_counts: Counter[str] = Counter()

    def fit_lines(self, lines: Iterable[str], max_lines: int = 0) -> int:
        trained_lines = 0
        for line in lines:
            tokens = tokenize(line, language=getattr(self, "language", "telugu"))
            if len(tokens) < 2:
                continue
            trained_lines += 1
            self.unigram_counts.update(tokens)
            padded = tokens + [END_TOKEN]
            for idx, word in enumerate(padded):
                max_context = min(self.max_order - 1, idx)
                for context_size in range(max_context + 1):
                    context = tuple(padded[idx - context_size : idx]) if context_size else tuple()
                    self.context_counts[context][word] += 1
            if max_lines and trained_lines >= max_lines:
                break
        return trained_lines

    def predict(self, text: str, top_k: int = 3) -> list[str]:
        contextual = self.predict_contextual(text, top_k=top_k)
        if contextual:
            return contextual
        return self._rank_words(self.unigram_counts, top_k)

    def predict_contextual(self, text: str, top_k: int = 3) -> list[str]:
        tokens = tokenize(text, language=getattr(self, "language", "telugu"))
        max_context = min(self.max_order - 1, len(tokens))

        for context_size in range(max_context, 0, -1):
            context = tuple(tokens[-context_size:]) if context_size else tuple()
            prediction_index = getattr(self, "context_predictions", {})
            if context in prediction_index:
                return list(prediction_index[context][:top_k])

            counter = self.context_counts.get(context)
            if not counter:
                continue
            return self._rank_words(counter, top_k)

        return []

    def compact(self, max_next_per_context: int = 8) -> None:
        prediction_index: dict[tuple[str, ...], tuple[str, ...]] = {}
        for context, counter in self.context_counts.items():
            words = tuple(self._rank_words(counter, max_next_per_context))
            if words:
                prediction_index[context] = words
        self.context_predictions = prediction_index
        self.context_counts = defaultdict(Counter)

    @staticmethod
    def _rank_words(counter: Counter[str], top_k: int) -> list[str]:
        ranked = sorted(
            ((word, count) for word, count in counter.items() if word != END_TOKEN),
            key=lambda item: (-item[1], item[0]),
        )
        return [word for word, _ in ranked[:top_k]]

    @staticmethod
    def _rank_scores(counter: Counter[str], top_k: int) -> list[str]:
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        return [word for word, _ in ranked[:top_k]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "NGramPredictor":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain an NGramPredictor")
        if not hasattr(obj, "language"):
            obj.language = "telugu"
        return obj


def build_ngram_model(
    raw_path: str | Path,
    output_path: str | Path,
    max_order: int = 5,
    max_lines: int = 15000,
    max_next_per_context: int = 8,
    language: str = "telugu",
) -> NGramPredictor:
    predictor = NGramPredictor(max_order=max_order, language=language)
    trained_lines = predictor.fit_lines(iter_language_lines(raw_path, language=language), max_lines=max_lines)
    if trained_lines == 0:
        raise FileNotFoundError(f"No {language} lines found in {raw_path}")
    predictor.compact(max_next_per_context=max_next_per_context)
    predictor.save(output_path)
    return predictor
