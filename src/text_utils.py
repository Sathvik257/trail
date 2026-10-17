from __future__ import annotations

import re
import sys
import unicodedata
from collections import deque
from pathlib import Path
from typing import Iterator

LANGUAGE_RANGES = {
    "telugu": ("\u0c00", "\u0c7f"),
    "hindi": ("\u0900", "\u097f"),
    "english": ("A", "z"),
}
LANGUAGE_LABELS = {
    "telugu": "Telugu",
    "hindi": "Hindi",
    "english": "English",
}
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?;:\u0964\u0965]+")
SPACE_RE = re.compile(r"\s+")
SENTENCE_BOUNDARY_CHARS = set(".!?;:\u0964\u0965")


def normalize_language(language: str) -> str:
    language = (language or "telugu").strip().lower()
    aliases = {
        "te": "telugu",
        "tel": "telugu",
        "tel_te": "telugu",
        "tel_telu": "telugu",
        "hi": "hindi",
        "hin": "hindi",
        "hin_deva": "hindi",
        "en": "english",
        "eng": "english",
    }
    language = aliases.get(language, language)
    if language not in LANGUAGE_RANGES:
        raise ValueError(f"Unsupported language: {language}")
    return language


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\ufeff", " ")


def _script_filter_re(language: str) -> re.Pattern[str]:
    if normalize_language(language) == "english":
        return re.compile(r"[^A-Za-z'\s]+")
    start, end = LANGUAGE_RANGES[normalize_language(language)]
    return re.compile(f"[^{start}-{end}\\s]+")


def clean_text(text: str, language: str = "telugu") -> str:
    """Keep only characters from the selected language script and normalize whitespace."""
    language = normalize_language(language)
    text = normalize_unicode(text)
    if language == "english":
        text = text.replace("’", "'").replace("`", "'").lower()
    text = _script_filter_re(language).sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def split_sentences(text: str, language: str = "telugu") -> list[str]:
    text = normalize_unicode(text)
    sentences: list[str] = []
    for sentence in SENTENCE_BOUNDARY_RE.split(text):
        cleaned = clean_text(sentence, language=language)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def split_telugu_sentences(text: str) -> list[str]:
    return split_sentences(text, language="telugu")


def split_hindi_sentences(text: str) -> list[str]:
    return split_sentences(text, language="hindi")


def split_english_sentences(text: str) -> list[str]:
    return split_sentences(text, language="english")


def tokenize(text: str, language: str = "telugu") -> list[str]:
    cleaned = clean_text(text, language=language)
    return cleaned.split() if cleaned else []


def read_telugu_lines(path: str | Path) -> list[str]:
    return list(iter_telugu_lines(path))


def read_hindi_lines(path: str | Path) -> list[str]:
    return list(iter_hindi_lines(path))


def read_english_lines(path: str | Path) -> list[str]:
    return list(iter_english_lines(path))


def iter_language_lines(path: str | Path, language: str = "telugu") -> Iterator[str]:
    path = Path(path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield from split_sentences(line, language=language)


def iter_telugu_lines(path: str | Path) -> Iterator[str]:
    yield from iter_language_lines(path, language="telugu")


def iter_hindi_lines(path: str | Path) -> Iterator[str]:
    yield from iter_language_lines(path, language="hindi")


def iter_english_lines(path: str | Path) -> Iterator[str]:
    yield from iter_language_lines(path, language="english")


def tail_tokens_from_file(path: str | Path, max_tokens: int = 200, language: str = "telugu") -> list[str]:
    recent_tokens: deque[str] = deque(maxlen=max_tokens)
    last_sentence_tokens: list[str] = []
    for line in iter_language_lines(path, language=language):
        tokens = tokenize(line, language=language)
        if tokens:
            last_sentence_tokens = tokens
            recent_tokens.extend(tokens)
    if last_sentence_tokens:
        return last_sentence_tokens[-max_tokens:]
    return list(recent_tokens)


def ends_with_sentence_boundary(path: str | Path) -> bool:
    last_char = ""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            for char in normalize_unicode(line):
                if not char.isspace():
                    last_char = char
    return last_char in SENTENCE_BOUNDARY_CHARS


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
