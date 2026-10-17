from __future__ import annotations

import argparse
import json

from agentic_predictor import AgenticPredictionService
from text_utils import configure_utf8_stdio, ends_with_sentence_boundary, project_root, tail_tokens_from_file


def predict_next_words(
    text: str,
    top_k: int = 3,
    mode: str = "auto",
    model_path: str | None = None,
    language: str = "telugu",
) -> tuple[list[str], str]:
    # model_path is kept for backward compatibility with earlier scripts.
    result = AgenticPredictionService(project_root(), language=language).predict(text, top_k=top_k, mode=mode)
    return result.predictions, result.strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Telugu or Hindi next words.")
    parser.add_argument("words", nargs="*", help="Input words.")
    parser.add_argument("--language", choices=["telugu", "hindi", "english"], default="telugu")
    parser.add_argument("--text", help="Input sentence.")
    parser.add_argument("--file", help="Read a large text file and predict from its final context.")
    parser.add_argument("--keep-final-token", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sentences", action="store_true", help="Also print full sentence continuations.")
    parser.add_argument("--sentence-count", type=int, default=5)
    parser.add_argument("--max-new-words", type=int, default=8)
    parser.add_argument("--mode", choices=["auto", "curated", "ngram", "lstm"], default="auto")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_input_text(args: argparse.Namespace) -> str:
    if args.file:
        file_tokens = tail_tokens_from_file(args.file, max_tokens=200, language=args.language)
        if not args.keep_final_token and len(file_tokens) > 1 and ends_with_sentence_boundary(args.file):
            file_tokens = file_tokens[:-1]
        return " ".join(file_tokens)
    return args.text or " ".join(args.words)


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    text = resolve_input_text(args).strip()
    if not text:
        prompts = {
            "telugu": "తెలుగు పదాలు ఇవ్వండి: ",
            "hindi": "हिंदी शब्द दर्ज करें: ",
            "english": "Enter English words: ",
        }
        prompt = prompts[args.language]
        text = input(prompt).strip()

    service = AgenticPredictionService(project_root(), language=args.language)
    result = service.predict(
        text,
        top_k=args.top_k,
        mode=args.mode,
    )
    sentence_result = (
        service.predict_sentences(
            text,
            count=args.sentence_count,
            max_new_words=args.max_new_words,
            mode=args.mode,
        )
        if args.sentences
        else None
    )
    if args.json:
        print(
            json.dumps(
                {
                    "language": args.language,
                    "input": result.input_text,
                    "strategy": result.strategy,
                    "confidence": result.confidence,
                    "predictions": result.predictions,
                    "sentence_predictions": sentence_result.predictions if sentence_result else [],
                },
                ensure_ascii=False,
            )
        )
        return

    labels = {
        "telugu": {
            "input": "ఇచ్చిన వాక్యం",
            "next": "తదుపరి పదాలు",
            "strategy": "ఎంచుకున్న విధానం",
            "confidence": "నమ్మకం",
            "sentences": "వాక్య సూచనలు",
            "none": "సరైన సూచనలు దొరకలేదు",
        },
        "hindi": {
            "input": "दिया गया वाक्य",
            "next": "अगले शब्द",
            "strategy": "चुनी गई विधि",
            "confidence": "विश्वास",
            "sentences": "वाक्य सुझाव",
            "none": "सही सुझाव नहीं मिले",
        },
        "english": {
            "input": "Input text",
            "next": "Next words",
            "strategy": "Selected method",
            "confidence": "Confidence",
            "sentences": "Sentence suggestions",
            "none": "No useful suggestions found",
        },
    }[args.language]

    print(f"{labels['input']}: {result.input_text}")
    if args.verbose:
        print(f"{labels['strategy']}: {result.strategy}")
        print(f"{labels['confidence']}: {result.confidence:.2f}")
    print(f"{labels['next']}:")
    if not result.predictions:
        print(labels["none"])
        return
    for rank, word in enumerate(result.predictions, start=1):
        print(f"{rank}. {word}")

    if sentence_result is not None:
        print(f"{labels['sentences']}:")
        if not sentence_result.predictions:
            print(labels["none"])
            return
        for rank, sentence in enumerate(sentence_result.predictions, start=1):
            print(f"{rank}. {sentence}")


if __name__ == "__main__":
    main()
