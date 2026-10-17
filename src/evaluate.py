from __future__ import annotations

import argparse
from pathlib import Path

from predict import predict_next_words
from text_utils import configure_utf8_stdio, iter_language_lines, project_root, tokenize


def iter_examples(raw_path: str | Path, min_context: int, max_examples: int, language: str):
    produced = 0
    for line in iter_language_lines(raw_path, language=language):
        tokens = tokenize(line, language=language)
        if len(tokens) <= min_context:
            continue
        for idx in range(min_context, len(tokens)):
            context = " ".join(tokens[:idx])
            target = tokens[idx]
            yield context, target
            produced += 1
            if max_examples and produced >= max_examples:
                return


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate Indic next-word prediction quality.")
    parser.add_argument("--language", choices=["telugu", "hindi", "english"], default="telugu")
    parser.add_argument("--raw-path", default=str(root / "data" / "telugu_raw.txt"))
    parser.add_argument("--mode", choices=["auto", "lstm", "ngram"], default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-context", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--show-misses", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()

    total = 0
    top1_hits = 0
    top3_hits = 0
    topk_hits = 0
    misses: list[tuple[str, str, list[str]]] = []

    for context, target in iter_examples(args.raw_path, args.min_context, args.max_examples, args.language):
        predictions, mode = predict_next_words(context, top_k=args.top_k, mode=args.mode, language=args.language)
        total += 1
        if predictions[:1] == [target]:
            top1_hits += 1
        if target in predictions[:3]:
            top3_hits += 1
        if target in predictions:
            topk_hits += 1
        elif len(misses) < args.show_misses:
            misses.append((context, target, predictions))

    if total == 0:
        raise SystemExit("No evaluation examples found. Add more Telugu text or lower --min-context.")

    print(f"Examples: {total}")
    print(f"Mode: {args.mode}")
    print(f"Top-1 accuracy: {top1_hits / total:.3f}")
    print(f"Top-3 accuracy: {top3_hits / total:.3f}")
    print(f"Top-{args.top_k} accuracy: {topk_hits / total:.3f}")

    if misses:
        print("\nSample misses:")
        for context, target, predictions in misses:
            joined = ", ".join(predictions) if predictions else "(none)"
            print(f"- Context: {context}")
            print(f"  Target: {target}")
            print(f"  Predicted: {joined}")


if __name__ == "__main__":
    main()
