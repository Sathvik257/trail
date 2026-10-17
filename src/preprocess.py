from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from ngram_predictor import build_ngram_model
from text_utils import configure_utf8_stdio, iter_language_lines, normalize_language, project_root, tokenize

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocab(sentences: list[list[str]], max_vocab: int, min_freq: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for tokens in sentences:
        counts.update(tokens)

    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counts.most_common(max_vocab - len(word_to_idx)):
        if count < min_freq:
            break
        word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def count_words(raw_path: str | Path, language: str = "telugu") -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    sentence_count = 0
    for line in iter_language_lines(raw_path, language=language):
        tokens = tokenize(line, language=language)
        if len(tokens) < 2:
            continue
        counts.update(tokens)
        sentence_count += 1
    return counts, sentence_count


def build_vocab_from_counts(counts: Counter[str], max_vocab: int, min_freq: int) -> dict[str, int]:
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counts.most_common(max_vocab - len(word_to_idx)):
        if count < min_freq:
            break
        word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def make_sequences(sentences: list[list[str]], word_to_idx: dict[str, int], seq_length: int) -> list[tuple[list[int], int]]:
    unk_id = word_to_idx[UNK_TOKEN]
    pad_id = word_to_idx[PAD_TOKEN]
    sequences: list[tuple[list[int], int]] = []

    for tokens in sentences:
        ids = [word_to_idx.get(token, unk_id) for token in tokens]
        for idx in range(1, len(ids)):
            context = ids[max(0, idx - seq_length) : idx]
            if len(context) < seq_length:
                context = [pad_id] * (seq_length - len(context)) + context
            sequences.append((context, ids[idx]))

    return sequences


def make_line_sequences(tokens: list[str], word_to_idx: dict[str, int], seq_length: int) -> list[tuple[list[int], int]]:
    unk_id = word_to_idx[UNK_TOKEN]
    pad_id = word_to_idx[PAD_TOKEN]
    ids = [word_to_idx.get(token, unk_id) for token in tokens]
    sequences: list[tuple[list[int], int]] = []
    for idx in range(1, len(ids)):
        context = ids[max(0, idx - seq_length) : idx]
        if len(context) < seq_length:
            context = [pad_id] * (seq_length - len(context)) + context
        sequences.append((context, ids[idx]))
    return sequences


def write_sequence_shards(
    raw_path: str | Path,
    word_to_idx: dict[str, int],
    seq_length: int,
    manifest_path: str | Path,
    shard_dir: str | Path,
    shard_size: int,
    legacy_path: str | Path,
    max_legacy_sequences: int,
    language: str = "telugu",
) -> tuple[int, int, bool]:
    manifest_path = Path(manifest_path)
    shard_dir = Path(shard_dir)
    legacy_path = Path(legacy_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, object]] = []
    current: list[tuple[list[int], int]] = []
    legacy_samples: list[tuple[list[int], int]] = []
    sentence_count = 0
    sequence_count = 0
    wrote_legacy = True

    def flush_shard() -> None:
        if not current:
            return
        shard_name = f"sequences_{len(shards):05d}.pkl"
        shard_path = shard_dir / shard_name
        with shard_path.open("wb") as handle:
            pickle.dump(list(current), handle, protocol=pickle.HIGHEST_PROTOCOL)
        shards.append(
            {
                "path": str(shard_path.relative_to(manifest_path.parent)),
                "count": len(current),
            }
        )
        current.clear()

    for line in iter_language_lines(raw_path, language=language):
        tokens = tokenize(line, language=language)
        if len(tokens) < 2:
            continue
        sentence_count += 1
        line_sequences = make_line_sequences(tokens, word_to_idx, seq_length)
        sequence_count += len(line_sequences)
        current.extend(line_sequences)

        if wrote_legacy and sequence_count <= max_legacy_sequences:
            legacy_samples.extend(line_sequences)
        elif wrote_legacy:
            wrote_legacy = False
            legacy_samples.clear()

        while len(current) >= shard_size:
            overflow = current[shard_size:]
            del current[shard_size:]
            flush_shard()
            current.extend(overflow)

    flush_shard()

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 1,
                "seq_length": seq_length,
                "total_sentences": sentence_count,
                "total_sequences": sequence_count,
                "shards": shards,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    if wrote_legacy:
        with legacy_path.open("wb") as handle:
            pickle.dump(legacy_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return sentence_count, sequence_count, wrote_legacy


def preprocess_corpus(
    raw_path: str | Path,
    sequences_path: str | Path,
    vocab_path: str | Path,
    ngram_path: str | Path,
    manifest_path: str | Path | None = None,
    shard_dir: str | Path | None = None,
    max_vocab: int = 30000,
    min_freq: int = 1,
    seq_length: int = 8,
    shard_size: int = 100000,
    max_legacy_sequences: int = 250000,
    max_ngram_lines: int = 15000,
    ngram_order: int = 5,
    ngram_predictions_per_context: int = 8,
    language: str = "telugu",
) -> tuple[int, int, int]:
    language = normalize_language(language)
    raw_path = Path(raw_path)
    sequences_path = Path(sequences_path)
    vocab_path = Path(vocab_path)
    manifest_path = Path(manifest_path) if manifest_path else sequences_path.parent / "sequences_manifest.json"
    shard_dir = Path(shard_dir) if shard_dir else sequences_path.parent / "sequence_shards"

    counts, counted_sentences = count_words(raw_path, language=language)
    if counted_sentences == 0:
        raise ValueError(f"No usable {language} sentences found in {raw_path}")

    word_to_idx = build_vocab_from_counts(counts, max_vocab=max_vocab, min_freq=min_freq)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with vocab_path.open("wb") as handle:
        pickle.dump(
            {
                "word_to_idx": word_to_idx,
                "idx_to_word": idx_to_word,
                "seq_length": seq_length,
                "max_vocab": max_vocab,
                "min_freq": min_freq,
                "total_token_count": sum(counts.values()),
                "language": language,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    sentence_count, sequence_count, wrote_legacy = write_sequence_shards(
        raw_path=raw_path,
        word_to_idx=word_to_idx,
        seq_length=seq_length,
        manifest_path=manifest_path,
        shard_dir=shard_dir,
        shard_size=shard_size,
        legacy_path=sequences_path,
        max_legacy_sequences=max_legacy_sequences,
        language=language,
    )

    build_ngram_model(
        raw_path,
        ngram_path,
        max_order=ngram_order,
        max_lines=max_ngram_lines,
        max_next_per_context=ngram_predictions_per_context,
        language=language,
    )
    if not wrote_legacy and sequences_path.exists():
        sequences_path.unlink()
    return sentence_count, sequence_count, len(word_to_idx)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Preprocess Indic text for next-word prediction.")
    parser.add_argument("--language", choices=["telugu", "hindi", "english"], default="telugu")
    parser.add_argument("--raw-path", default=str(root / "data" / "telugu_raw.txt"))
    parser.add_argument("--sequences-path", default=str(root / "data" / "sequences.pkl"))
    parser.add_argument("--manifest-path", default=str(root / "data" / "sequences_manifest.json"))
    parser.add_argument("--shard-dir", default=str(root / "data" / "sequence_shards"))
    parser.add_argument("--vocab-path", default=str(root / "data" / "vocab.pkl"))
    parser.add_argument("--ngram-path", default=str(root / "data" / "ngram_model.pkl"))
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=100000)
    parser.add_argument("--max-legacy-sequences", type=int, default=250000)
    parser.add_argument("--max-ngram-lines", type=int, default=15000)
    parser.add_argument("--ngram-order", type=int, default=5)
    parser.add_argument("--ngram-predictions-per-context", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    sentence_count, sequence_count, vocab_size = preprocess_corpus(
        raw_path=args.raw_path,
        sequences_path=args.sequences_path,
        vocab_path=args.vocab_path,
        ngram_path=args.ngram_path,
        manifest_path=args.manifest_path,
        shard_dir=args.shard_dir,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        seq_length=args.seq_length,
        shard_size=args.shard_size,
        max_legacy_sequences=args.max_legacy_sequences,
        max_ngram_lines=args.max_ngram_lines,
        ngram_order=args.ngram_order,
        ngram_predictions_per_context=args.ngram_predictions_per_context,
        language=args.language,
    )
    print(f"Cleaned sentences: {sentence_count}")
    print(f"Training sequences: {sequence_count}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Saved sequences: {args.sequences_path}")
    print(f"Saved sharded manifest: {args.manifest_path}")
    print(f"Saved vocab: {args.vocab_path}")
    print(f"Saved n-gram model: {args.ngram_path}")


if __name__ == "__main__":
    main()
