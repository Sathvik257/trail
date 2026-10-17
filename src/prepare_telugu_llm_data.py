"""Prepare Telugu-Llama2's public instruction corpora for word prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from preprocess import preprocess_corpus, write_sequence_shards
from sequence_data import load_pickle
from text_utils import clean_text, configure_utf8_stdio, project_root, split_sentences

SOURCES = [
    {
        "file": "alpaca.csv",
        "dataset": "Telugu-LLM-Labs/telugu_alpaca_yahma_cleaned_filtered_romanized",
        "sha256": "a890ac5d34f6ceee455d98e1b9a82044cbf99db760dec6ad87d18ab64dbe5b0e",
        "license": "cc-by-4.0",
    },
    {
        "file": "gpteacher.csv",
        "dataset": "Telugu-LLM-Labs/telugu_teknium_GPTeacher_general_instruct_filtered_romanized",
        "sha256": "b06688c8295a05458bf6a97548d2a925dcd45ce483cd28abaa4edb2efc82810d",
        "license": "mit",
    },
]
FIELDS = ("telugu_instruction", "telugu_input", "telugu_output")


def split_for_instruction(text: str) -> str:
    value = int(hashlib.sha256(("42:" + text).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if value < 5 else "validation" if value < 10 else "train"


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root() / "data_telugu_llm")
    args = parser.parse_args()
    base = args.data_dir.resolve()
    if (base / "provenance.json").exists():
        raise SystemExit("Prepared corpus already exists; use a fresh --data-dir to rebuild.")
    groups = {name: set() for name in ("train", "validation", "test")}
    sources = []
    for source in SOURCES:
        path = base / "source" / source["file"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != source["sha256"]:
            raise SystemExit(f"Source checksum mismatch: {path}")
        rows = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(FIELDS).issubset(reader.fieldnames or []):
                raise SystemExit(f"Missing Telugu text columns in {path}: {reader.fieldnames}")
            for row in reader:
                rows += 1
                instruction = clean_text(row["telugu_instruction"] or "", language="telugu")
                # All text from the same instruction remains in one split, across sources.
                split = split_for_instruction(instruction)
                for field in FIELDS:
                    for sentence in split_sentences(row[field] or "", language="telugu"):
                        if len(sentence.split()) >= 2:
                            groups[split].add(sentence)
        sources.append({**source, "rows": rows, "url": f"https://huggingface.co/datasets/{source['dataset']}"})
        print(f"Verified and read {rows:,} rows from {source['dataset']}", flush=True)

    # Keep held-out sentences out of training even when different instructions repeat them.
    train_overlaps = groups["train"] & (groups["validation"] | groups["test"])
    val_overlaps = groups["validation"] & groups["test"]
    groups["train"].difference_update(train_overlaps)
    groups["validation"].difference_update(val_overlaps)
    for name, sentences in groups.items():
        if not sentences:
            raise SystemExit(f"Empty split: {name}")
        lines = sorted(sentences)
        random.Random(42).shuffle(lines)
        directory = base if name == "train" else base / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "telugu_raw.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = {name: len(sentences) for name, sentences in groups.items()}
    del groups
    del train_overlaps, val_overlaps

    sentence_count, sequences, vocab_size = preprocess_corpus(
        base / "telugu_raw.txt", base / "sequences.pkl", base / "vocab.pkl", base / "ngram_model.pkl",
        max_vocab=20000, min_freq=2, seq_length=8, shard_size=50000,
        max_legacy_sequences=0, max_ngram_lines=15000, language="telugu",
    )
    vocabulary = load_pickle(base / "vocab.pkl")
    split_metrics = {"train": {"sentences": sentence_count, "sequences": sequences}}
    for name in ("validation", "test"):
        directory = base / name
        n_sentences, n_sequences, _ = write_sequence_shards(
            directory / "telugu_raw.txt", vocabulary["word_to_idx"], 8,
            directory / "sequences_manifest.json", directory / "sequence_shards",
            50000, directory / "sequences.pkl", 0, language="telugu",
        )
        split_metrics[name] = {"sentences": n_sentences, "sequences": n_sequences}
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources, "text_fields": list(FIELDS), "language": "telugu",
        "task": "next-word LSTM training (not Llama2 fine-tuning)",
        "split_method": "SHA256(seed=42, normalized instruction): 90/5/5; exact sentence overlaps removed",
        "splits": split_metrics, "unique_sentences": counts, "vocab_size": vocab_size,
        "vocab_fit": "training split only", "ngram_training_sentence_limit": 15000,
    }
    (base / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
