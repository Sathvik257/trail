"""Extract Telugu next-word training text from TeSent's train CSV.

TeSent is a sentiment dataset. Only its Content text is used here; labels,
rationales, validation, and test rows are deliberately excluded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from text_utils import split_sentences, tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TeSent Telugu training text")
    parser.add_argument("train_csv", type=Path, help="TeSent train.csv")
    parser.add_argument("--output", type=Path, default=Path("data_tesent/telugu_raw.txt"))
    args = parser.parse_args()

    if not args.train_csv.is_file():
        parser.error(f"TeSent training CSV not found: {args.train_csv}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = sentences = 0
    with args.train_csv.open("r", encoding="utf-8-sig", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "Content" not in reader.fieldnames:
            parser.error("TeSent CSV must contain a Content column")
        for row in reader:
            rows += 1
            for sentence in split_sentences(row.get("Content") or "", language="telugu"):
                if len(tokenize(sentence, language="telugu")) >= 2:
                    destination.write(sentence + "\n")
                    sentences += 1

    if not sentences:
        parser.error("No Telugu sentences with at least two words were found")
    print(f"Prepared {sentences} Telugu sentences from {rows} TeSent training rows: {args.output}")


if __name__ == "__main__":
    main()
