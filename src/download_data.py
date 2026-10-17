from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from text_utils import clean_text, configure_utf8_stdio, project_root


DEFAULT_SPLITS = {
    "telugu": "tel_Telu",
    "hindi": "hin_Deva",
    "english": "train",
}
DEFAULT_DATASETS = {
    "telugu": ("ai4bharat/IndicCorpV2", "indiccorp_v2"),
    "hindi": ("ai4bharat/IndicCorpV2", "indiccorp_v2"),
    "english": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
}
DEFAULT_DATA_DIRS = {
    "telugu": "data",
    "hindi": "data_hindi",
    "english": "data_english",
}


def extract_text(row: dict) -> str:
    for key in ("text", "content", "sentence"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download language-model text data.")
    parser.add_argument("--language", choices=["telugu", "hindi", "english"], default="telugu")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--subset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    if os.name == "nt" and not sys.flags.utf8_mode:
        os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

    configure_utf8_stdio()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The `datasets` package is missing. Run `pip install -r requirements.txt`.") from exc

    args = parse_args()
    root = project_root()
    default_dataset, default_subset = DEFAULT_DATASETS[args.language]
    dataset_name = args.dataset or default_dataset
    subset = args.subset or default_subset
    split = args.split or DEFAULT_SPLITS[args.language]
    if args.output:
        output_path = Path(args.output)
    else:
        data_dir = DEFAULT_DATA_DIRS[args.language]
        output_path = root / data_dir / f"{args.language}_raw_large.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {"split": split, "streaming": True}
    if args.language != "english":
        dataset_kwargs["encoding"] = "utf-8"
    dataset = load_dataset(dataset_name, subset, **dataset_kwargs)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            cleaned = clean_text(extract_text(row), language=args.language)
            if not cleaned:
                continue
            handle.write(cleaned + "\n")
            written += 1
            if written >= args.limit:
                break

    print(f"Downloaded {args.language} lines from {dataset_name}/{subset}/{split}: {written}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
