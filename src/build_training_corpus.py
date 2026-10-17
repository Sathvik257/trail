from __future__ import annotations

import argparse
from pathlib import Path

from text_utils import configure_utf8_stdio, iter_language_lines, project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Combine language corpora for stronger training.")
    parser.add_argument("--language", choices=["telugu", "hindi", "english"], default="telugu")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input text file. Pass multiple times to combine files.",
    )
    parser.add_argument("--output", default=str(root / "data" / "telugu_training.txt"))
    parser.add_argument(
        "--repeat-first",
        type=int,
        default=10,
        help="Repeat the first input corpus to preserve small project-specific examples.",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as output:
        for index, input_path in enumerate(args.input):
            repeat = args.repeat_first if index == 0 else 1
            for _ in range(max(1, repeat)):
                for line in iter_language_lines(input_path, language=args.language):
                    output.write(line + "\n")
                    written += 1

    print(f"Combined {args.language} lines: {written}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
