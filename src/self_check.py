from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from agentic_predictor import AgenticPredictionService
from text_utils import configure_utf8_stdio, project_root

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None


DIRECT_PACKAGES = ["torch", "flwr", "datasets", "streamlit", "numpy", "tqdm"]
REQUIRED_FILES = [
    "data/telugu_raw.txt",
    "data/telugu_training.txt",
    "data/sequences_manifest.json",
    "data/vocab.pkl",
    "data/ngram_model.pkl",
    "data/curated_ngram_model.pkl",
    "data_hindi/hindi_raw.txt",
    "data_hindi/hindi_training.txt",
    "data_hindi/sequences_manifest.json",
    "data_hindi/vocab.pkl",
    "data_hindi/ngram_model.pkl",
    "data_hindi/curated_ngram_model.pkl",
    "data_english/english_raw.txt",
    "data_english/english_training.txt",
    "data_english/sequences_manifest.json",
    "data_english/vocab.pkl",
    "data_english/ngram_model.pkl",
    "data_english/curated_ngram_model.pkl",
    "outputs/centralized_lstm.pt",
    "outputs/hindi_lstm.pt",
    "outputs/english_lstm.pt",
    "src/app.py",
    "src/predict.py",
    "src/fl_server.py",
    "src/fl_client.py",
]
SAMPLE_INPUTS = {
    "telugu": [
        "నేను తెలుగు",
        "సర్వర్ ప్రతి రౌండ్ తర్వాత కొత్త మోడల్",
        "మొబైల్ కీబోర్డ్ తెలుగు పదాలను",
        "ఫెడరేటెడ్ లెర్నింగ్ లో",
    ],
    "hindi": [
        "मैं हिंदी",
        "सर्वर हर राउंड के बाद नया मॉडल",
        "मोबाइल कीबोर्ड",
        "फेडरेटेड लर्निंग में",
    ],
    "english": [
        "I am learning English",
        "federated learning protects",
        "mobile keyboard",
        "in India",
    ],
}
LANGUAGE_RANGES = {
    "telugu": ("\u0c00", "\u0c7f"),
    "hindi": ("\u0900", "\u097f"),
    "english": ("A", "z"),
}
CHECKPOINTS = {
    "telugu": "centralized_lstm.pt",
    "hindi": "hindi_lstm.pt",
    "english": "english_lstm.pt",
}
DATA_DIRS = {
    "telugu": "data",
    "hindi": "data_hindi",
    "english": "data_english",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project health checks.")
    parser.add_argument("--health-url", default="http://localhost:8501/_stcore/health")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--require-health", action="store_true")
    return parser.parse_args()


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def has_language_char(text: str, language: str) -> bool:
    start, end = LANGUAGE_RANGES[language]
    return any(start <= char <= end for char in text)


def check_packages() -> bool:
    success = True
    for package in DIRECT_PACKAGES:
        try:
            version = metadata.version(package)
            ok(f"{package} {version}")
        except metadata.PackageNotFoundError:
            fail(f"Missing package: {package}")
            success = False
    return success


def check_files(root: Path) -> bool:
    success = True
    for relative_path in REQUIRED_FILES:
        path = root / relative_path
        if path.exists():
            ok(f"Found {relative_path}")
        else:
            fail(f"Missing {relative_path}")
            success = False
    return success


def check_manifest(root: Path) -> bool:
    success = True
    for language, data_dir in DATA_DIRS.items():
        manifest_path = root / data_dir / "sequences_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        total_sequences = int(manifest.get("total_sequences", 0))
        total_sentences = int(manifest.get("total_sentences", 0))
        shards = manifest.get("shards", [])
        if total_sequences <= 0 or total_sentences <= 0 or not shards:
            fail(f"{language} preprocessing manifest is incomplete")
            success = False
            continue

        ok(
            f"{language}: prepared {total_sequences:,} sequences from "
            f"{total_sentences:,} cleaned sentences across {len(shards)} shards"
        )
    return success


def check_checkpoint(root: Path) -> bool:
    if torch is None:
        fail("PyTorch is not importable")
        return False

    success = True
    required_keys = [
        "model_state_dict",
        "word_to_idx",
        "idx_to_word",
        "seq_length",
        "epoch",
        "best_val_loss",
    ]
    for language, filename in CHECKPOINTS.items():
        checkpoint_path = root / "outputs" / filename
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        missing = [key for key in required_keys if key not in checkpoint]
        if missing:
            fail(f"{language} checkpoint missing keys: {', '.join(missing)}")
            success = False
            continue

        ok(
            f"{language}: checkpoint epoch "
            f"{checkpoint['epoch']} | vocab {len(checkpoint['word_to_idx']):,} | "
            f"seq_length {checkpoint['seq_length']} | best_val_loss {checkpoint['best_val_loss']:.4f}"
        )
    return success


def check_predictions(root: Path) -> bool:
    success = True
    for language, samples in SAMPLE_INPUTS.items():
        service = AgenticPredictionService(root, language=language)
        for sample in samples:
            word_result = service.predict(sample, top_k=5, mode="auto")
            sentence_result = service.predict_sentences(sample, count=5, mode="auto")
            print(f"Input ({language}): {sample}")
            print(f"Words: {', '.join(word_result.predictions)}")
            print(f"Sentences: {' | '.join(sentence_result.predictions)}")
            if len(word_result.predictions) < 5:
                fail(f"Expected 5 word predictions for: {sample}")
                success = False
            elif not all(has_language_char(word, language) for word in word_result.predictions):
                fail(f"Wrong-language word prediction found for: {sample}")
                success = False
            elif len(sentence_result.predictions) < 5:
                fail(f"Expected 5 sentence predictions for: {sample}")
                success = False
            elif not all(has_language_char(sentence, language) for sentence in sentence_result.predictions):
                fail(f"Wrong-language sentence prediction found for: {sample}")
                success = False
            else:
                ok(f"{language} word + sentence prediction check passed for: {sample}")
    return success


def check_streamlit(url: str, required: bool) -> bool:
    try:
        with urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
        if response.status == 200 and body == "ok":
            ok(f"Streamlit health endpoint is live: {url}")
            return True
        message = f"Unexpected Streamlit health response: {response.status} {body}"
    except (OSError, URLError) as exc:
        message = f"Streamlit health endpoint is not reachable: {exc}"

    if required:
        fail(message)
        return False
    warn(message)
    return True


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    root = project_root()

    checks = [
        check_packages(),
        check_files(root),
        check_manifest(root),
        check_checkpoint(root),
        check_predictions(root),
    ]
    if not args.skip_health:
        checks.append(check_streamlit(args.health_url, required=args.require_health))

    if all(checks):
        print("\nOverall: PASS")
        return

    print("\nOverall: FAIL")
    sys.exit(1)


if __name__ == "__main__":
    main()
