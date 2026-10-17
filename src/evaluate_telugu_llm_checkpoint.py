"""Evaluate a completed Telugu next-word checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from centralized_train import collate_batch
from model import TeluguLSTM
from sequence_data import load_sequence_dataset
from text_utils import project_root


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "outputs" / "telugu_llm_lstm.pt")
    parser.add_argument("--test-data-dir", type=Path, default=root / "data_telugu_llm" / "test")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "telugu_llm_test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    total_batches = checkpoint.get("total_batches", 0)
    if checkpoint.get("epoch", 0) < checkpoint["training_config"]["epochs"] or checkpoint.get("partial_batch", 0):
        raise SystemExit(f"Checkpoint is incomplete (epoch {checkpoint.get('epoch')}, batch {checkpoint.get('partial_batch')}/{total_batches}).")
    vocab = checkpoint["word_to_idx"]
    model = TeluguLSTM(
        len(vocab), checkpoint["embedding_dim"], checkpoint["hidden_dim"],
        checkpoint["num_layers"], checkpoint["dropout"], vocab.get("<pad>", 0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    data = load_sequence_dataset(root, cache_size=1, data_dir=args.test_data_dir)
    batches = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    total = top1 = top5 = 0
    cross_entropy = 0.0
    with torch.inference_mode():
        for contexts, targets in batches:
            logits = model(contexts)
            cross_entropy += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            top1 += (logits.argmax(dim=1) == targets).sum().item()
            top5 += (logits.topk(5, dim=1).indices == targets[:, None]).any(dim=1).sum().item()
            total += len(targets)
    loss = cross_entropy / total
    metrics = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
        "test_sequences": total, "loss": loss, "perplexity": math.exp(loss),
        "top1_accuracy": top1 / total, "top5_accuracy": top5 / total,
        "dataset_provenance": checkpoint.get("dataset_provenance"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
