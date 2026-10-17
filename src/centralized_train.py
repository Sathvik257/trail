from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from preprocess import preprocess_corpus
from sequence_data import SliceDataset, load_pickle, load_sequence_dataset
from text_utils import configure_utf8_stdio, normalize_language, project_root

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


def ensure_preprocessed(root: Path, data_dir: Path, raw_path: Path, language: str) -> tuple[Path, Path]:
    sequences_path = data_dir / "sequences.pkl"
    manifest_path = data_dir / "sequences_manifest.json"
    vocab_path = data_dir / "vocab.pkl"
    ngram_path = data_dir / "ngram_model.pkl"
    if not manifest_path.exists() or not vocab_path.exists():
        preprocess_corpus(raw_path, sequences_path, vocab_path, ngram_path, language=language)
    return manifest_path, vocab_path


def collate_batch(batch):
    contexts, targets = zip(*batch)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train centralized Indic LSTM baseline.")
    parser.add_argument("--language", choices=["telugu", "hindi", "english"], default="telugu")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--raw-path", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional small-run limit for testing.")
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--validation-data-dir", default="", help="Held-out sequences encoded with the training vocabulary.")
    parser.add_argument("--provenance", default="", help="Dataset source/split JSON to embed in the checkpoint.")
    parser.add_argument("--threads", type=int, default=0, help="CPU thread limit; zero uses PyTorch's default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-cache-size", type=int, default=4)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle batches. Slower for large sharded data.")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every-batches", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    if torch is None:
        raise SystemExit("PyTorch is not installed. Run `pip install -r requirements.txt` first.")

    from model import TeluguLSTM

    args = parse_args()
    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    root = project_root()
    language = normalize_language(args.language)
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = root / data_dir
    else:
        data_dirs = {"telugu": "data", "hindi": "data_hindi", "english": "data_english"}
        data_dir = root / data_dirs[language]
    if args.raw_path:
        raw_path = Path(args.raw_path)
        if not raw_path.is_absolute():
            raw_path = root / raw_path
    else:
        training_path = data_dir / f"{language}_training.txt"
        raw_path = training_path if training_path.exists() else data_dir / f"{language}_raw.txt"
    checkpoint_names = {
        "telugu": "centralized_lstm.pt",
        "hindi": "hindi_lstm.pt",
        "english": "english_lstm.pt",
    }
    output_path = Path(args.output) if args.output else root / "outputs" / checkpoint_names[language]
    if not output_path.is_absolute():
        output_path = root / output_path

    _, vocab_path = ensure_preprocessed(root, data_dir=data_dir, raw_path=raw_path, language=language)
    dataset = load_sequence_dataset(root, cache_size=args.shard_cache_size, data_dir=data_dir)
    vocab_data = load_pickle(vocab_path)

    if args.max_samples > 0:
        dataset = SliceDataset(dataset, 0, args.max_samples)

    if args.validation_data_dir:
        train_dataset = dataset
        val_dataset = load_sequence_dataset(root, cache_size=args.shard_cache_size, data_dir=args.validation_data_dir)
    else:
        val_count = int(len(dataset) * args.validation_split)
        train_count = len(dataset) - val_count
        train_dataset = SliceDataset(dataset, 0, train_count)
        val_dataset = SliceDataset(dataset, train_count, len(dataset)) if val_count else None
    if not len(train_dataset) or (val_dataset is not None and not len(val_dataset)):
        raise SystemExit("Training and requested validation datasets must not be empty.")

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
        if val_dataset is not None
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}: {len(train_dataset):,} training sequences; "
          f"{len(val_dataset) if val_dataset is not None else 0:,} validation sequences", flush=True)

    model = TeluguLSTM(
        vocab_size=len(vocab_data["word_to_idx"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        for batch_idx, (contexts, targets) in enumerate(dataloader, start=1):
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            total_loss += loss.item()
            if args.log_every > 0 and batch_idx % args.log_every == 0:
                running_loss = total_loss / batch_idx
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch}/{args.epochs} - batch {batch_idx}/{len(dataloader)} - "
                    f"train loss: {running_loss:.4f} - elapsed: {elapsed:.1f}s",
                    flush=True,
                )
            if args.save_every_batches > 0 and batch_idx % args.save_every_batches == 0:
                save_checkpoint(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    vocab_data=vocab_data,
                    output_path=output_path,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    train_samples=len(train_dataset),
                    val_samples=len(val_dataset) if val_dataset is not None else 0,
                    partial_batch=batch_idx,
                    total_batches=len(dataloader),
                )
                print(
                    f"Checkpoint saved at epoch {epoch}, batch {batch_idx}/{len(dataloader)}",
                    flush=True,
                )

        avg_loss = total_loss / max(1, len(dataloader))
        val_loss = evaluate(model, val_loader, criterion, device) if val_loader is not None else None
        if val_loss is not None:
            best_val_loss = min(best_val_loss, val_loss)
            print(
                f"Epoch {epoch}/{args.epochs} - train loss: {avg_loss:.4f} - "
                f"val loss: {val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch}/{args.epochs} - train loss: {avg_loss:.4f}")

        save_checkpoint(
            args=args,
            model=model,
            optimizer=optimizer,
            vocab_data=vocab_data,
            output_path=output_path,
            epoch=epoch,
            best_val_loss=best_val_loss,
            train_samples=len(train_dataset),
            val_samples=len(val_dataset) if val_dataset is not None else 0,
        )

    print(f"Model saved: {output_path}")


def save_checkpoint(
    args,
    model,
    optimizer,
    vocab_data,
    output_path: Path,
    epoch: int,
    best_val_loss: float,
    train_samples: int,
    val_samples: int,
    partial_batch: int = 0,
    total_batches: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = None
    if args.provenance:
        provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word_to_idx": vocab_data["word_to_idx"],
            "idx_to_word": vocab_data["idx_to_word"],
            "seq_length": vocab_data["seq_length"],
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "training_samples": train_samples,
            "validation_samples": val_samples,
            "language": getattr(args, "language", "telugu"),
            "partial_batch": partial_batch,
            "total_batches": total_batches,
            "dataset_provenance": provenance,
            "training_config": vars(args),
        },
        temporary_path,
    )
    temporary_path.replace(output_path)


def evaluate(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            total_loss += criterion(model(contexts), targets).item()
    return total_loss / max(1, len(dataloader))


if __name__ == "__main__":
    main()
