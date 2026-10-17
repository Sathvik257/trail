from __future__ import annotations

import argparse
from collections import OrderedDict

from sequence_data import SliceDataset, load_pickle, load_sequence_dataset
from text_utils import configure_utf8_stdio, normalize_language, project_root

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - depends on local environment.
    np = None
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


def collate_batch(batch):
    contexts, targets = zip(*batch)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a Flower Indic LSTM client.")
    parser.add_argument("client_id", type=int)
    parser.add_argument("--language", choices=["telugu", "hindi", "english"], default="telugu")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--raw-path", default="")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-samples-per-client", type=int, default=0)
    parser.add_argument("--shard-cache-size", type=int, default=4)
    return parser.parse_args()


class FlowerTeluguClient:
    def __init__(self, model, dataloader, device, lr: float, local_epochs: int) -> None:
        import flwr as fl

        class Client(fl.client.NumPyClient):
            def get_parameters(self, config=None):
                return [value.cpu().numpy() for value in model.state_dict().values()]

            def set_parameters(self, parameters):
                keys = list(model.state_dict().keys())
                state_dict = OrderedDict(
                    {key: torch.tensor(value) for key, value in zip(keys, parameters)}
                )
                model.load_state_dict(state_dict, strict=True)

            def fit(self, parameters, config):
                self.set_parameters(parameters)
                train_one_client_epoch(model, dataloader, device, lr=lr, epochs=local_epochs)
                return self.get_parameters(), len(dataloader.dataset), {}

            def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                loss = evaluate(model, dataloader, device)
                return float(loss), len(dataloader.dataset), {"loss": float(loss)}

        self.client = Client()


def train_one_client_epoch(model, dataloader, device, lr: float, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(contexts), targets)
            loss.backward()
            optimizer.step()


def evaluate(model, dataloader, device) -> float:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            total_loss += criterion(model(contexts), targets).item()
    return total_loss / max(1, len(dataloader))


def main() -> None:
    configure_utf8_stdio()
    if torch is None or np is None:
        raise SystemExit("PyTorch and NumPy are required. Run `pip install -r requirements.txt`.")

    try:
        import flwr as fl
    except ImportError as exc:
        raise SystemExit("Flower is not installed. Run `pip install -r requirements.txt`.") from exc

    from model import TeluguLSTM
    from preprocess import preprocess_corpus

    args = parse_args()
    root = project_root()
    language = normalize_language(args.language)
    if args.data_dir:
        data_dir = root / args.data_dir
    else:
        data_dirs = {"telugu": "data", "hindi": "data_hindi", "english": "data_english"}
        data_dir = root / data_dirs[language]
    raw_path = root / args.raw_path if args.raw_path else data_dir / f"{language}_training.txt"
    if not raw_path.exists():
        raw_path = data_dir / f"{language}_raw.txt"
    sequences_path = data_dir / "sequences.pkl"
    manifest_path = data_dir / "sequences_manifest.json"
    vocab_path = data_dir / "vocab.pkl"
    if not manifest_path.exists() or not vocab_path.exists():
        preprocess_corpus(
            raw_path,
            sequences_path,
            vocab_path,
            data_dir / "ngram_model.pkl",
            language=language,
        )

    dataset = load_sequence_dataset(root, cache_size=args.shard_cache_size, data_dir=data_dir)
    vocab_data = load_pickle(vocab_path)

    client_size = len(dataset) // args.num_clients
    start = args.client_id * client_size
    end = len(dataset) if args.client_id == args.num_clients - 1 else start + client_size
    if args.max_samples_per_client > 0:
        end = min(end, start + args.max_samples_per_client)
    local_dataset = SliceDataset(dataset, start, end)
    if len(local_dataset) == 0:
        raise SystemExit(f"Client {args.client_id} received no samples.")

    dataloader = DataLoader(
        local_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeluguLSTM(
        vocab_size=len(vocab_data["word_to_idx"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    wrapped = FlowerTeluguClient(model, dataloader, device, lr=args.lr, local_epochs=args.local_epochs)
    print(f"Starting {language} client {args.client_id} with {len(local_dataset)} samples")
    fl.client.start_numpy_client(server_address=args.server_address, client=wrapped.client)


if __name__ == "__main__":
    main()
