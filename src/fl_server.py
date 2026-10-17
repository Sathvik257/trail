from __future__ import annotations

import argparse

from text_utils import configure_utf8_stdio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Flower FedAvg server.")
    parser.add_argument("--address", default="127.0.0.1:8080")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    try:
        import flwr as fl
    except ImportError as exc:
        raise SystemExit("Flower is not installed. Run `pip install -r requirements.txt`.") from exc

    args = parse_args()
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )
    print(f"Starting Flower server at {args.address} for {args.rounds} rounds")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
