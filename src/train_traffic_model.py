"""Train and persist the lightweight traffic-spike model."""

import argparse
from pathlib import Path

from .gen_td26 import load_gen_td26_requests
from .traffic import generate_requests
from .traffic_model import train_traffic_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/traffic_model.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--requests", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--window-size", type=float, default=1.0)
    args = parser.parse_args()

    if args.dataset:
        requests = load_gen_td26_requests(
            args.dataset, max_requests=args.max_requests or args.requests
        )
    else:
        requests = generate_requests(
            count=args.requests, seed=args.seed, arrival_rate=args.arrival_rate
        )
    model = train_traffic_model(requests, window_size=args.window_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Trained model on {len(requests)} requests: {args.output}")


if __name__ == "__main__":
    main()
