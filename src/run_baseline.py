"""Run the reproducible FIFO baseline experiment."""

import argparse
import json

from .fifo import calculate_metrics, schedule_fifo
from .traffic import generate_requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FIFO on synthetic API traffic")
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    args = parser.parse_args()

    requests = generate_requests(
        count=args.requests,
        seed=args.seed,
        arrival_rate=args.arrival_rate,
    )
    scheduled = schedule_fifo(requests)
    metrics = calculate_metrics(requests, scheduled)
    print(json.dumps(metrics.as_dict(), indent=2))


if __name__ == "__main__":
    main()
