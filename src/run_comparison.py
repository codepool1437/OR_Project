"""Compare FIFO and EDF on one identical synthetic workload."""

import argparse
import json

from .drr import schedule_drr
from .edf import schedule_edf
from .edf_drr import schedule_edf_drr
from .fifo import calculate_metrics, schedule_fifo
from .parallel import schedule_parallel
from .gen_td26 import load_gen_td26_requests
from .traffic import generate_requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FIFO, EDF, DRR, and hybrid EDF+DRR scheduling"
    )
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="path to GenTD26 lora_request_trace.csv; otherwise use synthetic traffic",
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument(
        "--horizon",
        type=float,
        default=None,
        help="simulation end time; omit to count all requests to completion",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel workers; values above 1 use the parallel dispatch simulator",
    )
    args = parser.parse_args()

    if args.dataset:
        requests = load_gen_td26_requests(
            args.dataset,
            max_requests=args.max_requests or args.requests,
        )
    else:
        requests = generate_requests(
            count=args.requests,
            seed=args.seed,
            arrival_rate=args.arrival_rate,
        )
    if args.workers == 1:
        schedules = {
            "FIFO": schedule_fifo(requests),
            "EDF": schedule_edf(requests),
            "DRR": schedule_drr(requests),
            "EDF+DRR": schedule_edf_drr(requests),
        }
    else:
        schedules = {
            policy: schedule_parallel(requests, policy, args.workers)
            for policy in ("FIFO", "EDF", "DRR", "EDF+DRR")
        }
    results = {
        name: calculate_metrics(
            requests,
            schedule,
            cpu_capacity=100.0 * args.workers,
            horizon=args.horizon,
        ).as_dict()
        for name, schedule in schedules.items()
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
