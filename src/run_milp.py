"""Compare live schedulers with the offline MILP benchmark."""

import argparse
import json

from .drr import schedule_drr
from .edf import schedule_edf
from .edf_drr import schedule_edf_drr
from .fifo import calculate_metrics, schedule_fifo
from .gen_td26 import load_gen_td26_requests
from .milp import solve_milp
from .traffic import generate_requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate schedulers against MILP on synthetic or GenTD26 requests"
    )
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="path to GenTD26 lora_request_trace.csv",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="sample size for GenTD26; keep small because MILP is exact",
    )
    parser.add_argument("--horizon", type=float, default=None)
    args = parser.parse_args()
    if args.dataset:
        requests = load_gen_td26_requests(
            args.dataset,
            max_requests=args.max_requests or args.requests,
        )
        source = "GenTD26"
    else:
        requests = generate_requests(
            count=args.requests, seed=args.seed, arrival_rate=args.arrival_rate
        )
        source = "synthetic"
    schedules = {
        "FIFO": schedule_fifo(requests),
        "EDF": schedule_edf(requests),
        "DRR": schedule_drr(requests),
        "EDF+DRR": schedule_edf_drr(requests),
        "MILP": solve_milp(requests),
    }
    output = {
        "source": source,
        "request_count": len(requests),
        "results": {
            name: calculate_metrics(
                requests, schedule, horizon=args.horizon
            ).as_dict()
            for name, schedule in schedules.items()
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
