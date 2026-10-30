"""Paired multi-seed evaluation of predictive vs. static EDF+DRR.

The traffic forecaster is trained on a held-out seed that never appears in the
evaluation set, and confidence intervals use Student-t quantiles.
"""

import argparse
import json
from pathlib import Path

from .edf_drr import schedule_edf_drr
from .fifo import calculate_metrics
from .predictive_edf_drr import schedule_predictive_edf_drr
from .research_metrics import mean_ci
from .traffic import generate_requests
from .traffic_model import train_traffic_model

TRAINING_SEED = 1_000_003


def _summary(deltas: list[float]) -> dict[str, object]:
    mean, low, high = mean_ci(deltas)
    return {
        "mean": mean,
        "95_percent_ci": [low, high],
        "wins": sum(value < 0 for value in deltas),
    }


def evaluate_schedulers(
    seeds: range,
    request_count: int,
    arrival_rate: float,
    horizon: float,
    window_size: float = 1.0,
) -> dict[str, object]:
    training = generate_requests(
        count=4 * request_count, seed=TRAINING_SEED, arrival_rate=arrival_rate
    )
    model = train_traffic_model(training, window_size=window_size)
    wait_deltas: list[float] = []
    sla_deltas: list[float] = []
    for seed in seeds:
        requests = generate_requests(
            count=request_count, seed=seed, arrival_rate=arrival_rate
        )
        baseline = calculate_metrics(requests, schedule_edf_drr(requests), horizon=horizon)
        predictive = calculate_metrics(
            requests,
            schedule_predictive_edf_drr(requests, model=model),
            horizon=horizon,
        )
        wait_deltas.append(predictive.average_waiting_time - baseline.average_waiting_time)
        sla_deltas.append(predictive.sla_violation_rate - baseline.sla_violation_rate)
    return {
        "seed_count": len(wait_deltas),
        "comparison": "Predictive EDF+DRR minus EDF+DRR",
        "average_waiting_time_delta": _summary(wait_deltas),
        "sla_violation_rate_delta": _summary(sla_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/research_evaluation.json"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--arrival-rate", type=float, default=20.0)
    parser.add_argument("--horizon", type=float, default=20.0)
    args = parser.parse_args()
    result = evaluate_schedulers(
        range(args.seeds), args.requests, args.arrival_rate, args.horizon
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote research evaluation to {args.output}")


if __name__ == "__main__":
    main()
