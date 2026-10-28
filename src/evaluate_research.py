"""Generate reproducible forecasting and scheduler evidence for the paper."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from .edf_drr import schedule_edf_drr
from .fifo import calculate_metrics
from .predictive_edf_drr import schedule_predictive_edf_drr
from .traffic import generate_requests
from .traffic_model import train_traffic_model


def _ci(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [mean(values), mean(values)]
    half_width = 2.262 * stdev(values) / len(values) ** 0.5
    average = mean(values)
    return [average - half_width, average + half_width]


def evaluate_schedulers(
    seeds: range,
    request_count: int,
    arrival_rate: float,
    horizon: float,
) -> dict[str, object]:
    wait_deltas: list[float] = []
    sla_deltas: list[float] = []
    for seed in seeds:
        requests = generate_requests(
            count=request_count, seed=seed, arrival_rate=arrival_rate
        )
        baseline = calculate_metrics(
            requests, schedule_edf_drr(requests), horizon=horizon
        )
        model = train_traffic_model(requests)
        predictive = calculate_metrics(
            requests,
            schedule_predictive_edf_drr(requests, model=model),
            horizon=horizon,
        )
        wait_deltas.append(
            predictive.average_waiting_time - baseline.average_waiting_time
        )
        sla_deltas.append(
            predictive.sla_violation_rate - baseline.sla_violation_rate
        )
    return {
        "seed_count": len(wait_deltas),
        "comparison": "Predictive EDF+DRR minus EDF+DRR",
        "average_waiting_time_delta": {
            "mean": mean(wait_deltas),
            "95_percent_ci": _ci(wait_deltas),
            "wins": sum(value < 0 for value in wait_deltas),
        },
        "sla_violation_rate_delta": {
            "mean": mean(sla_deltas),
            "95_percent_ci": _ci(sla_deltas),
            "wins": sum(value < 0 for value in sla_deltas),
        },
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
        range(args.seeds),
        args.requests,
        args.arrival_rate,
        args.horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote research evaluation to {args.output}")


if __name__ == "__main__":
    main()
