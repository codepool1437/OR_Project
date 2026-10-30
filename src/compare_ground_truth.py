"""Compare heuristic schedulers with the exact MILP ground truth."""

import argparse
import csv
from pathlib import Path
from typing import Iterable

from .drr import schedule_drr
from .edf import schedule_edf
from .edf_drr import schedule_edf_drr
from .fifo import calculate_metrics, schedule_fifo
from .gen_td26 import load_gen_td26_requests
from .dynamic_edf_drr import schedule_dynamic_edf_drr
from .milp import solve_milp_detailed
from .parallel import schedule_parallel
from .predictive_edf_drr import schedule_predictive_edf_drr
from .research_metrics import weighted_tardiness
from .traffic import generate_requests


def compare(requests, time_limit: float | None = 60.0) -> list[dict[str, object]]:
    schedules = {
        "FIFO": schedule_fifo(requests),
        "EDF": schedule_edf(requests),
        "DRR": schedule_drr(requests),
        "EDF+DRR": schedule_edf_drr(requests),
        "ATC": schedule_parallel(requests, "ATC", 1),
        "Predictive EDF+DRR": schedule_predictive_edf_drr(requests),
        "Dynamic EDF+DRR": schedule_dynamic_edf_drr(requests),
    }
    incumbent = min(schedules.values(), key=weighted_tardiness)
    milp = solve_milp_detailed(requests, 1, time_limit, incumbent)
    schedules["MILP (ground truth)"] = milp.schedule
    milp_objective = weighted_tardiness(milp.schedule)
    rows = []
    for policy, schedule in schedules.items():
        metrics = calculate_metrics(requests, schedule)
        objective = weighted_tardiness(schedule)
        rows.append(
            {
                "policy": policy,
                "weighted_tardiness": objective,
                "objective_gap": objective - milp_objective,
                "objective_gap_percent": (
                    100.0 * (objective - milp_objective) / milp_objective
                    if milp_objective > 1e-9
                    else None
                ),
                "average_waiting_time": metrics.average_waiting_time,
                "p95_waiting_time": metrics.p95_waiting_time,
                "sla_violation_rate": metrics.sla_violation_rate,
                "useful_work_ratio": metrics.useful_work_ratio,
                "throughput": metrics.throughput,
                "fairness_index": metrics.fairness_index,
                "milp_proven_optimal": milp.proven_optimal,
            }
        )
    return rows


def write_report(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = {
        "weighted_tardiness": min(rows, key=lambda row: row["weighted_tardiness"])["policy"],
        "average_waiting_time": min(rows, key=lambda row: row["average_waiting_time"])["policy"],
        "p95_waiting_time": min(rows, key=lambda row: row["p95_waiting_time"])["policy"],
        "sla_violation_rate": min(rows, key=lambda row: row["sla_violation_rate"])["policy"],
        "useful_work_ratio": max(rows, key=lambda row: row["useful_work_ratio"])["policy"],
        "throughput": max(rows, key=lambda row: row["throughput"])["policy"],
        "fairness_index": max(rows, key=lambda row: row["fairness_index"])["policy"],
    }
    lines = [
        "# Ground-Truth Scheduler Comparison",
        "",
        "| Policy | Weighted tardiness | Gap vs MILP | Avg wait | P95 wait | SLA rate | Useful work | Throughput | Fairness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gap_str = (
            f"{row['objective_gap_percent']:.2f}% | "
            if row["objective_gap_percent"] is not None
            else "N/A | "
        )
        lines.append(
            f"| {row['policy']} | {row['weighted_tardiness']:.6f} | {gap_str}"
        )
        lines[-1] += (
            f"{row['average_waiting_time']:.6f} | "
            f"{row['p95_waiting_time']:.6f} | {row['sla_violation_rate']:.4f} | "
            f"{row['useful_work_ratio']:.4f} | {row['throughput']:.6f} | "
            f"{row['fairness_index']:.6f} |"
        )
    lines.extend(["", "## Best policy by metric", ""])
    lines.extend(f"- **{metric}**: {policy}" for metric, policy in best.items())
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--compression", type=float, default=1.0,
                        help="scale arrival times (e.g. 0.12) to create congestion")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/ground_truth_comparison"))
    args = parser.parse_args()
    requests = (
        load_gen_td26_requests(args.dataset, max_requests=args.max_requests or args.requests)
        if args.dataset
        else generate_requests(args.requests, args.seed, args.arrival_rate)
    )
    if args.compression != 1.0:
        from dataclasses import replace

        requests = [
            replace(
                r,
                arrival_time=r.arrival_time * args.compression,
                deadline=r.deadline - r.arrival_time * (1.0 - args.compression),
            )
            for r in requests
        ]
    rows = compare(requests, args.time_limit)
    write_report(rows, args.output)
    print(f"Wrote {args.output.with_suffix('.md')} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
