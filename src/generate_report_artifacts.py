"""Generate paper-ready comparison tables and plots for GenTD26."""

import argparse
import csv
from pathlib import Path

from .fifo import calculate_metrics
from .gen_td26 import load_gen_td26_requests
from .parallel import schedule_parallel

POLICIES = ("FIFO", "EDF", "DRR", "EDF+DRR", "Predictive EDF+DRR")
WORKERS = (1, 2, 4, 8, 16, 32)


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, str]], path: Path, horizon: float) -> None:
    lines = [
        "# GenTD26 Finite-Horizon Scheduler Comparison",
        "",
        f"- Horizon: `{horizon:.0f}` seconds",
        f"- Offered requests: `{rows[0]['offered']}`",
        "- Policies: FIFO, EDF, DRR, EDF+DRR, Predictive EDF+DRR",
        "- Worker capacities: 1, 2, 4, 8, 16, 32",
        "",
        "| Workers | Policy | Completed | Backlog | Avg wait (s) | P95 wait (s) | SLA violation rate | Fairness |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['workers']} | {row['policy']} | {row['completed']} | "
            f"{row['backlog']} | {row['avg_wait_s']} | {row['p95_wait_s']} | "
            f"{float(row['sla_rate']) * 100:.3f}% | {row['fairness']} |"
        )
    lines.extend(
        [
            "",
            "Metrics use requests arriving before the horizon. Only requests",
            "completed by the horizon count as completed; unfinished requests are backlog.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(rows: list[dict[str, str]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("avg_wait_s", "Average waiting time (seconds)", "waiting_time.png"),
        ("sla_rate", "SLA violation rate", "sla_violations.png"),
        ("fairness", "Jain fairness index", "fairness.png"),
        ("backlog", "Backlog (requests)", "backlog.png"),
    )
    for field, ylabel, filename in metrics:
        figure, axis = plt.subplots(figsize=(8, 5))
        for policy in POLICIES:
            selected = [row for row in rows if row["policy"] == policy]
            x = [int(row["workers"]) for row in selected]
            y = [float(row[field]) for row in selected]
            if field == "sla_rate":
                y = [value * 100 for value in y]
                ylabel = "SLA violation rate (%)"
            axis.plot(x, y, marker="o", linewidth=2, label=policy)
        axis.set_xlabel("Worker count")
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(WORKERS))
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def generate(dataset: Path, output_dir: Path, horizon: float) -> None:
    requests = load_gen_td26_requests(dataset)
    rows: list[dict[str, str]] = []
    for workers in WORKERS:
        for policy in POLICIES:
            schedule = schedule_parallel(requests, policy, workers)
            metrics = calculate_metrics(
                requests,
                schedule,
                cpu_capacity=100 * workers,
                horizon=horizon,
            )
            rows.append(
                {
                    "horizon_s": f"{horizon:.0f}",
                    "workers": str(workers),
                    "policy": policy,
                    "offered": str(metrics.request_count),
                    "completed": str(metrics.completed_requests),
                    "backlog": str(sum(metrics.per_tier_backlog.values())),
                    "avg_wait_s": f"{metrics.average_waiting_time:.6f}",
                    "p95_wait_s": f"{metrics.p95_waiting_time:.6f}",
                    "sla_violations": str(metrics.sla_violations),
                    "sla_rate": f"{metrics.sla_violation_rate:.8f}",
                    "throughput": f"{metrics.throughput:.8f}",
                    "fairness": f"{metrics.fairness_index:.8f}",
                    "VIP_backlog": str(metrics.per_tier_backlog.get("VIP", 0)),
                    "Premium_backlog": str(metrics.per_tier_backlog.get("Premium", 0)),
                    "Free_backlog": str(metrics.per_tier_backlog.get("Free", 0)),
                }
            )

    _write_csv(rows, output_dir / "gentd26_scheduler_comparison.csv")
    _write_summary(rows, output_dir / "gentd26_scheduler_summary.md", horizon)
    _write_plots(rows, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GenTD26 paper artifacts")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--horizon", type=float, default=1_750_000.0)
    args = parser.parse_args()
    generate(args.dataset, args.output_dir, args.horizon)
    print(f"Generated report artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
