"""Render paper figures from the CSV artifacts written by ``src.experiments``."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
MAIN = [
    "FIFO", "EDF", "DRR", "EDF+DRR", "ATC",
    "Predictive EDF+DRR (forecast)", "Dynamic EDF+DRR",
]
STYLE = {
    "FIFO": ("tab:gray", "o"), "EDF": ("tab:blue", "s"), "DRR": ("tab:orange", "^"),
    "EDF+DRR": ("tab:green", "D"), "ATC": ("tab:purple", "v"),
    "Predictive EDF+DRR (forecast)": ("tab:brown", "P"),
    "Dynamic EDF+DRR": ("tab:red", "*"),
}


def _read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=200)
    plt.close(figure)


def capacity_figures() -> None:
    rows = _read("gentd26_capacity_sweep.csv")
    metrics = (
        ("sla_violation_rate", "SLA violation rate", "gentd26_sla.png", False),
        ("avg_wait", "Average waiting time (s)", "gentd26_wait.png", True),
        ("weighted_tardiness_per_request", "Weighted tardiness per request (s)", "gentd26_tardiness.png", True),
        ("backlog", "Backlog at horizon (requests)", "gentd26_backlog.png", False),
        ("ontime_fairness", "Jain index of per-tier on-time rates", "gentd26_fairness.png", False),
    )
    for field, label, filename, log in metrics:
        figure, axis = plt.subplots(figsize=(5.2, 3.4))
        for policy in MAIN:
            selected = sorted(
                (r for r in rows if r["policy"] == policy), key=lambda r: int(r["workers"])
            )
            color, marker = STYLE[policy]
            values = [max(float(r[field]), 1e-3) if log else float(r[field]) for r in selected]
            axis.plot(
                [int(r["workers"]) for r in selected], values, marker=marker,
                color=color, label=policy, linewidth=1.4, markersize=5,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32])
        axis.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        if log:
            axis.set_yscale("log")
        axis.set_xlabel("Workers")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        if field == "sla_violation_rate":
            axis.legend(fontsize=6)
        _save(figure, filename)


def synthetic_figures() -> None:
    rows = [r for r in _read("synthetic_summary.csv") if r["policy"] in MAIN]
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    for field, label, filename in (
        ("sla_violation_rate", "SLA violation rate", "synthetic_sla.png"),
        ("weighted_tardiness", "Weighted tardiness (log scale)", "synthetic_tardiness.png"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 3.4))
        width = 0.8 / len(MAIN)
        for offset, policy in enumerate(MAIN):
            values, errors = [], []
            for scenario in scenarios:
                row = next(r for r in rows if r["scenario"] == scenario and r["policy"] == policy)
                mean = float(row[field])
                values.append(max(mean, 1e-2) if field == "weighted_tardiness" else mean)
                errors.append(mean - float(row[f"{field}_ci_low"]))
            axis.bar(
                [i + offset * width for i in range(len(scenarios))], values, width,
                yerr=errors, color=STYLE[policy][0], label=policy, capsize=1.5,
                error_kw={"linewidth": 0.6},
            )
        axis.set_xticks([i + 0.4 - width / 2 for i in range(len(scenarios))])
        axis.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=6)
        axis.set_ylabel(label)
        if field == "weighted_tardiness":
            axis.set_yscale("log")
        else:
            axis.legend(fontsize=5, ncol=2)
        axis.grid(True, axis="y", alpha=0.3)
        _save(figure, filename)


def milp_figure() -> None:
    rows = _read("milp_summary.csv")
    policies = [
        "FIFO", "EDF", "DRR", "EDF+DRR", "ATC",
        "Predictive EDF+DRR (backlog proxy)", "Dynamic EDF+DRR",
    ]
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    groups = [(r["size"], r["machines"]) for r in rows]
    width = 0.8 / len(policies)
    for offset, policy in enumerate(policies):
        values = [max(float(r[f"{policy}::gap_vs_milp"]), 1e-2) for r in rows]
        axis.bar(
            [i + offset * width for i in range(len(groups))], values, width,
            label=policy.replace(" (backlog proxy)", " (proxy)"),
            color=STYLE.get(policy, ("tab:cyan", "x"))[0],
        )
    axis.set_xticks([i + 0.4 - width / 2 for i in range(len(groups))])
    axis.set_xticklabels([f"n={s}, m={m}" for s, m in groups], fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("Excess weighted tardiness vs MILP")
    axis.legend(fontsize=5, ncol=2)
    axis.grid(True, axis="y", alpha=0.3)
    _save(figure, "milp_gap.png")


def sensitivity_figure() -> None:
    rows = _read("dynamic_sensitivity.csv")
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    for policy, label, color in (
        ("Dynamic EDF+DRR (deferral only)", "deferral limit (s late)", "tab:red"),
    ):
        subset = [r for r in rows if r["policy"] == policy and r["value"] != "inf"]
        axis.plot(
            [float(r["mean_tardiness_ratio_vs_edf_drr"]) for r in subset],
            [float(r["mean_sla_rate_delta_vs_edf_drr"]) for r in subset],
            marker="o", color=color, label=label,
        )
        for r in subset:
            axis.annotate(r["value"], (float(r["mean_tardiness_ratio_vs_edf_drr"]),
                                       float(r["mean_sla_rate_delta_vs_edf_drr"])), fontsize=6)
        inf = next(r for r in rows if r["policy"] == policy and r["value"] == "inf")
        axis.scatter(
            [float(inf["mean_tardiness_ratio_vs_edf_drr"])],
            [float(inf["mean_sla_rate_delta_vs_edf_drr"])], color=color, marker="*", s=80,
            label="no limit (default)",
        )
    axis.axvline(1.0, color="k", linewidth=0.6)
    axis.set_xlabel("Weighted tardiness ratio vs EDF+DRR")
    axis.set_ylabel("SLA-rate change vs EDF+DRR")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=6)
    _save(figure, "dynamic_tradeoff.png")


def main() -> None:
    capacity_figures()
    synthetic_figures()
    milp_figure()
    sensitivity_figure()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
