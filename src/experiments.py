"""Full reproducible experiment suite for the paper.

Usage::

    python -m src.experiments synthetic --seeds 30
    python -m src.experiments gentd26 --dataset data/GenTD26/lora_request_trace.csv
    python -m src.experiments forecast
    python -m src.experiments milp --seeds 10
    python -m src.experiments all

Every sub-command writes CSV (and Markdown) artifacts into ``results/``. All
policies are evaluated on identical request batches, and policy differences
are reported as paired deltas with Student-t 95% confidence intervals.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
from pathlib import Path
import time
from typing import Callable, Iterable

from .fifo import _jain_fairness, calculate_metrics
from .gen_td26 import load_gen_td26_requests
from .models import Request, ScheduledRequest
from .parallel import schedule_parallel
from .research_metrics import (
    classification_metrics,
    mean_ci,
    regression_metrics,
    spike_labels,
    weighted_tardiness,
)
from .scenarios import SCENARIOS, generate_scenario
from .traffic_model import TrafficSpikeModel, train_traffic_model

RESULTS = Path("results")
SYNTHETIC_WINDOW = 2.5  # 24 buckets per 60 s burst period
GENTD26_WINDOW = 600.0

# (display name, engine policy, uses forecast model)
POLICY_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("FIFO", "FIFO", False),
    ("EDF", "EDF", False),
    ("DRR", "DRR", False),
    ("EDF+DRR", "EDF+DRR", False),
    ("ATC", "ATC", False),
    ("Predictive EDF+DRR (forecast)", "Predictive EDF+DRR", True),
    ("Predictive EDF+DRR (backlog proxy)", "Predictive EDF+DRR", False),
    ("Dynamic EDF+DRR", "Dynamic EDF+DRR", False),
    ("Dynamic EDF+DRR (aggressive quanta only)", "Dynamic EDF+DRR (aggressive quanta only)", False),
    ("Dynamic EDF+DRR (deferral only)", "Dynamic EDF+DRR (deferral only)", False),
)
REFERENCES = ("EDF+DRR", "EDF", "DRR")
PAIRED_METRICS = (
    "weighted_tardiness", "sla_violation_rate", "avg_wait", "p95_wait",
    "ontime_fairness", "ontime_ratio",
)


def evaluate_schedule(
    requests: list[Request],
    schedule: list[ScheduledRequest],
    workers: int,
    horizon: float | None = None,
) -> dict[str, float]:
    """Return the full metric row for one (workload, policy) run."""
    full = calculate_metrics(requests, schedule, cpu_capacity=100.0 * workers)
    cutoff = horizon if horizon is not None else max(r.arrival_time for r in requests)
    cut = calculate_metrics(
        requests, schedule, cpu_capacity=100.0 * workers, horizon=cutoff
    )
    tier_ontime: dict[str, list[int]] = {}
    for item in schedule:
        counts = tier_ontime.setdefault(item.request.tier, [0, 0])
        counts[1] += 1
        counts[0] += not item.sla_violated
    ontime_rates = [on / total for on, total in tier_ontime.values() if total]
    row: dict[str, float] = {
        "weighted_tardiness": weighted_tardiness(schedule),
        "weighted_tardiness_per_request": weighted_tardiness(schedule) / len(requests),
        "sla_violation_rate": full.sla_violation_rate,
        "ontime_ratio": 1.0 - full.sla_violation_rate,
        "avg_wait": full.average_waiting_time,
        "p95_wait": full.p95_waiting_time,
        "ontime_fairness": _jain_fairness(ontime_rates),
        "completion_fairness": cut.fairness_index,
        "backlog": float(sum(cut.per_tier_backlog.values())),
        "utilization": full.cpu_utilization,
    }
    for tier in ("VIP", "Premium", "Free"):
        row[f"{tier}_violation_rate"] = full.per_tier_sla_violation_rate.get(tier, 0.0)
        row[f"{tier}_wait"] = full.per_tier_waiting_time.get(tier, 0.0)
    return row


def run_policies(
    requests: list[Request],
    workers: int,
    model: TrafficSpikeModel | None,
    horizon: float | None = None,
    specs: Iterable[tuple[str, str, bool]] = POLICY_SPECS,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, engine, uses_model in specs:
        began = time.perf_counter()
        schedule = schedule_parallel(
            requests, engine, workers, model=model if uses_model else None
        )
        elapsed = time.perf_counter() - began
        row = evaluate_schedule(requests, schedule, workers, horizon)
        row["runtime_s"] = elapsed
        out[name] = row
    return out


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    samples: dict[str, list[dict[str, float]]], metrics: Iterable[str]
) -> list[dict[str, object]]:
    rows = []
    for policy, runs in samples.items():
        row: dict[str, object] = {"policy": policy, "n": len(runs)}
        for metric in metrics:
            mean, low, high = mean_ci([run[metric] for run in runs])
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return rows


def paired_deltas(
    samples: dict[str, list[dict[str, float]]],
    metrics: Iterable[str] = PAIRED_METRICS,
) -> list[dict[str, object]]:
    """Paired policy-minus-reference deltas, with CI and win counts."""
    rows = []
    for reference in REFERENCES:
        ref_runs = samples[reference]
        for policy, runs in samples.items():
            if policy == reference:
                continue
            for metric in metrics:
                deltas = [a[metric] - b[metric] for a, b in zip(runs, ref_runs)]
                mean, low, high = mean_ci(deltas)
                lower_is_better = metric not in {"ontime_fairness", "ontime_ratio"}
                wins = sum((d < -1e-12) if lower_is_better else (d > 1e-12) for d in deltas)
                losses = sum((d > 1e-12) if lower_is_better else (d < -1e-12) for d in deltas)
                significant = (high < 0) if lower_is_better else (low > 0)
                worse = (low > 0) if lower_is_better else (high < 0)
                rows.append(
                    {
                        "policy": policy,
                        "reference": reference,
                        "metric": metric,
                        "mean_delta": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "wins": wins,
                        "losses": losses,
                        "ties": len(deltas) - wins - losses,
                        "verdict": "better" if significant else "worse" if worse else "no significant difference",
                    }
                )
    return rows


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _md_table(rows: list[dict[str, object]], metrics: list[str]) -> list[str]:
    lines = [
        "| Policy | " + " | ".join(metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ]
    for row in rows:
        cells = [
            f"{_fmt(row[m])} [{_fmt(row[m + '_ci_low'])}, {_fmt(row[m + '_ci_high'])}]"
            for m in metrics
        ]
        lines.append(f"| {row['policy']} | " + " | ".join(cells) + " |")
    return lines


# --------------------------------------------------------------------------
# Synthetic scenarios
# --------------------------------------------------------------------------

def _train_synthetic_model(scenario: str, requests: int) -> TrafficSpikeModel:
    """Train on a held-out seed, never on evaluation workloads."""
    training = generate_scenario(scenario, requests, seed=100_000)
    return train_traffic_model(training, window_size=SYNTHETIC_WINDOW)


def _synthetic_job(args: tuple) -> tuple[str, int, dict[str, dict[str, float]]]:
    scenario, seed, count, workers, model = args
    requests = generate_scenario(scenario, count, seed)
    return scenario, seed, run_policies(requests, workers, model)


def run_synthetic(seeds: int, count: int, workers: int, jobs: int) -> None:
    models = {name: _train_synthetic_model(name, 4 * count) for name in SCENARIOS}
    tasks = [
        (scenario, seed, count, workers, models[scenario])
        for scenario in SCENARIOS
        for seed in range(seeds)
    ]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(_synthetic_job, tasks, chunksize=1))

    metrics = [
        "weighted_tardiness", "sla_violation_rate", "avg_wait", "p95_wait",
        "ontime_fairness", "completion_fairness", "backlog", "utilization",
        "VIP_violation_rate", "Premium_violation_rate", "Free_violation_rate",
        "VIP_wait", "Free_wait", "runtime_s",
    ]
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    md = [
        f"# Synthetic scenario comparison ({seeds} seeds, {count} requests, {workers} worker(s))",
        "",
        "Cells show mean [95% CI] over seeds. All policies see identical request batches.",
        "",
    ]
    for scenario in SCENARIOS:
        samples: dict[str, list[dict[str, float]]] = {name: [] for name, _, _ in POLICY_SPECS}
        for name, seed, runs in sorted(
            (r for r in results if r[0] == scenario), key=lambda r: r[1]
        ):
            for policy, row in runs.items():
                samples[policy].append(row)
        agg = aggregate(samples, metrics)
        for row in agg:
            summary_rows.append({"scenario": scenario, "workers": workers, **row})
        for row in paired_deltas(samples):
            paired_rows.append({"scenario": scenario, "workers": workers, **row})
        md += [f"## {scenario}: {SCENARIOS[scenario].description}", ""]
        md += _md_table(agg, ["weighted_tardiness", "sla_violation_rate", "avg_wait", "ontime_fairness"])
        md.append("")
    _write_csv(RESULTS / "synthetic_summary.csv", summary_rows)
    _write_csv(RESULTS / "synthetic_paired.csv", paired_rows)
    (RESULTS / "synthetic_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote synthetic results for {len(SCENARIOS)} scenarios to {RESULTS}")


# --------------------------------------------------------------------------
# GenTD26
# --------------------------------------------------------------------------

def _rebase(requests: list[Request]) -> list[Request]:
    origin = min(r.arrival_time for r in requests)
    return [
        replace(
            r,
            request_id=index,
            arrival_time=r.arrival_time - origin,
            deadline=r.deadline - origin,
        )
        for index, r in enumerate(sorted(requests, key=lambda r: r.arrival_time))
    ]


def _gentd26_chunk_job(args: tuple) -> tuple[int, int, dict[str, dict[str, float]]]:
    chunk_id, workers, requests, model = args
    return chunk_id, workers, run_policies(requests, workers, model)


def run_gentd26(dataset: Path, jobs: int, chunks: int, worker_counts: list[int]) -> None:
    requests = load_gen_td26_requests(dataset)
    print(f"Loaded {len(requests)} GenTD26 requests")
    train_end = int(0.4 * len(requests))
    model = train_traffic_model(requests[:train_end], window_size=GENTD26_WINDOW)

    # 1) Full-trace finite-horizon capacity sweep (the headline result).
    horizon = 1_750_000.0
    sweep_rows: list[dict[str, object]] = []
    for workers in worker_counts:
        runs = run_policies(requests, workers, model, horizon=horizon)
        offered = sum(r.arrival_time < horizon for r in requests)
        for policy, row in runs.items():
            sweep_rows.append(
                {"workers": workers, "horizon_s": horizon, "offered": offered, "policy": policy, **row}
            )
        print(f"  capacity sweep: {workers} worker(s) done", flush=True)
    _write_csv(RESULTS / "gentd26_capacity_sweep.csv", sweep_rows)

    # 2) Chronological test chunks for paired statistics (trace after the
    #    forecaster's training prefix, split into independent segments).
    test_requests = requests[train_end:]
    size = len(test_requests) // chunks
    segments = [
        _rebase(test_requests[i * size:(i + 1) * size]) for i in range(chunks)
    ]
    chunk_workers = [w for w in worker_counts if w <= 2] or [1]
    tasks = [
        (index, w, segment, model)
        for index, segment in enumerate(segments)
        for w in chunk_workers
    ]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(_gentd26_chunk_job, tasks, chunksize=1))
    metrics = [
        "weighted_tardiness_per_request", "sla_violation_rate", "avg_wait", "p95_wait",
        "ontime_fairness", "VIP_violation_rate", "Premium_violation_rate",
        "Free_violation_rate", "runtime_s",
    ]
    summary_rows, paired_rows = [], []
    md = [f"# GenTD26 chunked evaluation ({chunks} chronological test segments)", ""]
    for workers in chunk_workers:
        samples: dict[str, list[dict[str, float]]] = {name: [] for name, _, _ in POLICY_SPECS}
        for chunk_id, w, runs in sorted(
            (r for r in results if r[1] == workers), key=lambda r: r[0]
        ):
            for policy, row in runs.items():
                samples[policy].append(row)
        agg = aggregate(samples, metrics)
        summary_rows += [{"workers": workers, **row} for row in agg]
        paired_rows += [{"workers": workers, **row} for row in paired_deltas(samples)]
        md += [f"## {workers} worker(s)", ""]
        md += _md_table(agg, ["weighted_tardiness_per_request", "sla_violation_rate", "avg_wait", "ontime_fairness"])
        md.append("")
    _write_csv(RESULTS / "gentd26_chunk_summary.csv", summary_rows)
    _write_csv(RESULTS / "gentd26_chunk_paired.csv", paired_rows)
    (RESULTS / "gentd26_chunk_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote GenTD26 results")


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

def _bucket_counts(requests: list[Request], window: float) -> dict[str, list[float]]:
    buckets = int(max(r.arrival_time for r in requests) // window) + 1
    counts = {t: [0.0] * buckets for t in sorted({r.tier for r in requests})}
    for r in requests:
        counts[r.tier][int(r.arrival_time // window)] += 1.0
    return counts


def _forecast_rows(
    label: str, requests: list[Request], window: float, with_xgboost: bool
) -> list[dict[str, object]]:
    counts = _bucket_counts(requests, window)
    buckets = len(next(iter(counts.values())))
    train_end, val_end = int(0.6 * buckets), int(0.8 * buckets)
    cutoff = train_end * window
    train_requests = [r for r in requests if r.arrival_time < cutoff]
    ridge = train_traffic_model(train_requests, window_size=window)
    predictors: dict[str, Callable] = {
        "naive (last window)": lambda hist, idx: {t: (hist[t][-1] if hist[t] else 0.0) for t in counts},
        "moving average (3)": lambda hist, idx: {
            t: sum(hist[t][-3:]) / max(1, len(hist[t][-3:])) for t in counts
        },
        "ridge (proposed)": ridge.predict,
    }
    if with_xgboost:
        try:
            from .xgboost_model import train_xgboost_model

            xgb = train_xgboost_model(
                train_requests, RESULTS / "forecast" / f"xgb_{label}.json", window
            )
            predictors["xgboost"] = xgb.predict
        except RuntimeError as error:
            print(f"  xgboost skipped: {error}")

    rows = []
    for split, lo, hi in (("validation", train_end, val_end), ("test", val_end, buckets)):
        for name, predict in predictors.items():
            actual, predicted, actual_spike, predicted_spike = [], [], [], []
            for tier, series in counts.items():
                labels = spike_labels(series)
                for idx in range(max(lo, 3), hi):
                    history = {t: counts[t][:idx] for t in counts}
                    value = predict(history, idx)[tier]
                    previous = series[max(0, idx - 6):idx]
                    mean_prev = sum(previous) / len(previous)
                    std_prev = (sum((v - mean_prev) ** 2 for v in previous) / len(previous)) ** 0.5
                    actual.append(series[idx])
                    predicted.append(value)
                    actual_spike.append(labels[idx])
                    predicted_spike.append(len(previous) >= 2 and value > mean_prev + 2.0 * std_prev)
            reg = regression_metrics(actual, predicted)
            cls = classification_metrics(actual_spike, predicted_spike)
            rows.append(
                {
                    "workload": label, "split": split, "model": name, "samples": len(actual),
                    "spike_prevalence": sum(actual_spike) / max(1, len(actual_spike)),
                    **reg, **cls,
                }
            )
    return rows


def run_forecast(dataset: Path | None) -> None:
    rows = _forecast_rows(
        "synthetic_mixed_burst",
        generate_scenario("mixed_burst", 20_000, seed=7),
        SYNTHETIC_WINDOW,
        True,
    )
    rows += _forecast_rows(
        "synthetic_random_burst",
        generate_scenario("random_burst", 20_000, seed=7),
        SYNTHETIC_WINDOW,
        True,
    )
    if dataset and dataset.exists():
        rows += _forecast_rows(
            "gentd26", load_gen_td26_requests(dataset), GENTD26_WINDOW, True
        )
    _write_csv(RESULTS / "forecast_accuracy.csv", rows)
    lines = [
        "# Forecast accuracy (chronological 60/20/20 train/validation/test)",
        "",
        "| Workload | Split | Model | MAE | RMSE | Spike precision | Spike recall | Spike F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['workload']} | {r['split']} | {r['model']} | {r['mae']:.3f} | {r['rmse']:.3f} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    (RESULTS / "forecast_accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote forecast accuracy results")


# --------------------------------------------------------------------------
# MILP benchmark
# --------------------------------------------------------------------------

MILP_POLICIES = tuple(
    (n, e, m) for n, e, m in POLICY_SPECS
    if n in {
        "FIFO", "EDF", "DRR", "EDF+DRR", "ATC", "Predictive EDF+DRR (backlog proxy)",
        "Dynamic EDF+DRR",
    }
)


def _milp_instance(scenario: str, size: int, seed: int, compression: float) -> list[Request]:
    base = generate_scenario(scenario, size, seed)
    return [
        replace(
            r,
            arrival_time=r.arrival_time * compression,
            deadline=r.deadline - r.arrival_time * (1.0 - compression),
        )
        for r in base
    ]


def _milp_job(args: tuple) -> dict[str, object]:
    from .milp import solve_milp_detailed

    scenario, size, seed, machines, compression, limit = args
    requests = _milp_instance(scenario, size, seed, compression)
    heuristics = {}
    for name, engine, _ in MILP_POLICIES:
        schedule = schedule_parallel(requests, engine, machines)
        heuristics[name] = (schedule, weighted_tardiness(schedule))
    incumbent = min(heuristics.values(), key=lambda item: item[1])[0]
    result = solve_milp_detailed(requests, machines, limit, incumbent)
    row: dict[str, object] = {
        "scenario": scenario, "size": size, "seed": seed, "machines": machines,
        "milp_objective": result.objective,
        "milp_lower_bound": result.lower_bound,
        "milp_proven_optimal": result.proven_optimal,
        "milp_seconds": result.solve_seconds,
        "milp_solver": result.solver,
    }
    for name, (_, objective) in heuristics.items():
        row[f"obj::{name}"] = objective
    return row


def run_milp(seeds: int, sizes: list[int], machine_counts: list[int], limit: float, jobs: int) -> None:
    scenarios = ("overload", "tight_deadlines", "vip_spike", "mixed_burst")
    compression = 0.12
    tasks = [
        (scenario, size, seed, machines, compression, limit)
        for scenario in scenarios
        for size in sizes
        for machines in machine_counts
        for seed in range(seeds)
    ]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(_milp_job, tasks, chunksize=1))
    _write_csv(RESULTS / "milp_instances.csv", rows)
    summarize_milp(rows, sizes, machine_counts)


def summarize_milp(rows: list[dict[str, object]], sizes: list[int], machine_counts: list[int]) -> None:
    names = [n for n, _, _ in MILP_POLICIES]
    summary = []
    for size in sizes:
        for machines in machine_counts:
            subset = [r for r in rows if r["size"] == size and r["machines"] == machines]
            if not subset:
                continue
            entry: dict[str, object] = {
                "size": size, "machines": machines, "instances": len(subset),
                "proven_optimal": sum(bool(r["milp_proven_optimal"]) for r in subset),
                "mean_milp_seconds": sum(r["milp_seconds"] for r in subset) / len(subset),
                "mean_milp_objective": sum(r["milp_objective"] for r in subset) / len(subset),
                "mean_milp_lower_bound": sum(r["milp_lower_bound"] for r in subset) / len(subset),
            }
            for name in names:
                vs_bound = [r[f"obj::{name}"] - r["milp_lower_bound"] for r in subset]
                vs_best = [r[f"obj::{name}"] - r["milp_objective"] for r in subset]
                entry[f"{name}::gap_vs_bound"] = mean_ci(vs_bound)[0]
                mean, low, high = mean_ci(vs_best)
                entry[f"{name}::gap_vs_milp"] = mean
                entry[f"{name}::gap_vs_milp_ci_low"] = low
                entry[f"{name}::gap_vs_milp_ci_high"] = high
                entry[f"{name}::matches_milp"] = sum(
                    abs(d) < 1e-6 for d in vs_best
                )
                entry[f"{name}::beats_milp_incumbent"] = sum(d < -1e-6 for d in vs_best)
            summary.append(entry)
    _write_csv(RESULTS / "milp_summary.csv", summary)

    lines = [
        "# MILP benchmark (HiGHS, warm-started from the best heuristic)",
        "",
        "Mean excess weighted tardiness over the MILP incumbent (`gap vs MILP`, "
        "never negative because the MILP incumbent is at least as good as every "
        "warm-start heuristic) and over the proven dual bound (`gap vs bound`, a "
        "valid but possibly loose upper estimate of the true optimality gap). "
        "Both coincide when optimality is proven.",
        "",
        "## Gap vs MILP incumbent",
        "",
        "| Size | Machines | Instances | Proven optimal | " + " | ".join(names) + " |",
        "|---:|---:|---:|---:|" + "---:|" * len(names),
    ]
    for e in summary:
        cells = [f"{e[f'{n}::gap_vs_milp']:.3f}" for n in names]
        lines.append(
            f"| {e['size']} | {e['machines']} | {e['instances']} | {e['proven_optimal']} | "
            + " | ".join(cells) + " |"
        )
    lines += [
        "", "## Gap vs proven lower bound", "",
        "| Size | Machines | Mean MILP incumbent | Mean lower bound | " + " | ".join(names) + " |",
        "|---:|---:|---:|---:|" + "---:|" * len(names),
    ]
    for e in summary:
        cells = [f"{e[f'{n}::gap_vs_bound']:.3f}" for n in names]
        lines.append(
            f"| {e['size']} | {e['machines']} | {e['mean_milp_objective']:.3f} | "
            f"{e['mean_milp_lower_bound']:.3f} | " + " | ".join(cells) + " |"
        )
    text = chr(10).join(lines) + chr(10)
    (RESULTS / "milp_summary.md").write_text(text, encoding="utf-8")
    print("Wrote MILP benchmark results")


# --------------------------------------------------------------------------
# Dynamic EDF+DRR sensitivity (tuning seeds disjoint from evaluation seeds)
# --------------------------------------------------------------------------

def run_sensitivity(seeds: int, count: int) -> None:
    from . import dynamic_edf_drr, parallel

    tuning = range(1000, 1000 + seeds)
    data = {(sc, s): generate_scenario(sc, count, s) for sc in SCENARIOS for s in tuning}
    reference = {k: evaluate_schedule(v, schedule_parallel(v, "EDF+DRR", 1), 1) for k, v in data.items()}
    saved = (parallel.DEFER_LIMIT, parallel.QUANTA_FLOOR, dynamic_edf_drr.PENALTY_POWER)
    rows: list[dict[str, object]] = []
    try:
        configs = [("defer_limit_s", limit, "Dynamic EDF+DRR (deferral only)") for limit in (0.25, 0.5, 1.0, 2.0, 4.0, float("inf"))]
        configs += [("quanta_floor", floor, "Dynamic EDF+DRR") for floor in (0.25, 0.5, 0.75, 0.9, 1.0)]
        for parameter, value, policy in configs:
            parallel.DEFER_LIMIT = value if parameter == "defer_limit_s" else float("inf")
            parallel.QUANTA_FLOOR = value if parameter == "quanta_floor" else saved[1]
            if parameter == "quanta_floor":
                parallel.DEFER_LIMIT = float("inf")
            ratios, slas = [], []
            for key, requests in data.items():
                row = evaluate_schedule(requests, schedule_parallel(requests, policy, 1), 1)
                base = reference[key]
                if base["weighted_tardiness"] > 1.0:
                    ratios.append(row["weighted_tardiness"] / base["weighted_tardiness"])
                slas.append(row["sla_violation_rate"] - base["sla_violation_rate"])
            rows.append(
                {
                    "parameter": parameter, "value": value, "policy": policy,
                    "mean_tardiness_ratio_vs_edf_drr": sum(ratios) / len(ratios),
                    "mean_sla_rate_delta_vs_edf_drr": sum(slas) / len(slas),
                    "instances": len(slas),
                }
            )
    finally:
        parallel.DEFER_LIMIT, parallel.QUANTA_FLOOR, dynamic_edf_drr.PENALTY_POWER = saved
    _write_csv(RESULTS / "dynamic_sensitivity.csv", rows)
    print("Wrote dynamic sensitivity results")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["synthetic", "gentd26", "forecast", "milp", "sensitivity", "milp-summary", "all"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--dataset", type=Path, default=Path("data/GenTD26/lora_request_trace.csv"))
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--milp-seeds", type=int, default=10)
    parser.add_argument("--milp-sizes", type=int, nargs="+", default=[10, 14])
    parser.add_argument("--milp-machines", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--milp-time-limit", type=float, default=60.0)
    args = parser.parse_args()
    if args.command in ("synthetic", "all"):
        run_synthetic(args.seeds, args.requests, args.workers, args.jobs)
    if args.command == "milp-summary":
        with (RESULTS / "milp_instances.csv").open(encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
        rows = []
        for r in raw:
            row = {}
            for k, v in r.items():
                if k in ("scenario", "milp_solver"):
                    row[k] = v
                elif k == "milp_proven_optimal":
                    row[k] = v == "True"
                elif k in ("size", "seed", "machines"):
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            rows.append(row)
        summarize_milp(rows, args.milp_sizes, args.milp_machines)
    if args.command in ("sensitivity", "all"):
        run_sensitivity(8, args.requests)
    if args.command in ("gentd26", "all"):
        run_gentd26(args.dataset, args.jobs, args.chunks, [1, 2, 4, 8, 16, 32])
    if args.command in ("forecast", "all"):
        run_forecast(args.dataset)
    if args.command in ("milp", "all"):
        run_milp(args.milp_seeds, args.milp_sizes, args.milp_machines, args.milp_time_limit, args.jobs)


if __name__ == "__main__":
    main()
