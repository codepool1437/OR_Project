"""Write LaTeX tables for the paper directly from the result CSV files."""

import csv
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "tables"
MAIN = [
    "FIFO", "EDF", "DRR", "EDF+DRR", "ATC",
    "Predictive EDF+DRR (forecast)", "Dynamic EDF+DRR",
]
SHORT = {
    "Predictive EDF+DRR (forecast)": "Pred. EDF+DRR",
    "Predictive EDF+DRR (backlog proxy)": "Pred. (proxy)",
    "Dynamic EDF+DRR": "Dyn. EDF+DRR",
    "Dynamic EDF+DRR (deferral only)": "Dyn. (deferral)",
    "Dynamic EDF+DRR (aggressive quanta only)": "Dyn. (quanta)",
}


def _read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _name(policy: str) -> str:
    return SHORT.get(policy, policy).replace("&", r"\&")


def _write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bold_min(values: list[float], index: int, text: str, lower_better: bool = True) -> str:
    best = min(values) if lower_better else max(values)
    return rf"\textbf{{{text}}}" if abs(values[index] - best) < 1e-12 else text


def synthetic_tables() -> None:
    rows = _read("synthetic_summary.csv")
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    for field, name, fmt in (
        ("sla_violation_rate", "synthetic_sla.tex", "{:.3f}"),
        ("weighted_tardiness", "synthetic_tardiness.tex", "{:.0f}"),
    ):
        lines = [
            r"\begin{tabular}{l" + "r" * len(MAIN) + "}",
            r"\toprule",
            "Scenario & " + " & ".join(_name(p) for p in MAIN) + r" \\",
            r"\midrule",
        ]
        for scenario in scenarios:
            values = [
                float(next(r for r in rows if r["scenario"] == scenario and r["policy"] == p)[field])
                for p in MAIN
            ]
            cells = [_bold_min(values, i, fmt.format(v)) for i, v in enumerate(values)]
            lines.append(scenario.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        _write(name, lines)


def synthetic_paired_table() -> None:
    rows = _read("synthetic_paired.csv")
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Policy & $\Delta$SLA rate & Verdict & $\Delta$W.\ tardiness & Verdict \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        for policy in ("Predictive EDF+DRR (forecast)", "Dynamic EDF+DRR"):
            def pick(metric: str) -> dict[str, str]:
                return next(
                    r for r in rows
                    if r["scenario"] == scenario and r["policy"] == policy
                    and r["reference"] == "EDF+DRR" and r["metric"] == metric
                )

            sla, tard = pick("sla_violation_rate"), pick("weighted_tardiness")
            verdict = lambda r: {"better": "better", "worse": "worse"}.get(r["verdict"], "n.s.")
            lines.append(
                rf"{scenario.replace('_', chr(92) + '_')} & {_name(policy)} & "
                rf"{float(sla['mean_delta']):+.3f} & {verdict(sla)} & "
                rf"{float(tard['mean_delta']):+.1f} & {verdict(tard)} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("synthetic_paired.tex", lines)


def gentd26_table() -> None:
    rows = _read("gentd26_capacity_sweep.csv")
    lines = [
        r"\begin{tabular}{rlrrrrr}",
        r"\toprule",
        r"$W$ & Policy & SLA viol. & Avg wait (s) & W.\ tard./req (s) & Backlog & Tier fair. \\",
        r"\midrule",
    ]
    for workers in ("1", "2", "4"):
        subset = [r for p in MAIN for r in rows if r["workers"] == workers and r["policy"] == p]
        sla = [float(r["sla_violation_rate"]) for r in subset]
        wait = [float(r["avg_wait"]) for r in subset]
        tard = [float(r["weighted_tardiness_per_request"]) for r in subset]
        for i, r in enumerate(subset):
            lines.append(
                f"{workers if i == 0 else ''} & {_name(r['policy'])} & "
                f"{_bold_min(sla, i, f'{sla[i]*100:.1f}\\%')} & "
                f"{_bold_min(wait, i, f'{wait[i]:.0f}')} & "
                f"{_bold_min(tard, i, f'{tard[i]:.1f}')} & "
                f"{int(float(r['backlog']))} & {float(r['ontime_fairness']):.3f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    _write("gentd26_sweep.tex", lines)


def gentd26_paired_table() -> None:
    rows = _read("gentd26_chunk_paired.csv")
    lines = [
        r"\begin{tabular}{rllrrrrl}",
        r"\toprule",
        r"$W$ & Policy & Ref. & Metric & $\Delta$ mean & W/L & Verdict \\",
        r"\midrule",
    ]
    lines = [
        r"\begin{tabular}{rllrrl}",
        r"\toprule",
        r"$W$ & Policy vs EDF+DRR & Metric & $\Delta$ mean & W/L/T & Verdict \\",
        r"\midrule",
    ]
    for workers in sorted({r["workers"] for r in rows}, key=int):
        for policy in ("Predictive EDF+DRR (forecast)", "Dynamic EDF+DRR", "ATC"):
            for metric, label, fmt in (
                ("sla_violation_rate", "SLA rate", "{:+.3f}"),
                ("weighted_tardiness", "W. tard.", "{:+.0f}"),
            ):
                matches = [
                    r for r in rows
                    if r["workers"] == workers and r["policy"] == policy
                    and r["reference"] == "EDF+DRR" and r["metric"] == metric
                ]
                if not matches:
                    continue
                r = matches[0]
                lines.append(
                    f"{workers} & {_name(policy)} & {label} & {fmt.format(float(r['mean_delta']))} & "
                    f"{r['wins']}/{r['losses']}/{r['ties']} & "
                    f"{'n.s.' if r['verdict'].startswith('no') else r['verdict']} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("gentd26_paired.tex", lines)


def milp_table() -> None:
    rows = _read("milp_summary.csv")
    names = [
        "FIFO", "EDF", "DRR", "EDF+DRR", "ATC",
        "Predictive EDF+DRR (backlog proxy)", "Dynamic EDF+DRR",
    ]
    lines = [
        r"\begin{tabular}{rrrr" + "r" * len(names) + "}",
        r"\toprule",
        r"$n$ & $m$ & Opt./inst. & Mean MILP & " + " & ".join(_name(n) for n in names) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        values = [float(r[f"{n}::gap_vs_milp"]) for n in names]
        cells = [_bold_min(values, i, f"{v:.2f}") for i, v in enumerate(values)]
        lines.append(
            f"{r['size']} & {r['machines']} & {r['proven_optimal']}/{r['instances']} & "
            f"{float(r['mean_milp_objective']):.2f} & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("milp.tex", lines)


def forecast_table() -> None:
    rows = [r for r in _read("forecast_accuracy.csv") if r["split"] == "test"]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Workload & Model & MAE & RMSE & Prec. & Rec. & F1 \\",
        r"\midrule",
    ]
    last = None
    for r in rows:
        label = r["workload"].replace("synthetic_", "").replace("_", r"\_")
        lines.append(
            f"{label if r['workload'] != last else ''} & {r['model'].replace(' (proposed)', '')} & "
            f"{float(r['mae']):.2f} & {float(r['rmse']):.2f} & {float(r['precision']):.2f} & "
            f"{float(r['recall']):.2f} & {float(r['f1']):.2f} \\\\"
        )
        last = r["workload"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("forecast.tex", lines)


def sensitivity_table() -> None:
    rows = _read("dynamic_sensitivity.csv")
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Parameter & Value & W.\ tardiness ratio & $\Delta$SLA rate \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['parameter'].replace('_', chr(92) + '_')} & {r['value']} & "
            f"{float(r['mean_tardiness_ratio_vs_edf_drr']):.3f} & "
            f"{float(r['mean_sla_rate_delta_vs_edf_drr']):+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("sensitivity.tex", lines)


def main() -> None:
    synthetic_tables()
    synthetic_paired_table()
    gentd26_table()
    gentd26_paired_table()
    milp_table()
    forecast_table()
    sensitivity_table()
    print(f"Wrote LaTeX tables to {OUT}")


if __name__ == "__main__":
    main()
