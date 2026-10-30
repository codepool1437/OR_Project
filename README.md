# AI-Predictive API Gateway Scheduler

This repository contains a complete reproducible scheduling research platform:
synthetic and GenTD26 workloads, FIFO, EDF, DRR, static EDF+DRR, and
predictive/dynamic EDF+DRR schedulers, plus an offline MILP benchmark for
small-batch ground truth. Results support an IEEE research paper comparing all
policies across capacity sweeps, finite horizons, and paired multi-seed
evaluations.

The frozen research question, hypotheses, assumptions, scope, and evaluation
metrics are recorded in [`docs/research_design.md`](docs/research_design.md).

## Reproduce every result in the paper

```text
python -m pip install -r requirements.txt
python -m src.experiments synthetic --seeds 30      # 8 scenarios x 10 policy variants
python -m src.experiments sensitivity               # Dynamic EDF+DRR tuning (disjoint seeds)
python -m src.experiments gentd26 --jobs 8          # capacity sweep + chronological segments
python -m src.experiments forecast                  # MAE/RMSE + spike P/R/F1
python -m src.experiments milp --milp-seeds 8 --milp-sizes 10 14 18 --milp-machines 1 2 --milp-time-limit 30
python -m src.make_figures                          # results/figures/*.png
python -m src.make_tables                           # results/tables/*.tex (used by paper.tex)
python -m unittest discover -s tests -v
```

Policies (all in one engine, `src/parallel.py`, any worker count): FIFO, EDF,
DRR, EDF+DRR, ATC (weighted-tardiness dispatching rule), Predictive EDF+DRR
(forecast-driven quanta), and Dynamic EDF+DRR (`src/dynamic_edf_drr.py`:
urgency-driven quanta plus late-work deferral), with ablations
`Dynamic EDF+DRR (aggressive quanta only)` and `(deferral only)`.
The MILP (`src/milp.py`) supports multiple machines, warm starts, a time limit,
and reports the dual bound and whether optimality was proven (HiGHS; CBC
fallback). Eight synthetic scenarios live in `src/scenarios.py`.

Headline findings (see `paper.tex`, generated artifacts in `results/`): Dynamic
EDF+DRR lowers SLA-violation rate in every scenario but raises weighted
tardiness outside overload; ATC has the lowest weighted tardiness; the
forecast-driven variant gives no significant benefit over static EDF+DRR.

## Run the baseline

From the repository root:

```text
python -m src.run_baseline --requests 1000 --seed 42
```

The command prints JSON containing:

- `average_waiting_time`: mean time from arrival until service starts
- `p95_waiting_time`: 95th percentile waiting time
- `sla_violations`: requests completing after their deadline
- `sla_violation_rate`: SLA violations divided by total requests
- `throughput`: completed requests per elapsed experiment second
- `cpu_utilization`: consumed CPU-seconds divided by available CPU-seconds
- `fairness_index`: Jain's index over tier completion rates; 1.0 is perfectly equal

## Run tests

```text
python -m unittest discover -s tests -v
```

## Baseline assumptions

- One FIFO server processes one request at a time.
- Requests arrive according to an exponential inter-arrival distribution.
- Service time and CPU demand are sampled synthetically.
- VIP, Premium, and Free tiers have different deadline-slack distributions.
- The random seed makes experiments reproducible.
- All generated requests are allowed to complete; later milestones can add finite horizons and admission control.

The next comparison should reuse the same generated request batches and evaluate EDF and DRR against this FIFO baseline.

## Production workload: Alibaba GenTD26

The project can also evaluate the real request-level workload from Alibaba's
[GenTD26 dataset](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI).
Download the dataset from its official repository and pass the extracted
`lora_request_trace.csv` to the comparison runner:

```text
python -m src.run_comparison --dataset path/to/lora_request_trace.csv --max-requests 10000 --horizon 1000
```

The adapter in `src/gen_td26.py` uses these transparent preprocessing rules:

- `gmt_create` becomes relative `arrival_time`.
- `exec_time_seconds` becomes `service_time`.
- `predict_type` and other request attributes remain available in the source row;
	the common scheduler model uses the request timing fields.
- `groupId` is mapped deterministically using SHA-256: 20% VIP, 30% Premium,
	and 50% Free by hash bucket. This is an experimental assignment, not a
	claim about the original users' commercial tiers.
- Deadlines are arrival time plus `2x`, `4x`, or `8x` service time for VIP,
	Premium, or Free, respectively, with a 0.25-second minimum slack.
- Penalties are `10`, `4`, and `1` for VIP, Premium, and Free.
- CPU demand is a bounded proxy from execution time, inference steps, image
	count, prompt length, and LoRA count. GenTD26 reports GPU-oriented metrics,
	so this is not measured CPU usage and must be identified as a modeling
	assumption in the final report.

The synthetic generator remains the default and is used for controlled stress
tests. GenTD26 is used for the real-workload evaluation. The final report
should present both results separately and cite the GenTD26 SoCC'25 paper.

## Compare FIFO, EDF, DRR, EDF+DRR, and predictive EDF+DRR

All five schedulers can be evaluated on the exact same workload:

```text
python -m src.run_comparison --requests 1000 --seed 42
```

To evaluate behavior during a fixed operating window, pass a finite horizon:

```text
python -m src.run_comparison --requests 1000 --seed 42 --horizon 100
```

With a horizon, requests arriving before the cutoff are considered offered,
but only requests completed by the cutoff count toward throughput, SLA
violations, CPU utilization, and tier fairness. Requests still waiting at the
cutoff represent backlog and are not counted as completed.

To replay GenTD26 with a parallel worker pool, for example 16 workers:

```text
python -m src.run_comparison --dataset data/GenTD26/lora_request_trace.csv --max-requests 26823 --workers 16
```

For more realistic backlog/fairness results, combine workers with a finite
horizon. The reported CPU capacity scales as `100 * workers`; this is a
simulation abstraction, while GenTD26's actual infrastructure is GPU- and
container-based.

The finite-horizon capacity sweep is saved in
`results/gen_td26_finite_horizon_capacity.csv`. It uses a 1,750,000-second
replay window, which contains 24,522 offered requests and leaves measurable
backlog at low worker capacities. The horizon is deliberately before the end
of the trace so the system cannot drain the workload before measurement.

EDF is currently non-preemptive. It chooses the available request with the
earliest deadline whenever the server becomes free.

DRR maintains one FIFO queue per tier. Its default quantum is `0.60` for VIP,
`0.30` for Premium, and `0.15` for Free, giving higher tiers more service
opportunities while still allowing every non-empty tier to progress. DRR uses
service time as the request cost and carries unused deficit between rounds.

The simulator also defines a fairness-oriented sensitivity profile in
`src/drr.py`: VIP `0.20`, Premium `0.40`, and Free `0.60`. This profile is not
the default because it intentionally trades VIP priority for more equal
completion rates. Quantum selection must be evaluated across multiple seeds;
the profile must not be chosen from a single workload result.

Finite-horizon outputs include per-tier waiting time, completion rate, backlog,
and SLA violation rate. Completion rate and backlog use all requests offered
before the horizon; waiting and SLA violation rate are calculated over requests
completed by the horizon.

EDF+DRR combines the policies: DRR selects the next tier, then EDF selects the
earliest-deadline request within that tier. This prevents one busy tier from
monopolizing the server while still prioritizing urgent requests within each
tier.

Predictive EDF+DRR uses the same tier and deadline rules, but adjusts each
round's DRR quantum from the pending service demand. It is included in the
single-worker comparison, parallel-worker comparison, capacity report, and
MILP benchmark commands as a deterministic proxy for ML-based traffic
adaptation; it does not change the non-preemptive service model.

## Train the traffic-spike model

The project includes a dependency-free lightweight ridge-regression forecaster
that predicts next-window request demand per tier from recent bucket counts,
trend, and cyclical time features. Train it on synthetic or GenTD26 traffic:

```text
python -m src.train_traffic_model --requests 5000 --output results/traffic_model.json
python -m src.run_comparison --requests 1000 --traffic-model results/traffic_model.json
```

For GenTD26:

```text
python -m src.train_traffic_model --dataset data/GenTD26/lora_request_trace.csv --max-requests 10000
python -m src.run_comparison --dataset data/GenTD26/lora_request_trace.csv --max-requests 10000 --traffic-model results/traffic_model.json
```

The model is deliberately small and serializes to JSON; no ML runtime is
required in production. Predicted demand is combined with current backlog
demand, and each tier retains a minimum half-quantum floor to prevent
starvation. The model is a forecasting component, not evidence that an
XGBoost is available as an optional paper comparator through
`src.xgboost_model`; it is trained per tier and saved alongside metadata.
Install dependencies from `requirements.txt` before using it.

The paper evaluation must use chronological train/validation/test splits,
report MAE/RMSE for demand forecasts, define spikes using a preceding-window
threshold, and report precision, recall, and F1. The simulator also reports
expired requests and useful work ratio so overload results do not hide wasted
service.

For paper evidence, run the reproducible paired multi-seed evaluation:

```text
python -m src.evaluate_research --seeds 30 --output results/research_evaluation.json
```

Report the mean and 95% confidence interval of paired deltas against static
EDF+DRR. Do not claim a predictive improvement unless the interval excludes
zero in the favorable direction on unseen test traffic.

## MILP theoretical benchmark

Install the optimization dependency from `requirements.txt`, then run the
offline benchmark on a small batch:

```text
python -m pip install -r requirements.txt
python -m src.run_milp --requests 20 --seed 42
```

To run the exact benchmark on a sampled GenTD26 production batch:

```text
python -m src.run_milp --dataset path/to/lora_request_trace.csv --max-requests 20
```

The same augmented GenTD26 requests are sent to FIFO, EDF, DRR, EDF+DRR, and
MILP. Keep `--max-requests` small because the MILP contains pairwise ordering
variables and becomes expensive as the batch grows.

The MILP minimizes penalty-weighted SLA tardiness while enforcing request
arrival times and single-server non-overlap. It is the theoretical reference
for FIFO, EDF, DRR, and EDF+DRR. Because pairwise ordering variables grow
quadratically, use it for small batches (10–50 requests), not for the full
production trace or live scheduling.

Generate a paper-ready table comparing every policy with MILP ground truth:

```text
python -m src.compare_ground_truth --requests 20 --seed 42
```

The report includes weighted tardiness, objective gap versus MILP, mean and
P95 waiting time, SLA violation rate, useful work ratio, throughput, and
fairness. MILP is the ground truth only for its weighted-tardiness objective;
another policy may be better on a different operational metric.
