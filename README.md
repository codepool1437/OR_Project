# AI-Predictive API Gateway Scheduler

This repository contains the first project milestone: a reproducible synthetic workload and FIFO scheduling baseline.

The frozen research question, hypotheses, assumptions, scope, and evaluation
metrics are recorded in [`docs/research_design.md`](docs/research_design.md).

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

## Compare FIFO, EDF, DRR, and EDF+DRR

All four schedulers can be evaluated on the exact same workload:

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
