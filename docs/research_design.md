# Research Design

## Working title

**Predictive EDF+DRR Scheduling for Fair, SLA-Aware Multi-Tier API Gateways**

## Core problem

A multi-tier API gateway must schedule requests from users with different service
levels during traffic bursts. A scheduler that only prioritizes the earliest
SLA deadline can protect urgent premium requests but may starve lower-tier
requests. A scheduler that only preserves queue fairness can allow premium
requests to miss their deadlines. The research problem is therefore to reduce
weighted SLA violations while preserving measurable service fairness across
user tiers under limited worker capacity.

## Primary research question

> Under bursty, multi-tier API workloads and limited worker capacity, can a
> traffic-forecast-informed EDF+DRR scheduler reduce SLA violations while
> maintaining tier fairness better than reactive FIFO, EDF, DRR, and static
> EDF+DRR scheduling?

## Secondary research question

> How much improvement does predictive EDF+DRR provide over static EDF+DRR
> across predictable, unpredictable, and sustained traffic conditions?

The static policies are required ablations. The intended novelty is not the
claim that EDF or DRR individually are new; it is the integration of
short-term load forecasting with deadline-aware, tier-fair quantum adaptation.
The phrase "first" or "no prior work" must only be used after a documented
systematic literature review confirms the exact claim.

## Research objectives

1. Define a common request and SLA model for all experiments.
2. Compare FIFO, EDF, DRR, and EDF+DRR under identical workloads.
3. Measure the trade-off between SLA compliance, waiting time, throughput, and
   tier fairness as worker capacity changes.
4. Use a MILP model as an offline optimum for small request batches.
5. Evaluate the policies on controlled synthetic workloads and the real GenTD26
   request trace.
6. Add forecasting as the predictive policy-adaptation layer and evaluate it
  against static EDF+DRR through an ablation study.

## Hypotheses

### H1: Deadline performance

EDF will produce fewer SLA violations than FIFO under overloaded workloads
because it prioritizes requests with the earliest deadlines.

### H2: Fairness

DRR will provide more balanced per-tier completion rates than global EDF when
one tier generates a sustained backlog.

### H3: Hybrid performance

EDF+DRR will achieve a better SLA/fairness trade-off than EDF or DRR alone:
DRR selects the tier opportunity, while EDF selects the most urgent request
within that tier.

### H4: Capacity sensitivity

The advantage of scheduling policy will be largest near the capacity boundary.
When worker capacity is either severely insufficient or more than sufficient,
all policies will respectively struggle or perform similarly.

### H5: Predictive novelty

Forecast-informed DRR quantum adjustment will reduce SLA violations or waiting
time during predictable surges compared with static EDF+DRR, without causing a
material reduction in lower-tier completion fairness.

### H6: Forecast usefulness boundary

Forecast-informed EDF+DRR will provide little or no benefit for unpredictable
bursts or when worker capacity is already sufficient. This tests whether any
improvement comes from useful prediction rather than from changing the policy
alone.

## Algorithms in scope

### Baselines

- **FIFO:** arrival-order scheduling.
- **EDF:** global earliest-deadline-first scheduling.
- **DRR:** deficit round robin across tiers.

### Proposed static foundation

- **EDF+DRR:** DRR chooses the tier service opportunity; EDF chooses the
  earliest-deadline request within that tier.

### Proposed novel scheduler

- **Predictive EDF+DRR:** a traffic forecast adjusts DRR quanta within
  configured safety bounds. The forecast never bypasses deadline or fairness
  constraints.

## Common request model

Each request is represented by:

- `request_id`
- `arrival_time`
- `service_time`
- `cpu_demand` or normalized resource-demand proxy
- `tier`: VIP, Premium, or Free
- `deadline`
- `sla_penalty`
- `endpoint`

All policies receive the same request batch, worker capacity, horizon, and
initial conditions.

## Evaluation metrics

### Primary metrics

- SLA violation rate
- Penalty-weighted SLA tardiness
- Per-tier completion rate
- Jain fairness index over per-tier completion rates

### Secondary metrics

- Mean waiting time
- P95 waiting time
- Throughput
- Worker/resource utilization
- Backlog at the simulation horizon
- Per-tier waiting time and SLA violation rate
- MILP optimality gap for small batches
- Scheduler runtime

## Workload sources

### Primary real trace

Alibaba GenTD26 `lora_request_trace.csv` is the primary production workload
trace. It provides request timestamps, execution duration, request type, group
identity, and inference complexity fields.

GenTD26 does not provide original commercial SLA tiers, deadlines, penalties,
or direct CPU demand. Those values are added through documented preprocessing
rules and are treated as experimental assumptions, not observed production
facts.

### Controlled workloads

Synthetic workloads are retained for controlled stress tests:

- normal traffic
- sustained overload
- VIP spike
- Free-tier spike
- mixed-tier burst
- tight deadlines
- relaxed deadlines

Synthetic experiments use fixed seeds and the same request batches across all
policies.

## Frozen experimental assumptions

- Scheduling is non-preemptive.
- Workers are identical and process one request at a time.
- Worker capacities tested: 1, 2, 4, 8, 16, and 32.
- Finite-horizon experiments are required for meaningful backlog and fairness
  measurements.
- FIFO, EDF, DRR, static EDF+DRR, and predictive EDF+DRR are evaluated on
  identical inputs.
- MILP is an offline benchmark for small batches only; it is not a live policy.
- GenTD26 tier, deadline, penalty, and CPU mappings are explicitly modeled
  assumptions.
- Static EDF+DRR is the required ablation and fallback contribution; ML is the
  novelty layer, not a replacement for the static baseline.

## Scope boundary

This paper evaluates scheduling policy in a replay/simulation environment. It
does not claim to reproduce Alibaba's internal gateway, GPU cluster, or actual
commercial SLA policy. A FastAPI interface may demonstrate the scheduler later,
but live production traffic is outside the primary evaluation.

## Decision rule for the final claim

The predictive contribution is supported if predictive EDF+DRR achieves a
statistically meaningful improvement in the SLA/fairness trade-off over static
EDF+DRR and the reactive baselines on forecastable surge scenarios, without an
unacceptable degradation in fairness, throughput, or scheduler runtime. The
paper must report negative or neutral ML results honestly if this condition is
not met.
