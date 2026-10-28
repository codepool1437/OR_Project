# Ground-Truth Scheduler Comparison

| Policy | Weighted tardiness | Gap vs MILP | Avg wait | P95 wait | SLA rate | Useful work | Throughput | Fairness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
N/A | 0.320202 | 0.699886 | 0.2000 | 0.8000 | 7.881220 | 1.000000 |
N/A | 0.341848 | 0.880690 | 0.0000 | 1.0000 | 7.881220 | 1.000000 |
N/A | 0.303930 | 0.666385 | 0.0000 | 1.0000 | 7.881220 | 1.000000 |
N/A | 0.341603 | 0.880690 | 0.0000 | 1.0000 | 7.881220 | 1.000000 |
N/A | 0.374567 | 0.880690 | 0.0000 | 1.0000 | 7.881220 | 1.000000 |
N/A | 0.338738 | 0.831196 | 0.0000 | 1.0000 | 7.672157 | 1.000000 |

## Best policy by metric

- **weighted_tardiness**: EDF
- **average_waiting_time**: DRR
- **p95_waiting_time**: DRR
- **sla_violation_rate**: EDF
- **useful_work_ratio**: EDF
- **throughput**: FIFO
- **fairness_index**: FIFO
