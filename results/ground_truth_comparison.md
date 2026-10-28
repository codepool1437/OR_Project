# Ground-Truth Scheduler Comparison

| Policy | Weighted tardiness | Gap vs MILP | Avg wait | P95 wait | SLA rate | Useful work | Throughput | Fairness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 0.000000 | N/A | 0.102451 | 0.487280 | 0.0000 | 1.0000 | 4.528911 | 1.000000 |
| EDF | 0.000000 | N/A | 0.108821 | 0.640024 | 0.0000 | 1.0000 | 4.528911 | 1.000000 |
| DRR | 0.000000 | N/A | 0.104022 | 0.487280 | 0.0000 | 1.0000 | 4.528911 | 1.000000 |
| EDF+DRR | 0.000000 | N/A | 0.108821 | 0.640024 | 0.0000 | 1.0000 | 4.528911 | 1.000000 |
| Predictive EDF+DRR | 0.000000 | N/A | 0.101512 | 0.487280 | 0.0000 | 1.0000 | 4.528911 | 1.000000 |
| MILP (ground truth) | 0.000000 | N/A | 0.132601 | 0.529249 | 0.0000 | 1.0000 | 4.480822 | 1.000000 |

## Best policy by metric

- **weighted_tardiness**: FIFO
- **average_waiting_time**: Predictive EDF+DRR
- **p95_waiting_time**: FIFO
- **sla_violation_rate**: FIFO
- **useful_work_ratio**: FIFO
- **throughput**: FIFO
- **fairness_index**: FIFO
