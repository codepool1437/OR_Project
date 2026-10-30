# Ground-Truth Scheduler Comparison

| Policy | Weighted tardiness | Gap vs MILP | Avg wait | P95 wait | SLA rate | Useful work | Throughput | Fairness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 9.720954 | N/A | 1.113734 | 2.193094 | 0.2143 | 0.7857 | 4.556736 | 1.000000 |
| EDF | 0.000000 | N/A | 1.146958 | 2.388947 | 0.0000 | 1.0000 | 4.556736 | 1.000000 |
| DRR | 1.906884 | N/A | 1.154597 | 2.190995 | 0.0714 | 0.9286 | 4.556736 | 1.000000 |
| EDF+DRR | 0.000000 | N/A | 1.108072 | 2.388947 | 0.0000 | 1.0000 | 4.556736 | 1.000000 |
| ATC | 0.000000 | N/A | 1.143211 | 2.388947 | 0.0000 | 1.0000 | 4.556736 | 1.000000 |
| Predictive EDF+DRR | 1.062772 | N/A | 1.118054 | 2.388947 | 0.1429 | 0.8571 | 4.556736 | 1.000000 |
| Dynamic EDF+DRR | 0.000000 | N/A | 1.108072 | 2.388947 | 0.0000 | 1.0000 | 4.556736 | 1.000000 |
| MILP (ground truth) | 0.000000 | N/A | 1.272103 | 2.495101 | 0.0000 | 1.0000 | 4.554325 | 1.000000 |

## Best policy by metric

- **weighted_tardiness**: EDF
- **average_waiting_time**: EDF+DRR
- **p95_waiting_time**: DRR
- **sla_violation_rate**: EDF
- **useful_work_ratio**: EDF
- **throughput**: FIFO
- **fairness_index**: FIFO
