# MILP benchmark (HiGHS, warm-started from the best heuristic)

Mean excess weighted tardiness over the MILP incumbent (`gap vs MILP`, never negative because the MILP incumbent is at least as good as every warm-start heuristic) and over the proven dual bound (`gap vs bound`, a valid but possibly loose upper estimate of the true optimality gap). Both coincide when optimality is proven.

## Gap vs MILP incumbent

| Size | Machines | Instances | Proven optimal | FIFO | EDF | DRR | EDF+DRR | ATC | Predictive EDF+DRR (backlog proxy) | Dynamic EDF+DRR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 32 | 32 | 6.857 | 0.446 | 1.934 | 0.690 | 0.372 | 1.201 | 1.360 |
| 10 | 2 | 32 | 32 | 0.905 | 0.035 | 0.217 | 0.063 | 0.035 | 0.093 | 0.190 |
| 14 | 1 | 32 | 15 | 26.150 | 1.563 | 6.459 | 4.158 | 0.306 | 2.672 | 5.529 |
| 14 | 2 | 32 | 30 | 5.027 | 0.066 | 1.147 | 0.277 | 0.083 | 0.367 | 0.576 |
| 18 | 1 | 32 | 0 | 48.689 | 3.727 | 15.099 | 10.151 | 0.225 | 7.780 | 10.706 |
| 18 | 2 | 32 | 15 | 11.852 | 0.055 | 3.414 | 0.790 | 0.089 | 0.857 | 1.397 |

## Gap vs proven lower bound

| Size | Machines | Mean MILP incumbent | Mean lower bound | FIFO | EDF | DRR | EDF+DRR | ATC | Predictive EDF+DRR (backlog proxy) | Dynamic EDF+DRR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 0.352 | 0.352 | 6.857 | 0.446 | 1.934 | 0.690 | 0.372 | 1.201 | 1.360 |
| 10 | 2 | 0.000 | 0.000 | 0.905 | 0.035 | 0.217 | 0.063 | 0.035 | 0.093 | 0.190 |
| 14 | 1 | 4.023 | 0.410 | 29.762 | 5.175 | 10.071 | 7.770 | 3.918 | 6.284 | 9.142 |
| 14 | 2 | 0.293 | 0.037 | 5.283 | 0.322 | 1.402 | 0.533 | 0.339 | 0.623 | 0.832 |
| 18 | 1 | 11.643 | 0.358 | 59.974 | 15.012 | 26.384 | 21.436 | 11.510 | 19.064 | 21.990 |
| 18 | 2 | 0.612 | 0.036 | 12.428 | 0.631 | 3.989 | 1.366 | 0.664 | 1.432 | 1.972 |
