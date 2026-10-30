# Forecast accuracy (chronological 60/20/20 train/validation/test)

| Workload | Split | Model | MAE | RMSE | Spike precision | Spike recall | Spike F1 |
|---|---|---|---:|---:|---:|---:|---:|
| synthetic_mixed_burst | validation | naive (last window) | 2.319 | 3.276 | 0.206 | 0.050 | 0.081 |
| synthetic_mixed_burst | validation | moving average (3) | 2.133 | 2.925 | 0.000 | 0.000 | 0.000 |
| synthetic_mixed_burst | validation | ridge (proposed) | 1.969 | 2.849 | 0.438 | 0.050 | 0.090 |
| synthetic_mixed_burst | validation | xgboost | 1.784 | 2.520 | 0.478 | 0.309 | 0.376 |
| synthetic_mixed_burst | test | naive (last window) | 2.192 | 2.986 | 0.231 | 0.049 | 0.081 |
| synthetic_mixed_burst | test | moving average (3) | 2.028 | 2.704 | 0.000 | 0.000 | 0.000 |
| synthetic_mixed_burst | test | ridge (proposed) | 1.908 | 2.647 | 0.800 | 0.066 | 0.121 |
| synthetic_mixed_burst | test | xgboost | 1.686 | 2.317 | 0.561 | 0.377 | 0.451 |
| synthetic_random_burst | validation | naive (last window) | 2.178 | 2.906 | 0.087 | 0.017 | 0.028 |
| synthetic_random_burst | validation | moving average (3) | 1.949 | 2.567 | 0.000 | 0.000 | 0.000 |
| synthetic_random_burst | validation | ridge (proposed) | 1.989 | 2.929 | 0.000 | 0.000 | 0.000 |
| synthetic_random_burst | validation | xgboost | 1.993 | 2.612 | 0.270 | 0.140 | 0.185 |
| synthetic_random_burst | test | naive (last window) | 2.026 | 2.676 | 0.062 | 0.009 | 0.016 |
| synthetic_random_burst | test | moving average (3) | 1.806 | 2.407 | 0.000 | 0.000 | 0.000 |
| synthetic_random_burst | test | ridge (proposed) | 2.005 | 2.886 | 0.500 | 0.019 | 0.036 |
| synthetic_random_burst | test | xgboost | 1.943 | 2.519 | 0.278 | 0.204 | 0.235 |
| gentd26 | validation | naive (last window) | 2.142 | 4.317 | 0.264 | 0.110 | 0.155 |
| gentd26 | validation | moving average (3) | 2.103 | 4.097 | 0.000 | 0.000 | 0.000 |
| gentd26 | validation | ridge (proposed) | 3.623 | 7.239 | 0.052 | 0.201 | 0.083 |
| gentd26 | validation | xgboost | 4.143 | 6.675 | 0.092 | 0.450 | 0.153 |
| gentd26 | test | naive (last window) | 1.877 | 6.151 | 0.310 | 0.154 | 0.206 |
| gentd26 | test | moving average (3) | 1.945 | 5.596 | 0.000 | 0.000 | 0.000 |
| gentd26 | test | ridge (proposed) | 3.068 | 7.575 | 0.050 | 0.257 | 0.084 |
| gentd26 | test | xgboost | 3.739 | 7.197 | 0.072 | 0.451 | 0.125 |
