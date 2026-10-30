"""Forecast and traffic-spike metrics used by the research evaluation."""

from math import sqrt
from statistics import mean, stdev
from typing import Iterable, Sequence

# Two-sided 97.5% Student-t quantiles for small paired samples (df = n - 1).
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def weighted_tardiness(scheduled: Iterable) -> float:
    """Penalty-weighted tardiness: sum of penalty * max(0, completion - deadline)."""
    return sum(
        item.request.sla_penalty * max(0.0, item.completion_time - item.request.deadline)
        for item in scheduled
    )


def mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using the Student-t 95% interval."""
    average = mean(values)
    if len(values) < 2:
        return average, average, average
    df = len(values) - 1
    t_value = _T975.get(df, 1.96)
    half = t_value * stdev(values) / sqrt(len(values))
    return average, average - half, average + half


def spike_labels(
    counts: Sequence[float], window: int = 6, sigma: float = 2.0
) -> list[bool]:
    """Label a bucket as a spike using only preceding observations."""
    labels = []
    for index, value in enumerate(counts):
        history = list(counts[max(0, index - window):index])
        if len(history) < 2:
            labels.append(False)
            continue
        average = mean(history)
        deviation = sqrt(mean((item - average) ** 2 for item in history))
        labels.append(value > average + sigma * deviation)
    return labels


def classification_metrics(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    """Return precision, recall, and F1 for spike detection."""
    true_positive = sum(a and p for a, p in zip(actual, predicted))
    false_positive = sum(not a and p for a, p in zip(actual, predicted))
    false_negative = sum(a and not p for a, p in zip(actual, predicted))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Return MAE and RMSE for a chronological forecast."""
    errors = [prediction - value for value, prediction in zip(actual, predicted)]
    return {
        "mae": mean(abs(error) for error in errors) if errors else 0.0,
        "rmse": sqrt(mean(error * error for error in errors)) if errors else 0.0,
    }
