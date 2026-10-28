"""Forecast and traffic-spike metrics used by the research evaluation."""

from math import sqrt
from statistics import mean
from typing import Sequence


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
