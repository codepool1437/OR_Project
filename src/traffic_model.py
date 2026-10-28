"""Lightweight, dependency-free traffic forecasting for predictive DRR."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .models import Request


@dataclass(frozen=True)
class TrafficSpikeModel:
    """Per-tier ridge regressors forecasting requests in the next time window."""

    window_size: float
    tiers: tuple[str, ...]
    coefficients: dict[str, tuple[float, ...]]
    intercepts: dict[str, float]
    history_length: int = 3
    blend_weights: dict[str, float] | None = None

    def predict(
        self,
        history: Mapping[str, Sequence[float]],
        time_index: int,
    ) -> dict[str, float]:
        """Return non-negative next-window request-rate predictions."""
        predictions = {}
        for tier in self.tiers:
            values = list(history.get(tier, ()))
            recent = (values + [0.0] * self.history_length)[-self.history_length :]
            average = sum(recent) / self.history_length
            trend = recent[-1] - recent[0]
            angle = 2.0 * math.pi * time_index / 24.0
            features = recent + [average, trend, math.sin(angle), math.cos(angle)]
            value = self.intercepts[tier] + sum(
                coefficient * feature
                for coefficient, feature in zip(self.coefficients[tier], features)
            )
            weight = (self.blend_weights or {}).get(tier, 0.5)
            naive = recent[-1]
            predictions[tier] = max(0.0, weight * value + (1.0 - weight) * naive)
        return predictions

    def to_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "tiers": list(self.tiers),
            "history_length": self.history_length,
            "coefficients": {
                tier: list(values) for tier, values in self.coefficients.items()
            },
            "intercepts": self.intercepts,
            "blend_weights": self.blend_weights or {},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TrafficSpikeModel":
        tiers = tuple(str(tier) for tier in data["tiers"])  # type: ignore[index]
        history_length = int(data.get("history_length", 3))
        coefficients_data = data["coefficients"]  # type: ignore[index]
        intercepts_data = data["intercepts"]  # type: ignore[index]
        if not isinstance(coefficients_data, Mapping) or not isinstance(
            intercepts_data, Mapping
        ):
            raise ValueError("invalid traffic model coefficients")
        coefficients = {
            tier: tuple(float(value) for value in coefficients_data[tier])  # type: ignore[index]
            for tier in tiers
        }
        intercepts = {tier: float(intercepts_data[tier]) for tier in tiers}  # type: ignore[index]
        blend_data = data.get("blend_weights", {})
        blend_weights = (
            {tier: float(blend_data.get(tier, 0.5)) for tier in tiers}
            if isinstance(blend_data, Mapping)
            else None
        )
        return cls(
            window_size=float(data["window_size"]),  # type: ignore[index]
            tiers=tiers,
            coefficients=coefficients,
            intercepts=intercepts,
            history_length=history_length,
            blend_weights=blend_weights,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrafficSpikeModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("traffic model file must contain a JSON object")
        return cls.from_dict(data)


def _solve_ridge(
    rows: list[list[float]], targets: list[float], regularization: float
) -> tuple[float, ...]:
    size = len(rows[0])
    matrix = [[0.0] * (size + 1) for _ in range(size)]
    for row, target in zip(rows, targets):
        augmented = [1.0] + row
        for i in range(size):
            for j in range(size):
                matrix[i][j] += augmented[i] * augmented[j]
            matrix[i][-1] += augmented[i] * target
    for index in range(1, size):
        matrix[index][index] += regularization
    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda row: abs(matrix[row][pivot]))
        if abs(matrix[pivot_row][pivot]) < 1e-12:
            continue
        matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot]
        divisor = matrix[pivot][pivot]
        matrix[pivot] = [value / divisor for value in matrix[pivot]]
        for row in range(size):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            matrix[row] = [
                left - factor * right
                for left, right in zip(matrix[row], matrix[pivot])
            ]
    return tuple(matrix[index][-1] for index in range(size))


def train_traffic_model(
    requests: Iterable[Request],
    window_size: float = 1.0,
    regularization: float = 1e-3,
) -> TrafficSpikeModel:
    """Train next-window per-tier regressors from timestamped requests."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    ordered = list(requests)
    if not ordered:
        raise ValueError("at least one request is required")
    tiers = tuple(sorted({request.tier for request in ordered}))
    bucket_count = max(
        1, int(max(request.arrival_time for request in ordered) // window_size) + 1
    )
    counts = {tier: [0.0] * bucket_count for tier in tiers}
    for request in ordered:
        bucket = min(bucket_count - 1, int(request.arrival_time // window_size))
        counts[request.tier][bucket] += 1.0

    rows_by_tier: dict[str, list[list[float]]] = {tier: [] for tier in tiers}
    targets = {tier: [] for tier in tiers}
    for index in range(3, bucket_count - 1):
        for tier in tiers:
            recent = [counts[tier][index - offset] for offset in (3, 2, 1)]
            average = sum(recent) / 3.0
            trend = recent[-1] - recent[0]
            angle = 2.0 * math.pi * index / 24.0
            rows_by_tier[tier].append(
                recent + [average, trend, math.sin(angle), math.cos(angle)]
            )
            targets[tier].append(counts[tier][index + 1])
    if not any(rows_by_tier.values()):
        targets = {tier: [sum(counts[tier])] for tier in tiers}
        rows_by_tier = {tier: [[0.0] * 7] for tier in tiers}

    coefficients = {}
    intercepts = {}
    blend_weights = {}
    for tier in tiers:
        solution = _solve_ridge(
            rows_by_tier[tier], targets[tier], regularization
        )
        intercepts[tier] = solution[0]
        coefficients[tier] = tuple(solution[1:])
        best_weight = 0.5
        best_error = float("inf")
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            error = 0.0
            for row, target in zip(rows_by_tier[tier], targets[tier]):
                recent = row[:3]
                ridge = solution[0] + sum(
                    coefficient * feature
                    for coefficient, feature in zip(solution[1:], row)
                )
                prediction = weight * ridge + (1.0 - weight) * recent[-1]
                error += (prediction - target) ** 2
            if error < best_error:
                best_error = error
                best_weight = weight
        blend_weights[tier] = best_weight
    return TrafficSpikeModel(
        window_size=window_size,
        tiers=tiers,
        coefficients=coefficients,
        intercepts=intercepts,
        blend_weights=blend_weights,
    )
