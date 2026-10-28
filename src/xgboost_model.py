"""XGBoost traffic forecasting adapter for paper experiments."""

from dataclasses import dataclass
import json
from importlib import import_module
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .models import Request


def _features(history: Sequence[float], time_index: int) -> list[float]:
    recent = (list(history) + [0.0, 0.0, 0.0])[-3:]
    return recent + [sum(recent) / 3.0, recent[-1] - recent[0], time_index % 24]


@dataclass
class XGBoostTrafficModel:
    """Per-tier XGBoost next-window request-count forecaster."""

    window_size: float
    tiers: tuple[str, ...]
    models: dict[str, object]

    def predict(
        self, history: Mapping[str, Sequence[float]], time_index: int
    ) -> dict[str, float]:
        predictions = {}
        for tier in self.tiers:
            value = self.models[tier].predict(
                [_features(history.get(tier, ()), time_index)]
            )[0]
            predictions[tier] = max(0.0, float(value))
        return predictions


def train_xgboost_model(
    requests: Iterable[Request], output: str | Path, window_size: float = 60.0
) -> XGBoostTrafficModel:
    """Train and save one XGBoost regressor per tier."""
    try:
        xgb = import_module("xgboost")
    except ImportError as error:
        raise RuntimeError("install xgboost and scikit-learn first") from error
    ordered = list(requests)
    if not ordered or window_size <= 0:
        raise ValueError("requests must be non-empty and window_size must be positive")
    tiers = tuple(sorted({request.tier for request in ordered}))
    bucket_count = int(max(request.arrival_time for request in ordered) // window_size) + 1
    counts = {tier: [0.0] * bucket_count for tier in tiers}
    for request in ordered:
        counts[request.tier][int(request.arrival_time // window_size)] += 1.0
    models: dict[str, object] = {}
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for tier in tiers:
        rows = [
            _features(counts[tier][index - 3:index], index)
            for index in range(3, bucket_count - 1)
        ]
        targets = [counts[tier][index + 1] for index in range(3, bucket_count - 1)]
        if not rows:
            rows, targets = [[0.0] * 6], [0.0]
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            n_jobs=1,
        )
        model.fit(rows, targets)
        model.save_model(str(output_path.with_name(f"{output_path.stem}_{tier}.json")))
        models[tier] = model
    metadata = {"window_size": window_size, "tiers": list(tiers)}
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return XGBoostTrafficModel(window_size, tiers, models)


def load_xgboost_model(path: str | Path) -> XGBoostTrafficModel:
    """Load metadata and per-tier XGBoost model files."""
    try:
        xgb = import_module("xgboost")
    except ImportError as error:
        raise RuntimeError("install xgboost and scikit-learn first") from error
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    base = Path(path)
    models = {}
    for tier in metadata["tiers"]:
        model = xgb.XGBRegressor()
        model.load_model(str(base.with_name(f"{base.stem}_{tier}.json")))
        models[tier] = model
    return XGBoostTrafficModel(float(metadata["window_size"]), tuple(metadata["tiers"]), models)
