"""Load and transparently augment Alibaba GenTD26 request traces.

Expected source file: ``lora_request_trace.csv`` from GenTD26.
The loader converts production request records into the project's common
``Request`` model. SLA fields are not present in the public trace, so the
augmentation rules are explicit and deterministic rather than inferred.
"""

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Mapping

from .models import Request

DEFAULT_TIER_RULES: dict[str, tuple[float, float, float]] = {
    # tier: (hash upper bound, deadline multiplier, penalty)
    "VIP": (20.0, 2.0, 10.0),
    "Premium": (50.0, 4.0, 4.0),
    "Free": (100.0, 8.0, 1.0),
}


def _stable_percent(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) * 100.0


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        for format_string in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(value.strip(), format_string)
            except ValueError:
                continue
    raise ValueError(f"unsupported gmt_create timestamp: {value!r}")


def _number(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"invalid numeric value for {name}: {value!r}") from error


def _tier_for_group(group_id: str, rules: Mapping[str, tuple[float, float, float]]) -> str:
    percentile = _stable_percent(group_id or "missing-group")
    for tier, (upper_bound, _, _) in rules.items():
        if percentile < upper_bound:
            return tier
    raise ValueError("tier rules must cover the full 0-100 percentile range")


def _cpu_demand(row: Mapping[str, str], service_time: float) -> float:
    """Estimate normalized CPU demand from observable request complexity.

    GenTD26 reports GPU-oriented inference characteristics rather than CPU
    demand. This proxy is deliberately bounded to 10-100 CPU units and uses
    execution time, diffusion steps, image count, prompt length, and LoRA
    count. The formula is a modeling assumption, not measured CPU usage.
    """
    steps = _number(row, "num_inference_steps")
    images = _number(row, "num_images_per_prompt", 1.0)
    prompt_length = _number(row, "prompt_length")
    loras = _number(row, "num_lora")
    complexity = (
        service_time * 2.0
        + steps / 30.0
        + images
        + prompt_length / 100.0
        + loras * 0.5
    )
    return min(100.0, max(10.0, 10.0 + complexity * 10.0))


def load_gen_td26_requests(
    csv_path: str | Path,
    *,
    max_requests: int | None = None,
    tier_rules: Mapping[str, tuple[float, float, float]] = DEFAULT_TIER_RULES,
) -> list[Request]:
    """Load successful GenTD26 requests and augment them for scheduling.

    ``arrival_time`` is seconds relative to the first request. ``service_time``
    is ``exec_time_seconds``. Requests with missing/invalid timestamps or
    durations are rejected with a row-specific error; non-success statuses are
    skipped because they do not represent completed service requests.
    """
    if max_requests is not None and max_requests < 1:
        raise ValueError("max_requests must be positive")
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"gmt_create", "exec_time_seconds", "groupId"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GenTD26 CSV is missing required columns: {sorted(missing)}")

        raw_rows: list[tuple[datetime, dict[str, str]]] = []
        for row_number, row in enumerate(reader, start=2):
            status = row.get("predict_status", "").strip().upper()
            if status and status not in {"SUCCEED", "SUCCESS", "COMPLETED"}:
                continue
            try:
                timestamp = _parse_timestamp(row["gmt_create"])
                service_time = _number(row, "exec_time_seconds")
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid GenTD26 row {row_number}: {error}") from error
            if service_time <= 0:
                continue
            raw_rows.append((timestamp, row))
            if max_requests is not None and len(raw_rows) >= max_requests:
                break

    raw_rows.sort(key=lambda item: item[0])
    if not raw_rows:
        return []
    first_timestamp = raw_rows[0][0]
    requests: list[Request] = []
    for request_id, (timestamp, row) in enumerate(raw_rows):
        service_time = _number(row, "exec_time_seconds")
        tier = _tier_for_group(row.get("groupId", ""), tier_rules)
        _, deadline_multiplier, penalty = tier_rules[tier]
        arrival_time = (timestamp - first_timestamp).total_seconds()
        deadline = arrival_time + max(0.25, service_time * deadline_multiplier)
        requests.append(
            Request(
                request_id=request_id,
                tier=tier,
                arrival_time=arrival_time,
                service_time=service_time,
                cpu_demand=_cpu_demand(row, service_time),
                deadline=deadline,
                sla_penalty=penalty,
                endpoint=row.get("predict_type", "").strip(),
            )
        )
    return requests
