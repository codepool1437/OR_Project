"""FIFO scheduling and baseline metric calculations."""

from collections import defaultdict
from math import fsum
from statistics import quantiles
from typing import Iterable

from .models import Metrics, Request, ScheduledRequest


def schedule_fifo(requests: Iterable[Request]) -> list[ScheduledRequest]:
    """Schedule requests in arrival order on one FIFO server."""
    ordered = sorted(requests, key=lambda request: (request.arrival_time, request.request_id))
    current_time = 0.0
    scheduled: list[ScheduledRequest] = []
    for request in ordered:
        start_time = max(current_time, request.arrival_time)
        completion_time = start_time + request.service_time
        scheduled.append(ScheduledRequest(request, start_time, completion_time))
        current_time = completion_time
    return scheduled


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[94]


def _jain_fairness(values: list[float]) -> float:
    if not values:
        return 1.0
    denominator = len(values) * fsum(value * value for value in values)
    if denominator == 0:
        return 1.0
    total = fsum(values)
    return (total * total) / denominator


def calculate_metrics(
    requests: Iterable[Request],
    scheduled: Iterable[ScheduledRequest],
    cpu_capacity: float = 100.0,
    horizon: float | None = None,
) -> Metrics:
    """Calculate metrics for a single-server run.

    CPU utilization is total CPU-seconds consumed divided by available
    CPU-seconds during the experiment. With a horizon, only requests arriving
    before the cutoff are offered and only requests completed by the cutoff
    count toward completion, throughput, SLA violations, and fairness.
    Fairness is Jain's index over tier completion rates.
    """
    request_list = list(requests)
    scheduled_list = list(scheduled)
    if cpu_capacity <= 0:
        raise ValueError("cpu_capacity must be positive")
    if horizon is not None and horizon <= 0:
        raise ValueError("horizon must be positive")
    if not request_list:
        return Metrics(
            0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 1.0,
            {}, {}, {}, {},
        )

    if horizon is None:
        offered_requests = request_list
        completed_items = scheduled_list
    else:
        offered_requests = [request for request in request_list if request.arrival_time < horizon]
        completed_items = [
            item for item in scheduled_list
            if item.request.arrival_time < horizon and item.completion_time <= horizon
        ]
    if not offered_requests:
        return Metrics(
            0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 1.0,
            {}, {}, {}, {},
        )

    waits = [item.waiting_time for item in completed_items]
    violations = sum(item.sla_violated for item in completed_items)
    first_arrival = min(request.arrival_time for request in offered_requests)
    last_completion = max(
        (item.completion_time for item in completed_items),
        default=first_arrival,
    )
    end_time = horizon if horizon is not None else last_completion
    elapsed = max(last_completion - first_arrival, 1e-12)
    if horizon is not None:
        elapsed = max(end_time - first_arrival, 1e-12)
    cpu_items = (
        scheduled_list
        if horizon is None
        else [item for item in scheduled_list if item.request.arrival_time < horizon]
    )
    cpu_seconds = fsum(
        (
            max(
                0.0,
                min(item.completion_time, end_time) - item.start_time,
            )
            * item.request.cpu_demand
        )
        for item in cpu_items
    )

    offered_by_tier: dict[str, int] = defaultdict(int)
    completed_by_tier: dict[str, int] = defaultdict(int)
    for request in offered_requests:
        offered_by_tier[request.tier] += 1
    for item in completed_items:
        completed_by_tier[item.request.tier] += 1
    completion_rates = [
        completed_by_tier[tier] / offered_by_tier[tier]
        for tier in sorted(offered_by_tier)
    ]
    per_tier_waiting_time = {
        tier: (
            fsum(
                item.waiting_time
                for item in completed_items
                if item.request.tier == tier
            )
            / completed_by_tier[tier]
            if completed_by_tier[tier]
            else 0.0
        )
        for tier in sorted(offered_by_tier)
    }
    per_tier_completion_rate = {
        tier: completed_by_tier[tier] / offered_by_tier[tier]
        for tier in sorted(offered_by_tier)
    }
    per_tier_backlog = {
        tier: offered_by_tier[tier] - completed_by_tier[tier]
        for tier in sorted(offered_by_tier)
    }
    per_tier_sla_violation_rate = {
        tier: (
            sum(
                item.sla_violated
                for item in completed_items
                if item.request.tier == tier
            )
            / completed_by_tier[tier]
            if completed_by_tier[tier]
            else 0.0
        )
        for tier in sorted(offered_by_tier)
    }

    return Metrics(
        request_count=len(offered_requests),
        completed_requests=len(completed_items),
        average_waiting_time=fsum(waits) / len(waits) if waits else 0.0,
        p95_waiting_time=_p95(waits),
        sla_violations=violations,
        sla_violation_rate=violations / len(offered_requests),
        throughput=len(completed_items) / elapsed,
        cpu_utilization=min(cpu_seconds / (elapsed * cpu_capacity), 1.0),
        fairness_index=_jain_fairness(completion_rates),
        per_tier_waiting_time=per_tier_waiting_time,
        per_tier_completion_rate=per_tier_completion_rate,
        per_tier_backlog=per_tier_backlog,
        per_tier_sla_violation_rate=per_tier_sla_violation_rate,
    )
