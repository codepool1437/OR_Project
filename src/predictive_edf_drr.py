"""Predictive Hybrid EDF-within-tier and DRR-across-tier scheduling."""

from collections import defaultdict
import heapq
from typing import Iterable, Mapping

from .drr import DEFAULT_QUANTA
from .models import Request, ScheduledRequest


def predict_traffic_and_adjust_quanta(
    pending_requests: list[Request],
    base_quanta: dict[str, float],
) -> dict[str, float]:
    """Adjust tier quanta using the service demand in the current backlog.

    This is a deterministic proxy for a forecasting model. Each tier retains
    at least half of its configured quantum, while tiers with more pending
    service demand receive a larger share of the next scheduling round.
    """
    if not pending_requests:
        return dict(base_quanta)

    demand = defaultdict(float)
    for request in pending_requests:
        demand[request.tier] += request.service_time

    total_demand = sum(demand.values())
    if total_demand == 0:
        return dict(base_quanta)

    return {
        tier: max(
            base_quantum * 0.5,
            base_quantum * 0.5
            + (base_quantum * demand[tier] / total_demand * 1.5),
        )
        for tier, base_quantum in base_quanta.items()
    }


def schedule_predictive_edf_drr(
    requests: Iterable[Request],
    quanta: Mapping[str, float] | None = None,
) -> list[ScheduledRequest]:
    """Schedule with predictive DRR between tiers and EDF within each tier.

    The current backlog determines the quantum added for the selected tier at
    the start of each round. Requests are not preempted once service starts.
    """
    ordered = sorted(
        requests, key=lambda request: (request.arrival_time, request.request_id)
    )
    if not ordered:
        return []

    base_quanta = dict(DEFAULT_QUANTA if quanta is None else quanta)
    if not base_quanta or any(value <= 0 for value in base_quanta.values()):
        raise ValueError("quanta must contain positive values")

    observed_tiers = {request.tier for request in ordered}
    missing_tiers = observed_tiers - base_quanta.keys()
    if missing_tiers:
        raise ValueError(f"missing quantum for tier(s): {sorted(missing_tiers)}")

    tier_order = list(base_quanta)
    queues: dict[str, list[tuple[float, float, int, Request]]] = {
        tier: [] for tier in tier_order
    }
    deficits: dict[str, float] = defaultdict(float)
    scheduled: list[ScheduledRequest] = []
    current_time = 0.0
    next_index = 0
    cursor = 0

    def enqueue_available() -> None:
        nonlocal next_index
        while (
            next_index < len(ordered)
            and ordered[next_index].arrival_time <= current_time
        ):
            request = ordered[next_index]
            heapq.heappush(
                queues[request.tier],
                (request.deadline, request.arrival_time, request.request_id, request),
            )
            next_index += 1

    while next_index < len(ordered) or any(queues.values()):
        if not any(queues.values()) and next_index < len(ordered):
            current_time = max(current_time, ordered[next_index].arrival_time)
        enqueue_available()

        pending_requests = [
            request
            for queue in queues.values()
            for _, _, _, request in queue
        ]
        dynamic_quanta = predict_traffic_and_adjust_quanta(
            pending_requests, base_quanta
        )

        selected_index = next(
            (cursor + offset) % len(tier_order)
            for offset in range(len(tier_order))
            if queues[tier_order[(cursor + offset) % len(tier_order)]]
        )
        tier = tier_order[selected_index]
        deficits[tier] += dynamic_quanta[tier]
        queue = queues[tier]

        while queue and queue[0][3].service_time <= deficits[tier]:
            _, _, _, request = heapq.heappop(queue)
            deficits[tier] -= request.service_time
            start_time = max(current_time, request.arrival_time)
            completion_time = start_time + request.service_time
            scheduled.append(ScheduledRequest(request, start_time, completion_time))
            current_time = completion_time
            enqueue_available()

        cursor = (selected_index + 1) % len(tier_order)

    return scheduled
