"""Hybrid EDF-within-tier and DRR-across-tier scheduling."""

from collections import defaultdict
import heapq
from typing import Iterable, Mapping

from .drr import DEFAULT_QUANTA
from .models import Request, ScheduledRequest


def schedule_edf_drr(
    requests: Iterable[Request],
    quanta: Mapping[str, float] | None = None,
) -> list[ScheduledRequest]:
    """Schedule with DRR between tiers and EDF within each tier.

    DRR chooses which tier receives a service opportunity. Within that tier,
    the request with the earliest deadline is selected. This is intentionally
    different from global EDF: a single tier cannot monopolize the server,
    even when it contains the globally earliest deadline.
    """
    ordered = sorted(requests, key=lambda request: (request.arrival_time, request.request_id))
    if not ordered:
        return []

    configured_quanta = dict(DEFAULT_QUANTA if quanta is None else quanta)
    if not configured_quanta or any(value <= 0 for value in configured_quanta.values()):
        raise ValueError("quanta must contain positive values")
    observed_tiers = {request.tier for request in ordered}
    missing_tiers = observed_tiers - configured_quanta.keys()
    if missing_tiers:
        raise ValueError(f"missing quantum for tier(s): {sorted(missing_tiers)}")

    tier_order = list(configured_quanta)
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
        while next_index < len(ordered) and ordered[next_index].arrival_time <= current_time:
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

        selected_index = next(
            (cursor + offset) % len(tier_order)
            for offset in range(len(tier_order))
            if queues[tier_order[(cursor + offset) % len(tier_order)]]
        )
        tier = tier_order[selected_index]
        deficits[tier] += configured_quanta[tier]
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
