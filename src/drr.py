"""Deficit Round Robin scheduling across request tiers."""

from collections import defaultdict, deque
from typing import Iterable, Mapping

from .models import Request, ScheduledRequest


DEFAULT_QUANTA: dict[str, float] = {
    "VIP": 0.60,
    "Premium": 0.30,
    "Free": 0.15,
}

# Sensitivity profile that targets similar completion rates across tiers under
# overload. It is not the default because it gives Free more service budget
# than VIP; report it separately as a fairness-oriented operating point.
FAIRNESS_QUANTA: dict[str, float] = {
    "VIP": 0.20,
    "Premium": 0.40,
    "Free": 0.60,
}


def schedule_drr(
    requests: Iterable[Request],
    quanta: Mapping[str, float] | None = None,
) -> list[ScheduledRequest]:
    """Schedule requests with Deficit Round Robin.

    Each tier owns a FIFO queue. On each visit, a tier receives its quantum;
    requests at the head of that queue consume deficit equal to their service
    time. Unused deficit carries into later rounds, so a request larger than
    one quantum is not starved. The server remains non-preemptive.
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
    queues: dict[str, deque[Request]] = {
        tier: deque() for tier in tier_order
    }
    deficits: dict[str, float] = defaultdict(float)
    scheduled: list[ScheduledRequest] = []
    current_time = 0.0
    next_index = 0
    cursor = 0

    while next_index < len(ordered) or any(queues.values()):
        if not any(queues.values()) and next_index < len(ordered):
            current_time = max(current_time, ordered[next_index].arrival_time)

        while next_index < len(ordered) and ordered[next_index].arrival_time <= current_time:
            queues[ordered[next_index].tier].append(ordered[next_index])
            next_index += 1

        active_indices = [
            index for index, tier in enumerate(tier_order) if queues[tier]
        ]
        if not active_indices:
            continue

        selected_index = next(
            (cursor + offset) % len(tier_order)
            for offset in range(len(tier_order))
            if queues[tier_order[(cursor + offset) % len(tier_order)]]
        )
        tier = tier_order[selected_index]
        deficits[tier] += configured_quanta[tier]
        queue = queues[tier]

        while queue and queue[0].service_time <= deficits[tier]:
            request = queue.popleft()
            deficits[tier] -= request.service_time
            start_time = max(current_time, request.arrival_time)
            completion_time = start_time + request.service_time
            scheduled.append(ScheduledRequest(request, start_time, completion_time))
            current_time = completion_time
            while next_index < len(ordered) and ordered[next_index].arrival_time <= current_time:
                queues[ordered[next_index].tier].append(ordered[next_index])
                next_index += 1

        cursor = (selected_index + 1) % len(tier_order)

    return scheduled
