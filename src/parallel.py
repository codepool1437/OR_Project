"""Parallel-worker dispatch for the live scheduling policies."""

from collections import defaultdict, deque
import heapq
from typing import Iterable, Mapping

from .drr import DEFAULT_QUANTA
from .models import Request, ScheduledRequest
from .predictive_edf_drr import predict_traffic_and_adjust_quanta
from .traffic_model import TrafficSpikeModel


def schedule_parallel(
    requests: Iterable[Request],
    policy: str,
    workers: int,
    quanta: Mapping[str, float] | None = None,
    model: TrafficSpikeModel | None = None,
) -> list[ScheduledRequest]:
    """Schedule requests on a pool of identical non-preemptive workers.

    ``policy`` is one of ``FIFO``, ``EDF``, ``DRR``, ``EDF+DRR``, or
    ``Predictive EDF+DRR``. A worker dispatches the next available request
    whenever it becomes free. DRR chooses the tier and hybrid policies choose
    the request deadline within the selected tier.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    supported_policies = {"FIFO", "EDF", "DRR", "EDF+DRR", "Predictive EDF+DRR"}
    if policy not in supported_policies:
        raise ValueError(f"unsupported policy: {policy}")

    ordered = sorted(requests, key=lambda item: (item.arrival_time, item.request_id))
    if not ordered:
        return []
    tier_order = list((DEFAULT_QUANTA if quanta is None else quanta).keys())
    configured_quanta = dict(DEFAULT_QUANTA if quanta is None else quanta)
    if policy in {"DRR", "EDF+DRR", "Predictive EDF+DRR"}:
        observed = {request.tier for request in ordered}
        missing = observed - configured_quanta.keys()
        if missing:
            raise ValueError(f"missing quantum for tier(s): {sorted(missing)}")

    worker_heap = [(0.0, worker_id) for worker_id in range(workers)]
    heapq.heapify(worker_heap)
    ready: list[Request] = []
    tier_queues: dict[str, deque[Request]] = {
        tier: deque() for tier in tier_order
    }
    tier_heaps: dict[str, list[tuple[float, float, int, Request]]] = {
        tier: [] for tier in tier_order
    }
    deficits: dict[str, float] = defaultdict(float)
    cursor = 0
    next_index = 0
    scheduled: list[ScheduledRequest] = []
    history: dict[str, list[float]] = defaultdict(list)
    last_bucket = -1

    def add_arrivals(until: float) -> None:
        nonlocal last_bucket, next_index
        while next_index < len(ordered) and ordered[next_index].arrival_time <= until:
            request = ordered[next_index]
            if policy == "DRR":
                tier_queues[request.tier].append(request)
            elif policy in {"EDF+DRR", "Predictive EDF+DRR"}:
                heapq.heappush(
                    tier_heaps[request.tier],
                    (request.deadline, request.arrival_time, request.request_id, request),
                )
            else:
                ready.append(request)
            if policy == "Predictive EDF+DRR" and model is not None:
                bucket = int(request.arrival_time // model.window_size)
                while last_bucket < bucket:
                    last_bucket += 1
                    for tier in configured_quanta:
                        history[tier].append(0.0)
                history[request.tier][-1] += 1.0
            next_index += 1

    def choose_request() -> Request | None:
        nonlocal cursor
        if policy == "FIFO":
            ready.sort(key=lambda item: (item.arrival_time, item.request_id))
            return ready.pop(0) if ready else None
        if policy == "EDF":
            if not ready:
                return None
            index = min(
                range(len(ready)),
                key=lambda position: (
                    ready[position].deadline,
                    ready[position].arrival_time,
                    ready[position].request_id,
                ),
            )
            return ready.pop(index)

        queues_are_ready = tier_queues if policy == "DRR" else tier_heaps
        if not any(queues_are_ready.values()):
            return None
        for offset in range(len(tier_order)):
            index = (cursor + offset) % len(tier_order)
            tier = tier_order[index]
            if queues_are_ready[tier]:
                cursor = (index + 1) % len(tier_order)
                if policy == "Predictive EDF+DRR":
                    pending = [
                        request
                        for queue in tier_heaps.values()
                        for _, _, _, request in queue
                    ]
                    dynamic_quanta = predict_traffic_and_adjust_quanta(
                        pending,
                        configured_quanta,
                        model=model,
                        history=history,
                        time_index=max(last_bucket, 0),
                    )
                    deficits[tier] += dynamic_quanta[tier]
                else:
                    deficits[tier] += configured_quanta[tier]
                if policy == "DRR":
                    request = tier_queues[tier].popleft()
                else:
                    _, _, _, request = heapq.heappop(tier_heaps[tier])
                return request
        return None

    while next_index < len(ordered) or ready or any(tier_queues.values()) or any(tier_heaps.values()):
        available_time, worker_id = heapq.heappop(worker_heap)
        add_arrivals(available_time)
        request = choose_request()
        if request is None and next_index < len(ordered):
            available_time = max(available_time, ordered[next_index].arrival_time)
            add_arrivals(available_time)
            request = choose_request()
        if request is None:
            heapq.heappush(worker_heap, (available_time, worker_id))
            continue

        start_time = max(available_time, request.arrival_time)
        completion_time = start_time + request.service_time
        scheduled.append(ScheduledRequest(request, start_time, completion_time))
        heapq.heappush(worker_heap, (completion_time, worker_id))

    return sorted(scheduled, key=lambda item: (item.start_time, item.request.request_id))
