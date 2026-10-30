"""Unified multi-worker dispatch engine for every scheduling policy.

With ``workers=1`` this engine reproduces the dedicated single-server
schedulers (verified by the test-suite), so all policies can be compared on
one simulator for any worker count.
"""

from collections import defaultdict, deque
import heapq
from math import ceil, exp
from typing import Iterable, Mapping

from .drr import DEFAULT_QUANTA
from .dynamic_edf_drr import urgency_adjusted_quanta
from .models import Request, ScheduledRequest
from .predictive_edf_drr import predict_traffic_and_adjust_quanta
from .traffic_model import TrafficSpikeModel

BASE_POLICIES: tuple[str, ...] = ("FIFO", "EDF", "DRR", "EDF+DRR")
PREDICTIVE_POLICY = "Predictive EDF+DRR"
DYNAMIC_POLICY = "Dynamic EDF+DRR"
ATC_POLICY = "ATC"
ABLATION_POLICIES: tuple[str, ...] = (
    "Dynamic EDF+DRR (aggressive quanta only)",
    "Dynamic EDF+DRR (deferral only)",
)
# Main comparison set used in the paper.
ALL_POLICIES: tuple[str, ...] = BASE_POLICIES + (
    PREDICTIVE_POLICY,
    DYNAMIC_POLICY,
    ATC_POLICY,
)
SUPPORTED_POLICIES = set(ALL_POLICIES) | set(ABLATION_POLICIES)

_TIER_HEAP_POLICIES = {
    "EDF+DRR",
    PREDICTIVE_POLICY,
    DYNAMIC_POLICY,
    *ABLATION_POLICIES,
}
_DRR_POLICIES = _TIER_HEAP_POLICIES | {"DRR"}
ATC_LOOKAHEAD = 2.0
# A hopeless request is deferred only until it is this many seconds late, after
# which it is served again (aging bounds the extra tardiness deferral causes).
DEFER_LIMIT = float("inf")
# Dynamic EDF+DRR keeps each tier within [0.9, 0.9 + 3 * 0.1] of its base quantum;
# more aggressive adaptation was found to hurt (see the sensitivity experiment).
QUANTA_FLOOR = 0.9
AGGRESSIVE_QUANTA_FLOOR = 0.5


URGENCY_WINDOW = 64


class DeferralQueue:
    """Per-tier EDF queue that serves hopeless (already-late) work last.

    A request is *savable* while ``now + service_time <= deadline``. Since time
    only moves forward, a request becomes hopeless exactly once, so three lazy
    heaps give amortized O(log n) operations instead of an O(n) rescan.
    """

    def __init__(self) -> None:
        self._main: list[tuple[float, float, int, Request]] = []
        self._latest: list[tuple[float, int]] = []
        self._late: list[tuple[float, float, int, Request]] = []
        self._state: dict[int, str] = {}

    def __bool__(self) -> bool:
        return bool(self._state)

    def push(self, entry: tuple[float, float, int, Request]) -> None:
        request = entry[3]
        heapq.heappush(self._main, entry)
        heapq.heappush(self._latest, (request.deadline - request.service_time, request.request_id))
        self._state[request.request_id] = "main"

    def _refresh(self, now: float, defer_limit: float) -> None:
        while self._latest and self._latest[0][0] < now:
            _, request_id = heapq.heappop(self._latest)
            if self._state.get(request_id) == "main":
                self._state[request_id] = "late"
        # Move newly hopeless requests out of the main heap lazily on access.
        while self._main and self._state.get(self._main[0][2]) != "main":
            entry = heapq.heappop(self._main)
            if self._state.get(entry[2]) == "late":
                heapq.heappush(self._late, entry)

    def peek(self, now: float, defer_limit: float):
        self._refresh(now, defer_limit)
        # Aging: a hopeless request more than defer_limit late is served again.
        while self._late and self._state.get(self._late[0][2]) != "late":
            heapq.heappop(self._late)
        if self._late and now + self._late[0][3].service_time - self._late[0][0] >= defer_limit:
            return self._late[0]
        if self._main:
            return self._main[0]
        return self._late[0] if self._late else None

    def pop(self, entry) -> None:
        request_id = entry[2]
        state = self._state.pop(request_id)
        heap = self._main if state == "main" else self._late
        if heap and heap[0] is entry:
            heapq.heappop(heap)

    def window(self, limit: int) -> list[Request]:
        return [e[3] for e in self._main[:limit] if self._state.get(e[2]) == "main"] + [
            e[3] for e in self._late[:limit] if self._state.get(e[2]) == "late"
        ]


def schedule_parallel(
    requests: Iterable[Request],
    policy: str,
    workers: int,
    quanta: Mapping[str, float] | None = None,
    model: TrafficSpikeModel | None = None,
) -> list[ScheduledRequest]:
    """Schedule requests on a pool of identical non-preemptive workers.

    A worker dispatches the next request whenever it becomes free. DRR-family
    policies keep a visit state across dispatches: a visited tier receives its
    quantum once, then keeps serving while its head request fits in the
    deficit counter, exactly as in the single-server implementations.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy}")

    ordered = sorted(requests, key=lambda item: (item.arrival_time, item.request_id))
    if not ordered:
        return []
    configured_quanta = dict(DEFAULT_QUANTA if quanta is None else quanta)
    tier_order = list(configured_quanta)
    if policy in _DRR_POLICIES:
        if any(value <= 0 for value in configured_quanta.values()):
            raise ValueError("quanta must contain positive values")
        missing = {request.tier for request in ordered} - configured_quanta.keys()
        if missing:
            raise ValueError(f"missing quantum for tier(s): {sorted(missing)}")

    quanta_adaptive = policy in {PREDICTIVE_POLICY, DYNAMIC_POLICY, ABLATION_POLICIES[0]}
    defers_late = policy in {DYNAMIC_POLICY, ABLATION_POLICIES[1]}

    worker_heap = [(0.0, worker_id) for worker_id in range(workers)]
    heapq.heapify(worker_heap)
    fifo_queue: deque[Request] = deque()
    edf_heap: list[tuple[float, float, int, Request]] = []
    atc_ready: list[Request] = []
    tier_queues: dict[str, deque[Request]] = {tier: deque() for tier in tier_order}
    tier_heaps: dict = {
        tier: (DeferralQueue() if defers_late else []) for tier in tier_order
    }
    deficits: dict[str, float] = defaultdict(float)
    visit_quanta: dict[str, float] = dict(configured_quanta)
    cursor = 0
    in_visit = False
    next_index = 0
    pending_count = 0
    service_sum = 0.0
    arrived = 0
    scheduled: list[ScheduledRequest] = []
    history: dict[str, list[float]] = defaultdict(list)
    last_bucket = -1

    def add_arrivals(until: float) -> None:
        nonlocal last_bucket, next_index, pending_count, service_sum, arrived
        while next_index < len(ordered) and ordered[next_index].arrival_time <= until:
            request = ordered[next_index]
            key = (request.deadline, request.arrival_time, request.request_id, request)
            if policy == "FIFO":
                fifo_queue.append(request)
            elif policy == "EDF":
                heapq.heappush(edf_heap, key)
            elif policy == ATC_POLICY:
                atc_ready.append(request)
            elif policy == "DRR":
                tier_queues[request.tier].append(request)
            elif defers_late:
                tier_heaps[request.tier].push(key)
            else:
                heapq.heappush(tier_heaps[request.tier], key)
            if policy == PREDICTIVE_POLICY and model is not None:
                bucket = int(request.arrival_time // model.window_size)
                while last_bucket < bucket:
                    last_bucket += 1
                    for tier in configured_quanta:
                        history[tier].append(0.0)
                history[request.tier][-1] += 1.0
            pending_count += 1
            service_sum += request.service_time
            arrived += 1
            next_index += 1

    def tier_candidate(tier: str, now: float) -> tuple | Request | None:
        """Return the tier's next request without removing it."""
        if policy == "DRR":
            queue = tier_queues[tier]
            return queue[0] if queue else None
        heap = tier_heaps[tier]
        if not heap:
            return None
        if defers_late:
            return heap.peek(now, DEFER_LIMIT)
        return heap[0]

    def remove_candidate(tier: str, candidate) -> Request:
        if policy == "DRR":
            return tier_queues[tier].popleft()
        if defers_late:
            tier_heaps[tier].pop(candidate)
            return candidate[3]
        return heapq.heappop(tier_heaps[tier])[3]

    def pending_window() -> list[Request]:
        pending: list[Request] = []
        for heap in tier_heaps.values():
            if defers_late:
                pending.extend(heap.window(URGENCY_WINDOW))
            else:
                pending.extend(e[3] for e in heap[:URGENCY_WINDOW])
        return pending

    def begin_visit(now: float) -> None:
        nonlocal visit_quanta
        if policy == PREDICTIVE_POLICY:
            pending = (
                pending_window()
                if model is not None
                else [e[3] for heap in tier_heaps.values() for e in heap]
            )
            visit_quanta = predict_traffic_and_adjust_quanta(
                pending, configured_quanta, model=model, history=history,
                time_index=max(last_bucket, 0),
            )
        elif quanta_adaptive:
            pending = pending_window()
            floor = (
                AGGRESSIVE_QUANTA_FLOOR if policy == ABLATION_POLICIES[0] else QUANTA_FLOOR
            )
            visit_quanta = urgency_adjusted_quanta(
                pending, configured_quanta, now, floor=floor
            )
        else:
            visit_quanta = configured_quanta

    def choose_drr(now: float) -> Request | None:
        nonlocal cursor, in_visit
        failed_visits = 0
        while True:
            tier = tier_order[cursor]
            candidate = tier_candidate(tier, now)
            if candidate is None:
                in_visit = False
                cursor = (cursor + 1) % len(tier_order)
                if pending_count == 0:
                    return None
                continue
            request = candidate if policy == "DRR" else candidate[3]
            if not in_visit:
                begin_visit(now)
                deficits[tier] += visit_quanta[tier]
                in_visit = True
            if request.service_time <= deficits[tier]:
                deficits[tier] -= request.service_time
                return remove_candidate(tier, candidate)
            in_visit = False
            cursor = (cursor + 1) % len(tier_order)
            failed_visits += 1
            if failed_visits >= len(tier_order):
                fast_forward(now)
                failed_visits = 0

    def fast_forward(now: float) -> None:
        """Skip whole DRR cycles in which no tier can dispatch.

        Queue state, time, and therefore quanta are unchanged between such
        cycles, so adding ``k`` quanta at once is exactly equivalent to ``k``
        empty visits. This matters when service times far exceed the quanta.
        """
        rounds = None
        for tier in tier_order:
            candidate = tier_candidate(tier, now)
            if candidate is None:
                continue
            request = candidate if policy == "DRR" else candidate[3]
            quantum = visit_quanta[tier]
            needed = max(0, ceil((request.service_time - deficits[tier]) / quantum) - 1)
            rounds = needed if rounds is None else min(rounds, needed)
        if rounds:
            for tier in tier_order:
                if tier_candidate(tier, now) is not None:
                    deficits[tier] += rounds * visit_quanta[tier]

    def choose_request(now: float) -> Request | None:
        nonlocal pending_count, cursor, in_visit
        if pending_count == 0:
            if in_visit:
                # The system drained: the current DRR visit ends here.
                in_visit = False
                cursor = (cursor + 1) % len(tier_order)
            return None
        if policy == "FIFO":
            request = fifo_queue.popleft()
        elif policy == "EDF":
            request = heapq.heappop(edf_heap)[3]
        elif policy == ATC_POLICY:
            mean_service = service_sum / max(1, arrived)

            def index(item: Request) -> tuple[float, int]:
                slack = max(item.deadline - item.service_time - now, 0.0)
                value = (item.sla_penalty / item.service_time) * exp(
                    -slack / (ATC_LOOKAHEAD * mean_service)
                )
                return (-value, item.request_id)

            position = min(range(len(atc_ready)), key=lambda i: index(atc_ready[i]))
            request = atc_ready.pop(position)
        else:
            request = choose_drr(now)
            if request is None:
                return None
        pending_count -= 1
        return request

    while next_index < len(ordered) or pending_count > 0:
        available_time, worker_id = heapq.heappop(worker_heap)
        add_arrivals(available_time)
        request = choose_request(available_time)
        if request is None and next_index < len(ordered):
            available_time = max(available_time, ordered[next_index].arrival_time)
            add_arrivals(available_time)
            request = choose_request(available_time)
        if request is None:
            heapq.heappush(worker_heap, (available_time, worker_id))
            continue

        start_time = max(available_time, request.arrival_time)
        completion_time = start_time + request.service_time
        scheduled.append(ScheduledRequest(request, start_time, completion_time))
        heapq.heappush(worker_heap, (completion_time, worker_id))

    return sorted(scheduled, key=lambda item: (item.start_time, item.request.request_id))
