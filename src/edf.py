"""Non-preemptive Earliest Deadline First scheduling."""

from typing import Iterable

from .models import Request, ScheduledRequest


def schedule_edf(requests: Iterable[Request]) -> list[ScheduledRequest]:
    """Schedule the available request with the earliest SLA deadline first.

    The server is non-preemptive: once a request starts, it runs to
    completion. Requests that arrive while it is busy become candidates for
    the next scheduling decision.
    """
    ordered = sorted(requests, key=lambda request: (request.arrival_time, request.request_id))
    pending: list[Request] = []
    scheduled: list[ScheduledRequest] = []
    current_time = 0.0
    next_index = 0

    while next_index < len(ordered) or pending:
        if not pending and next_index < len(ordered):
            current_time = max(current_time, ordered[next_index].arrival_time)

        while next_index < len(ordered) and ordered[next_index].arrival_time <= current_time:
            pending.append(ordered[next_index])
            next_index += 1

        if not pending:
            continue

        request = min(
            pending,
            key=lambda item: (item.deadline, item.arrival_time, item.request_id),
        )
        pending.remove(request)
        start_time = current_time
        completion_time = start_time + request.service_time
        scheduled.append(ScheduledRequest(request, start_time, completion_time))
        current_time = completion_time

    return scheduled
