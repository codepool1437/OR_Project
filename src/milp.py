"""Offline MILP benchmark for minimum weighted SLA tardiness."""

from typing import Iterable

from .models import Request, ScheduledRequest


def solve_milp(requests: Iterable[Request]) -> list[ScheduledRequest]:
    """Return the optimal non-preemptive single-server order for a small batch.

    The model minimizes weighted tardiness, where each request's SLA penalty
    weights lateness after its deadline. Arrival times are release dates. All
    requests are admitted and completed, making this an ideal scheduling
    benchmark for comparing the live heuristics. This exact model is intended
    for small offline batches, not real-time execution.
    """
    try:
        import pulp
    except ImportError as error:
        raise RuntimeError(
            "PuLP is required for the MILP benchmark; install requirements.txt"
        ) from error

    request_list = sorted(
        requests, key=lambda request: (request.arrival_time, request.request_id)
    )
    if not request_list:
        return []
    if any(request.service_time <= 0 for request in request_list):
        raise ValueError("all service times must be positive")

    count = len(request_list)
    horizon = max(request.arrival_time for request in request_list) + sum(
        request.service_time for request in request_list
    )
    big_m = horizon + max(request.service_time for request in request_list)
    problem = pulp.LpProblem("sla_tardiness", pulp.LpMinimize)
    start = {
        index: pulp.LpVariable(f"start_{index}", lowBound=0)
        for index in range(count)
    }
    tardiness = {
        index: pulp.LpVariable(f"tardiness_{index}", lowBound=0)
        for index in range(count)
    }
    before = {
        (left, right): pulp.LpVariable(
            f"before_{left}_{right}", cat=pulp.LpBinary
        )
        for left in range(count)
        for right in range(left + 1, count)
    }

    for index, request in enumerate(request_list):
        problem += start[index] >= request.arrival_time
        completion = start[index] + request.service_time
        problem += tardiness[index] >= completion - request.deadline

    for left in range(count):
        for right in range(left + 1, count):
            order = before[(left, right)]
            left_request = request_list[left]
            right_request = request_list[right]
            problem += (
                start[left] + left_request.service_time
                <= start[right] + big_m * (1 - order)
            )
            problem += (
                start[right] + right_request.service_time
                <= start[left] + big_m * order
            )

    # The tiny secondary term removes arbitrary idle time among schedules
    # with equal SLA tardiness, while leaving the SLA objective dominant.
    problem += pulp.lpSum(
        request_list[index].sla_penalty * tardiness[index]
        + 1e-6 * (start[index] + request_list[index].service_time)
        for index in range(count)
    )
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"MILP solver did not find an optimal solution: {pulp.LpStatus[status]}")

    scheduled = [
        ScheduledRequest(
            request=request_list[index],
            start_time=pulp.value(start[index]),
            completion_time=pulp.value(start[index]) + request.service_time,
        )
        for index, request in enumerate(request_list)
    ]
    return sorted(scheduled, key=lambda item: (item.start_time, item.request.request_id))
