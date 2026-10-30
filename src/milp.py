"""Offline MILP benchmark for minimum penalty-weighted SLA tardiness.

The model is the classical disjunctive formulation of non-preemptive
scheduling with release dates on ``m`` identical machines
(``Rm | r_j | sum w_j T_j``). It is NP-hard, so it is only practical for small
batches; its role is to provide a reference value against which the online
policies are measured. The solver can be warm-started from a heuristic
schedule and given a time limit, in which case the result states whether
optimality was proven.
"""

from dataclasses import dataclass
import time
from typing import Iterable

from .models import Request, ScheduledRequest


@dataclass(frozen=True)
class MilpResult:
    """Outcome of one MILP solve."""

    schedule: list[ScheduledRequest]
    objective: float  # true weighted tardiness of the returned schedule
    proven_optimal: bool
    status: str
    solve_seconds: float
    machines: int
    lower_bound: float | None = None  # proven dual bound (HiGHS only)
    solver: str = ""


def weighted_tardiness_of(schedule: Iterable[ScheduledRequest]) -> float:
    return sum(
        item.request.sla_penalty * max(0.0, item.completion_time - item.request.deadline)
        for item in schedule
    )


def _machine_assignment(schedule: list[ScheduledRequest], machines: int) -> dict[int, int]:
    """Replay a heuristic schedule onto machine indices (for warm starts)."""
    free_at = [0.0] * machines
    assignment: dict[int, int] = {}
    for item in sorted(schedule, key=lambda s: (s.start_time, s.request.request_id)):
        candidates = [k for k in range(machines) if free_at[k] <= item.start_time + 1e-9]
        if not candidates:
            return {}
        machine = max(candidates, key=lambda k: free_at[k])
        assignment[item.request.request_id] = machine
        free_at[machine] = item.completion_time
    return assignment


def solve_milp_detailed(
    requests: Iterable[Request],
    machines: int = 1,
    time_limit: float | None = 60.0,
    warm_start: list[ScheduledRequest] | None = None,
    solver_name: str = "auto",
) -> MilpResult:
    """Solve the benchmark MILP and report whether optimality was proven."""
    try:
        import pulp
    except ImportError as error:
        raise RuntimeError(
            "PuLP is required for the MILP benchmark; install requirements.txt"
        ) from error
    if machines < 1:
        raise ValueError("machines must be at least 1")

    request_list = sorted(
        requests, key=lambda request: (request.arrival_time, request.request_id)
    )
    if not request_list:
        return MilpResult([], 0.0, True, "Optimal", 0.0, machines)
    if any(request.service_time <= 0 for request in request_list):
        raise ValueError("all service times must be positive")

    count = len(request_list)
    horizon = max(r.arrival_time for r in request_list) + sum(
        r.service_time for r in request_list
    ) / machines + max(r.service_time for r in request_list)
    big_m = horizon + max(r.service_time for r in request_list)
    problem = pulp.LpProblem("sla_tardiness", pulp.LpMinimize)
    start = {i: pulp.LpVariable(f"start_{i}", lowBound=0) for i in range(count)}
    tardiness = {i: pulp.LpVariable(f"tardiness_{i}", lowBound=0) for i in range(count)}
    before = {
        (i, j): pulp.LpVariable(f"before_{i}_{j}", cat=pulp.LpBinary)
        for i in range(count)
        for j in range(i + 1, count)
    }
    assign = {}
    if machines > 1:
        assign = {
            (i, k): pulp.LpVariable(f"x_{i}_{k}", cat=pulp.LpBinary)
            for i in range(count)
            for k in range(machines)
        }
        for i in range(count):
            problem += pulp.lpSum(assign[(i, k)] for k in range(machines)) == 1
            # Symmetry breaking: request i may only use machines 0..i.
            for k in range(i + 1, machines):
                problem += assign[(i, k)] == 0

    for i, request in enumerate(request_list):
        problem += start[i] >= request.arrival_time
        problem += tardiness[i] >= start[i] + request.service_time - request.deadline

    for i in range(count):
        for j in range(i + 1, count):
            pi, pj = request_list[i].service_time, request_list[j].service_time
            if machines == 1:
                problem += start[i] + pi <= start[j] + big_m * (1 - before[(i, j)])
                problem += start[j] + pj <= start[i] + big_m * before[(i, j)]
            else:
                for k in range(machines):
                    same = 2 - assign[(i, k)] - assign[(j, k)]
                    problem += start[i] + pi <= start[j] + big_m * (1 - before[(i, j)]) + big_m * same
                    problem += start[j] + pj <= start[i] + big_m * before[(i, j)] + big_m * same

    # Pure weighted-tardiness objective (a secondary idle-time term would only
    # slow the optimality proof). Idle time in the solution is removed below by
    # left-shifting, which cannot increase tardiness.
    problem += pulp.lpSum(
        request_list[i].sla_penalty * tardiness[i] for i in range(count)
    )

    use_warm = False
    if warm_start:
        assignment = _machine_assignment(warm_start, machines) if machines > 1 else {}
        if machines == 1 or assignment:
            by_id = {item.request.request_id: item for item in warm_start}
            for i, request in enumerate(request_list):
                item = by_id[request.request_id]
                start[i].setInitialValue(item.start_time)
                tardiness[i].setInitialValue(
                    max(0.0, item.completion_time - request.deadline)
                )
                if machines > 1:
                    for k in range(machines):
                        assign[(i, k)].setInitialValue(
                            1 if assignment[request.request_id] == k else 0
                        )
            for (i, j), variable in before.items():
                a = by_id[request_list[i].request_id]
                b = by_id[request_list[j].request_id]
                variable.setInitialValue(1 if a.start_time <= b.start_time else 0)
            use_warm = True

    available = pulp.listSolvers(onlyAvailable=True)
    use_highs = solver_name == "highs" or (solver_name == "auto" and "HiGHS" in available)
    if use_highs:
        solver = pulp.HiGHS(
            msg=False, timeLimit=time_limit, warmStart=use_warm,
            gapRel=1e-6, gapAbs=1e-6,
        )
    else:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, warmStart=use_warm)
    began = time.perf_counter()
    status = problem.solve(solver)
    elapsed = time.perf_counter() - began
    status_name = pulp.LpStatus[status]
    lower_bound = None
    proven = False
    if use_highs:
        info = problem.solverModel.getInfo()
        lower_bound = max(0.0, float(info.mip_dual_bound))
        model_status = problem.solverModel.modelStatusToString(
            problem.solverModel.getModelStatus()
        )
        proven = model_status == "Optimal"
    else:
        proven = status_name == "Optimal" and getattr(problem, "sol_status", 1) == 1
    if any(pulp.value(start[i]) is None for i in range(count)):
        raise RuntimeError(f"MILP solver returned no solution: {status_name}")

    machine_of = {
        i: (
            max(range(machines), key=lambda k: pulp.value(assign[(i, k)]) or 0.0)
            if machines > 1
            else 0
        )
        for i in range(count)
    }
    raw_start = {i: pulp.value(start[i]) for i in range(count)}
    free_at = [0.0] * machines
    scheduled = []
    for i in sorted(range(count), key=lambda i: (raw_start[i], i)):
        request = request_list[i]
        begin = max(request.arrival_time, free_at[machine_of[i]])
        free_at[machine_of[i]] = begin + request.service_time
        scheduled.append(
            ScheduledRequest(request, begin, begin + request.service_time)
        )
    scheduled.sort(key=lambda item: (item.start_time, item.request.request_id))
    objective = weighted_tardiness_of(scheduled)
    if warm_start and weighted_tardiness_of(warm_start) < objective - 1e-9:
        # Never report an incumbent worse than the supplied heuristic schedule.
        scheduled = sorted(warm_start, key=lambda s: (s.start_time, s.request.request_id))
        objective = weighted_tardiness_of(scheduled)
    if proven:
        lower_bound = objective
    return MilpResult(
        scheduled,
        objective,
        proven,
        "Optimal" if proven else "TimeLimit-Feasible",
        elapsed,
        machines,
        lower_bound,
        "HiGHS" if use_highs else "CBC",
    )


def solve_milp(requests: Iterable[Request], machines: int = 1) -> list[ScheduledRequest]:
    """Return an optimal non-preemptive schedule for a small offline batch."""
    result = solve_milp_detailed(requests, machines=machines, time_limit=None)
    if not result.proven_optimal:
        raise RuntimeError(f"MILP solver did not find an optimal solution: {result.status}")
    return result.schedule
