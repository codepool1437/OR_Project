import unittest
from dataclasses import replace

from src.drr import schedule_drr
from src.dynamic_edf_drr import (
    request_criticality,
    schedule_dynamic_edf_drr,
    urgency_adjusted_quanta,
)
from src.edf import schedule_edf
from src.edf_drr import schedule_edf_drr
from src.fifo import schedule_fifo
from src.milp import solve_milp_detailed
from src.models import Request
from src.parallel import ALL_POLICIES, schedule_parallel
from src.predictive_edf_drr import schedule_predictive_edf_drr
from src.research_metrics import mean_ci, weighted_tardiness
from src.scenarios import SCENARIOS, generate_scenario
from src.traffic import generate_requests


def _key(schedule):
    ordered = sorted(schedule, key=lambda i: (i.start_time, i.request.request_id))
    return [(i.request.request_id, round(i.start_time, 6)) for i in ordered]


def _request(rid, tier, arrival, service, deadline, penalty=1.0):
    return Request(rid, tier, arrival, service, 10.0, deadline, penalty)


def _congest(requests, factor):
    return [
        replace(
            r,
            arrival_time=r.arrival_time * factor,
            deadline=r.deadline - r.arrival_time * (1.0 - factor),
        )
        for r in requests
    ]


class UnifiedEngineTests(unittest.TestCase):
    def test_single_worker_engine_matches_dedicated_schedulers(self) -> None:
        dedicated = {
            "FIFO": schedule_fifo,
            "EDF": schedule_edf,
            "DRR": schedule_drr,
            "EDF+DRR": schedule_edf_drr,
            "Predictive EDF+DRR": schedule_predictive_edf_drr,
        }
        for seed in range(3):
            requests = generate_requests(300, seed, 6.0)
            for name, function in dedicated.items():
                self.assertEqual(
                    _key(function(requests)),
                    _key(schedule_parallel(requests, name, 1)),
                    name,
                )

    def test_every_policy_serves_each_request_once_without_overlap(self) -> None:
        requests = generate_scenario("mixed_burst", 300, 5)
        for workers in (1, 3):
            for policy in ALL_POLICIES:
                schedule = schedule_parallel(requests, policy, workers)
                self.assertEqual(
                    sorted(i.request.request_id for i in schedule),
                    [r.request_id for r in requests],
                )
                self.assertTrue(
                    all(i.start_time >= i.request.arrival_time - 1e-9 for i in schedule)
                )
                events = sorted(
                    [(i.start_time, 1) for i in schedule]
                    + [(i.completion_time, -1) for i in schedule],
                    key=lambda e: (round(e[0], 9), e[1]),
                )
                busy = peak = 0
                for _, delta in events:
                    busy += delta
                    peak = max(peak, busy)
                self.assertLessEqual(peak, workers)

    def test_parallel_drr_honours_quanta_not_plain_round_robin(self) -> None:
        requests = [
            _request(i, "VIP" if i % 2 == 0 else "Free", 0.0, 0.1, 100.0)
            for i in range(40)
        ]
        schedule = schedule_parallel(
            requests, "DRR", 2, quanta={"VIP": 0.4, "Free": 0.1}
        )
        first = [i.request.tier for i in schedule[:10]]
        self.assertGreater(first.count("VIP"), first.count("Free"))


class DynamicPolicyTests(unittest.TestCase):
    def test_criticality_is_zero_for_hopeless_requests(self) -> None:
        late = _request(0, "VIP", 0.0, 1.0, 1.5)
        self.assertEqual(request_criticality(late, now=1.0), 0.0)
        urgent = _request(1, "VIP", 0.0, 1.0, 1.1)
        self.assertGreater(request_criticality(urgent, now=0.0), 0.9)

    def test_quanta_stay_within_bounds_and_floor(self) -> None:
        base = {"VIP": 0.6, "Premium": 0.3, "Free": 0.15}
        pending = [_request(i, "VIP", 0.0, 0.1, 0.2) for i in range(5)]
        quanta = urgency_adjusted_quanta(pending, base, now=0.0, floor=0.5)
        for tier, value in quanta.items():
            self.assertGreaterEqual(value, base[tier] * 0.5 - 1e-12)
            self.assertLessEqual(value, base[tier] * (0.5 + 3 * 0.5) + 1e-12)
        self.assertGreater(quanta["VIP"] / base["VIP"], quanta["Free"] / base["Free"])
        self.assertEqual(urgency_adjusted_quanta([], base, 0.0), base)

    def test_deferral_serves_savable_request_before_hopeless_one(self) -> None:
        # The server is busy until t=1.0. By then request 0 (deadline 0.5) is
        # hopeless; request 1 (deadline 1.4) can still be saved.
        blocker = _request(9, "Free", 0.0, 1.0, 50.0)
        hopeless = _request(0, "VIP", 0.1, 0.2, 0.5)
        savable = _request(1, "VIP", 0.1, 0.2, 1.4)
        batch = [blocker, hopeless, savable]
        static = {i.request.request_id: i for i in schedule_edf_drr(batch)}
        dynamic = {i.request.request_id: i for i in schedule_dynamic_edf_drr(batch)}
        self.assertLess(static[0].start_time, static[1].start_time)
        self.assertLess(dynamic[1].start_time, dynamic[0].start_time)
        self.assertFalse(dynamic[1].sla_violated)

    def test_dynamic_improves_on_time_count_under_overload(self) -> None:
        requests = generate_scenario("overload", 600, 3)
        static = schedule_parallel(requests, "EDF+DRR", 1)
        dynamic = schedule_parallel(requests, "Dynamic EDF+DRR", 1)

        def on_time(schedule):
            return sum(not i.sla_violated for i in schedule)

        self.assertGreater(on_time(dynamic), on_time(static))


class ScenarioAndMetricTests(unittest.TestCase):
    def test_scenarios_are_deterministic_and_valid(self) -> None:
        for name in SCENARIOS:
            first = generate_scenario(name, 120, 11)
            self.assertEqual(first, generate_scenario(name, 120, 11))
            self.assertEqual(len(first), 120)
            arrivals = [r.arrival_time for r in first]
            self.assertEqual(arrivals, sorted(arrivals))
        with self.assertRaises(ValueError):
            generate_scenario("unknown")

    def test_vip_spike_concentrates_vip_inside_bursts(self) -> None:
        requests = generate_scenario("vip_spike", 3000, 2)
        inside = [r for r in requests if r.arrival_time % 60.0 < 15.0]
        outside = [r for r in requests if r.arrival_time % 60.0 >= 15.0]

        def share(items):
            return sum(r.tier == "VIP" for r in items) / len(items)

        self.assertGreater(share(inside), share(outside) + 0.2)

    def test_mean_ci_and_weighted_tardiness(self) -> None:
        mean, low, high = mean_ci([1.0, 2.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertLess(low, 2.0)
        self.assertGreater(high, 2.0)
        requests = [_request(0, "VIP", 0.0, 1.0, 0.5, penalty=10.0)]
        self.assertAlmostEqual(weighted_tardiness(schedule_fifo(requests)), 5.0)


class MilpTests(unittest.TestCase):
    def test_milp_is_a_lower_bound_for_every_policy(self) -> None:
        requests = _congest(generate_scenario("overload", 8, 4), 0.1)
        for machines in (1, 2):
            result = solve_milp_detailed(requests, machines=machines, time_limit=30)
            self.assertTrue(result.proven_optimal)
            for policy in ALL_POLICIES:
                heuristic = weighted_tardiness(
                    schedule_parallel(requests, policy, machines)
                )
                self.assertLessEqual(
                    result.objective, heuristic + 1e-6, (policy, machines)
                )
            self.assertEqual(len(result.schedule), len(requests))

    def test_warm_start_never_worsens_the_incumbent(self) -> None:
        requests = _congest(generate_scenario("tight_deadlines", 9, 2), 0.1)
        warm = schedule_parallel(requests, "EDF", 1)
        result = solve_milp_detailed(requests, 1, 30, warm)
        self.assertLessEqual(result.objective, weighted_tardiness(warm) + 1e-9)


if __name__ == "__main__":
    unittest.main()
