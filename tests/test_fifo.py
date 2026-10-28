import unittest
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

from src.drr import FAIRNESS_QUANTA, schedule_drr
from src.edf import schedule_edf
from src.edf_drr import schedule_edf_drr
from src.fifo import calculate_metrics, schedule_fifo
from src.gen_td26 import load_gen_td26_requests
from src.models import REQUEST_FIELDS, Request
from src.milp import solve_milp
from src.parallel import schedule_parallel
from src.predictive_edf_drr import (
    predict_traffic_and_adjust_quanta,
    schedule_predictive_edf_drr,
)
from src.traffic import generate_requests
from src.traffic_model import TrafficSpikeModel, train_traffic_model
from src.research_metrics import classification_metrics, regression_metrics, spike_labels


class FifoBaselineTests(unittest.TestCase):
    def test_fifo_preserves_arrival_order(self) -> None:
        requests = [
            Request(1, "Free", 0.0, 1.0, 50.0, 10.0, 1.0),
            Request(2, "VIP", 0.1, 0.5, 50.0, 1.0, 10.0),
        ]
        scheduled = schedule_fifo(requests)
        self.assertEqual([item.request.request_id for item in scheduled], [1, 2])
        self.assertAlmostEqual(scheduled[1].waiting_time, 0.9)

    def test_metrics_capture_sla_violation_and_utilization(self) -> None:
        requests = [Request(1, "VIP", 0.0, 2.0, 50.0, 1.0, 10.0)]
        scheduled = schedule_fifo(requests)
        metrics = calculate_metrics(requests, scheduled)
        self.assertEqual(metrics.sla_violations, 1)
        self.assertEqual(metrics.sla_violation_rate, 1.0)
        self.assertAlmostEqual(metrics.throughput, 0.5)
        self.assertAlmostEqual(metrics.cpu_utilization, 0.5)
        self.assertEqual(metrics.fairness_index, 1.0)

    def test_finite_horizon_counts_only_completed_requests(self) -> None:
        requests = [
            Request(1, "VIP", 0.0, 2.0, 50.0, 1.0, 10.0),
            Request(2, "Free", 0.1, 1.0, 50.0, 10.0, 1.0),
        ]
        scheduled = schedule_fifo(requests)
        metrics = calculate_metrics(requests, scheduled, horizon=2.5)
        self.assertEqual(metrics.request_count, 2)
        self.assertEqual(metrics.completed_requests, 1)
        self.assertEqual(metrics.sla_violations, 1)
        self.assertAlmostEqual(metrics.throughput, 0.4)
        self.assertAlmostEqual(metrics.cpu_utilization, 0.5)
        self.assertAlmostEqual(metrics.fairness_index, 0.5)
        self.assertEqual(metrics.per_tier_backlog, {"Free": 1, "VIP": 0})
        self.assertEqual(metrics.per_tier_completion_rate, {"Free": 0.0, "VIP": 1.0})
        self.assertEqual(metrics.per_tier_waiting_time, {"Free": 0.0, "VIP": 0.0})

    def test_generation_is_reproducible(self) -> None:
        first = generate_requests(count=10, seed=7)
        second = generate_requests(count=10, seed=7)
        self.assertEqual(first, second)

    def test_milp_returns_a_non_overlapping_schedule(self) -> None:
        requests = [
            Request(1, "Free", 0.0, 1.0, 50.0, 10.0, 1.0),
            Request(2, "VIP", 0.0, 1.0, 50.0, 0.5, 10.0),
            Request(3, "Premium", 0.2, 0.5, 50.0, 5.0, 4.0),
        ]
        scheduled = solve_milp(requests)
        self.assertEqual(len(scheduled), 3)
        for previous, current in zip(scheduled, scheduled[1:]):
            self.assertLessEqual(previous.completion_time, current.start_time + 1e-7)
        self.assertTrue(all(item.start_time >= item.request.arrival_time for item in scheduled))

    def test_parallel_fifo_uses_multiple_workers(self) -> None:
        requests = [
            Request(1, "Free", 0.0, 2.0, 50.0, 10.0, 1.0),
            Request(2, "VIP", 0.0, 2.0, 50.0, 10.0, 10.0),
        ]
        scheduled = schedule_parallel(requests, "FIFO", workers=2)
        self.assertEqual(len(scheduled), 2)
        self.assertTrue(all(item.start_time == 0.0 for item in scheduled))

    def test_parallel_edf_prioritizes_ready_deadlines(self) -> None:
        requests = [
            Request(1, "Free", 0.0, 2.0, 50.0, 10.0, 1.0),
            Request(2, "VIP", 0.0, 2.0, 50.0, 1.0, 10.0),
            Request(3, "Premium", 0.1, 1.0, 50.0, 0.5, 4.0),
        ]
        scheduled = schedule_parallel(requests, "EDF", workers=1)
        self.assertEqual(scheduled[0].request.request_id, 2)

    def test_parallel_predictive_hybrid_schedules_all_requests(self) -> None:
        requests = [
            Request(1, "VIP", 0.0, 0.5, 50.0, 10.0, 10.0),
            Request(2, "Free", 0.0, 0.2, 50.0, 10.0, 1.0),
            Request(3, "Premium", 0.1, 0.3, 50.0, 10.0, 4.0),
        ]
        scheduled = schedule_parallel(
            requests, "Predictive EDF+DRR", workers=2
        )
        self.assertEqual(
            {item.request.request_id for item in scheduled},
            {request.request_id for request in requests},
        )

    def test_static_policies_schedule_the_same_request_set(self) -> None:
        requests = generate_requests(count=40, seed=2026, arrival_rate=20.0)
        schedules = {
            "FIFO": schedule_fifo(requests),
            "EDF": schedule_edf(requests),
            "DRR": schedule_drr(requests),
            "EDF+DRR": schedule_edf_drr(requests),
        }
        expected_ids = {request.request_id for request in requests}
        for policy, scheduled in schedules.items():
            actual_ids = [item.request.request_id for item in scheduled]
            self.assertEqual(len(actual_ids), len(expected_ids), policy)
            self.assertEqual(set(actual_ids), expected_ids, policy)
            self.assertEqual(len(actual_ids), len(set(actual_ids)), policy)

    def test_predictive_edf_drr_schedules_the_same_request_set(self) -> None:
        requests = generate_requests(count=40, seed=2026, arrival_rate=20.0)
        scheduled = schedule_predictive_edf_drr(requests)
        self.assertEqual(
            {item.request.request_id for item in scheduled},
            {request.request_id for request in requests},
        )
        self.assertEqual(
            len(scheduled),
            len({item.request.request_id for item in scheduled}),
        )

    def test_predictive_quanta_preserve_floor_and_follow_demand(self) -> None:
        base = {"VIP": 1.0, "Premium": 1.0, "Free": 1.0}
        requests = [
            Request(1, "VIP", 0.0, 3.0, 1.0, 10.0, 1.0),
            Request(2, "Free", 0.0, 1.0, 1.0, 10.0, 1.0),
        ]
        adjusted = predict_traffic_and_adjust_quanta(requests, base)
        self.assertGreater(adjusted["VIP"], adjusted["Free"])
        self.assertGreaterEqual(adjusted["Premium"], 0.5)

    def test_traffic_model_trains_predicts_and_round_trips(self) -> None:
        requests = generate_requests(count=80, seed=9, arrival_rate=4.0)
        model = train_traffic_model(requests, window_size=1.0)
        predictions = model.predict({"VIP": [1.0, 2.0, 4.0]}, 3)
        self.assertEqual(set(predictions), set(model.tiers))
        self.assertTrue(all(value >= 0.0 for value in predictions.values()))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "traffic_model.json"
            model.save(path)
            loaded = TrafficSpikeModel.load(path)
        self.assertEqual(loaded.to_dict(), model.to_dict())

    def test_research_evaluation_returns_paired_statistics(self) -> None:
        from src.evaluate_research import evaluate_schedulers

        result = evaluate_schedulers(range(2), 80, 20.0, 5.0)
        self.assertEqual(result["seed_count"], 2)
        self.assertIn("95_percent_ci", result["average_waiting_time_delta"])

    def test_research_metrics_measure_spikes_and_forecasts(self) -> None:
        labels = spike_labels([1, 1, 1, 5, 1, 1])
        metrics = classification_metrics(labels, labels)
        self.assertEqual(metrics["f1"], 1.0)
        errors = regression_metrics([1.0, 3.0], [1.0, 2.0])
        self.assertAlmostEqual(errors["mae"], 0.5)

    def test_xgboost_model_trains_and_predicts(self) -> None:
        from src.xgboost_model import train_xgboost_model

        requests = generate_requests(count=80, seed=3, arrival_rate=4.0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "xgb.json"
            model = train_xgboost_model(requests, path, window_size=1.0)
            predictions = model.predict({"VIP": [1.0, 2.0, 3.0]}, 4)
        self.assertEqual(set(predictions), set(model.tiers))

    def test_ground_truth_comparison_contains_all_policies(self) -> None:
        from src.compare_ground_truth import compare

        requests = generate_requests(count=8, seed=4, arrival_rate=5.0)
        rows = compare(requests)
        policies = {row["policy"] for row in rows}
        self.assertEqual(
            policies,
            {
                "FIFO",
                "EDF",
                "DRR",
                "EDF+DRR",
                "Predictive EDF+DRR",
                "MILP (ground truth)",
            },
        )
        best_objective = min(row["weighted_tardiness"] for row in rows)
        milp_objective = next(
            row["weighted_tardiness"]
            for row in rows
            if row["policy"] == "MILP (ground truth)"
        )
        self.assertAlmostEqual(milp_objective, best_objective)

    def test_gen_td26_loader_augments_request_fields(self) -> None:
        csv_text = (
            "gmt_create,predict_type,predict_status,exec_time_seconds,groupId,"
            "prompt_length,num_images_per_prompt,num_inference_steps,num_lora\n"
            "2024-11-15T16:57:50+00:00,TXT_2_IMG,SUCCEED,2.0,G0000,63,1,30,0\n"
            "2024-11-15T16:57:53+00:00,INPAINTING,PROCESSING,3.0,G0001,10,1,20,1\n"
            "2024-11-15T16:57:55+00:00,IMG_2_IMG,SUCCEED,1.0,G0002,20,2,40,2\n"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lora_request_trace.csv"
            path.write_text(csv_text, encoding="utf-8")
            requests = load_gen_td26_requests(path)

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].arrival_time, 0.0)
        self.assertEqual(requests[1].arrival_time, 5.0)
        self.assertEqual(requests[0].endpoint, "TXT_2_IMG")
        self.assertIn(requests[0].tier, {"VIP", "Premium", "Free"})
        self.assertGreaterEqual(requests[0].cpu_demand, 10.0)
        self.assertLessEqual(requests[0].cpu_demand, 100.0)
        self.assertGreater(requests[0].deadline, requests[0].arrival_time)

    def test_all_sources_produce_the_same_canonical_model(self) -> None:
        synthetic = generate_requests(count=1, seed=42)[0]
        production = load_gen_td26_requests(
            Path(__file__).parent / "fixtures" / "gen_td26_sample.csv",
            max_requests=1,
        )[0]
        dataclass_fields = tuple(field.name for field in fields(Request))
        self.assertEqual(dataclass_fields, REQUEST_FIELDS)
        self.assertEqual(tuple(vars(synthetic)), REQUEST_FIELDS)
        self.assertEqual(tuple(vars(production)), REQUEST_FIELDS)
        self.assertIsInstance(synthetic, Request)
        self.assertIsInstance(production, Request)

    def test_edf_selects_earliest_deadline_among_available_requests(self) -> None:
        requests = [
            Request(1, "Free", 0.0, 1.0, 50.0, 10.0, 1.0),
            Request(2, "VIP", 0.0, 0.5, 50.0, 2.0, 10.0),
        ]
        scheduled = schedule_edf(requests)
        self.assertEqual([item.request.request_id for item in scheduled], [2, 1])

    def test_edf_does_not_preempt_a_running_request(self) -> None:
        requests = [
            Request(1, "Free", 0.0, 2.0, 50.0, 10.0, 1.0),
            Request(2, "VIP", 0.1, 0.5, 50.0, 1.0, 10.0),
        ]
        scheduled = schedule_edf(requests)
        self.assertEqual([item.request.request_id for item in scheduled], [1, 2])

    def test_drr_rotates_between_tier_queues(self) -> None:
        requests = [
            Request(1, "VIP", 0.0, 0.1, 50.0, 10.0, 10.0),
            Request(2, "Premium", 0.0, 0.1, 50.0, 10.0, 4.0),
            Request(3, "Free", 0.0, 0.1, 50.0, 10.0, 1.0),
            Request(4, "VIP", 0.0, 0.1, 50.0, 10.0, 10.0),
        ]
        scheduled = schedule_drr(
            requests,
            quanta={"VIP": 0.1, "Premium": 0.1, "Free": 0.1},
        )
        self.assertEqual([item.request.request_id for item in scheduled], [1, 2, 3, 4])

    def test_drr_carries_deficit_for_large_requests(self) -> None:
        requests = [Request(1, "VIP", 0.0, 0.25, 50.0, 10.0, 10.0)]
        scheduled = schedule_drr(requests, quanta={"VIP": 0.1})
        self.assertEqual(len(scheduled), 1)
        self.assertAlmostEqual(scheduled[0].completion_time, 0.25)

    def test_fairness_quantum_profile_is_valid(self) -> None:
        self.assertEqual(set(FAIRNESS_QUANTA), {"VIP", "Premium", "Free"})
        self.assertTrue(all(value > 0 for value in FAIRNESS_QUANTA.values()))

    def test_hybrid_uses_edf_order_within_each_tier(self) -> None:
        requests = [
            Request(1, "VIP", 0.0, 0.1, 50.0, 10.0, 10.0),
            Request(2, "VIP", 0.0, 0.1, 50.0, 2.0, 10.0),
        ]
        scheduled = schedule_edf_drr(requests, quanta={"VIP": 0.1})
        self.assertEqual([item.request.request_id for item in scheduled], [2, 1])

    def test_hybrid_rotates_tiers_with_drr(self) -> None:
        requests = [
            Request(1, "VIP", 0.0, 0.1, 50.0, 10.0, 10.0),
            Request(2, "Free", 0.0, 0.1, 50.0, 1.0, 1.0),
            Request(3, "VIP", 0.0, 0.1, 50.0, 11.0, 10.0),
        ]
        scheduled = schedule_edf_drr(
            requests,
            quanta={"VIP": 0.1, "Free": 0.1},
        )
        self.assertEqual([item.request.request_id for item in scheduled], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
