"""Core request and scheduling result models for the OR gateway project."""

from dataclasses import dataclass
from math import isfinite


REQUEST_FIELDS: tuple[str, ...] = (
    "request_id",
    "tier",
    "arrival_time",
    "service_time",
    "cpu_demand",
    "deadline",
    "sla_penalty",
    "endpoint",
)


@dataclass(frozen=True)
class Request:
    """Canonical request representation consumed by every scheduler."""

    request_id: int
    tier: str
    arrival_time: float
    service_time: float
    cpu_demand: float
    deadline: float
    sla_penalty: float
    endpoint: str = ""

    def __post_init__(self) -> None:
        """Reject malformed requests at the source boundary."""
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if not self.tier:
            raise ValueError("tier must not be empty")
        if not self.endpoint:
            object.__setattr__(self, "endpoint", "unknown")
        numeric_values = {
            "arrival_time": self.arrival_time,
            "service_time": self.service_time,
            "cpu_demand": self.cpu_demand,
            "deadline": self.deadline,
            "sla_penalty": self.sla_penalty,
        }
        if any(not isfinite(value) for value in numeric_values.values()):
            raise ValueError("request numeric fields must be finite")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.service_time <= 0:
            raise ValueError("service_time must be positive")
        if self.cpu_demand < 0:
            raise ValueError("cpu_demand must be non-negative")
        if self.deadline < self.arrival_time:
            raise ValueError("deadline must not precede arrival_time")
        if self.sla_penalty < 0:
            raise ValueError("sla_penalty must be non-negative")


@dataclass(frozen=True)
class ScheduledRequest:
    """A request after a scheduler assigns start and completion times."""

    request: Request
    start_time: float
    completion_time: float

    @property
    def waiting_time(self) -> float:
        return self.start_time - self.request.arrival_time

    @property
    def sla_violated(self) -> bool:
        return self.completion_time > self.request.deadline


@dataclass(frozen=True)
class Metrics:
    """Aggregate metrics produced by a scheduling experiment."""

    request_count: int
    completed_requests: int
    average_waiting_time: float
    p95_waiting_time: float
    sla_violations: int
    sla_violation_rate: float
    throughput: float
    cpu_utilization: float
    fairness_index: float
    per_tier_waiting_time: dict[str, float]
    per_tier_completion_rate: dict[str, float]
    per_tier_backlog: dict[str, int]
    per_tier_sla_violation_rate: dict[str, float]
    expired_requests: int = 0
    useful_work_ratio: float = 1.0

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly representation of the metrics."""
        return {
            "request_count": self.request_count,
            "completed_requests": self.completed_requests,
            "average_waiting_time": self.average_waiting_time,
            "p95_waiting_time": self.p95_waiting_time,
            "sla_violations": self.sla_violations,
            "sla_violation_rate": self.sla_violation_rate,
            "throughput": self.throughput,
            "cpu_utilization": self.cpu_utilization,
            "fairness_index": self.fairness_index,
            "per_tier_waiting_time": self.per_tier_waiting_time,
            "per_tier_completion_rate": self.per_tier_completion_rate,
            "per_tier_backlog": self.per_tier_backlog,
            "per_tier_sla_violation_rate": self.per_tier_sla_violation_rate,
            "expired_requests": self.expired_requests,
            "useful_work_ratio": self.useful_work_ratio,
        }
