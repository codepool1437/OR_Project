"""Deterministic synthetic API traffic generation."""

import random
from typing import Sequence

from .models import Request


DEFAULT_TIERS: tuple[str, ...] = ("VIP", "Premium", "Free")
DEFAULT_TIER_WEIGHTS: tuple[float, ...] = (0.20, 0.30, 0.50)


def generate_requests(
    count: int = 1_000,
    seed: int = 42,
    arrival_rate: float = 5.0,
    tiers: Sequence[str] = DEFAULT_TIERS,
    tier_weights: Sequence[float] = DEFAULT_TIER_WEIGHTS,
) -> list[Request]:
    """Generate a reproducible mixed-tier request workload.

    Inter-arrival times follow an exponential distribution. Service times,
    CPU demand, and deadline slack are sampled from simple distributions so
    that the first experiment remains transparent and easy to replace later.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    if len(tiers) != len(tier_weights) or not tiers:
        raise ValueError("tiers and tier_weights must have the same non-zero length")
    if any(weight < 0 for weight in tier_weights) or sum(tier_weights) <= 0:
        raise ValueError("tier_weights must be non-negative and not all zero")

    rng = random.Random(seed)
    normalized_weights = [weight / sum(tier_weights) for weight in tier_weights]
    tier_deadline_slack = {"VIP": (0.25, 1.25), "Premium": (0.75, 2.0), "Free": (1.5, 4.0)}

    requests: list[Request] = []
    current_time = 0.0
    for request_id in range(count):
        current_time += rng.expovariate(arrival_rate)
        tier = rng.choices(list(tiers), weights=normalized_weights, k=1)[0]
        service_time = rng.uniform(0.05, 0.30)
        cpu_demand = rng.uniform(10.0, 60.0)
        slack_range = tier_deadline_slack.get(tier, (1.0, 3.0))
        deadline = current_time + rng.uniform(*slack_range)
        penalty = {"VIP": 10.0, "Premium": 4.0, "Free": 1.0}.get(tier, 1.0)
        requests.append(
            Request(
                request_id=request_id,
                tier=tier,
                arrival_time=current_time,
                service_time=service_time,
                cpu_demand=cpu_demand,
                deadline=deadline,
                sla_penalty=penalty,
                endpoint="synthetic",
            )
        )
    return requests
