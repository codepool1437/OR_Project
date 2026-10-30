"""Dynamic (closed-loop, urgency-aware) EDF+DRR components.

Dynamic EDF+DRR changes the static EDF+DRR policy in two independent ways:

1. **Urgency-driven quanta.** At the start of each DRR visit the quantum of
   every tier is rescaled from the penalty-weighted *criticality* of its
   pending requests. No traffic forecast is required: the controller reacts
   to the deadline pressure that is actually observable in the queues.
2. **Late-work deferral.** Inside a tier, EDF normally serves the request whose
   deadline has already passed first (the classical EDF overload "domino
   effect"). Dynamic EDF+DRR defers requests that can no longer meet their
   deadline until no feasible request remains in the tier.

Both mechanisms can be enabled separately, which gives the ablation variants
reported in the paper.
"""

from typing import Iterable, Mapping

from .models import Request, ScheduledRequest

MIN_QUANTUM_SCALE = 0.9
# Exponent applied to the SLA penalty when weighting urgency. Base quanta already
# favour high tiers, so a full penalty weight would double count tier priority.
PENALTY_POWER = 0.0


def request_criticality(request: Request, now: float) -> float:
    """Return a value in [0, 1] describing how urgent a pending request is.

    A request that can no longer finish by its deadline cannot be saved by
    serving it, so it contributes zero. Otherwise criticality decays with the
    remaining slack measured in units of the request's own service time.
    """
    slack = request.deadline - now - request.service_time
    if slack < 0:
        return 0.0
    return 1.0 / (1.0 + slack / request.service_time)


def urgency_adjusted_quanta(
    pending_requests: Iterable[Request],
    base_quanta: Mapping[str, float],
    now: float,
    floor: float = MIN_QUANTUM_SCALE,
) -> dict[str, float]:
    """Rescale DRR quanta by each tier's share of penalty-weighted criticality.

    ``scale_t = floor + K * (1 - floor) * share_t`` where ``K`` is the number
    of tiers. Equal shares reproduce the base quanta, a tier with no savable
    work keeps ``floor`` of its quantum (no starvation), and the maximum scale
    is bounded by ``floor + K * (1 - floor)``.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError("floor must be in (0, 1]")
    urgency = {tier: 0.0 for tier in base_quanta}
    for request in pending_requests:
        if request.tier in urgency:
            urgency[request.tier] += request.sla_penalty**PENALTY_POWER * request_criticality(
                request, now
            )
    total = sum(urgency.values())
    if total <= 0.0:
        return dict(base_quanta)
    tier_count = len(base_quanta)
    return {
        tier: quantum
        * (floor + tier_count * (1.0 - floor) * urgency[tier] / total)
        for tier, quantum in base_quanta.items()
    }


def schedule_dynamic_edf_drr(
    requests: Iterable[Request],
    quanta: Mapping[str, float] | None = None,
    workers: int = 1,
) -> list[ScheduledRequest]:
    """Schedule with Dynamic EDF+DRR (urgency quanta + late-work deferral)."""
    from .parallel import schedule_parallel

    return schedule_parallel(requests, "Dynamic EDF+DRR", workers, quanta=quanta)
