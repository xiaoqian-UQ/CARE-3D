"""Tool allocation for equal-rollout-budget GCS ablations (no ML dependencies)."""
from __future__ import annotations

import hashlib
import random

GCS_POLICIES = ("least_used", "random", "uniform")


def coverage_rng(seed: int, step: int, group_index: int) -> random.Random:
    # Do not use trainer UIDs: they are uuid4 values regenerated on every run.
    key = f"{seed}:{step}:{group_index}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def sample_coverage_modes(
    counts: dict[str, int],
    n: int,
    mode_pool: tuple[str, ...],
    *,
    policy: str = "least_used",
    rng: random.Random | None = None,
    uniform_offset: int = 0,
) -> list[str]:
    """Return exactly n forced slots; only least_used observes free-rollout counts.

    least_used preserves the original zero-first/inverse-frequency sampler,
    including cycling its sampled plan when n exceeds the number of tools.
    random draws IID uniformly *with* replacement. uniform uses round-robin
    slots, allowing a smaller coverage budget than the number of tools.
    """
    if policy not in GCS_POLICIES:
        raise ValueError(f"Unknown GCS coverage policy {policy!r}; choose {GCS_POLICIES}")
    if not mode_pool or len(set(mode_pool)) != len(mode_pool):
        raise ValueError("GCS mode pool must be nonempty and contain unique modes")
    if n < 0:
        raise ValueError("GCS coverage count must be nonnegative")
    rng = rng if rng is not None else random.Random()
    if policy == "random":
        return [rng.choice(mode_pool) for _ in range(n)]
    if policy == "uniform":
        return [mode_pool[(uniform_offset + i) % len(mode_pool)] for i in range(n)]

    pool = list(mode_pool)
    selected: list[str] = []
    zero_count_modes = [mode for mode in pool if counts.get(mode, 0) == 0]
    rng.shuffle(zero_count_modes)
    while zero_count_modes and len(selected) < n:
        mode = zero_count_modes.pop()
        selected.append(mode)
        pool.remove(mode)
    for _ in range(min(n, len(pool))):
        if len(selected) >= n:
            break
        weights = [1.0 / (counts.get(mode, 0) + 1.0) for mode in pool]
        pick = rng.random() * sum(weights)
        cumulative = 0.0
        chosen_idx = 0
        for idx, weight in enumerate(weights):
            cumulative += weight
            if pick <= cumulative:
                chosen_idx = idx
                break
        selected.append(pool.pop(chosen_idx))
    return [selected[i % len(selected)] for i in range(n)]
