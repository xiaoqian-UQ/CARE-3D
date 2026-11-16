import random
from collections import Counter

import pytest

from scenecot_verl.agent.gcs_coverage import coverage_rng, sample_coverage_modes

MODES = ("prob", "loc_rect", "loc_polar", "highlight")


@pytest.mark.parametrize("policy", ["least_used", "random", "uniform"])
@pytest.mark.parametrize("n", [0, 1, 2, 4, 7])
def test_exact_budget_and_pool(policy, n):
    pool = MODES[:3]
    selected = sample_coverage_modes({}, n, pool, policy=policy, rng=random.Random(42))
    assert len(selected) == n
    assert set(selected) <= set(pool)


def test_original_rule_still_prioritizes_unseen_tools():
    assert set(sample_coverage_modes(
        dict(prob=0, loc_rect=0, loc_polar=5, highlight=20), 2, MODES,
        rng=coverage_rng(42, 1, 0),
    )) == {"prob", "loc_rect"}


@pytest.mark.parametrize("policy", ["random", "uniform"])
def test_baselines_ignore_free_rollout_counts(policy):
    kwargs = dict(n=2, mode_pool=MODES, policy=policy, uniform_offset=3)
    a = sample_coverage_modes({"prob": 100}, rng=coverage_rng(42, 1, 0), **kwargs)
    b = sample_coverage_modes({"highlight": 100}, rng=coverage_rng(42, 1, 0), **kwargs)
    assert a == b


def test_random_is_iid_with_replacement_and_reproducible():
    kwargs = dict(counts={}, n=1000, mode_pool=MODES, policy="random")
    a = sample_coverage_modes(rng=coverage_rng(42, 1, 0), **kwargs)
    assert a == sample_coverage_modes(rng=coverage_rng(42, 1, 0), **kwargs)
    assert a != sample_coverage_modes(rng=coverage_rng(43, 1, 0), **kwargs)
    assert any(a[i] == a[i + 1] for i in range(len(a) - 1))
    assert all(180 < count < 320 for count in Counter(a).values())


def test_uniform_balances_across_groups_with_partial_tool_budget():
    selected = []
    for group in range(7):
        selected.extend(sample_coverage_modes({}, 2, MODES, policy="uniform", uniform_offset=42 + group * 2))
    counts = Counter(selected)
    assert max(counts.values()) - min(counts.values()) == 1
    assert len(selected) == 14  # Never expand two slots into all four tools.


def test_unknown_policy_fails():
    with pytest.raises(ValueError, match="Unknown GCS"):
        sample_coverage_modes({}, 2, MODES, policy="typo")
