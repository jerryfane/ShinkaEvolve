"""Unit tests for the pure PACE acceptance gate (``shinka/core/acceptance.py``).

Covers the statistical guarantee (Ville/martingale false-commit bound), power on
a planted better candidate, tie handling, budget exhaustion, deterministic
auto-disable, alpha/lam edge cases, and verdict JSON round-tripping.
"""

from __future__ import annotations

import json
import math
import random
from typing import Dict

import pytest

from shinka.core.acceptance import (
    REASON_AUTODISABLED_DETERMINISTIC,
    REASON_COMMITTED,
    REASON_INSUFFICIENT_INSTANCES,
    REASON_REJECTED_BUDGET,
    REASON_REJECTED_EVIDENCE,
    AcceptanceGateConfig,
    GateVerdict,
    PaceGate,
)


def _binary_vectors(
    wins: int, losses: int, ties: int = 0
) -> tuple[Dict[int, float], Dict[int, float]]:
    """Build paired vectors with a fixed number of win/loss/tie instances.

    Wins come first, then losses, then ties, over sorted integer keys - so the
    canonical (sorted) betting order matches the intended sequence.
    """
    candidate: Dict[int, float] = {}
    incumbent: Dict[int, float] = {}
    idx = 0
    for _ in range(wins):
        candidate[idx], incumbent[idx] = 1.0, 0.0
        idx += 1
    for _ in range(losses):
        candidate[idx], incumbent[idx] = 0.0, 1.0
        idx += 1
    for _ in range(ties):
        candidate[idx], incumbent[idx] = 0.5, 0.5
        idx += 1
    return candidate, incumbent


# ---------------------------------------------------------------------------
# (a) Martingale / Ville property: false-commit rate <= alpha under a fair coin.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("win_prob", [0.5, 0.3])
def test_false_commit_rate_bounded_by_alpha(win_prob: float) -> None:
    """Under H0 (fair coin) or worse, the empirical commit rate stays <= alpha.

    Ville's inequality bounds P(sup wealth >= 1/alpha) at alpha for any stopping
    time, so this is a statistical assertion with a generous safety margin.
    """
    alpha = 0.05
    cfg = AcceptanceGateConfig(
        enabled=True, alpha=alpha, lam=0.5, min_discordant=1
    )
    gate = PaceGate(cfg)

    rng = random.Random(20240701)
    n_candidates = 10000
    n_instances = 200
    commits = 0
    for _ in range(n_candidates):
        candidate: Dict[int, float] = {}
        incumbent: Dict[int, float] = {}
        for i in range(n_instances):
            # Every instance is discordant; candidate wins with prob win_prob.
            if rng.random() < win_prob:
                candidate[i], incumbent[i] = 1.0, 0.0
            else:
                candidate[i], incumbent[i] = 0.0, 1.0
        verdict = gate.verdict(candidate, incumbent)
        if verdict.committed:
            commits += 1

    rate = commits / n_candidates
    # 3-sigma binomial margin plus slack; Ville is not tight, so the empirical
    # rate is typically well below alpha.
    sigma = math.sqrt(alpha * (1.0 - alpha) / n_candidates)
    assert rate <= alpha + 3.0 * sigma + 0.01, (
        f"commit rate {rate:.4f} exceeded alpha={alpha} + margin "
        f"(win_prob={win_prob})"
    )


# ---------------------------------------------------------------------------
# (b) Power: a planted better candidate commits.
# ---------------------------------------------------------------------------


def test_planted_better_candidate_commits() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    # 27 wins / 3 losses (90%): more than enough evidence to cross 1/alpha=20.
    candidate, incumbent = _binary_vectors(wins=27, losses=3)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is True
    assert verdict.reason == REASON_COMMITTED
    assert verdict.final_wealth >= verdict.wealth_target
    assert verdict.n_wins == 27


def test_planted_better_candidate_commits_high_reliability() -> None:
    """Over many stochastic 90%-win candidates, commit rate is high (power)."""
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    rng = random.Random(7)
    n = 500
    n_instances = 32
    commits = 0
    for _ in range(n):
        candidate: Dict[int, float] = {}
        incumbent: Dict[int, float] = {}
        for i in range(n_instances):
            if rng.random() < 0.9:
                candidate[i], incumbent[i] = 1.0, 0.0
            else:
                candidate[i], incumbent[i] = 0.0, 1.0
        if gate.verdict(candidate, incumbent).committed:
            commits += 1
    assert commits / n > 0.9


# ---------------------------------------------------------------------------
# (c) Ties are skipped (no bet placed, do not affect wealth).
# ---------------------------------------------------------------------------


def test_ties_are_skipped() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    # 27 wins, 3 losses, plus 40 ties interleaved via key ordering.
    candidate, incumbent = _binary_vectors(wins=27, losses=3, ties=40)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.n_ties == 40
    assert verdict.n_discordant == 30
    assert verdict.n_pairs == 70
    # Trajectory holds only discordant bets (+ the leading 1.0 start value).
    non_start_bets = len(verdict.wealth_trajectory) - 1
    assert non_start_bets <= verdict.n_discordant
    assert verdict.committed is True


def test_ties_do_not_change_wealth() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    no_ties, no_ties_inc = _binary_vectors(wins=5, losses=5)
    with_ties, with_ties_inc = _binary_vectors(wins=5, losses=5, ties=25)
    v1 = gate.verdict(no_ties, no_ties_inc)
    v2 = gate.verdict(with_ties, with_ties_inc)
    assert v1.final_wealth == pytest.approx(v2.final_wealth)
    assert v1.wealth_trajectory == pytest.approx(v2.wealth_trajectory)


# ---------------------------------------------------------------------------
# (d) Budget exhaustion rejects.
# ---------------------------------------------------------------------------


def test_budget_exhaustion_rejects() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5, max_pairs=5)
    gate = PaceGate(cfg)
    # Alternating win/loss so wealth oscillates and never crosses 20 in 5 bets.
    candidate: Dict[int, float] = {}
    incumbent: Dict[int, float] = {}
    for i in range(20):
        if i % 2 == 0:
            candidate[i], incumbent[i] = 1.0, 0.0
        else:
            candidate[i], incumbent[i] = 0.0, 1.0
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_BUDGET
    # Exactly max_pairs bets were placed before giving up.
    assert len(verdict.wealth_trajectory) - 1 == 5


def test_evidence_exhaustion_rejects_without_budget_cap() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    # Alternating win/loss keeps wealth oscillating near 1, never crossing 20;
    # all discordant pairs are consumed without a budget cap.
    candidate: Dict[int, float] = {}
    incumbent: Dict[int, float] = {}
    for i in range(30):
        if i % 2 == 0:
            candidate[i], incumbent[i] = 1.0, 0.0
        else:
            candidate[i], incumbent[i] = 0.0, 1.0
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_EVIDENCE
    assert verdict.n_discordant == 30


# ---------------------------------------------------------------------------
# (e) Deterministic vectors auto-disable (pass-through commit).
# ---------------------------------------------------------------------------


def test_identical_vectors_autodisable() -> None:
    cfg = AcceptanceGateConfig(enabled=True, determinism_autodisable=True)
    gate = PaceGate(cfg)
    vec = {i: float(i) for i in range(10)}
    verdict = gate.verdict(dict(vec), dict(vec))
    assert verdict.committed is True
    assert verdict.reason == REASON_AUTODISABLED_DETERMINISTIC
    assert verdict.n_discordant == 0
    assert verdict.n_ties == 10


def test_low_discordant_autodisable() -> None:
    cfg = AcceptanceGateConfig(
        enabled=True, determinism_autodisable=True, min_discordant=5
    )
    gate = PaceGate(cfg)
    # Only 2 discordant pairs, below min_discordant=5.
    candidate, incumbent = _binary_vectors(wins=2, losses=0, ties=20)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is True
    assert verdict.reason == REASON_AUTODISABLED_DETERMINISTIC


def test_low_discordant_insufficient_when_autodisable_off() -> None:
    cfg = AcceptanceGateConfig(
        enabled=True, determinism_autodisable=False, min_discordant=5
    )
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=2, losses=0, ties=20)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is True
    assert verdict.reason == REASON_INSUFFICIENT_INSTANCES


def test_no_common_instances_insufficient() -> None:
    cfg = AcceptanceGateConfig(enabled=True)
    gate = PaceGate(cfg)
    verdict = gate.verdict({1: 1.0, 2: 2.0}, {3: 1.0, 4: 2.0})
    assert verdict.committed is True
    assert verdict.reason == REASON_INSUFFICIENT_INSTANCES
    assert verdict.n_pairs == 0


# ---------------------------------------------------------------------------
# (f) alpha / lam edge cases.
# ---------------------------------------------------------------------------


def test_lam_zero_never_commits() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.0)
    gate = PaceGate(cfg)
    # Even an overwhelming win streak: every multiplier is exactly 1.0.
    candidate, incumbent = _binary_vectors(wins=100, losses=0)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is False
    assert verdict.final_wealth == pytest.approx(1.0)
    assert verdict.reason == REASON_REJECTED_EVIDENCE


def test_alpha_one_commits_easily() -> None:
    # alpha=1 -> wealth target 1.0; a single win crosses it immediately.
    cfg = AcceptanceGateConfig(enabled=True, alpha=1.0, lam=0.5)
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=1, losses=0)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is True
    assert verdict.reason == REASON_COMMITTED
    assert verdict.wealth_target == pytest.approx(1.0)


def test_alpha_tiny_rarely_commits() -> None:
    # alpha=1e-9 -> wealth target 1e9; a modest win streak cannot cross it.
    cfg = AcceptanceGateConfig(enabled=True, alpha=1e-9, lam=0.5)
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=30, losses=0)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_EVIDENCE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": -0.1},
        {"alpha": 1.5},
        {"lam": -0.01},
        {"lam": 1.01},
        {"min_discordant": 0},
        {"max_pairs": 0},
        {"tie_atol": -1.0},
    ],
)
def test_invalid_config_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AcceptanceGateConfig(enabled=True, **kwargs)


def test_config_defaults_flag_off() -> None:
    cfg = AcceptanceGateConfig()
    assert cfg.enabled is False
    assert cfg.alpha == 0.05
    assert cfg.lam == 0.5
    assert cfg.max_pairs is None
    assert cfg.determinism_autodisable is True
    assert cfg.gate_bandit is False
    assert cfg.gate_scratchpad is False
    assert cfg.wealth_target == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# (g) Verdict JSON round-trip.
# ---------------------------------------------------------------------------


def test_verdict_json_round_trip() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=27, losses=3, ties=5)
    verdict = gate.verdict(candidate, incumbent)

    payload = json.dumps(verdict.to_dict())
    restored = GateVerdict.from_dict(json.loads(payload))

    assert restored.committed == verdict.committed
    assert restored.reason == verdict.reason
    assert restored.n_pairs == verdict.n_pairs
    assert restored.n_discordant == verdict.n_discordant
    assert restored.n_ties == verdict.n_ties
    assert restored.n_wins == verdict.n_wins
    assert restored.final_wealth == pytest.approx(verdict.final_wealth)
    assert restored.peak_wealth == pytest.approx(verdict.peak_wealth)
    assert restored.wealth_target == pytest.approx(verdict.wealth_target)
    assert restored.wealth_trajectory == pytest.approx(verdict.wealth_trajectory)


def test_pass_through_verdict_json_round_trip() -> None:
    cfg = AcceptanceGateConfig(enabled=True)
    gate = PaceGate(cfg)
    vec = {i: float(i) for i in range(6)}
    verdict = gate.verdict(dict(vec), dict(vec))
    restored = GateVerdict.from_dict(json.loads(json.dumps(verdict.to_dict())))
    assert restored == verdict


# ---------------------------------------------------------------------------
# Canonical betting order is independent of dict insertion order.
# ---------------------------------------------------------------------------


def test_verdict_order_invariant_to_insertion_order() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=10, losses=8)

    shuffled_keys = list(candidate.keys())
    random.Random(1).shuffle(shuffled_keys)
    cand_shuf = {k: candidate[k] for k in shuffled_keys}
    inc_shuf = {k: incumbent[k] for k in shuffled_keys}

    v1 = gate.verdict(candidate, incumbent)
    v2 = gate.verdict(cand_shuf, inc_shuf)
    assert v1.final_wealth == pytest.approx(v2.final_wealth)
    assert v1.wealth_trajectory == pytest.approx(v2.wealth_trajectory)
    assert v1.reason == v2.reason


def test_verdict_is_idempotent() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=27, losses=3)
    v1 = gate.verdict(candidate, incumbent)
    v2 = gate.verdict(candidate, incumbent)
    assert v1 == v2
