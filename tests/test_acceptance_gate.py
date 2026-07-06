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
    REASON_COMMITTED,
    REASON_INSUFFICIENT_INSTANCES,
    REASON_REJECTED_BUDGET,
    REASON_REJECTED_EVIDENCE,
    REASON_REJECTED_INSUFFICIENT,
    REASON_REJECTED_NO_EVIDENCE,
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
# (e) Zero-evidence handling (deterministic / too-few-discordant / no-common).
#
# Default ``zero_evidence_action="reject"``: a judged comparison that never
# places a bet WITHHOLDS the candidate rather than silently admitting it.
# ---------------------------------------------------------------------------


def test_identical_vectors_rejected_by_default() -> None:
    cfg = AcceptanceGateConfig(enabled=True)  # zero_evidence_action="reject"
    gate = PaceGate(cfg)
    vec = {i: float(i) for i in range(10)}
    verdict = gate.verdict(dict(vec), dict(vec))
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_NO_EVIDENCE
    assert verdict.n_discordant == 0
    assert verdict.n_ties == 10


def test_low_discordant_rejected_by_default() -> None:
    cfg = AcceptanceGateConfig(enabled=True, min_discordant=5)
    gate = PaceGate(cfg)
    # Only 2 discordant pairs, below min_discordant=5.
    candidate, incumbent = _binary_vectors(wins=2, losses=0, ties=20)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_NO_EVIDENCE


def test_zero_evidence_action_pass_admits() -> None:
    cfg = AcceptanceGateConfig(
        enabled=True, min_discordant=5, zero_evidence_action="pass"
    )
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=2, losses=0, ties=20)
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is True
    assert verdict.reason == REASON_INSUFFICIENT_INSTANCES


def test_zero_evidence_action_both_modes_no_common() -> None:
    reject = PaceGate(AcceptanceGateConfig(enabled=True))
    v_reject = reject.verdict({1: 1.0, 2: 2.0}, {3: 1.0, 4: 2.0})
    assert v_reject.committed is False
    assert v_reject.reason == REASON_REJECTED_INSUFFICIENT
    assert v_reject.n_pairs == 0

    passer = PaceGate(
        AcceptanceGateConfig(enabled=True, zero_evidence_action="pass")
    )
    v_pass = passer.verdict({1: 1.0, 2: 2.0}, {3: 1.0, 4: 2.0})
    assert v_pass.committed is True
    assert v_pass.reason == REASON_INSUFFICIENT_INSTANCES
    assert v_pass.n_pairs == 0


def test_determinism_autodisable_off_logs_error_for_all_ties(caplog) -> None:
    # All-ties with determinism_autodisable=False -> logged at ERROR, still
    # resolved by zero_evidence_action (reject by default).
    cfg = AcceptanceGateConfig(enabled=True, determinism_autodisable=False)
    gate = PaceGate(cfg)
    vec = {i: float(i) for i in range(6)}
    import logging as _logging

    with caplog.at_level(_logging.ERROR):
        verdict = gate.verdict(dict(vec), dict(vec))
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_NO_EVIDENCE
    assert any("unexpected determinism" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# (e2) Missing-vector task-shape failure (``on_missing_vectors``). Distinct from
# the degenerate-comparison zero-evidence path: here NEITHER side publishes any
# per-instance score, so the gate has no evidence surface at all.
# ---------------------------------------------------------------------------


def test_missing_vectors_error_is_the_default() -> None:
    # Default on_missing_vectors="error": both vectors empty -> raise loud with a
    # remediation message rather than silently accept-all or reject-all.
    gate = PaceGate(AcceptanceGateConfig(enabled=True))
    with pytest.raises(RuntimeError) as excinfo:
        gate.verdict({}, {})
    msg = str(excinfo.value)
    assert "instance_scores" in msg
    assert "on_missing_vectors" in msg


def test_missing_vectors_reject_override(caplog) -> None:
    import logging as _logging

    gate = PaceGate(
        AcceptanceGateConfig(enabled=True, on_missing_vectors="reject")
    )
    with caplog.at_level(_logging.WARNING):
        verdict = gate.verdict({}, {})
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_INSUFFICIENT
    assert verdict.n_pairs == 0
    assert any("missing per-instance vectors" in r.message for r in caplog.records)


def test_missing_vectors_pass_override(caplog) -> None:
    import logging as _logging

    gate = PaceGate(
        AcceptanceGateConfig(enabled=True, on_missing_vectors="pass")
    )
    with caplog.at_level(_logging.WARNING):
        verdict = gate.verdict({}, {})
    assert verdict.committed is True
    assert verdict.reason == REASON_INSUFFICIENT_INSTANCES
    assert verdict.n_pairs == 0
    assert any("missing per-instance vectors" in r.message for r in caplog.records)


def test_present_vectors_all_ties_still_uses_zero_evidence() -> None:
    # Vectors PRESENT but all tied -> degenerate comparison, governed by
    # zero_evidence_action (NOT on_missing_vectors); default rejects.
    gate = PaceGate(
        AcceptanceGateConfig(enabled=True, on_missing_vectors="error")
    )
    vec = {i: float(i) for i in range(4)}
    verdict = gate.verdict(dict(vec), dict(vec))
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_NO_EVIDENCE


def test_one_sided_empty_vector_is_zero_evidence_not_missing() -> None:
    # Only one side empty (task publishes scores, this candidate has none) is a
    # per-candidate zero-evidence case, NOT the task-shape missing-vector error.
    gate = PaceGate(
        AcceptanceGateConfig(enabled=True, on_missing_vectors="error")
    )
    verdict = gate.verdict({}, {1: 1.0, 2: 0.0})
    assert verdict.committed is False
    assert verdict.reason == REASON_REJECTED_INSUFFICIENT
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
        {"tie_rtol": -1.0},
        {"zero_evidence_action": "maybe"},
        {"on_missing_vectors": "boom"},
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
    assert cfg.tie_atol == pytest.approx(1e-9)
    assert cfg.tie_rtol == pytest.approx(1e-9)
    assert cfg.zero_evidence_action == "reject"
    assert cfg.on_missing_vectors == "error"
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


# ---------------------------------------------------------------------------
# (h) Boolean / numeric vectors are judged (not silently inert).
# ---------------------------------------------------------------------------


def test_bool_vectors_are_judged() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    # Candidate True (1.0) where incumbent False (0.0): a decisive win streak.
    candidate = {i: True for i in range(27)}
    incumbent = {i: False for i in range(27)}
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.committed is True
    assert verdict.reason == REASON_COMMITTED
    assert verdict.n_wins == 27
    assert verdict.n_discordant == 27


def test_bool_and_int_ties_and_losses() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    # Mixed bool/int: True vs 1 is a tie; False vs 1 is a loss; True vs 0 a win.
    candidate = {0: True, 1: True, 2: False}
    incumbent = {0: 1, 1: 0, 2: 1}
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.n_ties == 1  # True(1.0) vs 1
    assert verdict.n_discordant == 2
    assert verdict.n_wins == 1  # True(1.0) vs 0


# ---------------------------------------------------------------------------
# (i) NaN / inf pairs are skipped (never a win, loss, or tie).
# ---------------------------------------------------------------------------


def test_nan_pair_skipped_and_counted() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    candidate, incumbent = _binary_vectors(wins=5, losses=5)
    # Poison two pairs: one with NaN on the candidate, one with inf on incumbent.
    candidate[100], incumbent[100] = float("nan"), 0.0
    candidate[101], incumbent[101] = 1.0, float("inf")
    verdict = gate.verdict(candidate, incumbent)
    assert verdict.n_invalid == 2
    # The two poisoned pairs are neither wins, losses, nor ties.
    assert verdict.n_pairs == 10
    assert verdict.n_discordant == 10
    assert verdict.n_wins == 5


def test_nan_pair_does_not_change_wealth() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    clean_c, clean_i = _binary_vectors(wins=5, losses=5)
    dirty_c, dirty_i = _binary_vectors(wins=5, losses=5)
    dirty_c[999], dirty_i[999] = float("nan"), float("nan")
    v_clean = gate.verdict(clean_c, clean_i)
    v_dirty = gate.verdict(dirty_c, dirty_i)
    assert v_dirty.wealth_trajectory == pytest.approx(v_clean.wealth_trajectory)
    assert v_dirty.n_invalid == 1


# ---------------------------------------------------------------------------
# (j) Relative tie tolerance (tie_rtol) classification.
# ---------------------------------------------------------------------------


def test_rtol_tie_classification() -> None:
    # A difference of 1.0 between two ~1e9 values is a tie under rtol=1e-9
    # (1.0 <= 1e-9 * 1e9 = 1.0) but NOT under atol alone.
    cfg = AcceptanceGateConfig(
        enabled=True, tie_atol=1e-9, tie_rtol=1e-9, min_discordant=1
    )
    gate = PaceGate(cfg)
    candidate = {0: 1_000_000_000.0, 1: 2.0}
    incumbent = {0: 999_999_999.0, 1: 0.0}
    verdict = gate.verdict(candidate, incumbent)
    # Instance 0 is a tie by rtol; instance 1 (2.0 vs 0.0) is a real win.
    assert verdict.n_ties == 1
    assert verdict.n_discordant == 1
    assert verdict.n_wins == 1


def test_rtol_zero_makes_large_diff_discordant() -> None:
    cfg = AcceptanceGateConfig(
        enabled=True, tie_atol=1e-9, tie_rtol=0.0, min_discordant=1
    )
    gate = PaceGate(cfg)
    candidate = {0: 1_000_000_000.0}
    incumbent = {0: 999_999_999.0}
    verdict = gate.verdict(candidate, incumbent)
    # Without rtol, the 1.0 gap far exceeds atol -> discordant win.
    assert verdict.n_ties == 0
    assert verdict.n_discordant == 1
    assert verdict.n_wins == 1


# ---------------------------------------------------------------------------
# (k) Run-level summary counters.
# ---------------------------------------------------------------------------


def test_summary_counters_partition_judged() -> None:
    cfg = AcceptanceGateConfig(enabled=True, alpha=0.05, lam=0.5)
    gate = PaceGate(cfg)
    # One commit, one evidence rejection, one zero-evidence (no common keys).
    gate.verdict(*_binary_vectors(wins=27, losses=3))
    # Alternating win/loss oscillates near 1 and never crosses the target.
    osc_c: Dict[int, float] = {}
    osc_i: Dict[int, float] = {}
    for i in range(30):
        if i % 2 == 0:
            osc_c[i], osc_i[i] = 1.0, 0.0
        else:
            osc_c[i], osc_i[i] = 0.0, 1.0
    reject = gate.verdict(osc_c, osc_i)
    assert reject.reason == REASON_REJECTED_EVIDENCE
    gate.verdict({1: 1.0}, {2: 1.0})
    s = gate.summary()
    assert s["judged"] == 3
    assert s["committed"] == 1
    assert s["rejected"] == 1
    assert s["no_evidence"] == 1
    assert s["committed"] + s["rejected"] + s["no_evidence"] == s["judged"]


# ---------------------------------------------------------------------------
# (l) EvolutionConfig unknown-key coercion error.
# ---------------------------------------------------------------------------


def test_evolution_config_unknown_gate_key_raises() -> None:
    from shinka.core.config import EvolutionConfig

    with pytest.raises(ValueError) as exc_info:
        EvolutionConfig(
            acceptance_gate={"enabled": True, "not_a_field": 1, "typo": 2}
        )
    msg = str(exc_info.value)
    assert "acceptance_gate: unknown keys" in msg
    assert "not_a_field" in msg and "typo" in msg
    assert "valid keys" in msg


def test_evolution_config_valid_gate_mapping_coerced() -> None:
    from shinka.core.config import EvolutionConfig

    cfg = EvolutionConfig(
        acceptance_gate={"enabled": True, "zero_evidence_action": "pass"}
    )
    assert isinstance(cfg.acceptance_gate, AcceptanceGateConfig)
    assert cfg.acceptance_gate.zero_evidence_action == "pass"
