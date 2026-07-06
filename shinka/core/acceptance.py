"""Anytime-valid PACE acceptance gate (pure e-process core).

This module implements the statistical heart of the PACE acceptance gate: a
betting martingale (e-process) over the McNemar-discordant pairs of a paired
candidate/incumbent per-instance score comparison. It is a *pure* module - it
holds no runner, database, or config-plumbing state and never performs I/O, so
its behaviour is fully determined by the two score vectors and the
:class:`AcceptanceGateConfig` it is constructed with.

The gate answers a single question per candidate: does the paired evidence show,
at an anytime-valid level ``alpha``, that the candidate beats its incumbent? The
wealth process starts at ``1.0`` and, for every discordant pair (candidate and
incumbent disagree on that instance), multiplies by ``1 + lam * (2 * win - 1)``
where ``win`` is ``1`` when the candidate wins that instance and ``0`` otherwise.
Ties (concordant pairs) place no bet. The candidate commits as soon as the
wealth crosses ``1 / alpha``; if the discordant-pair budget is exhausted without
crossing, the candidate is rejected.

Under the fair-coin null (each discordant pair a 50/50 win) the wealth is a
non-negative martingale, so Ville's inequality bounds the false-commit rate at
``alpha`` for any stopping rule. See ``tests/test_acceptance_gate.py``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)

# Verdict reason codes (kept as module-level constants so callers and tests
# reference the same strings the gate emits).
REASON_COMMITTED = "committed"
REASON_REJECTED_BUDGET = "rejected_budget"
REASON_REJECTED_EVIDENCE = "rejected_evidence"
REASON_AUTODISABLED_DETERMINISTIC = "autodisabled_deterministic"
REASON_INSUFFICIENT_INSTANCES = "insufficient_instances"

VERDICT_REASONS = (
    REASON_COMMITTED,
    REASON_REJECTED_BUDGET,
    REASON_REJECTED_EVIDENCE,
    REASON_AUTODISABLED_DETERMINISTIC,
    REASON_INSUFFICIENT_INSTANCES,
)


@dataclass
class AcceptanceGateConfig:
    """Configuration for the PACE acceptance gate.

    Attributes:
        enabled: Master switch. When ``False`` the gate is not consulted at all
            and behaviour must be byte-identical to stock ShinkaEvolve. This flag
            is enforced by the runner; the pure :meth:`PaceGate.verdict` method
            assumes it is only called when the gate is active.
        alpha: Anytime-valid significance level. The wealth commit target is
            ``1 / alpha`` (e.g. ``alpha=0.05`` -> target ``20``). Must be in
            ``(0, 1]``.
        lam: Betting fraction applied per discordant pair. Must be in
            ``[0, 1]``. ``lam=0`` places null bets and therefore never commits;
            ``lam=1`` lets the wealth hit ``0`` on a loss.
        max_pairs: Optional cap on the number of discordant bets (the paired
            evaluation budget). ``None`` means no cap. When the cap is reached
            without crossing, the verdict is ``rejected_budget``.
        min_discordant: Minimum number of discordant pairs required for an
            evidence-based verdict. Below this the paired signal is treated as
            too weak: either auto-disabled (if ``determinism_autodisable``) or
            passed through as ``insufficient_instances``.
        tie_atol: Absolute tolerance below which a per-instance score difference
            is treated as a tie (concordant, no bet).
        determinism_autodisable: When ``True`` (default), a comparison with no
            discordant pairs - identical vectors or a deterministic fitness - or
            fewer than ``min_discordant`` discordant pairs auto-disables the gate
            (pass-through commit, logged).
        gate_bandit: Door 3 sub-flag - whether the bandit reward is gated. Read
            by the runner; the pure gate does not use it.
        gate_scratchpad: Door 4 sub-flag - whether the meta scratchpad is gated.
            Read by the runner; the pure gate does not use it.
    """

    enabled: bool = False
    alpha: float = 0.05
    lam: float = 0.5
    max_pairs: int | None = None
    min_discordant: int = 1
    tie_atol: float = 1e-9
    determinism_autodisable: bool = True
    gate_bandit: bool = False
    gate_scratchpad: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not (0.0 <= self.lam <= 1.0):
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")
        if self.min_discordant < 1:
            raise ValueError(
                f"min_discordant must be >= 1, got {self.min_discordant}"
            )
        if self.max_pairs is not None and self.max_pairs < 1:
            raise ValueError(
                f"max_pairs must be >= 1 or None, got {self.max_pairs}"
            )
        if self.tie_atol < 0.0:
            raise ValueError(f"tie_atol must be >= 0, got {self.tie_atol}")

    @property
    def wealth_target(self) -> float:
        """Wealth threshold (``1 / alpha``) that triggers a commit."""
        return 1.0 / self.alpha


@dataclass
class GateVerdict:
    """Result of a single PACE gate judgement.

    Fully JSON-serialisable via :meth:`to_dict`; intended to be stored on
    ``program.metadata['pace']`` and appended to ``pace_log.jsonl``.

    Attributes:
        committed: Whether the candidate is admitted. ``True`` for a genuine
            evidence crossing (``committed``) and for pass-through reasons
            (``autodisabled_deterministic``, ``insufficient_instances``);
            ``False`` for genuine rejections (``rejected_budget``,
            ``rejected_evidence``).
        reason: One of :data:`VERDICT_REASONS`.
        n_pairs: Number of common paired instances (``n_discordant + n_ties``).
        n_discordant: Number of discordant pairs in the common set.
        n_ties: Number of concordant/tied pairs in the common set.
        n_wins: Number of discordant pairs the candidate won.
        final_wealth: Wealth after the last bet placed (``1.0`` if none).
        peak_wealth: Maximum wealth reached along the trajectory.
        wealth_target: The commit threshold (``1 / alpha``) used for this
            judgement.
        wealth_trajectory: Wealth after each discordant bet, in canonical order,
            with the leading ``1.0`` starting value included.
    """

    committed: bool
    reason: str
    n_pairs: int
    n_discordant: int
    n_ties: int
    n_wins: int
    final_wealth: float
    peak_wealth: float
    wealth_target: float
    wealth_trajectory: List[float] = field(default_factory=lambda: [1.0])

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable ``dict`` representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GateVerdict":
        """Reconstruct a verdict from :meth:`to_dict` output."""
        return cls(
            committed=bool(data["committed"]),
            reason=str(data["reason"]),
            n_pairs=int(data["n_pairs"]),
            n_discordant=int(data["n_discordant"]),
            n_ties=int(data["n_ties"]),
            n_wins=int(data["n_wins"]),
            final_wealth=float(data["final_wealth"]),
            peak_wealth=float(data["peak_wealth"]),
            wealth_target=float(data["wealth_target"]),
            wealth_trajectory=[float(w) for w in data["wealth_trajectory"]],
        )


class PaceGate:
    """Pure e-process acceptance gate.

    Construct once per run with an :class:`AcceptanceGateConfig`, then call
    :meth:`verdict` per candidate. The class is stateless across calls (the
    verdict is a deterministic function of the two score vectors and the config),
    which makes verdicts idempotent and safe to recompute on resume.
    """

    def __init__(self, config: AcceptanceGateConfig):
        self.config = config

    def verdict(
        self,
        candidate_vector: Mapping[Any, float],
        incumbent_vector: Mapping[Any, float],
    ) -> GateVerdict:
        """Judge a candidate against its incumbent on paired per-instance scores.

        Args:
            candidate_vector: Mapping from instance key (e.g. eval seed) to the
                candidate's per-instance score.
            incumbent_vector: Mapping from instance key to the incumbent's
                per-instance score, over the same instances.

        Returns:
            A :class:`GateVerdict`. The pairing is taken over the sorted
            intersection of the two key sets - the canonical order that keeps the
            wealth a supermartingale. Bets are *never* re-ordered by observed
            effect size.
        """
        cfg = self.config
        common = sorted(set(candidate_vector) & set(incumbent_vector))
        n_pairs = len(common)

        if n_pairs == 0:
            logger.debug(
                "PACE gate: no common instances between candidate and "
                "incumbent; passing through as insufficient_instances."
            )
            return self._pass_through(
                REASON_INSUFFICIENT_INSTANCES, n_pairs=0, n_ties=0
            )

        # Classify every common pair (order-independent counts for reporting).
        diffs = [candidate_vector[k] - incumbent_vector[k] for k in common]
        is_tie = [abs(d) <= cfg.tie_atol for d in diffs]
        n_ties = sum(is_tie)
        n_discordant = n_pairs - n_ties
        n_wins = sum(
            1 for d, tie in zip(diffs, is_tie) if not tie and d > 0.0
        )

        # Not enough discordant signal to run a meaningful evidence test.
        if n_discordant < cfg.min_discordant:
            if cfg.determinism_autodisable:
                logger.debug(
                    "PACE gate: %d discordant pairs (< min_discordant=%d); "
                    "fitness appears deterministic, auto-disabling "
                    "(pass-through).",
                    n_discordant,
                    cfg.min_discordant,
                )
                return self._pass_through(
                    REASON_AUTODISABLED_DETERMINISTIC,
                    n_pairs=n_pairs,
                    n_ties=n_ties,
                    n_discordant=n_discordant,
                    n_wins=n_wins,
                )
            logger.debug(
                "PACE gate: %d discordant pairs (< min_discordant=%d) and "
                "determinism auto-disable off; passing through as "
                "insufficient_instances.",
                n_discordant,
                cfg.min_discordant,
            )
            return self._pass_through(
                REASON_INSUFFICIENT_INSTANCES,
                n_pairs=n_pairs,
                n_ties=n_ties,
                n_discordant=n_discordant,
                n_wins=n_wins,
            )

        # Run the betting martingale in canonical (sorted) order.
        target = cfg.wealth_target
        wealth = 1.0
        peak = 1.0
        trajectory: List[float] = [1.0]
        bets = 0
        for diff, tie in zip(diffs, is_tie):
            if tie:
                continue
            win = 1.0 if diff > 0.0 else 0.0
            wealth *= 1.0 + cfg.lam * (2.0 * win - 1.0)
            bets += 1
            trajectory.append(wealth)
            if wealth > peak:
                peak = wealth
            if wealth >= target:
                return GateVerdict(
                    committed=True,
                    reason=REASON_COMMITTED,
                    n_pairs=n_pairs,
                    n_discordant=n_discordant,
                    n_ties=n_ties,
                    n_wins=n_wins,
                    final_wealth=wealth,
                    peak_wealth=peak,
                    wealth_target=target,
                    wealth_trajectory=trajectory,
                )
            if cfg.max_pairs is not None and bets >= cfg.max_pairs:
                return GateVerdict(
                    committed=False,
                    reason=REASON_REJECTED_BUDGET,
                    n_pairs=n_pairs,
                    n_discordant=n_discordant,
                    n_ties=n_ties,
                    n_wins=n_wins,
                    final_wealth=wealth,
                    peak_wealth=peak,
                    wealth_target=target,
                    wealth_trajectory=trajectory,
                )

        # Consumed all discordant pairs without crossing the wealth target.
        return GateVerdict(
            committed=False,
            reason=REASON_REJECTED_EVIDENCE,
            n_pairs=n_pairs,
            n_discordant=n_discordant,
            n_ties=n_ties,
            n_wins=n_wins,
            final_wealth=wealth,
            peak_wealth=peak,
            wealth_target=target,
            wealth_trajectory=trajectory,
        )

    def _pass_through(
        self,
        reason: str,
        *,
        n_pairs: int,
        n_ties: int,
        n_discordant: int = 0,
        n_wins: int = 0,
    ) -> GateVerdict:
        """Build a pass-through (committed) verdict that admits the candidate.

        Pass-through means "behave like stock ShinkaEvolve", which admits a
        correct candidate - hence ``committed=True`` with an empty betting
        trajectory.
        """
        return GateVerdict(
            committed=True,
            reason=reason,
            n_pairs=n_pairs,
            n_discordant=n_discordant,
            n_ties=n_ties,
            n_wins=n_wins,
            final_wealth=1.0,
            peak_wealth=1.0,
            wealth_target=self.config.wealth_target,
            wealth_trajectory=[1.0],
        )
