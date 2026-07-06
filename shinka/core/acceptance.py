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
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)

# Verdict reason codes (kept as module-level constants so callers and tests
# reference the same strings the gate emits).
REASON_COMMITTED = "committed"
REASON_REJECTED_BUDGET = "rejected_budget"
REASON_REJECTED_EVIDENCE = "rejected_evidence"
REASON_INSUFFICIENT_INSTANCES = "insufficient_instances"
# Zero-evidence rejections (``zero_evidence_action="reject"``): a judged
# comparison that never placed a bet because there were no common instances
# (``rejected_insufficient_instances``) or too few discordant pairs
# (``rejected_no_evidence``).
REASON_REJECTED_INSUFFICIENT = "rejected_insufficient_instances"
REASON_REJECTED_NO_EVIDENCE = "rejected_no_evidence"

VERDICT_REASONS = (
    REASON_COMMITTED,
    REASON_REJECTED_BUDGET,
    REASON_REJECTED_EVIDENCE,
    REASON_INSUFFICIENT_INSTANCES,
    REASON_REJECTED_INSUFFICIENT,
    REASON_REJECTED_NO_EVIDENCE,
)

# Allowed values for ``AcceptanceGateConfig.zero_evidence_action``.
ZERO_EVIDENCE_ACTIONS = ("reject", "pass")

# Allowed values for ``AcceptanceGateConfig.on_missing_vectors``. Governs the
# task-shape failure where a judged comparison has *no* per-instance score
# vectors on either side (the task publishes no per-instance scores at all).
ON_MISSING_VECTORS_ACTIONS = ("error", "reject", "pass")


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
            is treated as a tie (concordant, no bet). A pair is a tie iff
            ``abs(d) <= tie_atol + tie_rtol * max(abs(a), abs(b))``.
        tie_rtol: Relative tie tolerance, scaled by the larger of the two
            per-instance magnitudes (see ``tie_atol``). Must be ``>= 0``.
        zero_evidence_action: What to do when a judged comparison ends without
            any bet placed *despite both sides publishing per-instance vectors* -
            no common instances (``n_pairs == 0``), all valid pairs tied, or
            fewer than ``min_discordant`` discordant pairs. ``"reject"`` (default)
            withholds the candidate (``committed=False``); ``"pass"`` admits it
            (``committed=True``). Either way a warning is logged - a zero-evidence
            resolution is never silent. This governs the *degenerate comparison*
            case; the distinct *missing-vector* (task-shape failure) case is
            governed by ``on_missing_vectors``.
        on_missing_vectors: What to do when a judged comparison has *no*
            per-instance score vectors on *either* side (candidate and incumbent
            are both empty), i.e. the task publishes no per-instance scores at
            all. This is a task-shape failure, not a degenerate comparison, so it
            is handled separately from ``zero_evidence_action``:

            * ``"error"`` (default) raises :class:`RuntimeError` at the first
              occurrence with a remediation message - an enabled gate must never
              be silently inert (accepting everything) nor silently starve
              evolution (rejecting everything) because the wiring is wrong; it
              fails loud so the misconfiguration is caught immediately.
            * ``"reject"`` withholds the candidate (``committed=False``) with a
              ``logger.warning``.
            * ``"pass"`` admits the candidate (``committed=True``) with a
              ``logger.warning``.

            The degenerate-comparison semantics (all-ties / low-discordant with
            vectors *present*) are unaffected by this flag and keep their
            ``zero_evidence_action`` resolution.
        determinism_autodisable: Back-compat flag that now only controls the log
            severity of the all-ties case (every valid pair is a tie, i.e. the
            fitness looks deterministic). When ``True`` (default) the all-ties
            case is treated as ordinary zero evidence (warning + the configured
            ``zero_evidence_action``); when ``False`` it is additionally logged at
            ``error`` level as an unexpected determinism. It no longer forces a
            pass-through commit - admission is governed solely by
            ``zero_evidence_action``.
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
    tie_rtol: float = 1e-9
    zero_evidence_action: str = "reject"
    on_missing_vectors: str = "error"
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
        if self.tie_rtol < 0.0:
            raise ValueError(f"tie_rtol must be >= 0, got {self.tie_rtol}")
        if self.zero_evidence_action not in ZERO_EVIDENCE_ACTIONS:
            raise ValueError(
                "zero_evidence_action must be one of "
                f"{list(ZERO_EVIDENCE_ACTIONS)}, got "
                f"{self.zero_evidence_action!r}"
            )
        if self.on_missing_vectors not in ON_MISSING_VECTORS_ACTIONS:
            raise ValueError(
                "on_missing_vectors must be one of "
                f"{list(ON_MISSING_VECTORS_ACTIONS)}, got "
                f"{self.on_missing_vectors!r}"
            )

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
            evidence crossing (``committed``) and for zero-evidence pass-through
            (``insufficient_instances`` when ``zero_evidence_action="pass"``);
            ``False`` for genuine rejections (``rejected_budget``,
            ``rejected_evidence``) and zero-evidence rejections
            (``rejected_insufficient_instances``, ``rejected_no_evidence``).
        reason: One of :data:`VERDICT_REASONS`.
        n_pairs: Number of *valid* common paired instances
            (``n_discordant + n_ties``); excludes pairs skipped as invalid.
        n_discordant: Number of discordant pairs in the valid common set.
        n_ties: Number of concordant/tied pairs in the valid common set.
        n_wins: Number of discordant pairs the candidate won.
        n_invalid: Number of common pairs skipped because either side was NaN or
            infinite. Skipped pairs are never a win, loss, or tie.
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
    n_invalid: int = 0
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
            n_invalid=int(data.get("n_invalid", 0)),
            final_wealth=float(data["final_wealth"]),
            peak_wealth=float(data["peak_wealth"]),
            wealth_target=float(data["wealth_target"]),
            wealth_trajectory=[float(w) for w in data["wealth_trajectory"]],
        )


class PaceGate:
    """Pure e-process acceptance gate.

    Construct once per run with an :class:`AcceptanceGateConfig`, then call
    :meth:`verdict` per candidate. Each returned verdict is a deterministic
    function of the two score vectors and the config, so verdicts are idempotent
    and safe to recompute on resume. The gate does keep light run-level tallies
    (``judged``/``committed``/``rejected``/``no_evidence``) as a side effect of
    :meth:`verdict`, exposed via :meth:`summary`; these counters do not affect
    any individual verdict.
    """

    def __init__(self, config: AcceptanceGateConfig):
        self.config = config
        self._n_judged = 0
        self._n_committed = 0
        self._n_rejected = 0
        self._n_no_evidence = 0

    def _pass_through(self, reason: str) -> GateVerdict:
        """Build a committed pass-through verdict for a structural (non-bet) case.

        Used for comparisons the e-process never places a bet for but which must
        be admitted unconditionally by the runner - most notably the
        no-incumbent / initial-program case. No wealth is ever at risk, so the
        trajectory is the bare starting ``1.0`` and the pair counts are zero.
        """
        target = self.config.wealth_target
        return GateVerdict(
            committed=True,
            reason=reason,
            n_pairs=0,
            n_discordant=0,
            n_ties=0,
            n_wins=0,
            final_wealth=1.0,
            peak_wealth=1.0,
            wealth_target=target,
            wealth_trajectory=[1.0],
        )

    def summary(self) -> Dict[str, int]:
        """Return run-level tallies accumulated across :meth:`verdict` calls.

        Keys: ``judged`` (total comparisons judged), ``committed`` (genuine
        evidence crossings), ``rejected`` (budget/evidence rejections), and
        ``no_evidence`` (zero-evidence resolutions, whether rejected or passed).
        The three sub-counts partition ``judged``.
        """
        return {
            "judged": self._n_judged,
            "committed": self._n_committed,
            "rejected": self._n_rejected,
            "no_evidence": self._n_no_evidence,
        }

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
        self._n_judged += 1

        # Task-shape failure vs. degenerate comparison: if *neither* side
        # publishes any per-instance score the task itself has no evidence
        # surface, which is a wiring/config error rather than a legitimately
        # inconclusive comparison. Resolve it via ``on_missing_vectors`` so an
        # enabled gate is never silently inert (accept-all) nor silently starves
        # evolution (reject-all). A comparison with vectors *present* that merely
        # produces no bets (all-ties / low-discordant) is handled below by
        # ``zero_evidence_action`` and keeps those semantics unchanged.
        if not candidate_vector and not incumbent_vector:
            return self._missing_vectors_verdict()

        common = sorted(set(candidate_vector) & set(incumbent_vector))

        # Classify every common pair into invalid (NaN/inf on either side),
        # tie, or discordant. Invalid pairs are skipped entirely - never a win,
        # loss, or tie - and counted separately in ``n_invalid``. The remaining
        # valid pairs keep their canonical (sorted-key) order for betting.
        valid: List[tuple[float, bool]] = []  # (diff, is_tie) in canonical order
        n_invalid = 0
        n_ties = 0
        n_wins = 0
        for key in common:
            a = candidate_vector[key]
            b = incumbent_vector[key]
            if not (math.isfinite(a) and math.isfinite(b)):
                n_invalid += 1
                continue
            diff = a - b
            tie = abs(diff) <= cfg.tie_atol + cfg.tie_rtol * max(
                abs(a), abs(b)
            )
            valid.append((diff, tie))
            if tie:
                n_ties += 1
            elif diff > 0.0:
                n_wins += 1
        n_pairs = len(valid)
        n_discordant = n_pairs - n_ties

        # Zero-evidence: either no common valid instances at all, or too few
        # discordant pairs to run a meaningful test. Governed by
        # ``zero_evidence_action`` and never resolved silently.
        if n_pairs == 0 or n_discordant < cfg.min_discordant:
            return self._zero_evidence_verdict(
                no_instances=(n_pairs == 0),
                n_pairs=n_pairs,
                n_ties=n_ties,
                n_discordant=n_discordant,
                n_wins=n_wins,
                n_invalid=n_invalid,
            )

        # Run the betting martingale in canonical (sorted) order.
        target = cfg.wealth_target
        wealth = 1.0
        peak = 1.0
        trajectory: List[float] = [1.0]
        bets = 0
        for diff, tie in valid:
            if tie:
                continue
            win = 1.0 if diff > 0.0 else 0.0
            wealth *= 1.0 + cfg.lam * (2.0 * win - 1.0)
            bets += 1
            trajectory.append(wealth)
            if wealth > peak:
                peak = wealth
            if wealth >= target:
                self._n_committed += 1
                return GateVerdict(
                    committed=True,
                    reason=REASON_COMMITTED,
                    n_pairs=n_pairs,
                    n_discordant=n_discordant,
                    n_ties=n_ties,
                    n_wins=n_wins,
                    n_invalid=n_invalid,
                    final_wealth=wealth,
                    peak_wealth=peak,
                    wealth_target=target,
                    wealth_trajectory=trajectory,
                )
            if cfg.max_pairs is not None and bets >= cfg.max_pairs:
                self._n_rejected += 1
                return GateVerdict(
                    committed=False,
                    reason=REASON_REJECTED_BUDGET,
                    n_pairs=n_pairs,
                    n_discordant=n_discordant,
                    n_ties=n_ties,
                    n_wins=n_wins,
                    n_invalid=n_invalid,
                    final_wealth=wealth,
                    peak_wealth=peak,
                    wealth_target=target,
                    wealth_trajectory=trajectory,
                )

        # Consumed all discordant pairs without crossing the wealth target.
        self._n_rejected += 1
        return GateVerdict(
            committed=False,
            reason=REASON_REJECTED_EVIDENCE,
            n_pairs=n_pairs,
            n_discordant=n_discordant,
            n_ties=n_ties,
            n_wins=n_wins,
            n_invalid=n_invalid,
            final_wealth=wealth,
            peak_wealth=peak,
            wealth_target=target,
            wealth_trajectory=trajectory,
        )

    def _missing_vectors_verdict(self) -> GateVerdict:
        """Resolve a judged comparison with no per-instance vectors on either side.

        This is the task-shape failure: the task publishes no per-instance scores
        at all, so the gate has nothing to bet on. Governed by
        ``config.on_missing_vectors``:

        * ``"error"`` -> raise :class:`RuntimeError` with a remediation message
          (the gate fails loud rather than becoming silently inert or silently
          starving evolution).
        * ``"reject"`` -> withhold (``committed=False``) with a warning.
        * ``"pass"`` -> admit (``committed=True``) with a warning.
        """
        cfg = self.config
        action = cfg.on_missing_vectors

        if action == "error":
            raise RuntimeError(
                "PACE gate: judged comparison has no per-instance score vectors "
                "on either side (candidate and incumbent are both empty), so an "
                "enabled gate can neither accept nor reject on evidence. This is "
                "a task-shape failure: the task publishes no per-instance "
                "scores. Remediation: publish "
                "metrics['private']['instance_scores'] (a mapping from instance "
                "key to per-instance score) from your evaluation, or set "
                "AcceptanceGateConfig.on_missing_vectors explicitly to 'reject' "
                "or 'pass' to opt into the degenerate accept-all/reject-all "
                "behaviour deliberately."
            )

        self._n_no_evidence += 1
        committed = action == "pass"
        reason = (
            REASON_INSUFFICIENT_INSTANCES
            if committed
            else REASON_REJECTED_INSUFFICIENT
        )
        logger.warning(
            "PACE gate: missing per-instance vectors (both candidate and "
            "incumbent empty; task publishes no per-instance scores); "
            "on_missing_vectors=%s -> %s.",
            action,
            "commit" if committed else "reject",
        )
        return GateVerdict(
            committed=committed,
            reason=reason,
            n_pairs=0,
            n_discordant=0,
            n_ties=0,
            n_wins=0,
            n_invalid=0,
            final_wealth=1.0,
            peak_wealth=1.0,
            wealth_target=cfg.wealth_target,
            wealth_trajectory=[1.0],
        )

    def _zero_evidence_verdict(
        self,
        *,
        no_instances: bool,
        n_pairs: int,
        n_ties: int,
        n_discordant: int,
        n_wins: int,
        n_invalid: int,
    ) -> GateVerdict:
        """Resolve a judged comparison that never placed a bet.

        The admission decision is governed solely by
        ``config.zero_evidence_action`` (``"reject"`` -> withhold, ``"pass"`` ->
        admit). The resolution is always logged, never silent. The all-ties case
        (every valid pair a tie) is additionally logged at ``error`` level when
        ``determinism_autodisable`` is ``False`` (unexpected determinism).
        """
        cfg = self.config
        self._n_no_evidence += 1

        if cfg.zero_evidence_action == "reject":
            committed = False
            reason = (
                REASON_REJECTED_INSUFFICIENT
                if no_instances
                else REASON_REJECTED_NO_EVIDENCE
            )
        else:  # "pass"
            committed = True
            reason = REASON_INSUFFICIENT_INSTANCES

        all_ties = n_pairs > 0 and n_discordant == 0
        if all_ties and not cfg.determinism_autodisable:
            logger.error(
                "PACE gate: all %d valid pair(s) are ties but "
                "determinism_autodisable=False (unexpected determinism); "
                "treating as zero evidence (action=%s -> %s).",
                n_pairs,
                cfg.zero_evidence_action,
                "commit" if committed else "reject",
            )
        else:
            logger.warning(
                "PACE gate: zero evidence (reason=%s): n_pairs=%d "
                "n_discordant=%d (min=%d) n_invalid=%d; action=%s -> %s.",
                reason,
                n_pairs,
                n_discordant,
                cfg.min_discordant,
                n_invalid,
                cfg.zero_evidence_action,
                "commit" if committed else "reject",
            )

        return GateVerdict(
            committed=committed,
            reason=reason,
            n_pairs=n_pairs,
            n_discordant=n_discordant,
            n_ties=n_ties,
            n_wins=n_wins,
            n_invalid=n_invalid,
            final_wealth=1.0,
            peak_wealth=1.0,
            wealth_target=cfg.wealth_target,
            wealth_trajectory=[1.0],
        )
