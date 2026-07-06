"""A/B/C winner's-curse demo for the PACE acceptance gate (no LLM calls).

This is a *synthetic, controlled* experiment. It fabricates a stream of
candidate programs whose TRUE quality is known by construction (a per-instance
success probability ``p_true``) and runs each candidate through three competing
acceptance rules that **decompose** the effect of the PACE gate:

* **``greedy`` (frozen)** - accept iff the candidate's dev score beats the
  incumbent's *frozen stored* dev score (the ubiquitous "did the number go up?"
  rule). That stored score is itself a winner's-cursed maximum, so it is an
  inflated bar.
* **``greedy_reeval`` (re-eval, no gate)** - re-evaluates the incumbent *fresh
  every round* on the same paired instances (the exact eval cost and protocol
  PACE pays), then applies the *greedy* rule: accept iff the candidate's dev
  mean beats the incumbent's *re-evaluated* dev mean. This removes the
  frozen-inflated-score confound but keeps the naive decision rule.
* **``pace`` (re-eval + gate)** - re-evaluates the incumbent fresh every round
  (identically to ``greedy_reeval``) and accepts iff the *real* PACE acceptance
  gate commits (an anytime-valid e-process over the McNemar-discordant pairs of
  the paired comparison, ``alpha=0.05``, ``lam=0.5``).

Because ``greedy_reeval`` and ``pace`` share the *same* eval cost and re-eval
protocol and differ **only** in the decision rule, the contrast between them
isolates the gate's marginal effect. The three-arm decomposition reads as:

    greedy  --[remove frozen-score confound]-->  greedy_reeval
    greedy_reeval  --[swap greedy rule for the e-process gate]-->  pace

So ``pace`` vs ``greedy_reeval`` is the clean, deconfounded measure of what the
gate itself buys; ``greedy`` vs ``greedy_reeval`` shows what re-evaluation alone
does (spoiler: on its own it does not tame false commits - it can make churn
worse, because a re-evaluated incumbent regresses toward its true mean and is
easier to beat by noise).

**What ``alpha`` does and does not promise (read this before quoting a rate).**
The gate's e-process gives an anytime-valid guarantee **per comparison**: by
Ville's inequality the probability that a *single* null candidate ever commits
is bounded by ``alpha``. It does **not** bound the number of false commits over
a whole run. A run streams ~``T`` sequential comparisons, so the *expected*
false commits under the null is roughly ``(per-comparison false-commit rate) x
T`` - a per-comparison bound, multiplied by the number of comparisons. There is
no run-level (family-wise) control here; that is the job of the wealth-ledger
layer (a global betting budget across comparisons), which is **out of scope**
for this mechanism demo. Do not read the per-run false-commit count as "near
``alpha``": it is an accumulation of many per-comparison bets.

Mirroring the PACE paper's two-regime design (arXiv 2606.08106 §5.1/§5.2), the
demo runs **all three arms under two regimes**:

* **``planted``** - the original setup: most candidates are nulls, a few planted
  rounds are genuine improvements/regressions. Here all arms can capture the
  planted improvement, but the greedy rules pay for it with far more false
  commits. This regime demonstrates *power + false-commit suppression*.
* **``null_only``** - every candidate is a true null (``p_true == p_base``, the
  incumbent's own true quality) plus a few true regressions; there is **no**
  genuine improvement anywhere in the stream. True quality can never rise, so
  every greedy commit is a false improvement (churn) and its reported-best dev
  score is *pure winner's curse* - a large reported-minus-true gap over a true
  quality pinned at baseline. PACE commits far fewer and keeps the reported best
  much closer to the true baseline.

**Real paired difficulty.** Dev instances are heterogeneous: each round draws a
per-instance difficulty offset ``e_i ~ Normal(0, sigma_inst)`` (clipped so the
success probability stays in ``[0.05, 0.95]``) that is **shared** by the
candidate and the incumbent on the same instance index. Given that shared
difficulty, the two systems draw *independent* Bernoulli outcomes. Because a
hard instance depresses both systems together, the paired difference has lower
variance than an unpaired one - which is exactly the correlation McNemar pairing
exploits, so the gate's pairing is *genuinely* variance-reducing here rather
than a formality.

Crucially, this demo does **not** re-implement the gate. It imports
``shinka.core.acceptance.PaceGate`` and drives the *real*
``ShinkaEvolveRunner._apply_pace_gate`` method against a *real*
``ProgramDatabase`` (with genuine ``gate_passed`` / archive / best-program
semantics), reusing the harness pattern from
``tests/test_pace_gate_integration.py``.

What this DOES show: the effect of the acceptance-gate decision rule, in
isolation (deconfounded via the third arm), on a ground-truth-labelled candidate
stream.
What this does NOT show: full LLM-driven evolution, real task difficulty, or that
the gate helps on any particular real benchmark. It is a mechanism demo.

Run:  ``uv run python examples/pace_winners_curse/run_demo.py``
Outputs (written next to this file): ``results.json`` and ``winners_curse.png``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from shinka.core import AcceptanceGateConfig, PaceGate
from shinka.core.async_runner import (
    PACE_INSTANCE_KEY,
    ShinkaEvolveRunner,
)
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase


# ---------------------------------------------------------------------------
# Real-gate harness: reuse the runner's gate methods verbatim (no reimplementation).
# ---------------------------------------------------------------------------


class _GateHarness:
    """Minimal object exposing the runner's PACE gate methods verbatim.

    Mirrors ``tests/test_pace_gate_integration.py``: it binds the real
    ``ShinkaEvolveRunner`` gate methods onto a light object that provides only
    the three attributes those methods touch (``pace_gate``, ``async_db``,
    ``results_dir``), so the demo exercises the production gate code path.
    """

    _pace_instance_vector = ShinkaEvolveRunner._pace_instance_vector
    _pace_pass_through = ShinkaEvolveRunner._pace_pass_through
    _append_pace_log = ShinkaEvolveRunner._append_pace_log
    _apply_pace_gate = ShinkaEvolveRunner._apply_pace_gate

    def __init__(self, pace_gate, async_db, results_dir):
        self.pace_gate = pace_gate
        self.async_db = async_db
        self.results_dir = results_dir


# ---------------------------------------------------------------------------
# Experiment configuration.
# ---------------------------------------------------------------------------


NULL = "null"
IMPROVEMENT = "improvement"
REGRESSION = "regression"

# Regimes (mirroring the PACE paper's two-regime design, arXiv 2606.08106
# §5.1/§5.2). ``planted`` seeds the stream with a few genuine improvements;
# ``null_only`` has NO genuine improvement anywhere (every candidate is a true
# null, plus a few true regressions).
PLANTED = "planted"
NULL_ONLY = "null_only"
REGIMES = (PLANTED, NULL_ONLY)


@dataclass
class DemoConfig:
    """All knobs for the synthetic experiment (defaults per the design doc)."""

    regime: str = PLANTED  # "planted" | "null_only"
    n_dev: int = 40  # paired dev instances per comparison
    n_rounds: int = 60  # proposal rounds per run (T)
    n_seeds: int = 20  # independent replications per arm
    p_base: float = 0.5  # incumbent's true per-instance success probability
    delta: float = 0.30  # true effect size for planted improvements/regressions
    sigma_inst: float = 0.15  # per-instance difficulty std (paired heterogeneity)
    n_improvements: int = 3  # planted true improvements (ignored in null_only)
    n_regressions: int = 3  # planted true regressions per run
    alpha: float = 0.05  # anytime-valid significance level (gate)
    lam: float = 0.5  # betting fraction (gate)
    seed0: int = 20240  # base seed; run r uses seed0 + r

    def improvement_p(self) -> float:
        return min(1.0, self.p_base + self.delta)

    def regression_p(self) -> float:
        return max(0.0, self.p_base - self.delta)

    def planted_improvements(self) -> int:
        """Effective number of planted improvements for this regime.

        ``null_only`` has *no* genuine improvement by construction, so it forces
        zero regardless of ``n_improvements`` (the ``planted`` regime uses the
        knob verbatim). Keeps the candidate stream honest to the regime label.
        """
        return 0 if self.regime == NULL_ONLY else self.n_improvements


# ---------------------------------------------------------------------------
# Ground-truth candidate stream + paired Bernoulli dev evaluation.
# ---------------------------------------------------------------------------


def build_stream(cfg: DemoConfig, run_seed: int) -> List[Dict[str, Any]]:
    """Return the per-round candidate spec list for one run.

    Each candidate carries a TRUE per-instance success probability ``p_true``
    (ground truth by construction) and a ``kind`` label. Most rounds are nulls
    (``p_true == p_base``); a few planted rounds are true improvements
    (``+delta``) or regressions (``-delta``). Planted round positions are drawn
    from the run seed so the same stream is replayed identically for all arms.

    In the ``null_only`` regime there are **no** improvements (every candidate is
    a true null plus a few true regressions), so ``planted_improvements()``
    returns zero. The ``planted`` regime is byte-identical to the original
    single-regime stream.
    """
    rng = np.random.default_rng(run_seed)
    n_improvements = cfg.planted_improvements()
    kinds = [NULL] * cfg.n_rounds
    n_planted = n_improvements + cfg.n_regressions
    positions = rng.choice(cfg.n_rounds, size=n_planted, replace=False)
    for i, pos in enumerate(positions):
        kinds[pos] = IMPROVEMENT if i < n_improvements else REGRESSION

    stream = []
    for r, kind in enumerate(kinds):
        if kind == IMPROVEMENT:
            p_true = cfg.improvement_p()
        elif kind == REGRESSION:
            p_true = cfg.regression_p()
        else:
            p_true = cfg.p_base
        stream.append({"round": r, "kind": kind, "p_true": p_true})
    return stream


def eval_pair(
    cand_p: float,
    inc_p: float,
    n_dev: int,
    rng: np.random.Generator,
    sigma_inst: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """One paired dev evaluation of candidate + incumbent on a shared instance set.

    Each instance ``i`` draws a *shared* difficulty offset
    ``e_i ~ Normal(0, sigma_inst)``; the per-instance success probability is
    ``clip(p_true + e_i, 0.05, 0.95)`` for whichever system, so a hard instance
    (negative ``e_i``) depresses candidate *and* incumbent together. Given that
    shared difficulty the two systems draw **independent** Bernoulli outcomes.

    The shared difficulty is what makes the McNemar pairing genuinely
    variance-reducing: the paired candidate-minus-incumbent difference has lower
    variance than an unpaired one, so the gate's discordant-pair evidence is
    cleaner. The independent per-system noise on top is exactly what still lets a
    lucky null out-score the incumbent and trigger the winner's curse.

    The offsets are drawn first, then the candidate's Bernoulli draws, then the
    incumbent's, all off a single per-round generator - so the candidate vector
    is *identical across arms* for a given ``(run_seed, round)`` (it depends only
    on the offsets and the candidate draws, never on ``inc_p``), keeping the
    three arms strictly comparable on the same candidate stream.
    """
    offsets = rng.normal(0.0, sigma_inst, size=n_dev)
    cand_p_vec = np.clip(cand_p + offsets, 0.05, 0.95)
    inc_p_vec = np.clip(inc_p + offsets, 0.05, 0.95)
    cand_draws = rng.random(n_dev)
    inc_draws = rng.random(n_dev)
    cand = {
        f"inst_{i:02d}": float(cand_draws[i] < cand_p_vec[i]) for i in range(n_dev)
    }
    inc = {
        f"inst_{i:02d}": float(inc_draws[i] < inc_p_vec[i]) for i in range(n_dev)
    }
    return cand, inc


# ---------------------------------------------------------------------------
# Program construction for the real ProgramDatabase.
# ---------------------------------------------------------------------------


def _make_db(tmpdir: str) -> ProgramDatabase:
    return ProgramDatabase(
        config=DatabaseConfig(
            db_path=str(Path(tmpdir) / "demo.db"),
            num_islands=1,
            archive_size=200,
            parent_selection_strategy="power_law",
        ),
        embedding_model="",
    )


def _candidate_program(
    pid: str,
    spec: Dict[str, Any],
    vector: Dict[str, float],
    dev_score: float,
    parent_id: str,
) -> Program:
    return Program(
        id=pid,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=dev_score,
        generation=spec["round"] + 1,
        island_idx=0,
        parent_id=parent_id,
        private_metrics={PACE_INSTANCE_KEY: dict(vector)},
        metadata={"p_true": spec["p_true"], "kind": spec["kind"]},
    )


# ---------------------------------------------------------------------------
# Metrics accumulator.
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    arm: str
    run_seed: int
    reported_best_dev: float  # dev score of the final reported best
    reported_best_true: float  # ITS true quality (p_true) -- the audit label
    winners_curse_gap: float  # reported_best_dev - reported_best_true
    n_commits: int
    false_commits: int  # committed with p_true <= incumbent's (audit-labelled)
    false_commit_rate: float  # false_commits / max(n_commits, 1)
    improvements_committed: int  # planted improvements accepted at their round
    reached_optimum: int  # 1 iff the reported best's true quality == improvement_p
    dev_evals: int  # total per-instance dev evaluations consumed

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finish(
    arm: str,
    run_seed: int,
    db: ProgramDatabase,
    n_commits: int,
    false_commits: int,
    improvements_committed: int,
    cfg: DemoConfig,
    dev_evals: int,
) -> RunResult:
    best = db.get_best_program()
    reported_dev = float(best.combined_score)
    reported_true = float((best.metadata or {}).get("p_true", cfg.p_base))
    # Capture = did the arm's shipped best actually reach the planted optimum?
    # (An evolving incumbent makes "committed/planted" misleading: once the
    # incumbent reaches p_base+delta, a later planted improvement is genuinely a
    # null vs. the incumbent, so re-committing it would be a false commit, not a
    # capture. Reaching the optimum is the honest capture signal.)
    reached = int(reported_true >= cfg.improvement_p() - 1e-9)
    return RunResult(
        arm=arm,
        run_seed=run_seed,
        reported_best_dev=reported_dev,
        reported_best_true=reported_true,
        winners_curse_gap=reported_dev - reported_true,
        n_commits=n_commits,
        false_commits=false_commits,
        false_commit_rate=false_commits / max(n_commits, 1),
        improvements_committed=improvements_committed,
        reached_optimum=reached,
        dev_evals=dev_evals,
    )


# ---------------------------------------------------------------------------
# Greedy arms: frozen (``greedy``) and re-evaluated (``greedy_reeval``).
# ---------------------------------------------------------------------------


def run_greedy_variant(
    cfg: DemoConfig,
    run_seed: int,
    stream: List[Dict[str, Any]],
    *,
    reeval: bool,
) -> RunResult:
    """Greedy hill-climb using the "did the dev number go up?" rule.

    ``reeval=False`` (arm ``greedy``): compares the candidate against the
    incumbent's *frozen stored* dev score (1x eval cost/round). ``reeval=True``
    (arm ``greedy_reeval``): re-evaluates the incumbent fresh on this round's
    paired instances - the same 2x eval cost and protocol PACE pays - and
    compares against that re-evaluated mean. Both use the naive greedy accept
    rule; only the baseline (frozen vs. re-evaluated) differs, which is exactly
    the confound the third arm removes.
    """
    arm = "greedy_reeval" if reeval else "greedy"
    prefix = "r" if reeval else "g"
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            root = _candidate_program(
                "p0", {"round": -1, "kind": NULL, "p_true": cfg.p_base},
                {}, cfg.p_base, parent_id=None,
            )
            db.add(root)  # root incumbent, dev == p_base by construction

            n_commits = false_commits = captured = dev_evals = 0
            for spec in stream:
                rng = np.random.default_rng((run_seed, spec["round"]))
                incumbent = db.get_best_program()
                inc_true = float((incumbent.metadata or {}).get("p_true", cfg.p_base))
                # Candidate vector is identical across all arms (see eval_pair);
                # inc_vec is the paired incumbent re-eval, used only when reeval.
                cand_vec, inc_vec = eval_pair(
                    spec["p_true"], inc_true, cfg.n_dev, rng, cfg.sigma_inst
                )
                cand_dev = float(np.mean(list(cand_vec.values())))
                if reeval:
                    inc_dev = float(np.mean(list(inc_vec.values())))
                    dev_evals += 2 * cfg.n_dev  # candidate + paired incumbent re-eval
                else:
                    inc_dev = float(incumbent.combined_score)  # frozen stored score
                    dev_evals += cfg.n_dev  # candidate only; incumbent score reused

                # The ubiquitous rule: did the dev number go up?
                if cand_dev > inc_dev:
                    cand = _candidate_program(
                        f"{prefix}{spec['round']}", spec, cand_vec, cand_dev,
                        incumbent.id,
                    )
                    db.add(cand)  # gate_passed defaults to 1 -> eligible
                    n_commits += 1
                    if spec["p_true"] <= inc_true:
                        false_commits += 1
                    if spec["kind"] == IMPROVEMENT:
                        captured += 1
            return _finish(
                arm, run_seed, db, n_commits, false_commits, captured, cfg, dev_evals
            )
        finally:
            db.close()


# ---------------------------------------------------------------------------
# PACE arm: re-eval + real PaceGate verdict.
# ---------------------------------------------------------------------------


def run_pace(cfg: DemoConfig, run_seed: int, stream: List[Dict[str, Any]]) -> RunResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        gate = PaceGate(
            AcceptanceGateConfig(
                enabled=True, alpha=cfg.alpha, lam=cfg.lam,
                # Paired null candidates can legitimately produce zero discordant
                # pairs; treat that as "no evidence to promote" -> reject.
                zero_evidence_action="reject",
                on_missing_vectors="reject",
            )
        )
        harness = _GateHarness(gate, async_db, tmpdir)
        try:
            root = _candidate_program(
                "p0", {"round": -1, "kind": NULL, "p_true": cfg.p_base},
                {}, cfg.p_base, parent_id=None,
            )
            db.add(root, defer_maintenance=True)
            asyncio.run(harness._apply_pace_gate(root, parent=None))  # auto-pass
            db.run_post_add_maintenance(root)

            n_commits = false_commits = captured = dev_evals = 0
            for spec in stream:
                rng = np.random.default_rng((run_seed, spec["round"]))
                incumbent = db.get_best_program()
                inc_true = float((incumbent.metadata or {}).get("p_true", cfg.p_base))
                # Re-evaluate the incumbent on the SAME paired instance set this
                # round (shared difficulty offsets) and hand that fresh vector to
                # the gate as the comparison baseline -- the extra eval cost PACE
                # pays, identical to greedy_reeval's.
                cand_vec, inc_vec = eval_pair(
                    spec["p_true"], inc_true, cfg.n_dev, rng, cfg.sigma_inst
                )
                cand_dev = float(np.mean(list(cand_vec.values())))
                incumbent.private_metrics = {PACE_INSTANCE_KEY: dict(inc_vec)}
                dev_evals += 2 * cfg.n_dev  # candidate + paired incumbent re-eval

                cand = _candidate_program(
                    f"c{spec['round']}", spec, cand_vec, cand_dev, incumbent.id
                )
                db.add(cand, defer_maintenance=True)
                verdict = asyncio.run(
                    harness._apply_pace_gate(cand, parent=incumbent)
                )
                db.run_post_add_maintenance(cand)

                if verdict is not None and verdict.committed:
                    n_commits += 1
                    if spec["p_true"] <= inc_true:
                        false_commits += 1
                    if spec["kind"] == IMPROVEMENT:
                        captured += 1
            return _finish(
                "pace", run_seed, db, n_commits, false_commits, captured, cfg, dev_evals
            )
        finally:
            # Real executor shutdown: AsyncProgramDatabase exposes close_async()
            # (there is no sync close()); await it so the thread pools are joined.
            asyncio.run(async_db.close_async())
            db.close()


# ---------------------------------------------------------------------------
# Orchestration + reporting.
# ---------------------------------------------------------------------------


ARMS = ["greedy", "greedy_reeval", "pace"]


def _run_arm(arm: str, cfg: DemoConfig, run_seed: int, stream) -> RunResult:
    if arm == "greedy":
        return run_greedy_variant(cfg, run_seed, stream, reeval=False)
    if arm == "greedy_reeval":
        return run_greedy_variant(cfg, run_seed, stream, reeval=True)
    return run_pace(cfg, run_seed, stream)


def run_regime(cfg: DemoConfig) -> Dict[str, Any]:
    """Run all three arms across all seeds for ONE regime; return a results dict."""
    runs: List[RunResult] = []
    for r in range(cfg.n_seeds):
        run_seed = cfg.seed0 + r
        stream = build_stream(cfg, run_seed)  # identical stream for all arms
        for arm in ARMS:
            runs.append(_run_arm(arm, cfg, run_seed, stream))
    return {
        "config": asdict(cfg),
        "runs": [rr.as_dict() for rr in runs],
        "summary": _summarise(runs),
    }


def run_experiment(base_cfg: DemoConfig) -> Dict[str, Any]:
    """Run BOTH regimes; return a JSON-serialisable dict keyed by regime.

    The shared knobs (``n_dev``, ``n_rounds``, ``n_seeds``, ...) come from
    ``base_cfg``; each regime is a copy with its ``regime`` label set (and, for
    ``null_only``, ``n_improvements`` zeroed so the recorded config is honest to
    the "no genuine improvement" construction the stream actually uses).
    """
    out: Dict[str, Any] = {}
    for regime in REGIMES:
        overrides = {"regime": regime}
        if regime == NULL_ONLY:
            overrides["n_improvements"] = 0
        out[regime] = run_regime(replace(base_cfg, **overrides))
    return out


def _summarise(runs: List[RunResult]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in ARMS:
        arm_runs = [r for r in runs if r.arm == arm]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(r, attr) for r in arm_runs]))

        def std(attr: str) -> float:
            return float(np.std([getattr(r, attr) for r in arm_runs]))

        total_commits = int(sum(r.n_commits for r in arm_runs))
        total_false = int(sum(r.false_commits for r in arm_runs))
        out[arm] = {
            "reported_best_dev_mean": mean("reported_best_dev"),
            "reported_best_dev_std": std("reported_best_dev"),
            "reported_best_true_mean": mean("reported_best_true"),
            "winners_curse_gap_mean": mean("winners_curse_gap"),
            "winners_curse_gap_std": std("winners_curse_gap"),
            "n_commits_mean": mean("n_commits"),
            "n_commits_std": std("n_commits"),
            "false_commits_mean": mean("false_commits"),
            "false_commits_std": std("false_commits"),
            # Per-run rate averaged over seeds (kept for reference); the honest
            # cross-seed rate is the POOLED one below (total false / total
            # commits), which is not distorted by zero-commit seeds.
            "false_commit_rate_mean": mean("false_commit_rate"),
            "false_commits_total": total_false,
            "n_commits_total": total_commits,
            "false_commit_rate_pooled": total_false / max(total_commits, 1),
            # Fraction of runs whose shipped best actually reached the optimum.
            "reached_optimum_rate": mean("reached_optimum"),
            "dev_evals_mean": mean("dev_evals"),
        }
    return out


# ---------------------------------------------------------------------------
# Chart.
# ---------------------------------------------------------------------------


_ARM_LABELS = {
    "greedy": "greedy\n(frozen)",
    "greedy_reeval": "greedy_reeval\n(re-eval, no gate)",
    "pace": "PACE\n(re-eval + gate)",
}
_ARM_LEGEND = {
    "greedy": "greedy (frozen)",
    "greedy_reeval": "greedy_reeval (re-eval, no gate)",
    "pace": "PACE (re-eval + gate)",
}
_ARM_COLORS = {"greedy": "#d1495b", "greedy_reeval": "#edae49", "pace": "#2e86ab"}
_TRUE_COLOR = "#8d99ae"
_REGIME_TITLES = {
    PLANTED: "Regime: planted (a few genuine improvements exist)",
    NULL_ONLY: "Regime: null_only (no genuine improvement anywhere)",
}


def _series(runs: List[Dict[str, Any]], arm: str, attr: str) -> np.ndarray:
    return np.array([r[attr] for r in runs if r["arm"] == arm], dtype=float)


def _pooled_rate(runs: List[Dict[str, Any]], arm: str) -> float:
    """Pooled false-commit rate: total false commits / total commits over seeds.

    Unlike the mean of per-run rates, this is undistorted by zero-commit seeds
    (whose per-run rate is defined as 0), so it is the honest cross-seed rate.
    """
    total_commits = float(_series(runs, arm, "n_commits").sum())
    total_false = float(_series(runs, arm, "false_commits").sum())
    return total_false / max(total_commits, 1.0)


def _panel_reported_vs_true(ax, runs: List[Dict[str, Any]], title: str) -> None:
    """Reported best dev vs. its true quality, per arm (the inflation panel)."""
    from matplotlib.patches import Patch

    x = np.arange(len(ARMS))
    width = 0.38
    rep_mean = [_series(runs, a, "reported_best_dev").mean() for a in ARMS]
    rep_err = [_series(runs, a, "reported_best_dev").std() for a in ARMS]
    true_mean = [_series(runs, a, "reported_best_true").mean() for a in ARMS]
    true_err = [_series(runs, a, "reported_best_true").std() for a in ARMS]

    ax.bar(x - width / 2, rep_mean, width, yerr=rep_err, capsize=4,
           color=[_ARM_COLORS[a] for a in ARMS])
    ax.bar(x + width / 2, true_mean, width, yerr=true_err, capsize=4,
           color=_TRUE_COLOR)
    for i, a in enumerate(ARMS):
        gap = rep_mean[i] - true_mean[i]
        ax.annotate(
            f"gap {gap:+.3f}",
            (x[i], max(rep_mean[i] + rep_err[i], true_mean[i]) + 0.015),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=_ARM_COLORS[a],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_ARM_LABELS[a] for a in ARMS], fontsize=8)
    ax.set_ylabel("per-instance success rate")
    ax.set_ylim(0, 1.25)
    ax.set_title(title, fontsize=10)
    # Explicit, correctly-coloured swatches: one per arm (the reported bars) plus
    # the shared grey "true quality" bar. (A single ``color=[list]`` bar call
    # would legend as one wrong-coloured swatch, so we build handles by hand.)
    handles = [Patch(color=_ARM_COLORS[a], label=f"reported best: {_ARM_LEGEND[a]}")
               for a in ARMS]
    handles.append(Patch(color=_TRUE_COLOR, label="TRUE quality (audit)"))
    ax.legend(handles=handles, fontsize=7, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)


def _panel_false_commits(ax, runs: List[Dict[str, Any]], title: str) -> None:
    """False commits per run, per arm (audit-labelled churn)."""
    from matplotlib.patches import Patch

    x = np.arange(len(ARMS))
    fc_mean = [_series(runs, a, "false_commits").mean() for a in ARMS]
    fc_err = [_series(runs, a, "false_commits").std() for a in ARMS]
    ax.bar(x, fc_mean, 0.6, yerr=fc_err, capsize=4,
           color=[_ARM_COLORS[a] for a in ARMS])
    top = max(fc_mean) if max(fc_mean) > 0 else 1.0
    for i, a in enumerate(ARMS):
        pooled = _pooled_rate(runs, a)
        ax.annotate(
            f"{fc_mean[i]:.1f}/run\n(pooled {pooled:.0%})",
            (x[i], fc_mean[i] + fc_err[i] + top * 0.04 + 0.03),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=_ARM_COLORS[a],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_ARM_LABELS[a] for a in ARMS], fontsize=8)
    ax.set_ylabel("false commits per run (mean)")
    ax.set_ylim(0, top + top * 0.35 + 1)
    ax.set_title(title, fontsize=10)
    handles = [Patch(color=_ARM_COLORS[a], label=_ARM_LEGEND[a]) for a in ARMS]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def make_chart(results: Dict[str, Any], out_path: Path) -> None:
    """2x2 chart: rows = regimes, cols = (reported vs. true; false commits)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    fig.suptitle(
        "PACE acceptance gate vs. greedy: the winner's curse, deconfounded "
        "(3 arms) across two regimes\n(synthetic, ground-truth by construction; "
        "arXiv 2606.08106 §5.1/§5.2)",
        fontsize=12, fontweight="bold",
    )

    for row, regime in enumerate(REGIMES):
        runs = results[regime]["runs"]
        rt = _REGIME_TITLES[regime]
        _panel_reported_vs_true(
            axes[row, 0], runs,
            f"{rt}\n(1) Reported best vs. true quality (inflation = winner's curse)",
        )
        _panel_false_commits(
            axes[row, 1], runs,
            f"{rt}\n(2) False commits (committed with p_true <= incumbent)",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_demo(cfg: DemoConfig, out_dir: Path) -> Dict[str, Any]:
    """Run BOTH regimes and write ``results.json`` (keyed by regime) +
    ``winners_curse.png`` (a 2x2 figure: rows = regimes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_experiment(cfg)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    make_chart(results, out_dir / "winners_curse.png")
    return results


def main() -> None:
    cfg = DemoConfig()
    out_dir = Path(__file__).resolve().parent
    results = run_demo(cfg, out_dir)
    print("PACE winner's-curse demo complete (three arms, two regimes).")
    print(f"  config: T={cfg.n_rounds} rounds, n_dev={cfg.n_dev}, "
          f"{cfg.n_seeds} seeds/arm, delta={cfg.delta}, "
          f"sigma_inst={cfg.sigma_inst}, alpha={cfg.alpha}")
    for regime in REGIMES:
        s = results[regime]["summary"]
        print(f"  == regime: {regime} ==")
        for arm in ARMS:
            a = s[arm]
            print(
                f"    [{arm:>13}] reported={a['reported_best_dev_mean']:.3f}"
                f"±{a['reported_best_dev_std']:.3f} "
                f"true={a['reported_best_true_mean']:.3f} "
                f"gap={a['winners_curse_gap_mean']:+.3f} | "
                f"commits={a['n_commits_mean']:.1f} "
                f"false={a['false_commits_mean']:.1f}"
                f"±{a['false_commits_std']:.1f} "
                f"(pooled {a['false_commit_rate_pooled']:.0%}) | "
                f"reached_opt={a['reached_optimum_rate']:.0%} | "
                f"dev_evals={a['dev_evals_mean']:.0f}"
            )
    print(f"  wrote: {out_dir / 'results.json'}")
    print(f"  wrote: {out_dir / 'winners_curse.png'}")


if __name__ == "__main__":
    main()
