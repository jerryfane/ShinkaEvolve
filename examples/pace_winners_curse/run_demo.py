"""A/B winner's-curse demo for the PACE acceptance gate (no LLM calls).

This is a *synthetic, controlled* experiment. It fabricates a stream of
candidate programs whose TRUE quality is known by construction (a per-instance
success probability ``p_true``) and runs each candidate through two competing
acceptance rules:

* **Arm A - stock greedy**: accept a candidate iff its dev score beats the
  incumbent's dev score (the ubiquitous "did the number go up?" rule).
* **Arm B - PACE-gated**: accept iff the *real* PACE acceptance gate commits
  (an anytime-valid e-process over the McNemar-discordant pairs of a paired
  candidate/incumbent comparison, ``alpha=0.05``, ``lam=0.5``).

Both arms hill-climb an evolving incumbent. The point is to expose the
**winner's curse**: greedy selection on noisy dev scores systematically admits
lucky-noise candidates and inflates the reported best score above its true
quality, whereas the statistically-disciplined gate holds the false-commit rate
near ``alpha`` and keeps the reported score honest.

Mirroring the PACE paper's two-regime design (arXiv 2606.08106 §5.1/§5.2), the
demo runs **both arms under two regimes**:

* **``planted``** - the original setup: most candidates are nulls, a few planted
  rounds are genuine improvements/regressions. Here both arms capture the
  planted improvement, but greedy pays for it with far more false commits. This
  regime demonstrates *equal power + false-commit suppression*.
* **``null_only``** - every candidate is a true null (``p_true == p_base``, the
  incumbent's own true quality) plus a few true regressions; there is **no**
  genuine improvement anywhere in the stream. True quality can never rise, so
  every greedy commit is a false improvement (churn) and its reported-best dev
  score is *pure winner's curse* - a large reported-minus-true gap over a true
  quality pinned at baseline. PACE commits far fewer (its false-commit rate is
  bounded by the anytime-valid ``alpha`` over the sequential comparisons) and
  keeps the reported best much closer to the true baseline.

The champion-gap metric (reported-best dev minus its true quality) only cleanly
separates the arms when *no true improvement dominates* the stream - which is
exactly why both regimes exist: ``planted`` shows the arms are equally powered,
``null_only`` isolates the champion winner's curse.

Crucially, this demo does **not** re-implement the gate. It imports
``shinka.core.acceptance.PaceGate`` and drives the *real*
``ShinkaEvolveRunner._apply_pace_gate`` method against a *real*
``ProgramDatabase`` (with genuine ``gate_passed`` / archive / best-program
semantics), reusing the harness pattern from
``tests/test_pace_gate_integration.py``.

What this DOES show: the effect of the acceptance-gate decision rule, in
isolation, on a ground-truth-labelled candidate stream.
What this does NOT show: full LLM-driven evolution, real task difficulty, or
that the gate helps on any particular real benchmark. It is a mechanism demo.

Run:  ``uv run python examples/pace_winners_curse/run_demo.py``
Outputs (written next to this file): ``results.json`` and ``winners_curse.png``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List

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
    n_seeds: int = 5  # independent replications per arm
    p_base: float = 0.5  # incumbent's true per-instance success probability
    delta: float = 0.30  # true effect size for planted improvements/regressions
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
    from the run seed so the same stream is replayed identically for both arms.

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


def eval_vector(p_true: float, n_dev: int, rng: np.random.Generator) -> Dict[str, float]:
    """One dev evaluation: independent Bernoulli(p_true) over a shared instance set.

    The keys ``inst_00..inst_{n-1}`` are shared across candidate and incumbent
    (the paired-seed eval model: the *same* instance set is scored for both), but
    each system draws its own Bernoulli outcome per instance -- exactly the noise
    that lets a lucky null out-score the incumbent and trigger the winner's curse.
    """
    return {
        f"inst_{i:02d}": float(rng.random() < p_true) for i in range(n_dev)
    }


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
# Arm A: stock greedy (accept iff dev score > incumbent dev score).
# ---------------------------------------------------------------------------


def run_greedy(cfg: DemoConfig, run_seed: int, stream: List[Dict[str, Any]]) -> RunResult:
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
                cand_vec = eval_vector(spec["p_true"], cfg.n_dev, rng)
                dev_evals += cfg.n_dev  # candidate eval; incumbent uses stored score
                cand_dev = float(np.mean(list(cand_vec.values())))

                incumbent = db.get_best_program()
                inc_dev = float(incumbent.combined_score)
                inc_true = float((incumbent.metadata or {}).get("p_true", cfg.p_base))

                # The ubiquitous rule: did the dev number go up?
                if cand_dev > inc_dev:
                    cand = _candidate_program(
                        f"g{spec['round']}", spec, cand_vec, cand_dev, incumbent.id
                    )
                    db.add(cand)  # gate_passed defaults to 1 -> eligible
                    n_commits += 1
                    if spec["p_true"] <= inc_true:
                        false_commits += 1
                    if spec["kind"] == IMPROVEMENT:
                        captured += 1
            return _finish(
                "greedy", run_seed, db, n_commits, false_commits, captured, cfg, dev_evals
            )
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Arm B: PACE-gated (accept iff the REAL PaceGate verdict commits).
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
                cand_vec = eval_vector(spec["p_true"], cfg.n_dev, rng)
                cand_dev = float(np.mean(list(cand_vec.values())))

                incumbent = db.get_best_program()
                inc_true = float((incumbent.metadata or {}).get("p_true", cfg.p_base))
                # Re-evaluate the incumbent on the SAME instance set this round
                # (paired eval) and hand that fresh vector to the gate as the
                # comparison baseline. This is the extra eval cost PACE pays.
                inc_vec = eval_vector(inc_true, cfg.n_dev, rng)
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
            async_db.close() if hasattr(async_db, "close") else None
            db.close()


# ---------------------------------------------------------------------------
# Orchestration + reporting.
# ---------------------------------------------------------------------------


def run_regime(cfg: DemoConfig) -> Dict[str, Any]:
    """Run both arms across all seeds for ONE regime; return a results dict."""
    runs: List[RunResult] = []
    for r in range(cfg.n_seeds):
        run_seed = cfg.seed0 + r
        stream = build_stream(cfg, run_seed)  # identical stream for both arms
        runs.append(run_greedy(cfg, run_seed, stream))
        runs.append(run_pace(cfg, run_seed, stream))
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
    for arm in ("greedy", "pace"):
        arm_runs = [r for r in runs if r.arm == arm]

        def mean(attr: str) -> float:
            return float(np.mean([getattr(r, attr) for r in arm_runs]))

        out[arm] = {
            "reported_best_dev_mean": mean("reported_best_dev"),
            "reported_best_true_mean": mean("reported_best_true"),
            "winners_curse_gap_mean": mean("winners_curse_gap"),
            "n_commits_mean": mean("n_commits"),
            "false_commits_mean": mean("false_commits"),
            "false_commit_rate_mean": mean("false_commit_rate"),
            # Fraction of runs whose shipped best actually reached the optimum.
            "reached_optimum_rate": mean("reached_optimum"),
            "dev_evals_mean": mean("dev_evals"),
        }
    return out


_ARMS = ["greedy", "pace"]
_ARM_LABELS = {"greedy": "Arm A\nstock greedy", "pace": "Arm B\nPACE-gated"}
_ARM_COLORS = {"greedy": "#d1495b", "pace": "#2e86ab"}
_REGIME_TITLES = {
    PLANTED: "Regime: planted (a few genuine improvements exist)",
    NULL_ONLY: "Regime: null_only (no genuine improvement anywhere)",
}


def _series(runs: List[Dict[str, Any]], arm: str, attr: str) -> np.ndarray:
    return np.array([r[attr] for r in runs if r["arm"] == arm], dtype=float)


def _panel_reported_vs_true(ax, runs: List[Dict[str, Any]], title: str) -> None:
    """Reported best dev vs. its true quality, per arm (the inflation panel)."""
    x = np.arange(len(_ARMS))
    width = 0.36
    rep_mean = [_series(runs, a, "reported_best_dev").mean() for a in _ARMS]
    rep_err = [_series(runs, a, "reported_best_dev").std() for a in _ARMS]
    true_mean = [_series(runs, a, "reported_best_true").mean() for a in _ARMS]
    true_err = [_series(runs, a, "reported_best_true").std() for a in _ARMS]

    ax.bar(x - width / 2, rep_mean, width, yerr=rep_err, capsize=4,
           color=[_ARM_COLORS[a] for a in _ARMS], label="reported best dev score")
    ax.bar(x + width / 2, true_mean, width, yerr=true_err, capsize=4,
           color="#bcbcbc", label="TRUE quality (audit)")
    for i, a in enumerate(_ARMS):
        gap = rep_mean[i] - true_mean[i]
        ax.annotate(
            f"gap {gap:+.3f}",
            (x[i], max(rep_mean[i] + rep_err[i], true_mean[i]) + 0.015),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=_ARM_COLORS[a],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_ARM_LABELS[a] for a in _ARMS])
    ax.set_ylabel("per-instance success rate")
    ax.set_ylim(0, 1.18)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)


def _panel_false_commits(ax, runs: List[Dict[str, Any]], title: str) -> None:
    """False commits per run, per arm (audit-labelled churn)."""
    x = np.arange(len(_ARMS))
    fc_mean = [_series(runs, a, "false_commits").mean() for a in _ARMS]
    fc_err = [_series(runs, a, "false_commits").std() for a in _ARMS]
    ax.bar(x, fc_mean, 0.5, yerr=fc_err, capsize=4,
           color=[_ARM_COLORS[a] for a in _ARMS])
    top = max(fc_mean) if max(fc_mean) > 0 else 1.0
    for i, a in enumerate(_ARMS):
        rate = _series(runs, a, "false_commit_rate").mean()
        ax.annotate(
            f"{fc_mean[i]:.1f}  ({rate:.0%} of commits)",
            (x[i], fc_mean[i] + fc_err[i] + top * 0.04 + 0.03),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=_ARM_COLORS[a],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_ARM_LABELS[a] for a in _ARMS])
    ax.set_ylabel("false commits per run (mean)")
    ax.set_ylim(0, top + top * 0.25 + 1)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def make_chart(results: Dict[str, Any], out_path: Path) -> None:
    """2x2 chart: rows = regimes, cols = (reported vs. true; false commits)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.4))
    fig.suptitle(
        "PACE acceptance gate vs. stock greedy: the winner's curse across two "
        "regimes\n(synthetic, ground-truth by construction; "
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
    print("PACE winner's-curse demo complete (two regimes).")
    print(f"  config: T={cfg.n_rounds} rounds, n_dev={cfg.n_dev}, "
          f"{cfg.n_seeds} seeds/arm, delta={cfg.delta}, alpha={cfg.alpha}")
    for regime in REGIMES:
        s = results[regime]["summary"]
        print(f"  == regime: {regime} ==")
        for arm in ("greedy", "pace"):
            a = s[arm]
            print(
                f"    [{arm:>6}] reported={a['reported_best_dev_mean']:.3f} "
                f"true={a['reported_best_true_mean']:.3f} "
                f"gap={a['winners_curse_gap_mean']:+.3f} | "
                f"commits={a['n_commits_mean']:.1f} "
                f"false={a['false_commits_mean']:.1f} "
                f"({a['false_commit_rate_mean']:.0%}) | "
                f"reached_opt={a['reached_optimum_rate']:.0%} | "
                f"dev_evals={a['dev_evals_mean']:.0f}"
            )
    print(f"  wrote: {out_dir / 'results.json'}")
    print(f"  wrote: {out_dir / 'winners_curse.png'}")


if __name__ == "__main__":
    main()
