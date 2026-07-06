"""Integration tests for the PACE acceptance gate runner wiring.

These exercise the exact ``ShinkaEvolveRunner`` gate methods
(:meth:`_apply_pace_gate` and helpers) against a real ``ProgramDatabase`` /
``AsyncProgramDatabase`` and the real DB side-effect doors (archive admission,
best-program tracking, parent sampling). They stop short of spinning up a full
evolution loop (no LLM, no job scheduler) by binding the runner's gate methods
onto a light harness that provides only the three attributes those methods
touch: ``pace_gate``, ``async_db`` and ``results_dir``.

Coverage mirrors the design doc's integration plan:

* gate OFF -> pass-through: no verdict, no ``gate_passed`` flip, no
  ``pace_log.jsonl``, and a lucky candidate is admitted to the archive exactly
  as in stock ShinkaEvolve;
* gate ON, planted lucky-noise candidate -> rejected, withheld from the archive,
  the best-program lookup, and the parent pool;
* gate ON, planted true improvement -> committed and archived;
* initial (parent-less) program -> auto-pass;
* resume idempotence -> a persisted verdict is reused, never re-judged.
"""

import asyncio
import tempfile
from pathlib import Path

from shinka.core import AcceptanceGateConfig, PaceGate
from shinka.core.async_runner import (
    PACE_INSTANCE_KEY,
    PACE_REASON_NO_INCUMBENT,
    ShinkaEvolveRunner,
)
from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.parents import CombinedParentSelector


# ---------------------------------------------------------------------------
# Harness: reuse the real runner gate methods without building a full runner.
# ---------------------------------------------------------------------------


class _GateHarness:
    """Minimal object exposing the runner's PACE gate methods verbatim."""

    _pace_instance_vector = ShinkaEvolveRunner._pace_instance_vector
    _pace_pass_through = ShinkaEvolveRunner._pace_pass_through
    _append_pace_log = ShinkaEvolveRunner._append_pace_log
    _apply_pace_gate = ShinkaEvolveRunner._apply_pace_gate
    _pace_gate_bandit = ShinkaEvolveRunner._pace_gate_bandit
    _pace_gate_scratchpad = ShinkaEvolveRunner._pace_gate_scratchpad

    def __init__(self, pace_gate, async_db, results_dir):
        self.pace_gate = pace_gate
        self.async_db = async_db
        self.results_dir = results_dir


def _make_db(tmpdir: str, name: str = "pace.db") -> ProgramDatabase:
    return ProgramDatabase(
        config=DatabaseConfig(
            db_path=str(Path(tmpdir) / name),
            num_islands=1,
            archive_size=40,
            parent_selection_strategy="power_law",
        ),
        embedding_model="",
    )


def _config(enabled: bool = True, **kwargs) -> AcceptanceGateConfig:
    return AcceptanceGateConfig(
        enabled=enabled, alpha=0.05, lam=0.5, **kwargs
    )


def _instance_program(
    program_id: str,
    scores: dict,
    *,
    parent_id=None,
    generation: int = 1,
    combined_score: float = 1.0,
) -> Program:
    return Program(
        id=program_id,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=combined_score,
        generation=generation,
        island_idx=0,
        parent_id=parent_id,
        private_metrics={PACE_INSTANCE_KEY: dict(scores)},
    )


def _selector(db: ProgramDatabase) -> CombinedParentSelector:
    return CombinedParentSelector(
        cursor=db.cursor,
        conn=db.conn,
        config=db.config,
        get_program_func=db.get,
        best_program_id=db.best_program_id,
        get_best_program_func=db.get_best_program,
    )


# Ten paired instances. The incumbent (parent) scores 0.0 everywhere.
_SEEDS = [f"seed_{i:02d}" for i in range(10)]
_INCUMBENT = {s: 0.0 for s in _SEEDS}
# A candidate that wins every discordant pair: decisive true improvement.
_WINNER = {s: 1.0 for s in _SEEDS}
# A lucky-noise candidate: a couple of big spikes (high mean) but it loses the
# paired McNemar test on the majority of instances.
_LUCKY = {s: (5.0 if i < 2 else -1.0) for i, s in enumerate(_SEEDS)}


# ---------------------------------------------------------------------------
# Gate OFF -> byte-identical to stock.
# ---------------------------------------------------------------------------


def test_gate_off_is_passthrough():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            cand = _instance_program("lucky", _LUCKY, parent_id="p0")
            db.add(cand, defer_maintenance=True)

            harness = _GateHarness(None, async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            # Gate off: no verdict, no flag flip, no log file, no metadata.
            assert verdict is None
            assert cand.gate_passed == 1
            assert (cand.metadata or {}).get("pace") is None
            assert not (Path(tmpdir) / "pace_log.jsonl").exists()

            # Door 1 admits the lucky candidate exactly like stock.
            db.run_post_add_maintenance(cand)
            archive_ids = {p.id for p in db._get_archive_programs()}
            assert "lucky" in archive_ids
        finally:
            async_db.close() if hasattr(async_db, "close") else None
            db.close()


# ---------------------------------------------------------------------------
# Initial (parent-less) program auto-passes.
# ---------------------------------------------------------------------------


def test_initial_program_auto_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            init = _instance_program(
                "p0", _INCUMBENT, parent_id=None, generation=0
            )
            db.add(init, defer_maintenance=True)

            harness = _GateHarness(PaceGate(_config()), async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(init))

            assert verdict is not None
            assert verdict.committed
            assert verdict.reason == PACE_REASON_NO_INCUMBENT
            assert init.gate_passed == 1
            assert init.metadata["pace"]["reason"] == PACE_REASON_NO_INCUMBENT
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Gate ON: lucky-noise candidate rejected and excluded everywhere.
# ---------------------------------------------------------------------------


def test_lucky_candidate_rejected_and_excluded():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            # High mean (winner's curse) but loses the paired test.
            cand = _instance_program(
                "lucky", _LUCKY, parent_id="p0", combined_score=99.0
            )
            db.add(cand, defer_maintenance=True)

            harness = _GateHarness(PaceGate(_config()), async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            assert verdict is not None
            assert not verdict.committed
            assert not verdict.reason.startswith("committed")

            # In-memory flag flipped and persisted to the row.
            assert cand.gate_passed == 0
            assert db.get("lucky").gate_passed == 0

            # Verdict recorded to metadata + the append-only jsonl log.
            assert cand.metadata["pace"]["committed"] is False
            log_path = Path(tmpdir) / "pace_log.jsonl"
            assert log_path.exists()
            assert "lucky" in log_path.read_text()

            # Door 1: withheld from the archive despite its huge mean score.
            db.run_post_add_maintenance(cand)
            archive_ids = {p.id for p in db._get_archive_programs()}
            assert "lucky" not in archive_ids

            # Door 2 / selection: best lookup and parent pool exclude it.
            assert db.get_best_program().id == "p0"
            top_ids = {p.id for p in db.get_top_programs(n=10, correct_only=True)}
            assert "lucky" not in top_ids
            selector = _selector(db)
            sampled = {selector.sample_parent(island_idx=0).id for _ in range(30)}
            assert sampled == {"p0"}
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Gate ON: decisive true improvement committed and archived.
# ---------------------------------------------------------------------------


def test_true_improvement_committed_and_archived():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            cand = _instance_program(
                "winner", _WINNER, parent_id="p0", combined_score=2.0
            )
            db.add(cand, defer_maintenance=True)

            harness = _GateHarness(PaceGate(_config()), async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            assert verdict is not None
            assert verdict.committed
            assert verdict.reason == "committed"
            assert cand.gate_passed == 1
            assert db.get("winner").gate_passed == 1

            # Door 1: admitted to the archive and promoted to best.
            db.run_post_add_maintenance(cand)
            archive_ids = {p.id for p in db._get_archive_programs()}
            assert "winner" in archive_ids
            assert db.get_best_program().id == "winner"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Missing per-instance vector -> zero evidence. With the default
# ``zero_evidence_action="reject"`` the candidate is withheld (no crash); with
# ``"pass"`` it is admitted as ``insufficient_instances``.
# ---------------------------------------------------------------------------


def _plain_candidate() -> Program:
    return Program(
        id="plain",
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=1.0,
        generation=1,
        island_idx=0,
        parent_id="p0",
    )


def test_missing_instance_vector_rejected_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            # Candidate publishes no instance vector at all (legacy task).
            cand = _plain_candidate()
            db.add(cand, defer_maintenance=True)

            harness = _GateHarness(PaceGate(_config()), async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            assert verdict is not None
            assert verdict.committed is False  # zero evidence -> withheld
            assert verdict.reason == "rejected_insufficient_instances"
            assert cand.gate_passed == 0
        finally:
            db.close()


def test_missing_instance_vector_passes_when_configured():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            cand = _plain_candidate()
            db.add(cand, defer_maintenance=True)

            harness = _GateHarness(
                PaceGate(_config(zero_evidence_action="pass")),
                async_db,
                tmpdir,
            )
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            assert verdict is not None
            assert verdict.committed  # pass-through, does not crash
            assert verdict.reason == "insufficient_instances"
            assert cand.gate_passed == 1
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Resume idempotence: a persisted verdict is reused, never re-judged.
# ---------------------------------------------------------------------------


def test_persisted_verdict_is_reused_not_rejudged():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            # Vectors that WOULD commit, but a rejected verdict is already
            # persisted: the gate must honour the stored verdict on resume.
            cand = _instance_program("winner", _WINNER, parent_id="p0")
            gate = PaceGate(_config())
            fresh = gate.verdict(_WINNER, _INCUMBENT)
            assert fresh.committed  # sanity: a re-judge would commit
            planted = gate.verdict(_LUCKY, _INCUMBENT)
            assert not planted.committed
            cand.metadata = {"pace": planted.to_dict()}
            db.add(cand, defer_maintenance=True)

            harness = _GateHarness(gate, async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            # Stored (rejected) verdict returned unchanged -> not re-judged.
            assert verdict is not None
            assert not verdict.committed
            assert verdict.reason == planted.reason
            assert cand.gate_passed == 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Sub-flag accessors (doors 3 and 4).
# ---------------------------------------------------------------------------


def test_subflag_accessors():
    off = _GateHarness(None, None, ".")
    assert off._pace_gate_bandit() is False
    assert off._pace_gate_scratchpad() is False

    both = _GateHarness(
        PaceGate(_config(gate_bandit=True, gate_scratchpad=True)), None, "."
    )
    assert both._pace_gate_bandit() is True
    assert both._pace_gate_scratchpad() is True

    neither = _GateHarness(PaceGate(_config()), None, ".")
    assert neither._pace_gate_bandit() is False
    assert neither._pace_gate_scratchpad() is False
