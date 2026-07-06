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
from types import SimpleNamespace

from shinka.core import AcceptanceGateConfig, PaceGate
from shinka.core.async_runner import (
    PACE_INSTANCE_KEY,
    PACE_REASON_NO_INCUMBENT,
    AsyncRunningJob,
    PersistedProgramEvent,
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


def _drive_scenario(tmpdir: str, *, drive_gate_off: bool) -> dict:
    """Run a fixed multi-program scenario and snapshot the observable DB state.

    The scenario plants a root incumbent, a decisive winner, and a lucky-noise
    candidate (huge mean, loses the paired test). When ``drive_gate_off`` is
    True each candidate is additionally routed through ``_apply_pace_gate`` with
    ``pace_gate=None`` (the exact wiring a gate-config-absent OR ``enabled=False``
    runner uses) before Door 1 maintenance; when False the gate methods are
    never touched (pure stock). The two must produce byte-identical archive
    contents, best program, top-k, and parent-sampling results.
    """
    import numpy as np

    db = _make_db(tmpdir)
    async_db = AsyncProgramDatabase(db, max_workers=1)
    try:
        harness = _GateHarness(None, async_db, tmpdir)
        specs = [
            ("p0", _INCUMBENT, None, 0, 0.0),
            ("winner", _WINNER, "p0", 1, 2.0),
            ("lucky", _LUCKY, "p0", 1, 99.0),
        ]
        for pid, scores, parent_id, gen, score in specs:
            prog = _instance_program(
                pid, scores, parent_id=parent_id, generation=gen,
                combined_score=score,
            )
            db.add(prog, defer_maintenance=True)
            if drive_gate_off:
                assert asyncio.run(harness._apply_pace_gate(prog)) is None
            db.run_post_add_maintenance(prog)

        np.random.seed(1234)
        selector = _selector(db)
        sampled = [selector.sample_parent(island_idx=0).id for _ in range(50)]
        return {
            "archive": sorted(p.id for p in db._get_archive_programs()),
            "best": db.get_best_program().id,
            "top": [
                p.id for p in db.get_top_programs(n=10, correct_only=True)
            ],
            "gate_flags": {
                pid: db.get(pid).gate_passed for pid, *_ in specs
            },
            "sampled": sampled,
        }
    finally:
        async_db.close() if hasattr(async_db, "close") else None
        db.close()


def test_gate_off_matches_stock_archive_best_and_selection():
    """Finding closure: a gate-off (config-absent / ``enabled=False``) run must be
    byte-identical to stock across every observable selection surface, not merely
    admit one candidate to the archive. Drive the identical scenario with and
    without the gate side-effect path and assert full equality of archive
    contents, best program, top-k ordering, gate flags, and a deterministic
    parent-sampling trace."""
    with tempfile.TemporaryDirectory() as td_stock:
        stock = _drive_scenario(td_stock, drive_gate_off=False)
    with tempfile.TemporaryDirectory() as td_off:
        gate_off = _drive_scenario(td_off, drive_gate_off=True)

    assert gate_off == stock
    # And the gate never left a fingerprint: all rows stay eligible, no log.
    assert set(stock["gate_flags"].values()) == {1}
    assert "lucky" in stock["archive"] and "winner" in stock["archive"]


def test_runner_wiring_absent_and_disabled_both_yield_no_gate(monkeypatch, tmp_path):
    """The runner must collapse both a *missing* ``acceptance_gate`` and an
    explicit ``enabled=False`` config to ``pace_gate=None`` (the single source of
    the flag-off invariance). Exercised against the real ``ShinkaEvolveRunner``
    constructor so a future wiring change cannot silently diverge the two."""
    from shinka.core.async_runner import ShinkaEvolveRunner
    from shinka.core.config import EvolutionConfig
    from shinka.launch import LocalJobConfig

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def _pace_gate_for(gate_cfg, subdir):
        runner = ShinkaEvolveRunner(
            evo_config=EvolutionConfig(
                llm_models=["gpt-5-mini"],
                llm_dynamic_selection=None,
                meta_rec_interval=None,
                embedding_model=None,
                num_generations=1,
                results_dir=str(tmp_path / subdir),
                acceptance_gate=gate_cfg,
            ),
            job_config=LocalJobConfig(),
            db_config=DatabaseConfig(db_path=str(tmp_path / subdir / "db.sqlite")),
            verbose=False,
            banner_style="minimal",
        )
        return runner.pace_gate

    assert _pace_gate_for(None, "absent") is None
    assert (
        _pace_gate_for(AcceptanceGateConfig(enabled=False), "disabled") is None
    )


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
# Fail-closed pending row: a commit flips gate_passed 0 -> 1 atomically.
# ---------------------------------------------------------------------------


def test_commit_flips_pending_row_and_persists_verdict():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            # Fail-closed pending insert: judged-eligible row starts gate_passed=0
            # and is invisible to selection until the gate commits it.
            cand = _instance_program(
                "winner", _WINNER, parent_id="p0", combined_score=2.0
            )
            cand.gate_passed = 0
            db.add(cand, defer_maintenance=True)
            assert db.get("winner").gate_passed == 0  # pending, invisible

            harness = _GateHarness(PaceGate(_config()), async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            assert verdict is not None and verdict.committed
            # Atomic write flipped the column 0 -> 1 and persisted the verdict.
            row = db.get("winner")
            assert row.gate_passed == 1
            assert row.metadata["pace"]["committed"] is True
        finally:
            db.close()


def test_reject_persists_column_and_verdict_atomically():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            cand = _instance_program(
                "lucky", _LUCKY, parent_id="p0", combined_score=99.0
            )
            cand.gate_passed = 0
            db.add(cand, defer_maintenance=True)

            harness = _GateHarness(PaceGate(_config()), async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            assert verdict is not None and not verdict.committed
            # A single atomic write left the column AND metadata consistent.
            row = db.get("lucky")
            assert row.gate_passed == 0
            assert row.metadata["pace"]["committed"] is False
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Resume self-heal: a stored verdict repairs a disagreeing gate_passed column.
# ---------------------------------------------------------------------------


def test_resume_self_heals_disagreeing_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            gate = PaceGate(_config())
            rejected = gate.verdict(_LUCKY, _INCUMBENT)
            assert not rejected.committed
            # Simulate a crash between the two legacy writes: metadata records a
            # rejected verdict but the column is still the (default) eligible 1.
            cand = _instance_program("lucky", _LUCKY, parent_id="p0")
            cand.metadata = {"pace": rejected.to_dict()}
            cand.gate_passed = 1  # disagrees with the stored verdict
            db.add(cand, defer_maintenance=True)
            assert db.get("lucky").gate_passed == 1

            # Re-judging on resume honours the stored verdict AND repairs the
            # column via the atomic write (never re-runs the e-process).
            harness = _GateHarness(gate, async_db, tmpdir)
            verdict = asyncio.run(harness._apply_pace_gate(cand))

            assert verdict is not None and not verdict.committed
            assert cand.gate_passed == 0
            assert db.get("lucky").gate_passed == 0  # self-healed
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Door 3 (bandit reward): rejected candidate feeds a neutral parent-baseline
# reward, or skips the update entirely when there is no usable baseline.
# ---------------------------------------------------------------------------


class _FakeSelection:
    """Records ``update`` calls so Door-3 reward routing can be asserted."""

    def __init__(self):
        self.updates = []

    def update(self, arm, reward, baseline):
        self.updates.append({"arm": arm, "reward": reward, "baseline": baseline})

    def print_summary(self, console=None):
        pass


class _SideEffectHarness:
    """Runs the real ``_apply_persisted_program_side_effects`` door pipeline."""

    _pace_instance_vector = ShinkaEvolveRunner._pace_instance_vector
    _pace_pass_through = ShinkaEvolveRunner._pace_pass_through
    _append_pace_log = ShinkaEvolveRunner._append_pace_log
    _apply_pace_gate = ShinkaEvolveRunner._apply_pace_gate
    _pace_gate_bandit = ShinkaEvolveRunner._pace_gate_bandit
    _pace_gate_scratchpad = ShinkaEvolveRunner._pace_gate_scratchpad
    _apply_persisted_program_side_effects = (
        ShinkaEvolveRunner._apply_persisted_program_side_effects
    )
    _reconstruct_persisted_event_for_resume = (
        ShinkaEvolveRunner._reconstruct_persisted_event_for_resume
    )

    def __init__(self, pace_gate, async_db, results_dir, llm_selection):
        self.pace_gate = pace_gate
        self.async_db = async_db
        self.results_dir = results_dir
        self.llm_selection = llm_selection
        self.verbose = False
        self.meta_summarizer = None
        self.evo_config = SimpleNamespace(evolve_prompts=False, meta_rec_interval=0)
        self.console = None
        self.total_api_cost = 0.0

    async def _update_best_solution_async(self):
        return None

    async def _persist_program_metadata_async(self, program):
        return None


def _side_effect_event(program: Program) -> PersistedProgramEvent:
    job = AsyncRunningJob(
        job_id=str(program.id),
        exec_fname="",
        results_dir=".",
        start_time=0.0,
        proposal_started_at=0.0,
        evaluation_submitted_at=0.0,
        generation=program.generation,
        parent_id=program.parent_id,
    )
    return PersistedProgramEvent(
        job=job,
        program=program,
        evaluation_finished_at=0.0,
        postprocess_started_at=0.0,
        postprocess_finished_at=0.0,
    )


def test_door3_rejected_candidate_feeds_neutral_parent_reward():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0, combined_score=5.0))
            cand = _instance_program(
                "lucky", _LUCKY, parent_id="p0", combined_score=99.0
            )
            cand.metadata = {"model_name": "arm-A"}
            db.add(cand, defer_maintenance=True)

            sel = _FakeSelection()
            harness = _SideEffectHarness(
                PaceGate(_config(gate_bandit=True)), async_db, tmpdir, sel
            )
            asyncio.run(
                harness._apply_persisted_program_side_effects(
                    _side_effect_event(cand)
                )
            )

            # Rejected -> neutral reward is the incumbent's own score (delta 0),
            # NOT the candidate's inflated 99.0.
            assert len(sel.updates) == 1
            assert sel.updates[0]["reward"] == 5.0
            assert sel.updates[0]["baseline"] == 5.0
        finally:
            db.close()


def test_door3_skips_update_when_parent_has_no_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            # Incumbent publishes an instance vector but has no combined_score.
            db.add(
                _instance_program(
                    "p0", _INCUMBENT, generation=0, combined_score=None
                )
            )
            cand = _instance_program(
                "lucky", _LUCKY, parent_id="p0", combined_score=99.0
            )
            cand.metadata = {"model_name": "arm-A"}
            db.add(cand, defer_maintenance=True)

            sel = _FakeSelection()
            harness = _SideEffectHarness(
                PaceGate(_config(gate_bandit=True)), async_db, tmpdir, sel
            )
            asyncio.run(
                harness._apply_persisted_program_side_effects(
                    _side_effect_event(cand)
                )
            )

            # No usable neutral baseline -> the update is skipped entirely rather
            # than imputing a fabricated reward.
            assert sel.updates == []
        finally:
            db.close()


def test_door3_committed_candidate_uses_own_score():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0, combined_score=5.0))
            cand = _instance_program(
                "winner", _WINNER, parent_id="p0", combined_score=7.0
            )
            cand.metadata = {"model_name": "arm-A"}
            db.add(cand, defer_maintenance=True)

            sel = _FakeSelection()
            harness = _SideEffectHarness(
                PaceGate(_config(gate_bandit=True)), async_db, tmpdir, sel
            )
            asyncio.run(
                harness._apply_persisted_program_side_effects(
                    _side_effect_event(cand)
                )
            )

            # Committed -> legacy path: the candidate's own score is the reward.
            assert len(sel.updates) == 1
            assert sel.updates[0]["reward"] == 7.0
            assert sel.updates[0]["baseline"] == 5.0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Resume sweep: a pending, never-judged row is judged when re-processed.
# ---------------------------------------------------------------------------


def test_resume_sweep_judges_pending_unprocessed_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        async_db = AsyncProgramDatabase(db, max_workers=1)
        try:
            db.add(_instance_program("p0", _INCUMBENT, generation=0))
            # A fail-closed pending row that was inserted but never judged (no
            # metadata['pace'], gate_passed=0, missing side-effect flag) - exactly
            # what the resume sweep must re-process.
            cand = _instance_program(
                "winner", _WINNER, parent_id="p0", combined_score=2.0
            )
            cand.gate_passed = 0
            cand.metadata = {"source_job_id": "j0"}
            db.add(cand, defer_maintenance=True)

            sel = _FakeSelection()
            harness = _SideEffectHarness(
                PaceGate(_config()), async_db, tmpdir, sel
            )
            # Reconstruct the persisted event exactly as the resume sweep would,
            # then run the real side-effect pipeline over it.
            event = harness._reconstruct_persisted_event_for_resume(cand)
            asyncio.run(harness._apply_persisted_program_side_effects(event))

            # The previously-unjudged pending row is now committed: column flipped
            # 0 -> 1 and the verdict is persisted.
            row = db.get("winner")
            assert row.gate_passed == 1
            assert row.metadata["pace"]["committed"] is True
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
