"""Tests for the PACE ``gate_passed`` column: migration, back-compat, and
selection filtering (parent / top-k / best / island migration).

These tests exercise the database layer in isolation (no runner, no LLM). The
central invariant is that ``gate_passed`` defaults to ``1`` everywhere, so with
the gate off (nothing ever rejected) selection behaviour is identical to stock;
only rows explicitly marked ``gate_passed = 0`` are withheld from selection.
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.parents import CombinedParentSelector


def _pace(committed: bool) -> dict:
    """Minimal serialised gate verdict for ``set_gate_verdict`` calls."""
    return {
        "committed": committed,
        "reason": "committed" if committed else "rejected_evidence",
    }


def _program(program_id: str, generation: int = 0, score: float = 1.0) -> Program:
    return Program(
        id=program_id,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=score,
        generation=generation,
        island_idx=0,
    )


def _make_db(tmpdir: str, name: str = "gate.db") -> ProgramDatabase:
    return ProgramDatabase(
        config=DatabaseConfig(
            db_path=str(Path(tmpdir) / name),
            num_islands=1,
            parent_selection_strategy="power_law",
        ),
        embedding_model="",
    )


def _column_names(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(programs)")]
    finally:
        conn.close()
    return cols


# ---------------------------------------------------------------------------
# Default / dataclass round-trip
# ---------------------------------------------------------------------------


def test_gate_passed_defaults_to_one_on_insert():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            db.add(_program("p0"))
            assert db.get("p0").gate_passed == 1
            # Raw column value defaults to 1 as well.
            db.cursor.execute("SELECT gate_passed FROM programs WHERE id = 'p0'")
            assert db.cursor.fetchone()[0] == 1
        finally:
            db.close()


def test_program_dataclass_roundtrip_preserves_gate_passed():
    prog = _program("p0")
    prog.gate_passed = 0
    restored = Program.from_dict(prog.to_dict())
    assert restored.gate_passed == 0

    # Default field value is 1 when unset.
    assert Program(id="x", code="pass").gate_passed == 1


# ---------------------------------------------------------------------------
# Migration + back-compat
# ---------------------------------------------------------------------------


def test_migration_adds_gate_passed_and_backfills_existing_rows():
    """An existing DB without the column gets it added, defaulting old rows to 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "legacy.db")

        # Create a populated DB, then simulate a pre-migration schema by
        # dropping the column out from under it.
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=db_path, num_islands=1),
            embedding_model="",
        )
        db.add(_program("legacy0"))
        db.close()

        conn = sqlite3.connect(db_path)
        conn.execute("ALTER TABLE programs DROP COLUMN gate_passed")
        conn.commit()
        conn.close()
        assert "gate_passed" not in _column_names(db_path)

        # Re-opening runs migrations and re-adds the column.
        db2 = ProgramDatabase(
            config=DatabaseConfig(db_path=db_path, num_islands=1),
            embedding_model="",
        )
        try:
            assert "gate_passed" in _column_names(db_path)
            # The pre-existing row is backfilled to the default (eligible).
            assert db2.get("legacy0").gate_passed == 1
        finally:
            db2.close()


def test_migration_is_idempotent():
    """Running migrations repeatedly must not error or duplicate the column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "idem.db")
        for _ in range(3):
            db = ProgramDatabase(
                config=DatabaseConfig(db_path=db_path, num_islands=1),
                embedding_model="",
            )
            db.close()
        cols = _column_names(db_path)
        assert cols.count("gate_passed") == 1


# ---------------------------------------------------------------------------
# set_gate_verdict (sync + async): atomic column + metadata['pace'] write
# ---------------------------------------------------------------------------


def test_set_gate_verdict_persists_both_directions():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            db.add(_program("p0"))
            db.set_gate_verdict("p0", False, _pace(False))
            got = db.get("p0")
            assert got.gate_passed == 0
            # The verdict is merged under metadata['pace'] in the same write.
            assert got.metadata["pace"]["committed"] is False
            db.set_gate_verdict("p0", True, _pace(True))
            got = db.get("p0")
            assert got.gate_passed == 1
            assert got.metadata["pace"]["committed"] is True
        finally:
            db.close()


def test_set_gate_verdict_preserves_existing_metadata():
    """The atomic write must merge pace without clobbering other metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            prog = _program("p0")
            prog.metadata = {"model_name": "gpt", "source_job_id": "j0"}
            db.add(prog)
            db.set_gate_verdict("p0", False, _pace(False))
            got = db.get("p0")
            assert got.gate_passed == 0
            assert got.metadata["model_name"] == "gpt"
            assert got.metadata["source_job_id"] == "j0"
            assert got.metadata["pace"]["committed"] is False
        finally:
            db.close()


def test_set_gate_verdict_async_persists():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_db = _make_db(tmpdir, name="async_gate.db")
        async_db = AsyncProgramDatabase(sync_db=sync_db)

        async def _run():
            await async_db.add_program_async(_program("p0"))
            await async_db.set_gate_verdict_async("p0", False, _pace(False))

        try:
            asyncio.run(_run())
            got = sync_db.get("p0")
            assert got.gate_passed == 0
            assert got.metadata["pace"]["committed"] is False
        finally:
            asyncio.run(async_db.close_async())
            sync_db.close()


# ---------------------------------------------------------------------------
# Resume sweep query: rows missing the postprocess_side_effects_applied flag
# ---------------------------------------------------------------------------


def test_get_programs_pending_side_effects_selects_only_unprocessed():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_db = _make_db(tmpdir, name="pending.db")
        async_db = AsyncProgramDatabase(sync_db=sync_db)

        async def _run():
            # p_done has the applied flag set -> excluded from the sweep.
            done = _program("p_done")
            done.metadata = {"postprocess_side_effects_applied": True}
            await async_db.add_program_async(done)
            # p_pending has no flag -> included.
            pending = _program("p_pending")
            pending.metadata = {"source_job_id": "j1"}
            await async_db.add_program_async(pending)
            # p_bare has empty metadata -> also included.
            await async_db.add_program_async(_program("p_bare"))
            return await async_db.get_programs_pending_side_effects_async(100)

        try:
            rows = asyncio.run(_run())
            ids = {p.id for p in rows}
            assert "p_done" not in ids
            assert {"p_pending", "p_bare"} <= ids
        finally:
            asyncio.run(async_db.close_async())
            sync_db.close()


def test_get_programs_pending_side_effects_respects_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_db = _make_db(tmpdir, name="pending_limit.db")
        async_db = AsyncProgramDatabase(sync_db=sync_db)

        async def _run():
            for i in range(5):
                await async_db.add_program_async(_program(f"p{i}"))
            return await async_db.get_programs_pending_side_effects_async(2)

        try:
            rows = asyncio.run(_run())
            assert len(rows) == 2
        finally:
            asyncio.run(async_db.close_async())
            sync_db.close()


# ---------------------------------------------------------------------------
# Selection filtering: top-k / best / parent / migration
# ---------------------------------------------------------------------------


def test_get_top_programs_excludes_rejected_when_correct_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            db.add(_program("p0", score=1.0))
            db.add(_program("p1", score=2.0))
            db.add(_program("p2", score=3.0))
            db.set_gate_verdict("p2", False, _pace(False))  # reject the top scorer

            top_ids = {p.id for p in db.get_top_programs(n=10, correct_only=True)}
            assert "p2" not in top_ids
            assert {"p0", "p1"} <= top_ids

            # Without correct_only the rejected row is still visible (raw view).
            all_ids = {p.id for p in db.get_top_programs(n=10, correct_only=False)}
            assert "p2" in all_ids
        finally:
            db.close()


def test_get_best_program_skips_rejected_top_scorer():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            db.add(_program("p0", score=1.0))
            db.add(_program("p1", score=2.0))
            db.add(_program("p2", score=3.0))
            db.set_gate_verdict("p2", False, _pace(False))

            best = db.get_best_program()
            assert best is not None
            assert best.id == "p1"  # highest-scoring gate-eligible program
        finally:
            db.close()


def _selector(db: ProgramDatabase) -> CombinedParentSelector:
    return CombinedParentSelector(
        cursor=db.cursor,
        conn=db.conn,
        config=db.config,
        get_program_func=db.get,
        best_program_id=db.best_program_id,
        get_best_program_func=db.get_best_program,
    )


def test_parent_sampling_excludes_rejected():
    """A gate-rejected program must never be returned as a parent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)  # power_law strategy
        try:
            db.add(_program("p0", generation=0, score=1.0))
            db.add(_program("p1", generation=1, score=5.0))
            db.set_gate_verdict("p1", False, _pace(False))  # reject the higher-scoring child

            selector = _selector(db)
            sampled = {selector.sample_parent(island_idx=0).id for _ in range(30)}
            assert sampled == {"p0"}
            assert "p1" not in sampled
        finally:
            db.close()


def test_has_correct_programs_ignores_rejected_only_population():
    """If every correct program is gate-rejected, none count as eligible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            db.add(_program("p0"))
            db.set_gate_verdict("p0", False, _pace(False))
            selector = _selector(db)
            assert selector.has_correct_programs(island_idx=0) is False
            # A still-eligible program flips it back to True.
            db.add(_program("p1"))
            assert selector.has_correct_programs(island_idx=0) is True
        finally:
            db.close()


def test_default_population_is_fully_eligible_back_compat():
    """With nothing rejected (gate off), every selection surface sees all rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir)
        try:
            for i in range(4):
                db.add(_program(f"p{i}", generation=i, score=float(i)))

            top_correct = {p.id for p in db.get_top_programs(n=10, correct_only=True)}
            top_all = {p.id for p in db.get_top_programs(n=10, correct_only=False)}
            assert top_correct == top_all == {"p0", "p1", "p2", "p3"}

            selector = _selector(db)
            assert selector.has_correct_programs(island_idx=0) is True
            sampled = {selector.sample_parent(island_idx=0).id for _ in range(40)}
            # All eligible programs remain reachable as parents.
            assert sampled <= {"p0", "p1", "p2", "p3"}
        finally:
            db.close()
