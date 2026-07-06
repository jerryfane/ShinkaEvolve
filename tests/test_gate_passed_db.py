"""Tests for the PACE ``gate_passed`` column: migration, back-compat, and
selection filtering (parent / top-k / best / island migration).

These tests exercise the database layer in isolation (no runner, no LLM). The
central invariant is that ``gate_passed`` defaults to ``1`` everywhere, so with
the gate off (nothing ever rejected) selection behaviour is identical to stock;
only rows explicitly marked ``gate_passed = 0`` are withheld from selection.
"""

import asyncio
import re
import sqlite3
import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.async_dbase import AsyncProgramDatabase
from shinka.database.islands import CombinedIslandManager
from shinka.database.parents import CombinedParentSelector


# Tripwire regexes (shared by the scan tests and the semantics test below). The
# ``(?<![\w.])`` left boundary makes them match a bare/aliased ``correct`` but
# not a longer identifier ending in ``correct`` (``incorrect``/``is_correct``).
_COMBINED_PREDICATE_RE = re.compile(
    r"(?<![\w.])(?:\w+\.)?correct = 1 AND (?:\w+\.)?gate_passed = 1"
)
_LONE_ACCEPT_RE = re.compile(r"(?<![\w.])(?:\w+\.)?correct\s*=\s*1\b")


def _pace(committed: bool) -> dict:
    """Minimal serialised gate verdict for ``set_gate_verdict`` calls."""
    return {
        "committed": committed,
        "reason": "committed" if committed else "rejected_evidence",
    }


def _program(
    program_id: str,
    generation: int = 0,
    score: float = 1.0,
    parent_id=None,
) -> Program:
    return Program(
        id=program_id,
        code="def f():\n    return 1\n",
        correct=True,
        combined_score=score,
        generation=generation,
        island_idx=0,
        parent_id=parent_id,
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
            # Judged candidates carry a parent_id; only they are swept.
            # p_done has the applied flag set -> excluded from the sweep.
            done = _program("p_done", parent_id="root")
            done.metadata = {"postprocess_side_effects_applied": True}
            await async_db.add_program_async(done)
            # p_pending has no flag -> included.
            pending = _program("p_pending", parent_id="root")
            pending.metadata = {"source_job_id": "j1"}
            await async_db.add_program_async(pending)
            # p_bare has empty metadata -> also included.
            await async_db.add_program_async(_program("p_bare", parent_id="root"))
            return await async_db.get_programs_pending_side_effects_async(100)

        try:
            rows = asyncio.run(_run())
            ids = {p.id for p in rows}
            assert "p_done" not in ids
            assert {"p_pending", "p_bare"} <= ids
        finally:
            asyncio.run(async_db.close_async())
            sync_db.close()


def test_get_programs_pending_side_effects_excludes_parentless_rows():
    """Only judged candidates (parent_id NOT NULL) are swept. The gen-0 initial
    program and island copies have no parent and are pipeline-external - they
    never pass through the proposal/evaluation side-effect path, so the sweep
    must never re-enqueue them (doing so would re-judge / mutate rows the
    pipeline never owned)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_db = _make_db(tmpdir, name="pending_parentless.db")
        async_db = AsyncProgramDatabase(sync_db=sync_db)

        async def _run():
            # gen-0 initial program: parent_id is NULL, no applied flag.
            await async_db.add_program_async(_program("initial", generation=0))
            # An island copy: also parent_id NULL, pipeline-external.
            await async_db.add_program_async(_program("island_copy"))
            # A genuine judged candidate: parent_id set, no applied flag.
            await async_db.add_program_async(
                _program("candidate", generation=1, parent_id="initial")
            )
            return await async_db.get_programs_pending_side_effects_async(100)

        try:
            rows = asyncio.run(_run())
            ids = {p.id for p in rows}
            assert ids == {"candidate"}
            assert "initial" not in ids
            assert "island_copy" not in ids
        finally:
            asyncio.run(async_db.close_async())
            sync_db.close()


def test_set_gate_verdict_survives_concurrent_metadata_key():
    """F5: the atomic ``json_set`` write must not clobber a metadata key written
    by another writer just before/after the verdict. The old SELECT-merge-UPDATE
    read a stale snapshot and could silently drop such a key; ``json_set`` on the
    ``$.pace`` path preserves every sibling key, and the column + pace stay
    consistent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _make_db(tmpdir, name="atomic.db")
        try:
            prog = _program("p0")
            prog.metadata = {"before_key": "b"}
            db.add(prog)

            # A concurrent-ish writer touches a DIFFERENT key just before the
            # verdict write, going straight to the column (no in-memory merge).
            db.cursor.execute(
                "UPDATE programs SET metadata = "
                "json_set(COALESCE(metadata, '{}'), '$.after_key', ?) "
                "WHERE id = ?",
                ("a", "p0"),
            )
            db.conn.commit()

            db.set_gate_verdict("p0", True, _pace(True))

            got = db.get("p0")
            # Both sibling keys survive the pace merge.
            assert got.metadata["before_key"] == "b"
            assert got.metadata["after_key"] == "a"
            # The pace verdict is stored as a nested object (not a JSON string).
            assert got.metadata["pace"]["committed"] is True
            # Column and pace stay consistent in the single atomic write.
            assert got.gate_passed == 1
            db.cursor.execute(
                "SELECT gate_passed, json_extract(metadata, '$.pace.committed') "
                "FROM programs WHERE id = ?",
                ("p0",),
            )
            gate_passed, pace_committed = db.cursor.fetchone()
            assert gate_passed == 1
            assert pace_committed == 1  # json_extract of a JSON true -> 1
        finally:
            db.close()


def test_get_programs_pending_side_effects_respects_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_db = _make_db(tmpdir, name="pending_limit.db")
        async_db = AsyncProgramDatabase(sync_db=sync_db)

        async def _run():
            for i in range(5):
                await async_db.add_program_async(
                    _program(f"p{i}", parent_id="root")
                )
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


# ---------------------------------------------------------------------------
# Island initialization: a gate-rejected-only island is not "initialized"
# ---------------------------------------------------------------------------


def test_island_with_only_rejected_rows_counts_as_uninitialized():
    """An island whose only correct rows are gate-rejected must read as
    uninitialized so the assignment strategy reseeds it (aligning island
    initialization with parent-selection eligibility)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE programs (
            id TEXT PRIMARY KEY,
            correct INTEGER,
            gate_passed INTEGER DEFAULT 1,
            island_idx INTEGER
        )
        """
    )
    # Island 0 has an eligible program; island 1 has only a gate-rejected one.
    cursor.execute("INSERT INTO programs VALUES ('a', 1, 1, 0)")
    cursor.execute("INSERT INTO programs VALUES ('b', 1, 0, 1)")
    conn.commit()

    config = type("Cfg", (), {"num_islands": 2})()
    manager = CombinedIslandManager(cursor=cursor, conn=conn, config=config)

    # Island 1 (rejected-only) is not initialized.
    assert manager.get_initialized_islands() == [0]
    assert manager.are_all_islands_initialized() is False

    # A fresh program is reseeded onto the uninitialized island 1.
    fresh = type(
        "P", (), {"id": "new", "parent_id": None, "island_idx": None, "metadata": None}
    )()
    manager.assign_island(fresh)
    assert fresh.island_idx == 1

    # Once island 1 gains an eligible row, all islands read as initialized.
    cursor.execute("INSERT INTO programs VALUES ('c', 1, 1, 1)")
    conn.commit()
    assert sorted(manager.get_initialized_islands()) == [0, 1]
    assert manager.are_all_islands_initialized() is True

    conn.close()


# ---------------------------------------------------------------------------
# Percentile population: gate-rejected programs are excluded
# ---------------------------------------------------------------------------


def test_compute_percentile_excludes_rejected_population():
    """compute_percentile_async(correct_only=True) must rank against the
    eligible population only, ignoring gate-rejected rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_db = _make_db(tmpdir, name="percentile.db")
        async_db = AsyncProgramDatabase(sync_db=sync_db)

        async def _run():
            await async_db.add_program_async(_program("p0", score=1.0))
            await async_db.add_program_async(_program("p1", score=2.0))
            await async_db.add_program_async(_program("p2", score=3.0))
            # Reject the low scorer so only {2.0, 3.0} remain eligible.
            await async_db.set_gate_verdict_async("p0", False, _pace(False))
            return await async_db.compute_percentile_async(2.5, correct_only=True)

        try:
            pct = asyncio.run(_run())
            # Eligible scores {2.0, 3.0}: 2.5 beats one of two -> 0.5.
            # (Including the rejected 1.0 would give 2/3.)
            assert pct == 0.5
        finally:
            asyncio.run(async_db.close_async())
            sync_db.close()


# ---------------------------------------------------------------------------
# Island copy preserves the source row's gate verdict
# ---------------------------------------------------------------------------


def test_island_copy_preserves_rejected_gate_verdict():
    """Copying a program to other islands must carry the source's
    ``gate_passed`` value rather than defaulting the copies to eligible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ProgramDatabase(
            config=DatabaseConfig(
                db_path=str(Path(tmpdir) / "copy.db"), num_islands=3
            ),
            embedding_model="",
        )
        try:
            prog = _program("p0")
            prog.gate_passed = 0
            prog.island_idx = 0

            copies = db.island_manager.copy_program_to_islands(prog)
            assert len(copies) == 2

            for copy_id in copies:
                row = db.cursor.execute(
                    "SELECT gate_passed FROM programs WHERE id = ?", (copy_id,)
                ).fetchone()
                assert row["gate_passed"] == 0
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Regression tripwire: the eligibility predicate lives only in eligibility.py
# ---------------------------------------------------------------------------


def test_no_raw_eligibility_predicate_outside_module():
    """Guard against reintroducing the copy-pasted eligibility predicate. Every
    selection site must go through ``eligible_sql()`` so a missed gate filter
    can never silently surface gate-rejected programs again."""
    shinka_root = Path(__file__).resolve().parent.parent / "shinka"
    # Matches both the bare and table-aliased forms of the predicate. The
    # ``(?<![\w.])`` left boundary prevents matching a longer identifier that
    # merely ends in ``correct`` (e.g. ``incorrect``/``is_correct``); the
    # optional ``(?:\w+\.)?`` alias group still matches ``p.correct``.
    predicate = _COMBINED_PREDICATE_RE

    offenders = []
    for path in shinka_root.rglob("*.py"):
        if path.name == "eligibility.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if predicate.search(line):
                rel = path.relative_to(shinka_root.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Raw eligibility predicate found outside eligibility.py; use "
        "eligible_sql() instead:\n" + "\n".join(offenders)
    )


def test_no_lone_accept_side_correctness_filter():
    """Guard against a lone accept-side ``correct = 1`` filter that forgets the
    gate. The combined-predicate tripwire above only fires once a site already
    references ``gate_passed``; a hand-written ``WHERE ... correct = 1`` that
    never added the gate column (as ``display.py`` originally did) would slip
    through and surface gate-rejected programs. Any accept-side literal must
    instead route through ``eligible_sql()``. ``correct = 0`` (the incorrect-
    program paths) is deliberately not matched."""
    shinka_root = Path(__file__).resolve().parent.parent / "shinka"
    # The ``(?<![\w.])`` left boundary ensures a bare ``correct = 1`` (or an
    # aliased ``p.correct = 1``) is matched, but a longer identifier ending in
    # ``correct`` - ``incorrect = 1``, ``is_correct = 1`` - is not: the char
    # immediately before ``correct``/its alias must not be a word char or a dot.
    lone_accept = _LONE_ACCEPT_RE

    offenders = []
    for path in shinka_root.rglob("*.py"):
        if path.name == "eligibility.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if lone_accept.search(line):
                rel = path.relative_to(shinka_root.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Lone accept-side 'correct = 1' filter found; use eligible_sql() so "
        "the gate_passed filter is never dropped:\n" + "\n".join(offenders)
    )


def test_tripwire_regex_left_boundary_semantics():
    """The tripwire left boundary must fire on real accept-side filters but not
    on longer identifiers that merely *end* in ``correct``. Without the
    ``(?<![\\w.])`` boundary, ``incorrect = 1`` / ``is_correct = 1`` false-match
    the tail ``correct = 1`` and would wedge every legitimate incorrect-path
    filter."""
    # Positive: a bare and an aliased accept-side filter must match.
    assert _LONE_ACCEPT_RE.search("WHERE correct = 1")
    assert _LONE_ACCEPT_RE.search("WHERE p.correct = 1")
    assert _COMBINED_PREDICATE_RE.search("correct = 1 AND gate_passed = 1")
    assert _COMBINED_PREDICATE_RE.search(
        "p.correct = 1 AND p.gate_passed = 1"
    )

    # Negative: longer identifiers ending in ``correct`` must NOT match.
    assert not _LONE_ACCEPT_RE.search("WHERE incorrect = 1")
    assert not _LONE_ACCEPT_RE.search("WHERE is_correct = 1")
    assert not _COMBINED_PREDICATE_RE.search(
        "incorrect = 1 AND gate_passed = 1"
    )
    assert not _COMBINED_PREDICATE_RE.search(
        "is_correct = 1 AND gate_passed = 1"
    )
    # And the deliberately-excluded incorrect-path literal still does not match.
    assert not _LONE_ACCEPT_RE.search("WHERE correct = 0")
