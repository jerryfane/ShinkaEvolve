"""Shared SQL predicate for selection eligibility.

A program is eligible to be surfaced as a parent, inspiration, elite or "best"
program only when it is both ``correct`` and has passed the PACE acceptance
gate (``gate_passed = 1``). This predicate was historically copy-pasted across
dozens of call sites, which made it easy to miss one and silently surface
gate-rejected programs. Centralising it here removes that drift risk.

``gate_passed`` defaults to ``1`` (see the schema/migration in ``dbase.py``),
so the predicate is behaviour-neutral for runs with the gate disabled and for
databases created before the gate existed.
"""

SELECTABLE_SQL = "correct = 1 AND gate_passed = 1"


def eligible_sql(alias: str | None = None) -> str:
    """Return the selection-eligibility predicate.

    Args:
        alias: Optional table alias/name to qualify each column with. For
            example ``eligible_sql("p")`` yields
            ``"p.correct = 1 AND p.gate_passed = 1"``. When ``None`` the bare
            column form (``SELECTABLE_SQL``) is returned.

    Returns:
        A SQL boolean expression suitable for a ``WHERE``/``CASE WHEN`` clause.
    """
    if alias:
        return f"{alias}.correct = 1 AND {alias}.gate_passed = 1"
    return SELECTABLE_SQL
