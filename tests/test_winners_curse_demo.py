"""Fast smoke test for the PACE winner's-curse A/B demo (both regimes).

Runs a tiny config (T=10, 2 seeds) end to end over BOTH regimes and asserts the
demo:

* runs against the real gate + real ``ProgramDatabase`` without error,
* produces both artifacts (``results.json`` + ``winners_curse.png``),
* returns a well-formed results dict keyed by regime (``planted`` +
  ``null_only``), each with the right shape and audit-labelled fields, and
* reproduces, *in every regime*, the qualitative direction the demo exists to
  show: the PACE-gated arm commits no more false positives than stock greedy.

This is a *smoke* test, not a statistical one: with only 2 seeds the exact
numbers are noisy, so we assert the robust inequality (gated <= greedy) rather
than a magnitude. The ``null_only`` regime additionally checks the construction
invariant that true quality never rises above baseline (no genuine improvement
exists), so any greedy commit there is pure winner's-curse churn.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = REPO_ROOT / "examples" / "pace_winners_curse" / "run_demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("pace_winners_curse_demo", DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass string-annotation resolution (PEP 563 /
    # ``from __future__ import annotations``) can find the module in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_runs_and_shows_gate_effect(tmp_path):
    demo = _load_demo()
    cfg = demo.DemoConfig(
        n_dev=20,
        n_rounds=10,
        n_seeds=2,
        n_improvements=1,
        n_regressions=1,
    )

    results = demo.run_demo(cfg, tmp_path)

    # Artifacts written.
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "winners_curse.png").exists()
    on_disk = json.loads((tmp_path / "results.json").read_text())
    assert on_disk == results

    # Top-level shape: results are keyed by regime, both regimes present.
    assert set(results.keys()) == set(demo.REGIMES) == {"planted", "null_only"}

    for regime in demo.REGIMES:
        block = results[regime]
        runs = block["runs"]

        # Well-formed shape: 2 seeds x 2 arms, config labelled with the regime.
        assert block["config"]["regime"] == regime
        assert len(runs) == 2 * cfg.n_seeds
        assert {r["arm"] for r in runs} == {"greedy", "pace"}

        # Every run carries the audit-labelled fields.
        for r in runs:
            assert 0 <= r["false_commits"] <= r["n_commits"]
            assert "reached_optimum" in r
            assert "winners_curse_gap" in r

        greedy, pace = block["summary"]["greedy"], block["summary"]["pace"]

        # Qualitative direction the demo demonstrates, in EVERY regime: the gate
        # does not commit more false positives than greedy (it commits fewer).
        assert (
            pace["false_commit_rate_mean"]
            <= greedy["false_commit_rate_mean"] + 1e-9
        )
        assert pace["false_commits_mean"] <= greedy["false_commits_mean"] + 1e-9

    # null_only construction invariant: NO genuine improvement exists, so true
    # quality can never rise above baseline for either arm -- every commit there
    # is winner's-curse churn, not a real gain. (The config zeroes improvements.)
    null_block = results["null_only"]
    assert null_block["config"]["n_improvements"] == 0
    p_base = null_block["config"]["p_base"]
    for r in null_block["runs"]:
        assert r["reported_best_true"] <= p_base + 1e-9
        assert r["reached_optimum"] == 0  # no optimum to reach in the null regime
