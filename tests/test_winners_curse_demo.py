"""Fast smoke test for the PACE winner's-curse A/B/C demo (both regimes).

Runs a tiny config (T=10, 2 seeds) end to end over BOTH regimes and asserts the
demo:

* runs against the real gate + real ``ProgramDatabase`` without error,
* produces both artifacts (``results.json`` + ``winners_curse.png``),
* returns a well-formed results dict keyed by regime (``planted`` +
  ``null_only``), each with the right shape and audit-labelled fields (including
  the per-arm std + pooled false-commit reporting), and
* reproduces, *in every regime*, the qualitative direction the demo exists to
  show: the PACE-gated arm commits no more false positives than **either**
  greedy arm -- the ``pace`` vs ``greedy_reeval`` contrast (same 2x eval cost,
  gate-vs-no-gate) being the deconfounded measure of the gate's effect.

This is a *smoke* test, not a statistical one: with only 2 seeds the exact
numbers are noisy, so we assert robust inequalities (gated <= each greedy arm on
false-commit *counts*) rather than magnitudes. The ``null_only`` regime
additionally checks the construction invariant that true quality never rises
above baseline (no genuine improvement exists), so any greedy commit there is
pure winner's-curse churn.
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

    # Three arms are present and named as expected.
    assert demo.ARMS == ["greedy", "greedy_reeval", "pace"]

    for regime in demo.REGIMES:
        block = results[regime]
        runs = block["runs"]

        # Well-formed shape: 2 seeds x 3 arms, config labelled with the regime.
        assert block["config"]["regime"] == regime
        assert block["config"]["sigma_inst"] == cfg.sigma_inst  # paired difficulty
        assert len(runs) == len(demo.ARMS) * cfg.n_seeds
        assert {r["arm"] for r in runs} == set(demo.ARMS)

        # Every run carries the audit-labelled fields.
        for r in runs:
            assert 0 <= r["false_commits"] <= r["n_commits"]
            assert "reached_optimum" in r
            assert "winners_curse_gap" in r

        summ = block["summary"]
        greedy = summ["greedy"]
        greedy_reeval = summ["greedy_reeval"]
        pace = summ["pace"]

        # New reporting: every arm summary carries per-seed std + pooled
        # false-commit fields (means alone are no longer the whole story).
        for arm_summary in (greedy, greedy_reeval, pace):
            assert "false_commits_std" in arm_summary
            assert "false_commit_rate_pooled" in arm_summary
            assert "n_commits_total" in arm_summary

        # Eval-cost decomposition: greedy_reeval and pace pay the SAME (2x) eval
        # cost -- they differ only in the decision rule -- while frozen greedy
        # pays 1x. This is what makes pace-vs-greedy_reeval the deconfounded
        # measure of the gate's effect.
        assert greedy_reeval["dev_evals_mean"] == pace["dev_evals_mean"]
        assert greedy["dev_evals_mean"] < greedy_reeval["dev_evals_mean"]

        # Qualitative direction the demo demonstrates, in EVERY regime: the gate
        # commits no more false positives than EITHER greedy arm (it commits
        # fewer). pace <= greedy_reeval is the deconfounded, same-cost contrast.
        assert pace["false_commits_mean"] <= greedy_reeval["false_commits_mean"] + 1e-9
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
