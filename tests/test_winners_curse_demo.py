"""Fast smoke test for the PACE winner's-curse A/B demo.

Runs a tiny config (T=10, 2 seeds) end to end and asserts the demo:

* runs against the real gate + real ``ProgramDatabase`` without error,
* produces both artifacts (``results.json`` + ``winners_curse.png``),
* returns a well-formed results dict, and
* reproduces the qualitative direction the demo exists to show: the PACE-gated
  arm's false-commit rate is no worse than stock greedy's.

This is a *smoke* test, not a statistical one: with only 2 seeds the exact
numbers are noisy, so we assert the robust inequality (gated <= greedy) rather
than a magnitude.
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

    # Well-formed shape: 2 seeds x 2 arms.
    assert len(results["runs"]) == 2 * cfg.n_seeds
    arms = {r["arm"] for r in results["runs"]}
    assert arms == {"greedy", "pace"}

    summary = results["summary"]
    greedy, pace = summary["greedy"], summary["pace"]

    # Every run carries the audit-labelled fields.
    for r in results["runs"]:
        assert 0 <= r["false_commits"] <= r["n_commits"]
        assert "reached_optimum" in r
        assert "winners_curse_gap" in r

    # Qualitative direction the demo demonstrates: the gate does not commit more
    # false positives than greedy (it should commit far fewer).
    assert pace["false_commit_rate_mean"] <= greedy["false_commit_rate_mean"] + 1e-9
    assert pace["false_commits_mean"] <= greedy["false_commits_mean"] + 1e-9
