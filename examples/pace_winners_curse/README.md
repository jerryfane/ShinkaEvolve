# PACE acceptance gate: the winner's-curse A/B demo

A small, **fully synthetic, no-LLM** experiment that isolates and measures the
effect of ShinkaEvolve's PACE acceptance gate on the *winner's curse* — the
tendency of greedy "did the number go up?" selection to admit lucky-noise
candidates and report an inflated best score.

```bash
uv run python examples/pace_winners_curse/run_demo.py
```

Outputs (written into this directory):

- `results.json` — every run's metrics plus a per-arm summary and the config.
- `winners_curse.png` — a two-panel chart: (1) reported best vs. true quality,
  (2) false commits per arm.

## Why this is honest

The whole point is that **the ground truth is known by construction**, so we can
audit every acceptance decision — something you can never do on a real task.

- Each candidate has a fixed TRUE per-instance success probability `p_true`.
- A **dev evaluation** is `n_dev` (default 40) independent `Bernoulli(p_true)`
  draws over a *shared* instance set — the paired-seed eval model: candidate and
  incumbent are scored on the same instances, but each draws its own outcome per
  instance. That per-instance noise is exactly what lets a lucky null out-score
  the incumbent.
- An **audit** is just `p_true` itself — the ground truth, no sampling needed.

## What it exercises (no reimplementation)

The gated arm does **not** reimplement the gate. It imports the real
`shinka.core.acceptance.PaceGate` and drives the real
`ShinkaEvolveRunner._apply_pace_gate` against a real `ProgramDatabase` — genuine
`gate_passed` / archive / best-program semantics — reusing the harness pattern
from `tests/test_pace_gate_integration.py`. The greedy arm uses the same real
`ProgramDatabase` for best-program tracking.

## The setup

Per run: a stream of `T = 60` proposal rounds. Most candidates are **nulls**
(`p_true == p_base`, default 0.5); a few planted rounds are true **improvements**
(`+delta`, default `p_true = 0.8`) or **regressions** (`-delta`). Both arms
hill-climb an evolving incumbent over the *same* candidate stream:

- **Arm A — stock greedy**: accept iff candidate dev score `>` incumbent dev
  score (the ubiquitous rule).
- **Arm B — PACE-gated**: accept iff `PaceGate` commits (an anytime-valid
  e-process over the McNemar-discordant pairs, `alpha = 0.05`, `lam = 0.5`), on
  the same paired dev instances.

Five seeds × both arms. Metrics per run: reported-best dev vs. its true quality
(the inflation gap), commits, false commits (audit-labelled: committed with
`p_true <= incumbent`), whether the shipped best reached the planted optimum,
and total dev evaluations consumed.

## Representative result

From a default run (5 seeds/arm):

| metric | Arm A greedy | Arm B PACE |
| --- | --- | --- |
| reported best dev | 0.860 | 0.840 |
| true quality (audit) | 0.800 | 0.800 |
| **winner's-curse gap** | **+0.060** | **+0.040** |
| commits / run | 3.8 | 1.6 |
| **false commits / run** | **2.8 (73%)** | **0.6 (23%)** |
| reached the optimum | 100% | 100% |
| dev evaluations / run | 2400 | 4800 |

The headline: **both arms reliably capture the planted improvement, but greedy
pays for it with roughly 5× the false commits** — most of its commits (73%) are
lucky-noise candidates that are no better than the incumbent. The PACE gate
holds the false-commit rate down near its `alpha` guarantee and, because it
commits the first *statistically significant* candidate rather than chasing the
dev-score maximum, reports a less-inflated best. (Exact numbers vary with the
seeds; re-running reproduces the direction.)

Note the honest cost: the gate re-evaluates the incumbent on each round's paired
instances, so it consumes ~2× the dev evaluations of stored-score greedy. The
gate buys statistical discipline, not free lunch.

## What this does and does not show

- **Does**: that the *acceptance-gate decision rule*, in isolation and against a
  known ground truth, sharply cuts the false-commit rate and the winner's-curse
  inflation while still capturing genuine improvements.
- **Does not**: model full LLM-driven evolution, real task difficulty, code
  semantics, or claim a benefit on any particular real benchmark. The candidate
  stream is fabricated; `p_true` is assigned, not learned. This is a mechanism
  demo of the gate, nothing more.

## Configuration

All knobs live in `DemoConfig` in `run_demo.py` (`n_dev`, `n_rounds`, `n_seeds`,
`p_base`, `delta`, `n_improvements`, `n_regressions`, `alpha`, `lam`, `seed0`).
`delta` is the main lever: too small and the gate cannot detect the planted
improvement in `n_dev` instances (low capture); large enough (the default) and
the gate captures it while still rejecting the null noise greedy accepts.
