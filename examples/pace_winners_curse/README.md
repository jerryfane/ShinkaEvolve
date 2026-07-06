# PACE acceptance gate: the winner's-curse A/B demo (two regimes)

A small, **fully synthetic, no-LLM** experiment that isolates and measures the
effect of ShinkaEvolve's PACE acceptance gate on the *winner's curse* — the
tendency of greedy "did the number go up?" selection to admit lucky-noise
candidates and report an inflated best score.

Mirroring the PACE paper's two-regime design (arXiv 2606.08106 §5.1/§5.2), it
runs **both arms under two regimes** and writes a combined report.

```bash
uv run python examples/pace_winners_curse/run_demo.py
```

Outputs (written into this directory):

- `results.json` — **keyed by regime** (`planted`, `null_only`); each holds the
  regime's config, every run's metrics, and a per-arm summary.
- `winners_curse.png` — a **2×2** chart: rows = regimes, cols = (1) reported best
  vs. true quality, (2) false commits per run.

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

Per run: a stream of `T = 60` proposal rounds, `n_dev = 40`, 5 seeds, both arms.
Both arms hill-climb an evolving incumbent over the *same* candidate stream:

- **Arm A — stock greedy**: accept iff candidate dev score `>` incumbent dev
  score (the ubiquitous rule).
- **Arm B — PACE-gated**: accept iff `PaceGate` commits (an anytime-valid
  e-process over the McNemar-discordant pairs, `alpha = 0.05`, `lam = 0.5`), on
  the same paired dev instances.

Metrics per run: reported-best dev vs. its true quality (the **champion gap**),
commits, false commits (audit-labelled: committed with `p_true <= incumbent`),
whether the shipped best reached the planted optimum, and total dev evaluations.

### The two regimes

- **`planted`** — most candidates are nulls (`p_true == p_base`, default 0.5); a
  few planted rounds are true **improvements** (`+delta`, `p_true = 0.8`) or
  **regressions** (`-delta`). A genuine improvement *does* exist to be found.
- **`null_only`** — **no genuine improvement anywhere**: every candidate is a
  true null (`p_true == p_base`, the incumbent's own true quality) plus a few
  true regressions (`-delta`). True quality can therefore *never* rise above
  baseline, so every accepted candidate is by construction a false improvement.

## Results

From a default run (5 seeds/arm, both regimes). Re-running reproduces the
direction; exact numbers vary with the seeds.

### Regime `planted` — equal power + false-commit suppression

| metric | Arm A greedy | Arm B PACE |
| --- | --- | --- |
| reported best dev | 0.860 | 0.840 |
| true quality (audit) | 0.800 | 0.800 |
| champion gap | +0.060 | +0.040 |
| commits / run | 3.8 | 1.6 |
| **false commits / run** | **2.8** | **0.6** |
| **false-commit rate** | **72.7%** | **23.3%** |
| reached the optimum | 100% | 100% |
| dev evaluations / run | 2400 | 4800 |

**Both arms reliably capture the planted improvement (100% each), but greedy pays
for it with ~5× the false commits** — 72.7% of its commits are lucky-noise
candidates no better than the incumbent, versus 23.3% for PACE. This regime shows
the gate is *equally powered* (it still finds the real gain) while suppressing
false commits toward its `alpha` guarantee.

Note that here the **champion gap barely separates the arms** (+0.060 vs +0.040):
once a genuine improvement dominates the stream, *both* arms' reported best is
anchored near the real optimum (0.8), so the reported-vs-true gap is small for
both. The gap is not the discriminating metric in this regime — the false-commit
rate is.

### Regime `null_only` — the champion winner's curse + churn suppression

| metric | Arm A greedy | Arm B PACE |
| --- | --- | --- |
| reported best dev | 0.675 | 0.585 |
| true quality (audit) | 0.500 | 0.500 |
| **champion gap** | **+0.175** | **+0.085** |
| commits / run (all false) | 3.0 | 1.2 |
| **false commits / run** | **3.0** | **1.2** |
| false-commit rate | 100% | 60%* |
| reached the optimum | 0% (none exists) | 0% (none exists) |
| dev evaluations / run | 2400 | 4800 |

With no genuine improvement in the stream, true quality stays pinned at 0.500 for
both arms — yet **greedy reports a best of 0.675, a +0.175 champion gap that is
pure winner's curse**: it is the running maximum of noisy `Bernoulli(40, 0.5)`
draws, and *every one* of its ~3 commits is churn (100% false). PACE commits far
fewer (1.2/run) and keeps its reported best much closer to the truth (+0.085 gap,
roughly half of greedy's).

This is the regime where the **champion gap cleanly separates the arms** — and
that is the whole reason both regimes exist. The champion-gap metric only
discriminates when *no true improvement dominates*; when one does (regime
`planted`), both arms anchor near the real optimum and the gap collapses. So the
two regimes are complementary: `planted` proves equal power + false-commit
suppression, `null_only` isolates and quantifies the champion winner's curse.

**Honest caveats (not tuned to force the story):**

- PACE in `null_only` does **not** commit exactly zero. Its false-commit rate is
  bounded by the anytime-valid `alpha` *per comparison*, but the run streams
  ~60 sequential comparisons against a re-evaluated incumbent, so a few nulls
  slip through within budget (≈1.2/run, ≈2% per comparison — well under `alpha`).
  In this seed set PACE commits 0 in 2/5 seeds (reported = true = 0.500, gap 0)
  and 1–3 in the others. The separation is real (~2.5× fewer false commits, ~½
  the gap) but it is a *suppression*, not an *elimination*.
- \*The 60% "false-commit rate" for PACE is an averaging artifact of the two
  zero-commit seeds (rate is defined as 0 when a run makes no commits). In
  `null_only` **every** commit is false by construction, so the honest churn
  metric is the false-commit *count* (3.0 vs 1.2), not the rate.

### The cost of paired evaluation

The gate re-evaluates the incumbent on each round's paired instances, so it
consumes **~2× the dev evaluations** of stored-score greedy (4800 vs 2400 per run
in this config). The gate buys statistical discipline, not a free lunch.

## What this does and does not show

- **Does**: that the *acceptance-gate decision rule*, in isolation and against a
  known ground truth, sharply cuts the false-commit rate (both regimes) and the
  champion winner's-curse inflation (`null_only`), while still capturing genuine
  improvements at full power (`planted`).
- **Does not**: model full LLM-driven evolution, real task difficulty, code
  semantics, or claim a benefit on any particular real benchmark. The candidate
  stream is fabricated; `p_true` is assigned, not learned. This is a mechanism
  demo of the gate, nothing more.

## Configuration

All knobs live in `DemoConfig` in `run_demo.py` (`regime`, `n_dev`, `n_rounds`,
`n_seeds`, `p_base`, `delta`, `n_improvements`, `n_regressions`, `alpha`, `lam`,
`seed0`). `run_experiment` runs both regimes automatically; `null_only` forces
`n_improvements = 0` regardless of the knob (the `planted` regime uses it
verbatim). `delta` is the main lever for `planted`: too small and the gate cannot
detect the planted improvement in `n_dev` instances (low capture); large enough
(the default) and the gate captures it while still rejecting the null noise
greedy accepts.
