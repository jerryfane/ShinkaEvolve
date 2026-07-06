# PACE acceptance gate: the winner's-curse A/B/C demo (three arms, two regimes)

A small, **fully synthetic, no-LLM** experiment that isolates and measures the
effect of ShinkaEvolve's PACE acceptance gate on the *winner's curse* — the
tendency of greedy "did the number go up?" selection to admit lucky-noise
candidates and report an inflated best score.

**What `alpha` promises (read first).** The gate's e-process is anytime-valid
**per comparison**: by Ville's inequality, the probability that any *single*
null candidate ever commits is bounded by `alpha`. It does **not** bound the
number of false commits over a whole run. A run streams ~`T` sequential
comparisons, so the *expected* false commits under the null is roughly
`(per-comparison false-commit rate) × T` — a per-comparison bound multiplied by
the number of comparisons. There is **no run-level (family-wise) control here**;
that is the job of the wealth-ledger layer (a global betting budget across
comparisons), which is **out of scope** for this mechanism demo. So do not read
the per-run false-commit count as "near `alpha`" — it accumulates many
per-comparison bets.

Mirroring the PACE paper's two-regime design (arXiv 2606.08106 §5.1/§5.2), it
runs **three arms under two regimes** and writes a combined report.

```bash
uv run python examples/pace_winners_curse/run_demo.py
```

Outputs (written into this directory):

- `results.json` — **keyed by regime** (`planted`, `null_only`); each holds the
  regime's config, every run's metrics, and a per-arm summary (means, stds, and
  **pooled** false-commit counts).
- `winners_curse.png` — a **2×2** chart: rows = regimes, cols = (1) reported best
  vs. true quality, (2) false commits per run — three arms per panel.

## The three arms (a deconfounded decomposition)

The point of the third arm is to **deconfound** the gate's effect from the mere
act of re-evaluating the incumbent:

| arm | re-evaluates incumbent? | accept rule | eval cost/round |
| --- | --- | --- | --- |
| **`greedy`** (frozen) | no — uses the incumbent's *frozen stored* dev score | `cand_dev > frozen_inc_dev` | `1 × n_dev` |
| **`greedy_reeval`** (re-eval, no gate) | **yes**, fresh every round | `cand_dev > reevaluated_inc_dev` | `2 × n_dev` |
| **`pace`** (re-eval + gate) | **yes**, fresh every round | the real `PaceGate` commits | `2 × n_dev` |

The frozen greedy bar is itself a winner's-cursed maximum (an *inflated* bar).
`greedy_reeval` removes that confound by re-evaluating the incumbent on the same
paired instances — **the exact eval cost and protocol PACE pays** — while keeping
the naive greedy rule. Because `greedy_reeval` and `pace` differ in **nothing but
the decision rule**, the contrast between them is the gate's *marginal* effect:

```
greedy  ──[remove frozen-inflated-score confound]──▶  greedy_reeval
greedy_reeval  ──[swap greedy rule for the e-process gate]──▶  pace
```

**`pace` vs `greedy_reeval` is therefore the clean, isolated measure of what the
gate itself buys.** (`greedy` vs `greedy_reeval` shows what re-evaluation alone
does — and, as the numbers below show, on its own it makes churn *dramatically
worse*, because a re-evaluated incumbent regresses toward its true mean and is
much easier to beat by noise.)

## Why this is honest

The whole point is that **the ground truth is known by construction**, so we can
audit every acceptance decision — something you can never do on a real task.

- Each candidate has a fixed TRUE per-instance success probability `p_true`.
- A **dev evaluation** applies **real paired difficulty**: each round draws a
  per-instance offset `e_i ~ Normal(0, sigma_inst=0.15)`, clipped so the success
  probability `clip(p_true + e_i, 0.05, 0.95)` stays valid. The offset `e_i` is
  **shared** by candidate and incumbent on the same instance index (a hard
  instance depresses both together); given that shared difficulty, the two
  systems draw **independent** `Bernoulli` outcomes. This shared difficulty is
  what makes the McNemar pairing **genuinely variance-reducing**: the paired
  candidate-minus-incumbent difference has lower variance than an unpaired one,
  so the gate's discordant-pair evidence is cleaner. The independent per-system
  noise on top is exactly what still lets a lucky null out-score the incumbent.
- An **audit** is just `p_true` itself — the ground truth, no sampling needed.

## What it exercises (no reimplementation)

The gated arm does **not** reimplement the gate. It imports the real
`shinka.core.acceptance.PaceGate` and drives the real
`ShinkaEvolveRunner._apply_pace_gate` against a real `ProgramDatabase` — genuine
`gate_passed` / archive / best-program semantics — reusing the harness pattern
from `tests/test_pace_gate_integration.py`. The greedy arms use the same real
`ProgramDatabase` for best-program tracking.

## The setup

Per run: a stream of `T = 60` proposal rounds, `n_dev = 40`, **20 seeds**, all
three arms hill-climb an evolving incumbent over the *same* candidate stream (the
candidate vector for a given `(seed, round)` is identical across arms, so the
comparison is strictly apples-to-apples).

Metrics per run: reported-best dev vs. its true quality (the **champion gap**),
commits, false commits (audit-labelled: committed with `p_true <= incumbent`),
whether the shipped best reached the planted optimum, and total dev evaluations.
Summaries report per-arm **means ± std over the 20 seeds**, plus a **pooled**
false-commit rate (total false commits / total commits across seeds — undistorted
by zero-commit seeds, unlike a mean of per-run rates).

### The two regimes

- **`planted`** — most candidates are nulls (`p_true == p_base`, default 0.5); a
  few planted rounds are true **improvements** (`+delta`, `p_true = 0.8`) or
  **regressions** (`-delta`). A genuine improvement *does* exist to be found.
- **`null_only`** — **no genuine improvement anywhere**: every candidate is a
  true null (`p_true == p_base`, the incumbent's own true quality) plus a few
  true regressions (`-delta`). True quality can therefore *never* rise above
  baseline, so every accepted candidate is by construction a false improvement.

## Results

From a default run (**20 seeds/arm**, both regimes; means ± std). The run is
**fully seeded**: the identical command reproduces **byte-identical** results;
change `seed0` (in `DemoConfig`) to vary the sample.

### Regime `planted` — power preserved, false commits suppressed

| metric | `greedy` (frozen) | `greedy_reeval` (re-eval, no gate) | `pace` (re-eval + gate) |
| --- | --- | --- | --- |
| reported best dev | 0.831 ± 0.054 | 0.807 ± 0.068 | 0.778 ± 0.078 |
| true quality (audit) | 0.785 | 0.785 | 0.785 |
| champion gap | +0.046 ± 0.069 | +0.022 ± 0.085 | −0.008 ± 0.071 |
| commits / run | 3.4 ± 1.2 | 8.6 ± 6.0 | 1.4 ± 0.6 |
| **false commits / run** | **2.5 ± 1.2** | **7.4 ± 5.6** | **0.4 ± 0.6** |
| **false-commit rate (pooled)** | **72%** (49/68) | **87%** (148/171) | **30%** (8/27) |
| reached the optimum | 95% | 95% | 95% |
| dev evaluations / run | 2400 | 4800 | 4800 |

**All three arms capture the planted improvement equally (95% each — the gate is
not under-powered), but the greedy rules pay for it with far more false commits.**
The deconfounding is the headline: re-evaluating the incumbent *without* the gate
(`greedy_reeval`) does **not** help — it commits **18.5× the false commits of
`pace`** (7.4 vs 0.4/run), *more* than frozen greedy, because the re-evaluated
incumbent regresses toward its true mean and is easier to beat by noise. Only the
gate suppresses false commits (0.4/run, pooled 30%), at the same 4800-eval cost
as `greedy_reeval`. So the suppression is attributable to **the gate**, not to
re-evaluation.

Note the **champion gap barely separates the arms here** (+0.046 / +0.022 /
−0.008): once a genuine improvement dominates the stream, every arm's reported
best is anchored near the real optimum (0.8), so the reported-vs-true gap is small
for all. The gap is not the discriminating metric in this regime — the
false-commit rate is.

### Regime `null_only` — the champion winner's curse + churn suppression

| metric | `greedy` (frozen) | `greedy_reeval` (re-eval, no gate) | `pace` (re-eval + gate) |
| --- | --- | --- | --- |
| reported best dev | 0.676 ± 0.040 | 0.676 ± 0.040 | 0.591 ± 0.080 |
| true quality (audit) | 0.500 | 0.500 | 0.500 |
| **champion gap** | **+0.176 ± 0.040** | **+0.176 ± 0.040** | **+0.091 ± 0.080** |
| commits / run (all false) | 3.0 ± 1.1 | 24.9 ± 3.1 | 1.1 ± 0.8 |
| **false commits / run** | **3.0 ± 1.1** | **24.9 ± 3.1** | **1.1 ± 0.8** |
| false-commit rate (pooled) | 100% | 100% | 100% |
| reached the optimum | 0% (none exists) | 0% (none exists) | 0% (none exists) |
| dev evaluations / run | 2400 | 4800 | 4800 |

With no genuine improvement in the stream, true quality stays pinned at 0.500 for
every arm — yet **greedy reports a best of 0.676, a +0.176 champion gap that is
pure winner's curse** (the running maximum of noisy `Bernoulli(40, ~0.5)` draws),
and *every* commit is churn (100% false by construction).

The third arm makes the mechanism unmistakable. `greedy_reeval` has the **same
champion gap and the same reported best as frozen greedy** (+0.176; the two arms
commit the same all-time-max candidate) — so **re-evaluation alone closes none of
the gap** — but it commits **~25 false candidates per run**, an order of magnitude
more churn than frozen greedy's 3.0 and **22.6× `pace`'s** 1.1. `pace` is the only
arm that both cuts the churn (1.1/run) *and* halves the champion gap (+0.091 vs
+0.176). The gate — not the re-evaluation — is doing the work.

**Honest caveats (not tuned to force the story):**

- PACE in `null_only` does **not** commit exactly zero. Its per-*comparison*
  false-commit probability is bounded by the anytime-valid `alpha`, but the run
  streams ~60 sequential comparisons against a re-evaluated incumbent, so the
  expected per-run count is ≈ per-comparison-rate × T ≈ 1.1/run here — small, but
  **not** "near `alpha` per run" (there is no run-level guarantee; see the top
  note). The separation is a **suppression**, not an **elimination**: ~23× fewer
  false commits than `greedy_reeval` and about half the champion gap.
- In `null_only` **every** commit is false by construction, so the pooled rate is
  trivially 100% for all arms; the honest churn metric there is the false-commit
  *count* (3.0 / 24.9 / 1.1 per run), not the rate.

### The cost of paired evaluation

Both re-evaluating arms (`greedy_reeval`, `pace`) re-score the incumbent each
round, so they consume **2× the dev evaluations** of stored-score frozen greedy
(4800 vs 2400 per run in this config). `greedy_reeval` and `pace` pay the *same*
cost — the only difference between them is the decision rule, which is precisely
why their contrast isolates the gate. The gate buys statistical discipline for the
re-eval budget, not a free lunch.

## What this does and does not show

- **Does**: that the *acceptance-gate decision rule*, in isolation and
  **deconfounded** from re-evaluation (via the `greedy_reeval` arm), against a
  known ground truth, sharply cuts the false-commit rate (both regimes) and the
  champion winner's-curse inflation (`null_only`), while still capturing genuine
  improvements at full power (`planted`). It also shows re-evaluation *alone* is
  not a fix — it worsens churn.
- **Does not**: model full LLM-driven evolution, real task difficulty, code
  semantics, or claim a benefit on any particular real benchmark. The candidate
  stream is fabricated; `p_true` is assigned, not learned. It provides
  per-*comparison* false-commit control, **not** run-level control (that is the
  wealth-ledger layer's job, out of scope here). This is a mechanism demo of the
  gate, nothing more.

## Configuration

All knobs live in `DemoConfig` in `run_demo.py` (`regime`, `n_dev`, `n_rounds`,
`n_seeds`, `p_base`, `delta`, `sigma_inst`, `n_improvements`, `n_regressions`,
`alpha`, `lam`, `seed0`). `run_experiment` runs both regimes automatically;
`null_only` forces `n_improvements = 0` regardless of the knob (the `planted`
regime uses it verbatim). `delta` is the main lever for `planted`: too small and
the gate cannot detect the planted improvement in `n_dev` instances (low capture);
large enough (the default) and the gate captures it while still rejecting the null
noise the greedy arms accept. `sigma_inst` controls how heterogeneous the paired
instances are (how much the shared per-instance difficulty varies).
