# RQ1 Sphere H1 — literature-domain attribution

**Status:** Phase A complete 2026-08-26 (**20/20**, **GO**). Phase B **read** 2026-08-27 (**40/40**; finished 2026-08-26 23:51 +03).
**Does not amend** Holm, CA H1, maze empty 2×2, or Sphere H2.

Fontaine linear-projection Sphere ($D{=}20$, $100\times100$) as a second
domain for the **H1 leftover**: live parent-level surrogate scalars versus
stub placeholders, at matched `target_selection`. The LLM writes 20 local
deltas on an anonymous $\mathbb{R}^{20}$ parent, dated
`gpt-4o-mini-2024-07-18`. This is not Sphere H2 (no LLM) and not a named-field
CA/maze editor.

## Why Sphere (and why H1 is still a scalar leftover)

H1 on CA/maze is the incremental effect of **providing** two parent-level
scalars through a prompt that already contains true parent fitness. Sphere has
no named genotype fields; the analog is:

| Channel | Stub | Live (hints) |
|---------|------|----------------|
| True parent objective and measures | in prompt | in prompt |
| `predicted fitness`, `uncertainty` | constants $0.5$ / $1.0$ | ensemble MLP on parent $\theta$ |

The LLM actuator is parent-conditioned **by construction**: the model returns
`{"deltas":[20]}` with $|d_i|\le 1.5$ and at least four $|d_i|\ge 0.05$; the
child is `clip(parent + deltas, [-5.12, 5.12])`. Parse failure falls back to
the genetic Gaussian ($\sigma{=}0.5$). If the model cannot emit a valid local
edit, the 2×2 can floor; that is a domain result, not a silent CA replica.

## Phase A — genetic policy probe (cheap, required)

**Question:** on Sphere, with archive headroom, does `min_fitness_frontier`
vs `uniform_frontier` move coverage?

| Knob | Lock |
|------|------|
| Mix | 20 random (box) + 30 Gaussian $\sigma{=}0.5$ |
| Archive | empty $100\times100$ |
| Budget | **5{,}000** proposal slots; primary = terminal coverage |
| Seeds | 0–9 |
| Tier | `q1-sphere-genetic-policy` |

```bash
./scripts/run_experiment_batch.sh q1-sphere-genetic-policy 0 9
python scripts/analyze_sphere_genetic_policy.py
```

### Go / no-go (descriptive; not Holm)

Read paired seeds 0–9, minfit minus uniform, at 5k:

| Decision | Rule |
|----------|------|
| **GO** Phase B | $\lvert\mathrm{mean}\,\Delta\rvert \ge 5$ pp, **or** 8/10 same sign and $\lvert\mathrm{mean}\,\Delta\rvert \ge 3$ pp |
| **NO-GO** | $\lvert\mathrm{mean}\,\Delta\rvert < 2$ pp |
| **Borderline** | otherwise: report, do **not** spend LLM |

NO-GO means Sphere policy does not move coverage in this mix, so a stub/hints
2×2 cannot replay the CA/maze inversion. That is a valid negative.

### Result (2026-08-26)

**GO.** @ 5k: minfit **13.59 ± 0.46%** vs uniform **19.38 ± 0.31%**;
paired Δ **−5.79 ± 0.55 pp** (10/10). Headroom remains (10k-cell archive).
Magnitude is much smaller than maze genetic −40 pp; still clears the 5 pp
rule. Details: [`experiments/q1-sphere-genetic-policy/ANALYSIS.md`](experiments/q1-sphere-genetic-policy/ANALYSIS.md).

## Phase B — LLM 2×2 (only on GO)

Four cells, **not** Holm, **not** Sphere H2, **not** occupancy-matched to the
CA 38.84% floor:

`{stub, live} × {minfit, uniform}` on Sphere, dated `gpt-4o-mini-2024-07-18`,
call logs.

| Knob | Lock |
|------|------|
| Archive | **Empty** |
| Budget | **5{,}000** (100 × 50); 20R+30L after 100 random |
| Model | `gpt-4o-mini-2024-07-18` via `worldspace/specs/llm_world_generator_rq1_fixed_openai.yaml` |
| Cells | `llm_stub_minfit`, `llm_stub_uniform`, `llm_hints_minfit`, `llm_hints_uniform` |
| Seeds | 0–9 (40 runs) |
| Surrogate | off on stub; frozen `artifacts/surrogate/sphere_h1_mlp.joblib` on live (gate off) |
| H1 leftover | live − stub at each matched policy |
| Tier | `q1-rq1-sphere-factorial` |

Train the ensemble once (seed 0, 20k box samples, 3 MLPs) before hints runs:

```bash
uv run python scripts/run_sphere_rq1.py train-surrogate \
  --out artifacts/surrogate/sphere_h1_mlp.joblib
```

Launch (after Phase A GO **and** a passing live preflight):

```bash
uv run python scripts/preflight_sphere_llm.py
# gates: parse ≥ 0.90, fallback ≤ 0.10, mean L2 ≥ 0.1 (20 calls)
./scripts/run_rq1_sphere_factorial_nohup.sh
# override: SPHERE_FACTORIAL_WORKERS=4 ./scripts/run_rq1_sphere_factorial_nohup.sh
python scripts/analyze_rq1_sphere_factorial.py
```

Malformed chat JSON, HTTP 429/5xx/520–524, SSL/timeouts, and a 200 body
that is not a chat envelope (`LLM response …`) retry then fall back to
the genetic Gaussian; they must **not** abort the seed. Missing
`OPENAI_API_KEY` still aborts (launch script checks it). High
`fallback_rate` does not crash the grid — it contaminates leftover
(both arms become genetic); the analyzer warns at mean fallback
$>0.10$ or parse $<0.90$.

**Reporting:** Descriptive paired mean $\Delta$ coverage. Not a new Holm
family. Does not identify CA Holm $+15.09$ or maze $+39$ pp. **Journal PDF:
appendix table only** (`tab:sphere-h1` in `draft_v0.tex`). Not main text.
Maze remains the second-evaluator $2\times2$.

### Result (2026-08-27)

**40/40.** @ 5k, dated `gpt-4o-mini-2024-07-18`, parse $\ge 99.99\%$,
fallback $\le 0.01\%$:

| Cell | Coverage % |
|------|------------|
| stub minfit | $10.26 \pm 0.12$ |
| stub uniform | $12.94 \pm 0.23$ |
| live minfit | $10.26 \pm 0.15$ |
| live uniform | $12.73 \pm 0.23$ |

Policy (uniform − minfit): **$+2.68$ / $+2.47$ pp** (both 10/10).
H1 leftover (live − stub): **$+0.00$ pp** at minfit (6/4);
**$-0.21$ pp** at uniform (1/9; paired $t$ 95% CI $[-0.33,-0.09]$).
Same qualitative split as CA/maze (policy $\gg$ leftover); LLM policy
gap is compressed vs genetic Phase A ($-5.79$ pp). Not portable pp.
Details: [`experiments/q1-rq1-sphere-factorial/ANALYSIS.md`](experiments/q1-rq1-sphere-factorial/ANALYSIS.md).

## What this does not do

- Does not reuse Sphere H2 `me_uniform` / `me_filter` (no LLM, 32.5k, random elite parent).
- Does not claim portable percentage points.
- Does not test named-field JSON editing; deltas keep the emitter local so
  policy *can* act. If parse/fallback rates are high, report them; do not
  reread leftover as a scalar effect.
- Does not abort a seed on malformed model output. Persistent parse or
  transport failure uses genetic fallback. That is the same maze contract,
  plus Sphere also swallows a 200 chat envelope that is HTML / missing
  `choices` / empty `content` (`LLM response …`). Do not treat a
  high-fallback grid as an H1 leftover.
