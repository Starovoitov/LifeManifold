# Controlled Attribution of Performance Gains in LLM-Driven Quality-Diversity Search

## Research programme (planning document)

**Status:** planning, not a protocol, preregistration, or claim of confirmatory
evidence. The existing LifeManifold results are motivating evidence and may be
used to design the programme, but they are not prospective tests of the new
cross-domain claims. A separate dated protocol must be frozen after feasibility
pilots and before confirmatory runs.

**Working paper title:** *Controlled Attribution of Performance Gains in
LLM-Driven Quality-Diversity Search*

**Target genre:** systematic empirical methodology / benchmarking at the
intersection of evolutionary computation, quality-diversity (QD), and machine
learning. The intended primary venue is ACM TELO; TMLR is the
soundness-oriented alternative and IEEE TEVC is the high-risk archival target.

## 1. Objective

Build and evaluate a reusable controlled-attribution procedure for LLM-driven
QD pipelines. The paper must distinguish the effect of the LLM variation
operator from effects caused by orchestration choices that are often bundled
with it:

- archive initialization and warm start;
- parent / target selection;
- prompt-side information;
- generator identity and mutation scale;
- validity repair and fallback;
- evaluation gating;
- archive replacement;
- proposal, evaluation, LLM-call, token, wall-time, and monetary budgets;
- static versus state-dependent allocation of LLM calls.

The programme is not designed to prove that LLM emitters generally help or
generally fail. It is designed to estimate which component produced an
observed gain, how that attribution changes across domains, and when a bundled
comparison changes the ranking or scientific conclusion.

## 2. Contribution required for the broad title

The broad title is earned only if the final study contains all of the following:

1. A formal component taxonomy and a reusable attribution procedure.
2. Controlled estimates for multiple distinct performance gains, not only the
   historical `stub` versus `hints` contrast.
3. Public, recognized tasks beyond the LifeManifold CA / maze stack.
4. More than coverage: QD-score, elite quality, validity, anytime performance,
   and resource costs.
5. Common accounting across proposal, evaluation, LLM-call, token, wall-time,
   and dollar budgets.
6. Factorial or nested-factorial contrasts that expose interactions instead of
   assigning an interaction to one component.
7. A prospective confirmatory tier separated from all feasibility and design
   pilots.
8. Reproducible code, immutable task definitions, model/version records, seeds,
   cell-level summaries, and analysis scripts.
9. A cross-domain synthesis stating where the same attribution pattern holds,
   where it changes, and why.

Reproducing headline gains from two or three published LLM+QD /
LLM+evolutionary systems is an **optional maximum extension**, described in
Section 11. It is not required for the mandatory core programme.

## 3. Claims and non-claims

### 3.1 Candidate primary claims

The protocol will select exact wording after pilots, but the intended claim
classes are:

- **Bundling bias:** a complete-pipeline contrast can differ materially from
  the controlled LLM-component contrast.
- **Interaction:** target selection, archive state, and generator behavior can
  interact, so a gain cannot always be decomposed into additive component
  shares.
- **Budget dependence:** rankings can change between proposal-, evaluation-,
  call-, token-, and wall-time-indexed views.
- **Domain heterogeneity:** attribution patterns depend on representation,
  validity constraints, evaluator geometry, and archive occupancy.
- **Auditability:** the proposed procedure detects these changes using
  pre-specified cells, logs, and estimands.

### 3.2 Claims explicitly excluded

- No universal claim that LLMs outperform genetic emitters.
- No universal claim that target selection explains most published LLM gains.
- No portable percentage-point effect across archive sizes or domains.
- No causal claim about an omitted component.
- No attribution of an interaction to one factor merely because that factor
  was changed last.
- No use of the same data for operator calibration and confirmatory testing.

## 4. Mandatory attribution framework

### 4.1 Unit of analysis

The primary unit is one independently seeded QD run from a frozen initial
archive. Runs are paired or blocked by:

- domain instance;
- initial archive / occupancy floor;
- random seed;
- LLM model version;
- calendar/API block when remote inference is used.

Repeated candidates within one run are not treated as independent replicates.

### 4.2 Pipeline treatment vector

Every reported arm must be representable by an explicit treatment vector:

`initialization × selector × generator × prompt channel × repair/fallback ×
gate × replacement × allocation × budget`.

No arm label such as `hints`, `LLM`, or `full` is sufficient by itself.
Configuration manifests must contain the complete vector and its hash.

### 4.3 Estimands

For every claimed gain, report:

- the original or natural **bundled contrast**;
- the **controlled component contrast** with non-target components matched;
- relevant factor interactions;
- absolute cell means and paired differences;
- an anytime contrast under each applicable budget axis;
- validity/fallback rates and resource costs.

If a full factorial is infeasible, use a nested design and state exactly which
effects are not identified. Do not use an order-dependent subtraction as an
additive decomposition. Shapley-style summaries may be secondary only when
all required cells exist; raw factorial contrasts remain primary.

### 4.4 Mandatory modular experiment blocks

The programme avoids one unmanageable full factorial. Each selected domain
must support Block A and Block B. Block C is required on at least two domains,
one of which must be public.

**Block A — prompt channel × target selection**

- prompt channel: constant/ablated versus live task-relevant information;
- target selection: a neutral reference versus the policy used by the bundled
  pipeline;
- four cells with matched model, prompt template except the channel, archive,
  generator, repair, replacement, and budgets;
- primary outputs: policy effects at both channel levels, channel effects at
  both policies, and their interaction.

**Block B — generator attribution**

- LLM emitter;
- calibrated local genetic mutation;
- random or grammar-valid mutation baseline;
- identical archive, target selection, replacement, and proposal/evaluation
  budget;
- LLM and genetic step distributions calibrated on a separate held-out pilot
  set, with the uncalibrated baseline retained as a sensitivity analysis;
- primary outputs: QD-score/coverage AUC, final archive quality, validity,
  insertion behavior, and cost.

**Block C — LLM-call allocation**

- fixed total LLM-call budget;
- at minimum: uniform allocation, early allocation, and a state-aware
  allocation rule;
- identical non-LLM fallback and total proposal horizon;
- state-aware rule frozen before confirmatory runs;
- compare both used and forfeited calls and report token/cost imbalance.

**Block D — archive state / warm-start sensitivity**

- required sensitivity, not necessarily a full factor in every domain;
- at minimum empty start plus two nonterminal occupancy floors where reachable;
- floors generated independently of the tested LLM arm;
- occupancy is not assumed to be comparable across domains;
- measure whether attribution changes with occupancy rather than only whether
  terminal performance changes.

## 5. Mandatory performance and cost outcomes

### 5.1 Search outcomes

- normalized QD-score and raw QD-score;
- archive coverage;
- maximum elite quality;
- mean elite quality conditional on occupancy;
- validity / parse success;
- accepted insertion and strict-improvement rates;
- anytime AUC and terminal values;
- domain-specific success metrics defined before confirmatory runs.

### 5.2 Resource outcomes

- proposal slots;
- valid proposals;
- real evaluator calls;
- LLM calls attempted and completed;
- prompt, completion, and total tokens;
- fallback count and cause;
- evaluator wall time, LLM latency, and end-to-end wall time;
- monetary cost under a dated price table.

The main text must show at least proposal-, evaluator-, and LLM-call-indexed
views. Token, wall-time, and dollar views may be compact but cannot be omitted.

## 6. Mandatory domain portfolio

The final core study must include:

1. LifeManifold cellular automata as the motivating historical case.
2. The existing maze implementation as a structurally different internal
   transfer case.
3. At least **two public tasks from different benchmark families**, selected
   after the feasibility stage below.

The Fontaine Sphere task may remain a calibration or negative-control domain,
but it does not by itself satisfy the public-domain expansion requirement:
direct vector emission is an unnatural use of an LLM and its evaluator is too
similar to a synthetic optimizer benchmark.

### 6.1 Recommended public-task shortlist

#### Option A — NAS-Bench-201 (recommended)

Use neural architectures as discrete structured genotypes and table lookup for
public benchmark performance.

- **Why suitable:** public, cheap deterministic evaluation, discrete grammar,
  natural code/architecture edits for an LLM, and established connection to
  LLMatic.
- **Candidate quality:** validation/test accuracy with a pre-specified dataset
  and lookup convention.
- **Candidate descriptors:** FLOPs, parameter count, depth, skip-connect count,
  or operation histogram reduced to two declared archive axes.
- **Strength:** allows many seeds and factorial cells without retraining every
  proposed network.
- **Risks:** benchmark leakage into pretrained models, duplicate architecture
  aliases, and accidental use of test performance during search.
- **Required controls:** canonical architecture hash, validation-only search,
  held-out test readout, lookup-miss policy, and prompt variants that omit
  benchmark leaderboard knowledge.

#### Option B — PCG Benchmark: Sokoban and Zelda (recommended)

Use the open-source Procedural Content Generation Benchmark. Treat Sokoban and
Zelda as two tasks in one benchmark family, not as two independent domain
families.

- **Why suitable:** standardized quality/diversity/controllability outputs,
  discrete tile genotypes, public evaluators, and direct LLM-versus-GA
  relevance.
- **Candidate quality:** benchmark feasibility/playability score.
- **Candidate descriptors:** solution length, crate/enemy count, path length,
  density, or benchmark-provided controllability dimensions.
- **Strength:** close enough to the current maze code for implementation, but
  externally defined and independently maintained.
- **Risks:** sparse validity, few-shot copying, task-specific repair dominating
  the LLM effect, and correlated Sokoban/Zelda conclusions.
- **Required controls:** exact-level deduplication, copy-rate audit against
  prompt examples, repair-on/off accounting, and immutable evaluator version.

#### Option C — OpenELM Sodarace

Use LLM-generated Python programs that construct 2-D simulated robots.

- **Why suitable:** direct public LLM+QD task, code genotype, physics evaluator,
  and an archive that is not a tile-grid or analytic function.
- **Candidate quality:** distance or task-normalized locomotion performance.
- **Candidate descriptors:** morphology size, width/height, mass distribution,
  or gait/motion descriptors supported by the environment.
- **Strength:** strongest direct genre match for LLM-driven QD.
- **Risks:** old dependencies, simulator nondeterminism, sandboxing generated
  code, model/API drift, and potentially high per-run cost.
- **Required controls:** containerized evaluator, code execution limits,
  repeated-evaluation noise estimate, canonical program hash, and fixed
  terrain set.

Using Sodarace under the common LifeManifold attribution harness does not by
itself count as reproducing OpenELM's published headline gain. That stronger
claim belongs to the optional extension.

#### Option D — Feynman symbolic regression / QDSR

Use expression trees or grammar-constrained expressions on a public subset of
the Feynman symbolic-regression benchmark.

- **Why suitable:** LLMs can make semantic expression edits; QD can preserve
  diverse complexity/structure niches; evaluation is deterministic and public.
- **Candidate quality:** validation error plus a separately reported exact
  recovery criterion.
- **Candidate descriptors:** expression size, nonlinear operator count,
  dimensional complexity, or number of variables used.
- **Strength:** different representation and scientific task, with interpretable
  offspring and failure modes.
- **Risks:** arbitrary descriptor choice, symbolic simplification aliases,
  benchmark memorization, and a large task-by-seed hierarchy.
- **Required controls:** canonical symbolic simplification, train/validation
  separation, equation-level blocking, memorization-sensitive prompt, and
  fixed grammar.

#### Option E — molecular QD with SELFIES and GuacaMol-style properties

Use public molecular property evaluators with SELFIES or another
validity-preserving representation.

- **Why suitable:** structured language-like genotype, deterministic RDKit
  properties, established diversity objectives, and meaningful validity costs.
- **Candidate quality:** a pre-specified public property objective or constrained
  multi-property score.
- **Candidate descriptors:** molecular weight, logP, TPSA, scaffold class, or
  ring count.
- **Strength:** domain is far from CA/maze and exposes repair/validity effects.
- **Risks:** chemistry expertise requirement, train-data contamination,
  rediscovery of known molecules, metric gaming, and version-sensitive RDKit
  canonicalization.
- **Required controls:** canonical SMILES/InChI deduplication, novelty against a
  declared public reference set, property evaluator pinning, and no wet-lab or
  therapeutic claims.

#### Option F — pyribs planar arm repertoire (negative-control candidate)

- **Why suitable:** recognized QD benchmark with a deterministic evaluator.
- **Limitation:** emitting a vector of joint angles is not a natural LLM task.
- **Use:** negative control for representation dependence, not one of the two
  preferred public primary tasks unless pilots demonstrate stable valid local
  edits and a defensible prompt interface.

### 6.2 Initial recommendation

Start feasibility work with:

1. **NAS-Bench-201** — cheap, structured, and suitable for large factorials.
2. **PCG Benchmark Sokoban/Zelda** — public procedural artifacts and strong
   validity/repair questions.
3. **Sodarace** as the first reserve or stretch task if its environment can be
   reproduced reliably.
4. **Feynman symbolic regression** as the substitute if PCG or Sodarace fails
   feasibility.

This gives at least one lookup/data-backed task and one generative simulator
task. Molecules should be attempted only if evaluator and chemistry review are
available; the planar arm should remain a negative control.

## 7. Feasibility stage before protocol freeze

Feasibility data are design data and must never be pooled with the prospective
confirmatory tier.

For every candidate public task:

1. Pin repository, dataset, evaluator, and license versions.
2. Implement canonical genotype serialization and hashing.
3. Run random and genetic smoke tests.
4. Run at least 50 isolated LLM proposals across multiple parents.
5. Measure parse validity, exact duplicates, prompt-example copying, fallback,
   evaluator failures, step size, latency, and tokens.
6. Demonstrate archive headroom under the proposed horizon.
7. Confirm that both target policies visit meaningfully different parents or
   niches without one arm being mechanically invalid.
8. Estimate variance and cost for sample-size and budget planning.
9. Document any repair and ensure it is either matched across arms or included
   as an explicit factor.
10. Decide GO / REVISE / DROP using criteria frozen before the feasibility
    batch is read.

Minimum GO criteria to be finalized before execution:

- task and generated-artifact licenses permit release;
- deterministic evaluation, or measured noise with a re-evaluation design;
- LLM parse/validity sufficient to distinguish the emitter from fallback;
- no terminal archive saturation before the primary horizon;
- at least two viable selectors;
- total confirmatory cost within the declared cap;
- no hidden test-set feedback during search.

## 8. Model and API robustness

The mandatory core uses:

- one pinned primary model for all complete factorial cells;
- one second model on a pre-specified robustness subset containing all cells
  needed for the primary component contrast;
- temperature, decoding parameters, system prompt, tools, response schema, and
  provider version recorded;
- interleaved/calendar-blocked execution for remote APIs;
- retries and fallback causes logged separately;
- raw request/response payloads retained privately when release would expose
  secrets or licensing-sensitive text; sanitized hashes and usage metadata are
  public.

Model disagreement is a result, not a reason to average providers.

## 9. Statistical design

Exact hypotheses, multiplicity families, sample sizes, and smallest effect
sizes of interest will be frozen only after feasibility pilots.

Mandatory principles:

- paired contrasts on common seeds/initial archives wherever possible;
- domain-level estimates with confidence intervals and raw paired points;
- no inference from candidate-level pseudo-replication;
- interactions reported directly;
- equivalence tests only with domain-specific, substantively justified margins;
- correction within declared confirmatory families;
- no paper-wide pooling of unrelated hypotheses;
- hierarchical cross-domain synthesis as secondary to transparent per-domain
  results;
- sensitivity to failed calls, fallback, duplicates, and evaluator noise;
- all stopped, dropped, or infeasible tasks reported in a design-flow diagram.

For cheap public lookup tasks, target at least 20 confirmatory paired seeds.
For API- or simulator-heavy tasks, determine the sample size from pilot
variance and the declared precision target, with 10 paired seeds as a floor
unless the frozen analysis justifies another design.

## 10. Deliverables and phases

### Phase 0 — literature and claim inventory

- Catalogue LLM+QD comparisons and the components changed in each baseline.
- Record which papers match initialization, selection, budget, representation,
  repair, and model.
- Use this inventory to motivate the taxonomy; do not call it a prevalence
  estimate unless the search and inclusion procedure is systematic.

### Phase 1 — common attribution harness

- Typed treatment manifest and stable schema.
- Domain adapter interface.
- Canonical genotype and proposal log.
- Unified budget ledger.
- Factorial job builder.
- Analysis producing cell means, contrasts, interactions, anytime curves, and
  cost views.
- Unit/integration tests that prevent mixed domains, pilots, and confirmatory
  tiers from being globbed together.

### Phase 2 — public-domain feasibility

- Implement NAS-Bench-201 first.
- Implement PCG Benchmark Sokoban/Zelda second.
- Evaluate Sodarace and Feynman as reserve candidates.
- Select two public tasks from different families.

Design notes (not a protocol, not task selection):  
`artifacts/controlled_attribution/PHASE2_FOUNDATION.md`,  
`PHASE2_NAS_BENCH_201_FEASIBILITY.md`, `PHASE2_NAS_P2_1.md`, `PHASE2_NAS_P2_2.md`,
`PHASE2_PCG_FEASIBILITY.md`, `PHASE2_PCG_P2_3.md`, `PHASE2_PCG_P2_4.md`,
`PHASE2_P2_5.md`.

### Phase 3 — protocol freeze

Create a new dated protocol containing:

- final domains and immutable versions;
- treatment cells and estimands;
- model versions and prompts;
- seeds, initial archives, horizons, and budgets;
- quality gates and stopping rules;
- confirmatory families and multiplicity;
- release/privacy plan;
- exclusions and failure handling.

Commit/tag the protocol before the first confirmatory run. Amendments require
dated dual reporting; pilots remain permanently separate.

### Phase 4 — mandatory confirmatory programme

- Run Blocks A and B on every selected domain.
- Run Block C on at least two domains, including one public domain.
- Run Block D occupancy sensitivity where occupancy floors are meaningful.
- Execute the second-model robustness subset.
- Freeze raw run manifests before analysis.

### Phase 5 — synthesis and paper

- Separate historical evidence, feasibility evidence, and prospective evidence.
- Lead with the attribution method and cross-domain findings.
- Explain mechanisms using parent choice, offspring distance, validity, and
  archive-state diagnostics.
- Report heterogeneity without forcing a universal direction.
- Release an artifact bundle with one-command analysis reproduction.

## 11. Optional maximum extension: external headline-gain reproduction

This extension is optional and starts only after the mandatory core is
complete. Select two or three published systems whose code, models, datasets,
and headline estimands can be reconstructed.

Initial candidates:

- **OpenELM / Sodarace:** reproduce the published LLM evolutionary/QD comparison
  before applying the common attribution cells.
- **LLMatic / NAS-Bench-201:** reproduce the reported competitive NAS result and
  component ablations, then isolate archive, prompt, selector, and model effects.
- **Large Language Models as In-context AI Generators for QD:** assess code and
  model availability, identify its headline gain, reproduce it unchanged, then
  run the controlled audit.

Selection gates:

- peer-reviewed or otherwise clearly published claim;
- accessible code and legal task assets;
- identifiable numerical headline result;
- feasible original model or documented substitute analysis;
- total compute/API cost below a separately approved cap;
- enough implementation fidelity that failure to reproduce is interpretable.

For each system, preserve two layers:

1. **Faithful reproduction:** original code/configuration and original headline
   estimand, changed only where required for execution and fully disclosed.
2. **Controlled audit:** common manifests, matched components, and budget
   accounting.

Do not call a modern-model port a reproduction of the original headline gain.
Report it as a port or robustness study.

## 12. Decision gates for the title and venue

Use the broad working title only if:

- at least three distinct gains/components are attributed;
- the mandatory public-domain portfolio is complete;
- at least one non-LifeManifold domain shows a material bundled-versus-matched
  difference or a clearly informative counterexample;
- interactions and budget-axis changes are estimable;
- the final reusable procedure is more than a checklist;
- cross-domain conclusions are supported prospectively.

If those conditions fail, narrow the title to:

*Controlled Attribution of an Apparent Performance Gain in LLM-Driven
Quality-Diversity Search.*

Venue order after results:

1. ACM TELO for the complete cross-domain methodological study.
2. IEEE TEVC only if the general EC insight and evidence are archival in scale.
3. TMLR if the strongest contribution is a rigorous empirical audit with
   modest or heterogeneous effects.
4. A narrower EC/QD journal if the public-domain expansion does not hold.

## 13. Immediate next actions

1. Build the literature/claim inventory template.
2. Write the domain-adapter and treatment-manifest specification without
   changing existing Q1 experiment code.
3. Produce feasibility designs for NAS-Bench-201 and PCG Benchmark
   Sokoban/Zelda.
4. Estimate API calls, tokens, evaluator time, storage, and author effort for
   the mandatory core.
5. Decide the two public tasks only after the feasibility plans and cost model
   are reviewed.
6. Do not freeze hypotheses or launch new confirmatory runs from this planning
   document.

Dated note (2026-09-02): a Phase 3 **draft** exists at
`artifacts/controlled_attribution/PHASE3_PROTOCOL.md`. It is not a freeze and
not a licence to start Phase 4. The planning document remains non-confirmatory.
