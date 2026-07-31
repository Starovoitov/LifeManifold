#!/usr/bin/env bash
# Run one condition × seed for the Q1 experiment matrix.
# Usage: ./scripts/run_experiment_batch.sh TIER [first_seed] [last_seed]
#
# Grid tiers:  pilot | q1-min | q1-full | q1-full-filter | q1-repeat | shadow
# CVT tiers:   q1-cvt-min | q1-cvt | q1-cvt-filter | cvt-shadow | q1-prompt-ablation
# B2 tier:     q1-v3-pyribs  (CMA-ME + CMA-MAE via run_pyribs_baseline.py)
# B3 tiers:    q1-v3-sphere | q1-v3-rastrigin  (CA-independent standard benchmarks)
# v3 B1/RQ0:   q1-v3-vanilla              (random-only; no LLM)
# v3 B1b:      q1-v3-genetic-me           (20R+30G; no LLM; matched stub/hints slots)
# v3 sensitivity: q1-v3-genetic-me-uniform  (genetic_me + uniform_frontier; no surrogate)
# v3 factorial: q1-v3-genetic-me-filter    (−LLM + surrogate filter; 2×2 ablation cell)
# v3 mixed 2×2: q1-v3-mixed-2x2 (stub_uniform / hints / filter_stub / filter; archive_trace)
# v3 G1:       q1-v3-llm-deepseek-v4-pro  (stub+hints; --llm-provider deepseek)
#              q1-v3-llm-gpt-4o-mini       (stub+hints; --llm-provider openai)
# Post-arXiv:  q1-stub-uniform-sensitivity (stub_uniform only; target_selection parity)
# H1 matched:  q1-h1-matched-gpt-4o-mini (stub_uniform @ openai; reuse G1 hints; seeds 0–2)
# Anytime:     q1-anytime-ladder (vanilla + hints + cma_me; archive_trace; new root)
# Encoding:    q1-cma-encoding-ablation (CMA-ME decode/warm-start sanity; new root)
# C4 discrete: q1-v3-pyribs-discrete-cma (Bernoulli-decode CMA-ME; package C4)
# Native discrete: q1-v3-pyribs-native-discrete-cma (bit-flip DiscreteCMAEmitter)
# H2 threshold: q1-h2-threshold-sensitivity (genetic_me_filter @ τ∈{0.35,0.45,0.55}; seeds 0–2 default)
# RQ1b pilot: q1-hints-rich-pilot (hints_rich only; component user prompt)
# RQ1d pilot: q1-hints-parent-pilot (hints_parent only; parent metrics in hint block)
# RQ1e pilot: q1-hints-direction-pilot (hints_direction only; FD direction hints)
# RQ1f pilot: q1-v3-llm-weak-pilot (weak LLM × stub_uniform + hints interaction)
#
# q1-repeat: stub+hints only; 3 replicates per seed (default seeds 0–1) for LLM variance floor.
# q1-prompt-ablation: CVT archive + grid system prompt; stub+hints (default seed 0).
# q1-*-filter: filter arm only; requires completed stub + hints for each seed.
# q1-v3-pyribs: seeds × {cma_me,cma_mae}; default 32500 evals; override with PYRIBS_EVALUATIONS (must ÷ 250).
# q1-v3-sphere / q1-v3-rastrigin: override smoke budget with PYRIBS_STANDARD_EVALUATIONS (must ÷ 250).
# q1-v3-vanilla: RQ0 / B1; default seeds 0–9; CPU only (no API key).
# q1-v3-llm-deepseek-v4-pro / q1-v3-llm-gpt-4o-mini: G1; default seeds 0–4; full: 0 9
#   Requires: DEEPSEEK_API_KEY / OPENAI_API_KEY respectively.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REQUESTED_TIER="${1:-pilot}"
TIER="$REQUESTED_TIER"
SEED_START="${2:-0}"
SEED_END="${3:-$SEED_START}"
FILTER_ONLY=false
RUN_PYRIBS=false
RUN_PYRIBS_STANDARD=false
RUN_DUNGEON=false
RUN_MAZE=false
case "$TIER" in
  q1-full-filter)
    TIER=q1-full
    FILTER_ONLY=true
    ;;
  q1-cvt-filter)
    TIER=q1-cvt
    FILTER_ONLY=true
    ;;
esac

EXP_ROOT="$ROOT/artifacts/experiments"
GRID_BASELINE_ARCHIVE="$ROOT/artifacts/map_elites_nightly/baseline/map_elites_archive.jsonl"
CVT_BASELINE_ARCHIVE="$ROOT/artifacts/map_elites_nightly/cvt/baseline/map_elites_archive.jsonl"
BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
ARCHIVE_TYPE=grid

TRAIN_SCRIPT="$ROOT/scripts/train_surrogate.py"
RUN_SCRIPT="$ROOT/scripts/run_github_llm_map_elites.py"
PYRIBS_SCRIPT="$ROOT/scripts/run_pyribs_baseline.py"
PYRIBS_STANDARD_SCRIPT="$ROOT/scripts/run_pyribs_standard.py"
DUNGEON_SCRIPT="$ROOT/scripts/run_dungeon_qd.py"
MAZE_SCRIPT="$ROOT/scripts/run_maze_qd.py"
AGG_SCRIPT="$ROOT/scripts/aggregate_experiment_runs.py"

SCHEDULER_STUB_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub.yaml"
SCHEDULER_STUB_UNIFORM_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub_uniform.yaml"
SCHEDULER_HINTS_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm.yaml"
SCHEDULER_HINTS_RICH_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_hints_rich.yaml"
SCHEDULER_HINTS_PARENT_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_hints_parent.yaml"
SCHEDULER_HINTS_DIRECTION_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_hints_direction.yaml"
SCHEDULER_HINTS_WEAK_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_weak_hints.yaml"
SCHEDULER_FILTER_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter.yaml"
SCHEDULER_FILTER_STUB_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter_stub.yaml"
SCHEDULER_FILTER_GRAY_ZONE="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter_gray_zone.yaml"
SCHEDULER_SHADOW_HINTS_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow_hints.yaml"
SCHEDULER_SHADOW_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow.yaml"
SCHEDULER_VANILLA_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_vanilla.yaml"
SCHEDULER_GENETIC_ME_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_genetic_me.yaml"
SCHEDULER_GENETIC_ME_UNIFORM_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_genetic_me_uniform.yaml"
SCHEDULER_GENETIC_ME_FILTER_NIGHTLY="$ROOT/worldspace/specs/map_elites_scheduler_nightly_genetic_me_filter.yaml"
SCHEDULER_GENETIC_ME_FILTER_TAU035="$ROOT/worldspace/specs/map_elites_scheduler_nightly_genetic_me_filter_tau035.yaml"
SCHEDULER_GENETIC_ME_FILTER_TAU055="$ROOT/worldspace/specs/map_elites_scheduler_nightly_genetic_me_filter_tau055.yaml"

SCHEDULER_STUB_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub_cvt.yaml"
SCHEDULER_HINTS_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_cvt.yaml"
SCHEDULER_FILTER_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_filter_cvt.yaml"
SCHEDULER_SHADOW_HINTS_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow_hints_cvt.yaml"
SCHEDULER_SHADOW_CVT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_shadow_cvt.yaml"
SCHEDULER_STUB_CVT_GRID_PROMPT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_stub_cvt_grid_prompt.yaml"
SCHEDULER_HINTS_CVT_GRID_PROMPT="$ROOT/worldspace/specs/map_elites_scheduler_nightly_llm_cvt_grid_prompt.yaml"

SCHEDULER_STUB_PILOT="$ROOT/worldspace/specs/map_elites_scheduler_github_llm_stub.yaml"
SCHEDULER_HINTS_PILOT="$ROOT/worldspace/specs/map_elites_scheduler_github_llm.yaml"

RUN_VANILLA=false
RUN_GENETIC_ME=false
RUN_GENETIC_ME_UNIFORM=false
RUN_GENETIC_ME_FILTER=false
RUN_STUB_UNIFORM_ONLY=false
RUN_HINTS_RICH_ONLY=false
RUN_HINTS_PARENT_ONLY=false
RUN_HINTS_DIRECTION_ONLY=false
RUN_WEAK_HINTS_PILOT=false
RUN_ANYTIME_LADDER=false
RUN_CMA_ENCODING_ABLATION=false
RUN_CMA_DISCRETE=false
RUN_CMA_NATIVE_DISCRETE=false
RUN_CMA_PBCMA=false
RUN_H2_THRESHOLD_SWEEP=false
RUN_GRAY_ZONE_ONLY=false
RUN_MIXED_2X2=false
case "$TIER" in
  pilot)
    ITERATIONS=120
    EXP_DIR="$EXP_ROOT/pilot"
    SCHEDULER_STUB="$SCHEDULER_STUB_PILOT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_PILOT"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-min)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-min"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-full)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-full"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    SCHEDULER_FILTER="$SCHEDULER_FILTER_NIGHTLY"
    RUN_FILTER=true
    RUN_SHADOW=false
    ;;
  q1-repeat)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-repeat"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    REPLICATE_COUNT=3
    ;;
  shadow)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/shadow"
    SCHEDULER_HINTS="$SCHEDULER_SHADOW_HINTS_NIGHTLY"
    SCHEDULER_FILTER="$SCHEDULER_SHADOW_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=true
    ;;
  q1-cvt-min)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-cvt"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_STUB="$SCHEDULER_STUB_CVT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_CVT"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-cvt)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-cvt"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_STUB="$SCHEDULER_STUB_CVT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_CVT"
    SCHEDULER_FILTER="$SCHEDULER_FILTER_CVT"
    RUN_FILTER=true
    RUN_SHADOW=false
    ;;
  cvt-shadow)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/cvt-shadow"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_HINTS="$SCHEDULER_SHADOW_HINTS_CVT"
    SCHEDULER_FILTER="$SCHEDULER_SHADOW_CVT"
    RUN_FILTER=false
    RUN_SHADOW=true
    ;;
  q1-prompt-ablation)
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-prompt-ablation"
    ARCHIVE_TYPE=cvt
    BASELINE_ARCHIVE="$CVT_BASELINE_ARCHIVE"
    SCHEDULER_STUB="$SCHEDULER_STUB_CVT_GRID_PROMPT"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_CVT_GRID_PROMPT"
    RUN_FILTER=false
    RUN_SHADOW=false
    ;;
  q1-v3-pyribs)
    EXP_DIR="$EXP_ROOT/q1-v3-pyribs"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_PYRIBS=true
    ;;
  q1-v3-sphere)
    EXP_DIR="$EXP_ROOT/q1-v3-sphere"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_PYRIBS_STANDARD=true
    STANDARD_BENCHMARK=sphere
    ;;
  q1-v3-rastrigin)
    EXP_DIR="$EXP_ROOT/q1-v3-rastrigin"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_PYRIBS_STANDARD=true
    STANDARD_BENCHMARK=rastrigin
    ;;
  q1-v4-dungeon)
    EXP_DIR="$EXP_ROOT/${DUNGEON_EXPERIMENT_ROOT:-q1-v4-dungeon}"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_DUNGEON=true
    DUNGEON_CONDITIONS=(genetic genetic_filter llm_stub llm_hints llm_hints_filter)
    ;;
  q1-v4-dungeon-genetic|q1-v4-dungeon-genetic-filter|q1-v4-dungeon-llm-stub|q1-v4-dungeon-llm-hints|q1-v4-dungeon-llm-hints-filter)
    EXP_DIR="$EXP_ROOT/${DUNGEON_EXPERIMENT_ROOT:-q1-v4-dungeon}"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_DUNGEON=true
    dungeon_condition="${TIER#q1-v4-dungeon-}"
    dungeon_condition="${dungeon_condition//-/_}"
    DUNGEON_CONDITIONS=("$dungeon_condition")
    ;;
  q1-v4-maze|q1-v5-maze)
    EXP_DIR="$EXP_ROOT/${MAZE_EXPERIMENT_ROOT:-q1-v4-maze}"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_MAZE=true
    MAZE_CONDITIONS=(genetic genetic_filter llm_stub llm_hints llm_hints_filter)
    ;;
  q1-v4-maze-genetic|q1-v4-maze-random|q1-v4-maze-genetic-filter|q1-v4-maze-llm-stub|q1-v4-maze-llm-hints|q1-v4-maze-llm-hints-filter|q1-v5-maze-genetic|q1-v5-maze-genetic-filter|q1-v5-maze-llm-stub|q1-v5-maze-llm-hints|q1-v5-maze-llm-hints-filter)
    EXP_DIR="$EXP_ROOT/${MAZE_EXPERIMENT_ROOT:-q1-v4-maze}"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_MAZE=true
    maze_condition="${TIER#q1-v4-maze-}"
    maze_condition="${maze_condition#q1-v5-maze-}"
    maze_condition="${maze_condition//-/_}"
    MAZE_CONDITIONS=("$maze_condition")
    ;;
  q1-v3-vanilla)
    # B1 / RQ0: vanilla MAP-Elites (random emitter only).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-vanilla"
    SCHEDULER_VANILLA="$SCHEDULER_VANILLA_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_VANILLA=true
    ;;
  q1-v3-genetic-me)
    # B1b: genetic MAP-Elites (20R+30G; LLM slots → genetic; no surrogate).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-genetic-me"
    SCHEDULER_GENETIC_ME="$SCHEDULER_GENETIC_ME_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_GENETIC_ME=true
    ;;
  q1-v3-genetic-me-uniform)
    # Matched control: genetic ME + uniform_frontier, no surrogate and no LLM.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-genetic-me-uniform"
    SCHEDULER_GENETIC_ME_UNIFORM="$SCHEDULER_GENETIC_ME_UNIFORM_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_GENETIC_ME_UNIFORM=true
    ;;
  q1-v3-genetic-me-filter)
    # Factorial (−LLM, +surrogate filter): 20R+30G + threshold_gate; CPU-only ablation cell.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-genetic-me-filter"
    SCHEDULER_GENETIC_ME_FILTER="$SCHEDULER_GENETIC_ME_FILTER_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_GENETIC_ME_FILTER=true
    ;;
  q1-v3-llm-deepseek-v4-pro)
    # G1: DeepSeek V4 Pro @ official API (non-thinking). stub+hints only.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-llm/deepseek-v4-pro"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    LLM_PROVIDER=deepseek
    ;;
  q1-v3-llm-gpt-4o-mini)
    # G1: OpenAI gpt-4o-mini (budget). stub+hints only.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-llm/gpt-4o-mini"
    SCHEDULER_STUB="$SCHEDULER_STUB_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    LLM_PROVIDER=openai
    ;;
  q1-stub-uniform-sensitivity)
    # Post-arXiv: stub with uniform_frontier (surrogate off); does not mutate v2/G1 stub runs.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-stub-uniform-sensitivity"
    SCHEDULER_STUB_UNIFORM="$SCHEDULER_STUB_UNIFORM_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_STUB_UNIFORM_ONLY=true
    ;;
  q1-h1-matched-gpt-4o-mini)
    # Matched H1 confirmatory pilot: stub_uniform @ OpenAI gpt-4o-mini (reuse G1 hints).
    # Default seeds 0–2; compare to existing q1-v3-llm/gpt-4o-mini/hints/.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-llm/gpt-4o-mini"
    SCHEDULER_STUB_UNIFORM="$SCHEDULER_STUB_UNIFORM_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_STUB_UNIFORM_ONLY=true
    LLM_PROVIDER=openai
    ;;
  q1-h1-matched-deepseek-v4-pro)
    # Matched H1 exploratory pilot: stub_uniform @ DeepSeek V4 Pro (reuse G1 hints).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-llm/deepseek-v4-pro"
    SCHEDULER_STUB_UNIFORM="$SCHEDULER_STUB_UNIFORM_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_STUB_UNIFORM_ONLY=true
    LLM_PROVIDER=deepseek
    ;;
  q1-anytime-ladder)
    # Selective re-runs for archive_trace curves (vanilla + hints + cma_me); does not touch q1-full.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-anytime-ladder"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    SCHEDULER_VANILLA="$SCHEDULER_VANILLA_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_ANYTIME_LADDER=true
    ;;
  q1-cma-encoding-ablation)
    # F-RQ-ceiling honesty: alternate rule-bit decode + cold-start CMA-ME (seeds 0–4 default).
    EXP_DIR="$EXP_ROOT/q1-cma-encoding-ablation"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_CMA_ENCODING_ABLATION=true
    ;;
  q1-v3-pyribs-discrete-cma)
    # Package C4: Bernoulli-decode CMA-ME (discrete rule bits at eval; continuous CMA proposal).
    EXP_DIR="$EXP_ROOT/q1-v3-pyribs-discrete-cma"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_CMA_DISCRETE=true
    ;;
  q1-v3-pyribs-native-discrete-cma)
    # Native discrete-search CMA-ME (bit-flip emitter; discrete genotype in archive).
    EXP_DIR="$EXP_ROOT/q1-v3-pyribs-native-discrete-cma"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_CMA_NATIVE_DISCRETE=true
    ;;
  q1-v3-pyribs-pbcma)
    # pbCMA: latent-Gaussian CMA + bit threshold + margin (discrete archive genotype).
    EXP_DIR="$EXP_ROOT/q1-v3-pyribs-pbcma"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_CMA_PBCMA=true
    ;;
  q1-h2-threshold-sensitivity)
    # H2 robustness: genetic_me_filter @ τ∈{0.35,0.45,0.55} vs existing genetic_me_uniform.
    # τ=0.45 reuses q1-v3-genetic-me-filter runs when present (no duplicate sim).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-h2-threshold-sensitivity"
    ARCHIVE_TYPE=grid
    BASELINE_ARCHIVE="$GRID_BASELINE_ARCHIVE"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_H2_THRESHOLD_SWEEP=true
    ;;
  q1-hints-rich-pilot)
    # RQ1b: component-rich surrogate hints; 1-seed pilot default (seed 0).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-hints-rich-pilot"
    SCHEDULER_HINTS_RICH="$SCHEDULER_HINTS_RICH_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_HINTS_RICH_ONLY=true
    ;;
  q1-hints-parent-pilot)
    # RQ1d: parent archive metrics in hint block; 1-seed pilot default (seed 0).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-hints-parent-pilot"
    SCHEDULER_HINTS_PARENT="$SCHEDULER_HINTS_PARENT_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_HINTS_PARENT_ONLY=true
    ;;
  q1-hints-direction-pilot)
    # RQ1e: finite-difference direction hints; 1-seed pilot default (seed 0).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-hints-direction-pilot"
    SCHEDULER_HINTS_DIRECTION="$SCHEDULER_HINTS_DIRECTION_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_HINTS_DIRECTION_ONLY=true
    ;;
  q1-v3-llm-weak-pilot)
    # RQ1f / G2: weak model (qwen2.5-omni-7b) × stub_uniform + hints @ uniform_frontier.
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-llm-weak-pilot"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_WEAK_NIGHTLY"
    SCHEDULER_STUB_UNIFORM="$SCHEDULER_STUB_UNIFORM_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_WEAK_HINTS_PILOT=true
    LLM_PROVIDER=weak
    ;;
  q1-v3-h3-gray-zone-pilot)
    # H3 exploratory: force-eval gray zone vs frozen q1-full/hints (seeds 0–4 default).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-h3-gray-zone-pilot"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_GRAY_ZONE_ONLY=true
    ;;
  q1-v3-h3-gray-zone)
    # H3 confirmatory F-RQ3-gray (deferred journal extension).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-h3-gray-zone"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_GRAY_ZONE_ONLY=true
    ;;
  q1-v3-mixed-2x2)
    # Mixed LLM stack 2×2: soft hints × hard filter (archive_trace re-run tier).
    ITERATIONS=650
    EXP_DIR="$EXP_ROOT/q1-v3-mixed-2x2"
    SCHEDULER_STUB_UNIFORM="$SCHEDULER_STUB_UNIFORM_NIGHTLY"
    SCHEDULER_HINTS="$SCHEDULER_HINTS_NIGHTLY"
    SCHEDULER_FILTER_STUB="$SCHEDULER_FILTER_STUB_NIGHTLY"
    SCHEDULER_FILTER="$SCHEDULER_FILTER_NIGHTLY"
    RUN_FILTER=false
    RUN_SHADOW=false
    RUN_MIXED_2X2=true
    ;;
  *)
    echo "Unknown tier: $TIER" >&2
    echo "Use: pilot|q1-min|q1-full|q1-full-filter|q1-repeat|shadow|q1-cvt-min|q1-cvt|q1-cvt-filter|cvt-shadow|q1-prompt-ablation|q1-v3-pyribs|q1-v3-sphere|q1-v3-rastrigin|q1-v4-dungeon|q1-v4-dungeon-{genetic,genetic-filter,llm-stub,llm-hints,llm-hints-filter}|q1-v4-maze|q1-v5-maze|q1-v4-maze-{genetic,random,genetic-filter,llm-stub,llm-hints,llm-hints-filter}|q1-v3-vanilla|q1-v3-genetic-me|q1-v3-genetic-me-uniform|q1-v3-genetic-me-filter|q1-v3-llm-deepseek-v4-pro|q1-v3-llm-gpt-4o-mini|q1-stub-uniform-sensitivity|q1-h1-matched-gpt-4o-mini|q1-h1-matched-deepseek-v4-pro|q1-anytime-ladder|q1-cma-encoding-ablation|q1-v3-pyribs-discrete-cma|q1-v3-pyribs-native-discrete-cma|q1-v3-pyribs-pbcma|q1-h2-threshold-sensitivity|q1-hints-rich-pilot|q1-hints-parent-pilot|q1-hints-direction-pilot|q1-v3-llm-weak-pilot|q1-v3-h3-gray-zone-pilot|q1-v3-h3-gray-zone|q1-v3-mixed-2x2" >&2
    exit 1
    ;;
esac

# Default LLM provider → worldspace/specs/llm_world_generator_${LLM_PROVIDER}.yaml
LLM_PROVIDER="${LLM_PROVIDER:-qwen}"

REPLICATE_COUNT="${REPLICATE_COUNT:-1}"
if [[ "$REQUESTED_TIER" == "q1-repeat" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=1
fi
if [[ "$REQUESTED_TIER" == "q1-prompt-ablation" && $# -lt 2 ]]; then
  # Prefer documenting seeds 0–2; default remains 0 for cheap resume of existing run.
  SEED_START=0
  SEED_END=0
  echo "NOTE: q1-prompt-ablation default is seed 0 only; for a stronger claim run: $0 q1-prompt-ablation 0 2" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-llm-deepseek-v4-pro" && $# -lt 2 ]]; then
  # G1 minimal default: seeds 0–4 (protocol §6). Full: $0 q1-v3-llm-deepseek-v4-pro 0 9
  SEED_START=0
  SEED_END=4
  echo "NOTE: q1-v3-llm-deepseek-v4-pro default seeds 0–4 (G1 minimal); full matrix: $0 q1-v3-llm-deepseek-v4-pro 0 9" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-llm-gpt-4o-mini" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=4
  echo "NOTE: q1-v3-llm-gpt-4o-mini default seeds 0–4 (G1 minimal); full matrix: $0 q1-v3-llm-gpt-4o-mini 0 9" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-vanilla" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-vanilla default seeds 0–9 (RQ0 / B1 full matrix)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-genetic-me" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-genetic-me default seeds 0–9 (B1b genetic ME baseline)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-genetic-me-uniform" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-genetic-me-uniform default seeds 0–9 (matched target-selection control)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-genetic-me-filter" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-genetic-me-filter default seeds 0–9 (factorial −LLM/+filter cell)" >&2
fi
if [[ ("$REQUESTED_TIER" == "q1-v3-sphere" || "$REQUESTED_TIER" == "q1-v3-rastrigin") && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: $REQUESTED_TIER default seeds 0–9 (B3 standard benchmark matrix)" >&2
fi
if [[ "$REQUESTED_TIER" == q1-v4-dungeon* && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=4
  echo "NOTE: $REQUESTED_TIER default seeds 0–4 (B4 exploratory gate)" >&2
fi
if [[ ("$REQUESTED_TIER" == q1-v4-maze* || "$REQUESTED_TIER" == q1-v5-maze*) && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=2
  echo "NOTE: $REQUESTED_TIER default seeds 0–2 (maze symmetry smoke)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-stub-uniform-sensitivity" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=2
  echo "NOTE: q1-stub-uniform-sensitivity default seeds 0–2 (post-arXiv pilot); full: $0 q1-stub-uniform-sensitivity 0 9" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-h1-matched-gpt-4o-mini" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=2
  echo "NOTE: q1-h1-matched-gpt-4o-mini default seeds 0–2 (matched H1 pilot vs existing gpt-4o-mini hints)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-h1-matched-deepseek-v4-pro" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=2
  echo "NOTE: q1-h1-matched-deepseek-v4-pro default seeds 0–2 (matched H1 pilot vs existing deepseek hints)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-h3-gray-zone-pilot" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=4
  echo "NOTE: q1-v3-h3-gray-zone-pilot default seeds 0–4 (exploratory n=5; vs frozen q1-full/hints)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-h3-gray-zone" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-h3-gray-zone default seeds 0–9 (F-RQ3-gray confirmatory; shadow 8–18% pre-flight on seed 0)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-anytime-ladder" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=4
  echo "NOTE: q1-anytime-ladder default seeds 0–4 (vanilla + hints + cma_me); full: $0 q1-anytime-ladder 0 9" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-cma-encoding-ablation" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=4
  echo "NOTE: q1-cma-encoding-ablation default seeds 0–4 (3 CMA arms); full: $0 q1-cma-encoding-ablation 0 9" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-pyribs-discrete-cma" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-pyribs-discrete-cma default seeds 0–9 (Bernoulli-decode CMA-ME; package C4)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-pyribs-native-discrete-cma" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-pyribs-native-discrete-cma default seeds 0–9 (native bit-flip CMA-ME)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-h2-threshold-sensitivity" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=2
  echo "NOTE: q1-h2-threshold-sensitivity default seeds 0–2 (H2 τ pilot); full: $0 q1-h2-threshold-sensitivity 0 9" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-pyribs-pbcma" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-pyribs-pbcma default seeds 0–9 (pbCMA latent-Gaussian CMA-ME)" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-hints-rich-pilot" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=0
  echo "NOTE: q1-hints-rich-pilot default seed 0 only; extend: $0 q1-hints-rich-pilot 0 4" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-hints-parent-pilot" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=0
  echo "NOTE: q1-hints-parent-pilot default seed 0 only; extend: $0 q1-hints-parent-pilot 0 4" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-hints-direction-pilot" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=0
  echo "NOTE: q1-hints-direction-pilot default seed 0 only; extend: $0 q1-hints-direction-pilot 0 4" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-mixed-2x2" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=9
  echo "NOTE: q1-v3-mixed-2x2 default seeds 0–9 (4 arms × archive_trace); parallel: run_mixed_2x2_nohup.sh" >&2
fi
if [[ "$REQUESTED_TIER" == "q1-v3-llm-weak-pilot" && $# -lt 2 ]]; then
  SEED_START=0
  SEED_END=2
  echo "NOTE: q1-v3-llm-weak-pilot default seeds 0–2 (interaction pilot); single seed: $0 q1-v3-llm-weak-pilot 0 0" >&2
fi

if [[ "$LLM_PROVIDER" == "deepseek" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is required for LLM_PROVIDER=deepseek" >&2
  exit 1
fi
if [[ "$LLM_PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for LLM_PROVIDER=openai" >&2
  exit 1
fi
if [[ "$LLM_PROVIDER" == "weak" && -z "${QWEN_API_KEY:-}" ]]; then
  echo "QWEN_API_KEY is required for LLM_PROVIDER=weak (DashScope qwen2.5-omni-7b)" >&2
  exit 1
fi

apply_long_run_llm_defaults() {
  if [[ -z "${LIFEMANIFOLD_LOG_ITERATION_TIMING:-}" ]]; then
    export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
  fi
  if [[ -z "${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-}" ]]; then
    export LIFEMANIFOLD_LLM_PARALLEL_WORKERS=4
  fi
}

apply_vanilla_run_defaults() {
  if [[ -z "${LIFEMANIFOLD_LOG_ITERATION_TIMING:-}" ]]; then
    export LIFEMANIFOLD_LOG_ITERATION_TIMING=1
  fi
}

case "$TIER" in
  q1-min|q1-full|q1-repeat|shadow|q1-cvt-min|q1-cvt|cvt-shadow|q1-prompt-ablation|q1-v3-llm-deepseek-v4-pro|q1-v3-llm-gpt-4o-mini|q1-stub-uniform-sensitivity|q1-h1-matched-gpt-4o-mini|q1-h1-matched-deepseek-v4-pro|q1-hints-rich-pilot|q1-hints-parent-pilot|q1-hints-direction-pilot|q1-v3-llm-weak-pilot|q1-v3-mixed-2x2)
    apply_long_run_llm_defaults
    ;;
  q1-v3-vanilla|q1-v3-genetic-me|q1-v3-genetic-me-uniform|q1-v3-genetic-me-filter|q1-h2-threshold-sensitivity)
    apply_vanilla_run_defaults
    ;;
  q1-anytime-ladder)
    apply_vanilla_run_defaults
    apply_long_run_llm_defaults
    ;;
esac

if [[ "$FILTER_ONLY" == true && -z "${LIFEMANIFOLD_LLM_PARALLEL_WORKERS:-}" ]]; then
  export LIFEMANIFOLD_LLM_PARALLEL_WORKERS=2
fi

if [[ "$RUN_PYRIBS_STANDARD" != true && "$RUN_DUNGEON" != true && "$RUN_MAZE" != true && ! -f "$BASELINE_ARCHIVE" ]]; then
  echo "Missing baseline archive: $BASELINE_ARCHIVE" >&2
  if [[ "$ARCHIVE_TYPE" == "cvt" ]]; then
    echo "Run: ./scripts/run_cvt_baseline.sh" >&2
  else
    echo "Run: uv run python -m worldspace.scripts.run_map_elites_nightly --archive-type grid" >&2
  fi
  exit 1
fi

if [[ "$RUN_PYRIBS" != true && "$RUN_PYRIBS_STANDARD" != true && "$RUN_DUNGEON" != true && "$RUN_MAZE" != true ]]; then
  CHECKPOINT="$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.pkl"
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Training surrogate checkpoint..."
    uv run python "$TRAIN_SCRIPT" \
      --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
      --checkpoint-path "$CHECKPOINT" \
      --summary-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json" \
      --mlp-dropout-p 0.05 \
      --mlp-uncertainty-method ensemble_mc \
      --mlp-mc-samples 16 \
      --no-quality-gate
  fi

  CALIBRATION="$ROOT/artifacts/surrogate/checkpoints/calibration_v3_mc_d005.pkl"
  if [[ ("$RUN_FILTER" == true || "$RUN_SHADOW" == true || "$RUN_GENETIC_ME_FILTER" == true || "$RUN_H2_THRESHOLD_SWEEP" == true) && ! -f "$CALIBRATION" ]]; then
    echo "Training uncertainty calibration (required for filter/shadow arms)..."
    uv run python "$TRAIN_SCRIPT" \
      --buffer-path "$ROOT/artifacts/surrogate/buffer_nightly.jsonl" \
      --checkpoint-path "$CHECKPOINT" \
      --summary-path "$ROOT/artifacts/surrogate/checkpoints/nightly_v3_mc_d005.summary.json" \
      --calibrate \
      --calibration-path "$CALIBRATION" \
      --no-quality-gate
  fi
fi

mkdir -p "$EXP_DIR"

require_stub_hints_for_seed() {
  local seed="$1"
  local stub_summary="$EXP_DIR/stub/seed_${seed}/nightly_run_summary.json"
  local hints_summary="$EXP_DIR/hints/seed_${seed}/nightly_run_summary.json"
  if [[ ! -f "$stub_summary" || ! -f "$hints_summary" ]]; then
    echo "filter-only: missing completed stub/hints for seed $seed" >&2
    echo "  expected: $stub_summary" >&2
    echo "  expected: $hints_summary" >&2
    exit 1
  fi
}

remove_incomplete_run_dir() {
  local out="$1"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    return 0
  fi
  if [[ -d "$out" ]] && [[ -n "$(ls -A "$out" 2>/dev/null || true)" ]]; then
    echo "Removing incomplete run artifacts: $out" >&2
    rm -rf "$out"
  fi
}

run_one() {
  local condition="$1"
  local scheduler="$2"
  local seed="$3"
  local replicate="${4:-}"
  local out="$EXP_DIR/${condition}/seed_${seed}"
  if [[ -n "$replicate" ]]; then
    out="${out}/rep_${replicate}"
  fi
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=()
  if [[ "$condition" == "hints" || "$condition" == "filter" || "$condition" == "filter_stub" || "$condition" == "filter_gray_zone" || "$condition" == "hints_rich" || "$condition" == "hints_parent" || "$condition" == "hints_direction" ]]; then
    extra+=(--require-surrogate-quality-gate)
  fi
  if [[ -n "$replicate" ]]; then
    extra+=(--replicate "$replicate")
  fi
  echo "=== tier=$TIER archive=$ARCHIVE_TYPE condition=$condition seed=$seed replicate=${replicate:-none} llm=$LLM_PROVIDER ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$RUN_SCRIPT" \
      --scheduler "$scheduler" \
      --output-dir "$out" \
      --seed "$seed" \
      --iterations "$ITERATIONS" \
      --load-archive "$BASELINE_ARCHIVE" \
      --llm-provider "$LLM_PROVIDER" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

run_pyribs_one() {
  local algo="$1"
  local seed="$2"
  local out="$EXP_DIR/${algo}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=()
  if [[ -n "${PYRIBS_EVALUATIONS:-}" ]]; then
    extra+=(--evaluations "$PYRIBS_EVALUATIONS")
  fi
  echo "=== tier=$TIER archive=$ARCHIVE_TYPE algo=$algo seed=$seed evaluations=${PYRIBS_EVALUATIONS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$PYRIBS_SCRIPT" \
      --algo "$algo" \
      --seed "$seed" \
      --output-dir "$out" \
      --load-archive "$BASELINE_ARCHIVE" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

run_cma_encoding_one() {
  local condition="$1"
  local decode_mode="$2"
  local no_warmstart="$3"
  local seed="$4"
  local out="$EXP_DIR/${condition}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=(
    --algo cma_me
    --decode-mode "$decode_mode"
    --condition-label "$condition"
  )
  if [[ -n "${PYRIBS_EVALUATIONS:-}" ]]; then
    extra+=(--evaluations "$PYRIBS_EVALUATIONS")
  fi
  if [[ "$no_warmstart" == "true" ]]; then
    extra+=(--no-load-archive)
  else
    extra+=(--load-archive "$BASELINE_ARCHIVE")
  fi
  echo "=== tier=$TIER condition=$condition decode=$decode_mode warmstart=$([[ "$no_warmstart" == true ]] && echo off || echo on) seed=$seed evaluations=${PYRIBS_EVALUATIONS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$PYRIBS_SCRIPT" \
      --seed "$seed" \
      --output-dir "$out" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

run_cma_native_discrete_one() {
  local seed="$1"
  local out="$EXP_DIR/cma_me_discrete/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=(
    --algo cma_me
    --emitter-kind discrete_cma
    --decode-mode threshold
    --condition-label cma_me_discrete
  )
  if [[ -n "${PYRIBS_EVALUATIONS:-}" ]]; then
    extra+=(--evaluations "$PYRIBS_EVALUATIONS")
  fi
  extra+=(--load-archive "$BASELINE_ARCHIVE")
  echo "=== tier=$TIER condition=cma_me_discrete emitter=discrete_cma warmstart=on seed=$seed evaluations=${PYRIBS_EVALUATIONS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$PYRIBS_SCRIPT" \
      --seed "$seed" \
      --output-dir "$out" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

run_cma_pbcma_one() {
  local seed="$1"
  local out="$EXP_DIR/cma_me_pbcma/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=(
    --algo cma_me
    --emitter-kind pbcma
    --decode-mode threshold
    --condition-label cma_me_pbcma
  )
  if [[ -n "${PYRIBS_EVALUATIONS:-}" ]]; then
    extra+=(--evaluations "$PYRIBS_EVALUATIONS")
  fi
  extra+=(--load-archive "$BASELINE_ARCHIVE")
  echo "=== tier=$TIER condition=cma_me_pbcma emitter=pbcma warmstart=on seed=$seed evaluations=${PYRIBS_EVALUATIONS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$PYRIBS_SCRIPT" \
      --seed "$seed" \
      --output-dir "$out" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

link_h2_tau045_from_legacy() {
  local seed="$1"
  local out="$EXP_DIR/genetic_me_filter_tau045/seed_${seed}"
  local legacy="$EXP_ROOT/q1-v3-genetic-me-filter/genetic_me_filter/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  if [[ ! -f "$legacy/nightly_run_summary.json" ]]; then
    return 1
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$(dirname "$out")"
  ln -sfn "$(realpath "$legacy")" "$out"
  echo "Linked τ=0.45 seed $seed from $legacy (reuse q1-v3-genetic-me-filter)"
  return 0
}

run_h2_threshold_one() {
  local tau_tag="$1"
  local scheduler="$2"
  local condition="$3"
  local seed="$4"
  if [[ "$tau_tag" == "045" ]]; then
    if link_h2_tau045_from_legacy "$seed"; then
      return 0
    fi
    echo "NOTE: no legacy τ=0.45 run for seed $seed; running fresh" >&2
  fi
  run_one "$condition" "$scheduler" "$seed"
}

run_pyribs_standard_one() {
  local algo="$1"
  local seed="$2"
  local out="$EXP_DIR/${algo}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=()
  if [[ -n "${PYRIBS_STANDARD_EVALUATIONS:-}" ]]; then
    extra+=(--evaluations "$PYRIBS_STANDARD_EVALUATIONS")
  fi
  echo "=== tier=$TIER benchmark=$STANDARD_BENCHMARK algo=$algo seed=$seed evaluations=${PYRIBS_STANDARD_EVALUATIONS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$PYRIBS_STANDARD_SCRIPT" \
      --benchmark "$STANDARD_BENCHMARK" \
      --algo "$algo" \
      --seed "$seed" \
      --output-dir "$out" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

run_dungeon_one() {
  local condition="$1"
  local seed="$2"
  local scheduler="$ROOT/worldspace/specs/dungeon_scheduler_${condition}.yaml"
  local out="$EXP_DIR/${condition}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=()
  if [[ -n "${DUNGEON_PROPOSALS:-}" ]]; then
    extra+=(--proposals "$DUNGEON_PROPOSALS")
  fi
  echo "=== tier=$TIER domain=dungeon condition=$condition seed=$seed proposals=${DUNGEON_PROPOSALS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$DUNGEON_SCRIPT" \
      --scheduler "$scheduler" \
      --seed "$seed" \
      --output-dir "$out" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

run_maze_one() {
  local condition="$1"
  local seed="$2"
  local scheduler="$ROOT/worldspace/specs/maze_scheduler_${condition}.yaml"
  local out="$EXP_DIR/${condition}/seed_${seed}"
  if [[ -f "$out/nightly_run_summary.json" ]]; then
    echo "Skip existing: $out"
    return 0
  fi
  remove_incomplete_run_dir "$out"
  mkdir -p "$out"
  local extra=()
  if [[ -n "${MAZE_PROPOSALS:-}" ]]; then
    extra+=(--proposals "$MAZE_PROPOSALS")
  fi
  if [[ -n "${MAZE_MOCK_LLM:-}" ]]; then
    extra+=(--mock-llm)
  fi
  if [[ -n "${MAZE_SIM_COST_MS:-}" ]]; then
    extra+=(--sim-cost-ms "$MAZE_SIM_COST_MS")
  fi
  echo "=== tier=$TIER domain=maze condition=$condition seed=$seed proposals=${MAZE_PROPOSALS:-32500} ==="
  local lock_file="$out/.run.lock"
  (
    flock -n 9 || {
      echo "Another process holds $lock_file; refusing to start duplicate run." >&2
      exit 1
    }
    uv run python "$MAZE_SCRIPT" \
      --scheduler "$scheduler" \
      --seed "$seed" \
      --output-dir "$out" \
      "${extra[@]}"
  ) 9>"$lock_file"
}

if [[ "$RUN_MAZE" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for condition in "${MAZE_CONDITIONS[@]}"; do
      run_maze_one "$condition" "$seed"
    done
  done
elif [[ "$RUN_DUNGEON" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for condition in "${DUNGEON_CONDITIONS[@]}"; do
      run_dungeon_one "$condition" "$seed"
    done
  done
elif [[ "$RUN_PYRIBS_STANDARD" == true ]]; then
  standard_algos=(cma_me cma_mae)
  if [[ "$STANDARD_BENCHMARK" == "sphere" ]]; then
    standard_algos+=(me_random)
  fi
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for algo in "${standard_algos[@]}"; do
      run_pyribs_standard_one "$algo" "$seed"
    done
  done
elif [[ "$RUN_PYRIBS" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for algo in cma_me cma_mae; do
      run_pyribs_one "$algo" "$seed"
    done
  done
elif [[ "$RUN_MIXED_2X2" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    run_one stub_uniform "$SCHEDULER_STUB_UNIFORM" "$seed"
    run_one hints "$SCHEDULER_HINTS" "$seed"
    run_one filter_stub "$SCHEDULER_FILTER_STUB" "$seed"
    run_one filter "$SCHEDULER_FILTER" "$seed"
  done
elif [[ "$RUN_ANYTIME_LADDER" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    run_one vanilla "$SCHEDULER_VANILLA" "$seed"
    run_one hints "$SCHEDULER_HINTS" "$seed"
    run_pyribs_one cma_me "$seed"
  done
elif [[ "$RUN_CMA_ENCODING_ABLATION" == true ]]; then
  # Reference arm: frozen q1-v3-pyribs/cma_me (rint + warm-start). Not re-run here.
  CMA_ENCODING_ARMS=(
    "cma_me_threshold:threshold:false"
    "cma_me_bernoulli:bernoulli:false"
    "cma_me_cold:rint:true"
  )
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for arm_spec in "${CMA_ENCODING_ARMS[@]}"; do
      IFS=":" read -r condition decode_mode no_warmstart <<< "$arm_spec"
      run_cma_encoding_one "$condition" "$decode_mode" "$no_warmstart" "$seed"
    done
  done
elif [[ "$RUN_CMA_DISCRETE" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    run_cma_encoding_one "cma_me_bernoulli" "bernoulli" "false" "$seed"
  done
elif [[ "$RUN_CMA_NATIVE_DISCRETE" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    run_cma_native_discrete_one "$seed"
  done
elif [[ "$RUN_CMA_PBCMA" == true ]]; then
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    run_cma_pbcma_one "$seed"
  done
elif [[ "$RUN_H2_THRESHOLD_SWEEP" == true ]]; then
  H2_THRESHOLD_ARMS=(
    "035:$SCHEDULER_GENETIC_ME_FILTER_TAU035:genetic_me_filter_tau035"
    "045:$SCHEDULER_GENETIC_ME_FILTER_NIGHTLY:genetic_me_filter_tau045"
    "055:$SCHEDULER_GENETIC_ME_FILTER_TAU055:genetic_me_filter_tau055"
  )
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for arm_spec in "${H2_THRESHOLD_ARMS[@]}"; do
      IFS=":" read -r tau_tag scheduler condition <<< "$arm_spec"
      run_h2_threshold_one "$tau_tag" "$scheduler" "$condition" "$seed"
    done
  done
else
  for seed in $(seq "$SEED_START" "$SEED_END"); do
    for rep in $(seq 0 $((REPLICATE_COUNT - 1))); do
      rep_arg=""
      if [[ "$REPLICATE_COUNT" -gt 1 ]]; then
        rep_arg="$rep"
      fi
      if [[ "$RUN_SHADOW" == true ]]; then
        run_one hints "$SCHEDULER_HINTS" "$seed" "$rep_arg"
        run_one filter "$SCHEDULER_FILTER" "$seed" "$rep_arg"
      elif [[ "$RUN_VANILLA" == true ]]; then
        run_one vanilla "$SCHEDULER_VANILLA" "$seed" "$rep_arg"
      elif [[ "$RUN_GENETIC_ME" == true ]]; then
        run_one genetic_me "$SCHEDULER_GENETIC_ME" "$seed" "$rep_arg"
      elif [[ "$RUN_GENETIC_ME_UNIFORM" == true ]]; then
        run_one genetic_me_uniform "$SCHEDULER_GENETIC_ME_UNIFORM" "$seed" "$rep_arg"
      elif [[ "$RUN_GENETIC_ME_FILTER" == true ]]; then
        run_one genetic_me_filter "$SCHEDULER_GENETIC_ME_FILTER" "$seed" "$rep_arg"
      elif [[ "$RUN_STUB_UNIFORM_ONLY" == true ]]; then
        run_one stub_uniform "$SCHEDULER_STUB_UNIFORM" "$seed" "$rep_arg"
      elif [[ "$RUN_HINTS_RICH_ONLY" == true ]]; then
        run_one hints_rich "$SCHEDULER_HINTS_RICH" "$seed" "$rep_arg"
      elif [[ "$RUN_HINTS_PARENT_ONLY" == true ]]; then
        run_one hints_parent "$SCHEDULER_HINTS_PARENT" "$seed" "$rep_arg"
      elif [[ "$RUN_HINTS_DIRECTION_ONLY" == true ]]; then
        run_one hints_direction "$SCHEDULER_HINTS_DIRECTION" "$seed" "$rep_arg"
      elif [[ "$RUN_WEAK_HINTS_PILOT" == true ]]; then
        run_one stub_uniform "$SCHEDULER_STUB_UNIFORM" "$seed" "$rep_arg"
        run_one hints "$SCHEDULER_HINTS" "$seed" "$rep_arg"
      elif [[ "$RUN_GRAY_ZONE_ONLY" == true ]]; then
        run_one filter_gray_zone "$SCHEDULER_FILTER_GRAY_ZONE" "$seed" "$rep_arg"
      else
        if [[ "$FILTER_ONLY" == true ]]; then
          require_stub_hints_for_seed "$seed"
        else
          run_one stub "$SCHEDULER_STUB" "$seed" "$rep_arg"
          run_one hints "$SCHEDULER_HINTS" "$seed" "$rep_arg"
        fi
        if [[ "$RUN_FILTER" == true ]]; then
          run_one filter "$SCHEDULER_FILTER" "$seed" "$rep_arg"
        fi
      fi
    done
  done
fi

if [[ -f "$AGG_SCRIPT" && -z "${LIFEMANIFOLD_SKIP_EXPERIMENT_AGGREGATE:-}" ]]; then
  if ! uv run python "$AGG_SCRIPT" --root "$EXP_DIR" --output "$EXP_DIR/summary.csv"; then
    echo "WARNING: failed to write $EXP_DIR/summary.csv (runs may still be valid)" >&2
    exit 1
  fi
  echo "Wrote $EXP_DIR/summary.csv"
fi
