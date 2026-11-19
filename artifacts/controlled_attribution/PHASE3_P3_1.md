# Phase 3.1 — public-task adapters (pre-stamp)

Status: **complete for adapter surface**. Freeze is recorded in `phase3_protocol_draft.json` (`freeze_in_force: true`; tag `phase3-protocol-v0` at `197e2ef`). No confirmatory LLM yet.

## Delivered

1. **Capabilities** (`worldspace/attribution/capabilities.py`)
   - `nas201` / `pcg_sokoban` registered in `current_domain_capabilities()`.
   - `allocation_kinds`: `static`, `state_aware_median`.
   - NAS `repair_fallback_kinds` is `identity` only. Runners reject `genetic_fallback` as a repair kind.
   - PCG `repair_fallback_kinds`: `identity`, `structural_counts`. Confirmatory arms use `structural_counts`.
   - Init: `generated_floor` (+ `loaded_archive`).

2. **Native loops** (sidecar, not a universal QD kernel)
   - `worldspace/nas201/runner.py` → `run_nas201_qd`
   - `worldspace/pcg/runner.py` → `run_pcg_sokoban_qd`
   - Shared allocation helper: `worldspace/attribution/public_loop.py`
   - Artifacts: `nightly_run_summary.json`, `{nas201,pcg}_archive.jsonl`, `archive_trace.jsonl`

3. **Read-only normalizers**
   - `Nas201NormalizationAdapter`, `PcgSokobanNormalizationAdapter`
   - Shared: `worldspace/attribution/adapters/public.py`

4. **Live prompts + scanners** (SHA-256 below; insert into protocol by this amendment)

| File | SHA-256 |
|---|---|
| `prompts/nas201_llm_emitter_system.txt` | `373ecefdfb040e7b8a5cbc79ff400522cc509512c68f071209522c066cbe3b00` |
| `prompts/nas201_llm_emitter_user.txt` (constant) | `95e192f17552450d616bde1ff7e95fc1bf1b76dcb204d563d4ab56cd7f0ae1e5` |
| `prompts/nas201_llm_emitter_live_user.txt` | `a96af227263d9ae41fb49d903c6443590414b626588c8d658a43a865bc0d4233` |
| `prompts/pcg_sokoban_llm_emitter_system.txt` | `50baddc3d9cb159fded244938dce2253e83818b5c6b4fd9e90b204a2e6b8e687` |
| `prompts/pcg_sokoban_llm_emitter_user.txt` (constant) | `874cf140f29192b8ee193eb6922738c4ca7e623f9455ba5754f14d0b401b7584` |
| `prompts/pcg_sokoban_llm_emitter_live_user.txt` | `caf5778b9390aa334437bdbb595221d159d0616fbaa616df800a8b243edda694` |

   Scanners: `worldspace/nas201/live_prompt_scan.py`, `worldspace/pcg/live_prompt_scan.py`.

5. **Job builder cells without launch**
   - `worldspace/attribution/confirmatory_studies.py` — Block A/B/C studies
   - `scripts/build_confirmatory_jobs.py` — emit plans only (`launched=False`)

6. **Admission**
   - `worldspace/attribution/confirmatory_admission.py` — refuse Phase 2/feasibility JSONL, Q1 nightly summaries, mixed `evidence_tier`

7. **Tests**: `tests/test_attribution_confirmatory.py`

## Explicitly not done (P3.2 / Phase 4)

- Protocol stamp + git tag
- Price table lock
- Confirmatory LLM runs
- Prospective `attribution_events.jsonl` capture on public runners (summary/trace only today)
- Zelda / dungeon / sphere adapters

## Q1 guard

Scheduler YAML and archived Q1 `nightly_run_summary.json` remain byte-unchanged.
