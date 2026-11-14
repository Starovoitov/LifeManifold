# OSF / AsPredicted registration pack — Phase 3 protocol `ca-phase3-v0`

**Use this text for the external pre-data stamp.** Do not start Phase 4 LLM until the registry entry exists and names the SHA below.

## Title

Controlled Attribution Phase 3 confirmatory protocol (`ca-phase3-v0`): NAS-Bench-201 and PCG sokoban-v0 MAP-Elites factorial (Blocks A–C)

## Document pin

- Repository path: `artifacts/controlled_attribution/PHASE3_PROTOCOL.md`
- Protocol ID: `ca-phase3-v0`
- SHA-256 (UTF-8 file bytes):  
  `0377685986abf4ce2c256f0d8c63732a6fd05a28e4080ec4bd3dc890540cce91`
- Companion machine card: `artifacts/controlled_attribution/phase3_protocol_draft.json` (draft until freeze)
- Price table ID: `openai-public-list-2026-09-02`  
  (`price_table.json` SHA-256 `a60d1f99b89e7067b8fb2e1eac0c9a7f2b64c041f460101e864c2b413e549b94`)

## Hypotheses / estimands (summary)

Primary endpoint for all Holm contrasts: **QD-score anytime AUC**, proposal-indexed. Drop the 20-random floor, then equal-spaced trapezoidal area / (n − 1) on the remaining points (n = H = 200 when the search completes).

Paired seeds: `301001`–`301020` (overflow `301021`–`301040`). α=0.05. Test: paired Wilcoxon signed-rank. Multiplicity: Holm **within family only** (no paper-wide Holm; no TOST/SESOI).

| Family | Domain | Contrasts (m) |
|---|---|---|
| F-A-nas / F-A-pcg | each domain | (1) minfit−uniform \| constant; (2) live−constant \| uniform |
| F-B-nas / F-B-pcg | each domain | (1) LLM−genetic; (2) LLM−random |
| F-C-nas / F-C-pcg | each domain | (1) state_aware_median−static |

Block A 2×2 interaction is **descriptive only**. NAS coverage is **not** Holm. Zelda is **not** a second public task. Phase 2 feasibility artifacts are **not** pooled.

## Models

- Primary: request `gpt-4o-mini`; response pin `gpt-4o-mini-2024-07-18`; T=0.2; max_tokens=350.
- Robustness (not Holm): request `gpt-4o`.

## Cap

Stop new confirmatory LLM launches at first of: 90 000 completed LLM calls, 40 000 000 total tokens, or 14 calendar days after first confirmatory API call.

## Data collection status at registration

**No confirmatory Phase 4 runs yet.** Phase 2 outputs are design data only.

## Post-registration

1. Record registry URL/ID in `phase3_protocol_draft.json` (`external_stamp`).
2. Git tag `phase3-protocol-v0` on the commit containing the byte-identical protocol file.
3. Set `freeze_in_force: true` only after both.

## Verification command

```bash
sha256sum artifacts/controlled_attribution/PHASE3_PROTOCOL.md
# expect: 0377685986abf4ce2c256f0d8c63732a6fd05a28e4080ec4bd3dc890540cce91
```
