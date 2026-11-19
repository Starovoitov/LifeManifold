# Phase 3.2 — stamp package

**Date:** 2026-09-02  
**Status:** `freeze_in_force` **true** as of 2026-11-20. No confirmatory LLM yet.

## Reporting line

OSF registration [https://osf.io/8jaxy](https://osf.io/8jaxy), DOI `10.17605/OSF.IO/8JAXY`, registered 2026-11-14T23:46:44Z, names SHA-256 `0377685986abf4ce2c256f0d8c63732a6fd05a28e4080ec4bd3dc890540cce91`. Git tag `phase3-protocol-v0` points at commit `197e2ef04419fb37ae9492d8c3bbaf4484cdf588`, which contains that byte-identical `PHASE3_PROTOCOL.md`. This record does not edit that file.

## Why both were required

Programme rule (Q1 lesson): git SHA / mtime ≠ registration. Freeze requires **both**:

1. OSF or AsPredicted timestamp naming the protocol SHA below;
2. git tag `phase3-protocol-v0` on the commit that contains that **byte-identical** `PHASE3_PROTOCOL.md`.

This slice prepared the payload. The registry entry and the tag now both exist; the reporting line above is the record.

## Protocol pin

| Item | Value |
|---|---|
| File | `artifacts/controlled_attribution/PHASE3_PROTOCOL.md` |
| Protocol ID | `ca-phase3-v0` |
| SHA-256 | `0377685986abf4ce2c256f0d8c63732a6fd05a28e4080ec4bd3dc890540cce91` |
| Price table | `price_tables/2026-09-02/price_table.json` → `a60d1f99b89e7067b8fb2e1eac0c9a7f2b64c041f460101e864c2b413e549b94` |

## Delivered in this slice

1. Dated OpenAI public-list price table + page snapshots under `price_tables/2026-09-02/`.
2. Price hashes wired into protocol §8.
3. §14 claim sentences locked as stamp wording (activate only after registry+tag).
4. §13.2 stamp procedure written.
5. Registration pack: `PHASE3_OSF_REGISTRATION_PACK.md`.
6. `.gitignore` whitelist so the stamp package can enter git (7.9G Phase 2 trees stay ignored).

## Operator checklist

Done. The SHA pin matched, the pack was registered, and the reporting line plus tag `phase3-protocol-v0` are recorded above. `freeze_in_force` is true in `phase3_protocol_draft.json`.

## Not done

- Phase 4 runs
- Prospective public-runner event capture (optional)

## Q1 guard

Scheduler YAML and archived Q1 nightly summaries remain byte-unchanged.
