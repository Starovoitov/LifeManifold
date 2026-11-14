# Phase 3.2 — stamp package (external registry still required)

**Date:** 2026-09-02  
**Status:** stamp **package ready**; `freeze_in_force` still **false**. No confirmatory LLM. No git tag yet.

## Why freeze is still false

Programme rule (Q1 lesson): git SHA / mtime ≠ registration. Freeze requires **both**:

1. OSF or AsPredicted timestamp naming the protocol SHA below;
2. git tag `phase3-protocol-v0` on the commit that contains that **byte-identical** `PHASE3_PROTOCOL.md`.

This slice prepares (1)'s payload and (2)'s prerequisites. It does **not** invent an OSF ID or create the tag from chat.

## Protocol pin

| Item | Value |
|---|---|
| File | `artifacts/controlled_attribution/PHASE3_PROTOCOL.md` |
| Protocol ID | `ca-phase3-v0` |
| SHA-256 | `bcbd9f6376ff10364d37015d68fcde1ff9471e6140f9c4e99d05941031b8dc62` |
| Price table | `price_tables/2026-09-02/price_table.json` → `a60d1f99b89e7067b8fb2e1eac0c9a7f2b64c041f460101e864c2b413e549b94` |

## Delivered in this slice

1. Dated OpenAI public-list price table + page snapshots under `price_tables/2026-09-02/`.
2. Price hashes wired into protocol §8.
3. §14 claim sentences locked as stamp wording (activate only after registry+tag).
4. §13.2 stamp procedure written.
5. Registration pack: `PHASE3_OSF_REGISTRATION_PACK.md`.
6. `.gitignore` whitelist so the stamp package can enter git (7.9G Phase 2 trees stay ignored).

## Operator checklist (you)

1. Confirm `sha256sum artifacts/controlled_attribution/PHASE3_PROTOCOL.md` equals the pin above.
2. **Commit** the stamp-package files (ask the agent if you want the commit made for you). **Do not tag yet.**
3. Paste `PHASE3_OSF_REGISTRATION_PACK.md` into OSF Registrations or AsPredicted **before any Phase 4 LLM**.
4. Reply with registry URL/ID.
5. Then: Reporting line + `git tag phase3-protocol-v0` on that commit; set `freeze_in_force: true`.

## Not done

- External stamp itself
- `phase3-protocol-v0` tag
- Phase 4 runs
- Prospective public-runner event capture (optional)

## Q1 guard

Scheduler YAML and archived Q1 nightly summaries remain byte-unchanged.
