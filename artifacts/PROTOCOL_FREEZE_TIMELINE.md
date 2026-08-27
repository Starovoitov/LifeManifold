# Protocol freeze kinds and freeze↔data timeline

Auditor-facing ledger. **Not part of the journal PDF.** Holm families, endpoints, and numeric outcomes are unchanged by moving this file out of the manuscript appendix.

Full wording log: [`EXPERIMENT_PROTOCOL_Q1_v3.md`](EXPERIMENT_PROTOCOL_Q1_v3.md) §12.

External integrity pin for every row: Zenodo snapshot DOI [10.5281/zenodo.22119553](https://doi.org/10.5281/zenodo.22119553) (git tag `journal-v1.1` / `a714e548cfd6`). First version of the same concept: [10.5281/zenodo.21727011](https://doi.org/10.5281/zenodo.21727011) (`journal-v1`). That snapshot is **not** evidence of pre-data registration.

## Status

Confirmatory families were specified in dated internal protocol files before the recorded analysis window for each family. This is an internal audit trail of rule-before-analysis timing, **not** an externally timestamped pre-registration (no OSF/AsPredicted lock before data). Public `git` history and filesystem artifact mtimes are integrity aids; they are **mutable** and are not equivalent to an independent registry timestamp.

## Kinds

- **Lock** — freezes a confirmatory family's endpoints / Holm (or NI) rules in the internal protocol relative to that family's analysis window.
- **Extension** — adds a new package or repair path; does not rewrite already-frozen families.
- **Reporting** — records an outcome, descriptive case study, or post-hoc TOST on already-collected paired seeds; not a change of locked family rules.

Confirmatory Holm families for the bundled stack and dungeon AUC were not redefined by Extension/Reporting rows (H2 was never a Holm family; the 2026-08-09 row only clarifies descriptive status). H4 is the explicit exception: the 2026-08-07 Reporting row is a **methodological category correction** (representation-mismatched `rint` encoding made confirmatory “hints > CMA” a category error from the v3 lock onward), not a post-hoc status demotion after seeing FAIL. Paired deltas are unchanged.

**Dual-report:** under the original locked F-RQ4 (Holm *m*=4), the family remains a failure (only `RQ4_mae_fit` rejects; CMA-ME Δcov mean −12.96 pp). The manuscript retains that locked-family outcome and adds the descriptive end-to-end-baseline reading. A former pooled “paper-wide” Holm table is withdrawn (2026-08-09): multiplicity stays within-family only.

## Freeze / status events

Wording-only manuscript edits are omitted.

| Date | Kind | Scope | Note |
| --- | --- | --- | --- |
| 2026-07-12 | Lock | Primary-grid §§1–4 / F-RQ0 / F-RQ4 | Document-dated freeze (v3 header) |
| 2026-07-17 | Lock | Dungeon v4 confirmatory AUC | Bundled stack / H4 wording unchanged |
| 2026-07-17 | Reporting | Matched H2 genetic filter | Protocol §3.8 descriptive; no Holm family |
| 2026-07-22 | Reporting | No external pre-registration declared | Disclosure, not a family rewrite |
| 2026-07-22 | Extension | Multi-provider bundled *n*=10 | Per-provider Holm; not matched H1 |
| 2026-07-28 | Extension | Hold-out gate=0.95; F-RQ3-gray path | Production H3 not confirmatory |
| 2026-07-29 | Extension | Gray-zone exploratory pilot (*n*=10) | Not pooled into production H3 |
| 2026-07-30 | Extension | Maze H5 confirmatory; Sphere H2; cost wall | Maze mixed; wall descriptive |
| 2026-07-31 | Reporting | H3-gray outcome; matched H1 TOST | Outcome/descriptive; H3-prod still not confirmatory |
| 2026-08-07 | Reporting | H4 category correction → descriptive ceiling | Representation mismatch made confirmatory Holm *m*=4 a category error; deltas unchanged; locked F-RQ4 still FAIL |
| 2026-08-09 | Reporting | H2 status recorded as descriptive (§3.8) | Manuscript wording; no Holm; does not invent F-RQ2 |
| 2026-08-09 | Reporting | Withdraw pooled “paper-wide” Holm | Incomplete *m*=16 vector dropped; multiplicity within-family only |
| 2026-08-13 | Reporting | Cold mixed 2×2 *n*=10 readout | Descriptive empty-archive bound; not Holm; does not amend warm H1/H2 |
| 2026-08-17 | Extension+Rep. | Calendar-blocked RQ1 2×2 complete | Five independent floors; dated model; call logs. Not Holm; not TOST; locked families unchanged |
| 2026-08-18 | Extension+Rep. | Maze empty RQ1 2×2 complete | Empty 900-cell archive at 5k; dated model; call logs. Not Holm; not H5; locked families unchanged |
| 2026-08-26 | Extension | Sphere H1 second domain; Phase A GO | Genetic minfit vs uniform −5.79 pp @ 5k (10/10). Phase B not launched. Not Holm; not Sphere H2 |
| 2026-08-26 | Extension | Sphere H1 Phase B launched | `q1-rq1-sphere-factorial` 2 workers after preflight 20/20 parse 1.0. Grid running, not read. Not Holm; not Sphere H2 |
| 2026-08-26 | Reporting | Main-text “X is not Y” wall → §6.5 | Self-definitions collected once in Limitations; gating named once in Intro |
| 2026-08-26 | Reporting | Stacked hedges unstacked | One qualifier per thesis. Maze numbers no longer carry second-evaluator / not-portable / not-primary in the same sentence |
| 2026-08-26 | Reporting | H2 / Holm disclaimers once in §4.5 | Budget-axis / not co-equal RQ, and leftover–calendar–maze–H2 not Holm, said once. Pointers elsewhere |
| 2026-08-26 | Reporting | Authorial voice in Intro / Discussion | Expected scalars to matter; they did not. Bundled gap evaporates under matched policy |
| 2026-08-26 | Reporting | Main-text cross-refs thinned | At most one section/table/figure pointer per paragraph. Appendix maps no longer cited from the running argument |
| 2026-08-27 | Extension+Rep. | Sphere H1 LLM 2×2 readout | Empty 10k-cell archive at 5k; dated model; call logs. Policy +2.68/+2.47 pp; leftover +0.00/−0.21 pp. Not Holm; not Sphere H2; locked families unchanged |
| 2026-08-27 | Reporting | Sphere H1 not in journal PDF | Keep H2-only in the manuscript. H1 2×2 remains repo/protocol. Maze empty 2×2 stays the second-evaluator table |
| 2026-08-27 | Reporting | Sphere H1 appendix-only | Reverses same-day keep-out. One appendix table; not main text; not Holm. Maze empty 2×2 remains the second-evaluator table |
| 2026-08-27 | Reporting | Zenodo artifact pin v2 | `10.5281/zenodo.22119553` / tag `journal-v1.1` / `a714e548cfd6` supersedes v1 `21727011` (`journal-v1`) as the current artifact DOI |

## Freeze versus recorded data window

For each claim-driving family: internal-protocol freeze date; first/last artifact mtime window, used only as a filesystem proxy for the recorded data-generation window; a public git SHA that carries the protocol/analysis text; and amendment class. The external integrity pin applies to every row.

Missing by design: a logged “first result inspection” timestamp and an independent pre-data registry ID—we do not invent them. Artifact windows are local filesystem mtimes, not wall-clock run logs. Git SHAs are mutable history tips. Where the document freeze date precedes the first commit that carries that freeze text, both are reported—that gap is why an external pre-data stamp would strengthen a later revision.

Dates `YYYY-MM-DD` local.

| Family | Internal freeze | Data window | Git SHA | Kind |
| --- | --- | --- | --- | --- |
| Bundled stub/hints | v2 | `q1-full` 06-29–07-28 | historical | Lock\* |
| H4 (F-RQ4) | 2026-07-12 | `q1-v3-pyribs` 07-12–07-13 | v3 freeze text | Lock; later Reporting§ |
| H2 genetic filter | 2026-07-17 | `genetic_me_*` 07-16–07-17 | matched-H2 window | Descriptive¶ |
| Dungeon AUC (app.) | 2026-07-17 | `q1-v4-dungeon` 07-17–07-22 | v4 lock | Lock |
| Maze H5 (F-B5) | v5, before readout | `q1-v5-maze` 07-24–07-30 | `0d28707` | Lock |
| H3-gray | 2026-07-28 path | confirmatory 07-30–07-31 | `07d9550` / `716f7c3` | Ext.+Reporting |
| Matched H1 TOST | — (post-hoc) | `stub_uniform` / LLM (prior) | reporting commits | Reporting |
| Calendar-blocked 2×2 | 2026-08-13 | `q1-rq1-cf` 08-13–08-17 | protocol lock | Extension† |
| Maze empty 2×2 | 2026-08-17 | `q1-rq1-maze` 08-17–08-18 | protocol lock | Extension‡ |
| Sphere H1 2×2 | 2026-08-26 | `q1-rq1-sphere-factorial` 08-26 | protocol lock | Extension§ |

\* Bundled stack locked in v2 internal protocol; not a matched-H1 TOST lock.

§ H4 was locked as confirmatory Holm *m*=4 on 2026-07-12 (encoding asymmetry disclosed in the same freeze). Reporting row 2026-08-07 is a category correction to a descriptive ceiling—not a demotion of an inconvenient reject: paired deltas unchanged; locked F-RQ4 remains a failure (dual-report).

¶ Matched genetic H2 is protocol §3.8 **descriptive** (no Holm family); Kind is not a confirmatory lock. The mtime window starts 16 Jul, one day before the 17 Jul DONE text—an internal-audit overlap, not a silent upgrade to confirmatory.

† Calendar-blocked RQ1 2×2 is a locked extension (not a Holm family; not TOST).

‡ Maze empty RQ1 2×2 is a locked extension (not a Holm family; not H5; not CA occupancy).

§ Sphere H1 LLM 2×2 is a locked extension (not a Holm family; not Sphere H2; not CA occupancy). Phase A GO 2026-08-26; Phase B launched the same day; readout 2026-08-27.

First-inspection time not logged.

**Reading rule.** A Lock row plus a data window that starts at or after the freeze date supports the claim that the documented rules preceded the recorded artifact window, but does **not** establish when results were first inspected. H2 is not a Lock row. An Extension row must not be read as reopening H2. A Reporting row (including matched H1 TOST) must not be read as a new confirmatory family. The H4 Reporting row specifically corrects the **category** of a representation-mismatched confirmatory hypothesis (continuous-relaxation encoding); it does not erase the locked F-RQ4 dual-report.
