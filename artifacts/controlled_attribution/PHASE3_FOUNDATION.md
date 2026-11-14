# Как читать Phase 3 (дизайн протокола, freeze ещё не в силе)

**Дата:** 2026-09-02  
**Слайс:** P3.2 (stamp package). Адаптеры P3.1 готовы. **Внешний stamp и tag ещё не сделаны.**  
**Это не preregistration и не methods, пока нет внешнего stamp.**  
**Черновик протокола:** `PHASE3_PROTOCOL.md` (SHA в `PHASE3_PROTOCOL.sha256`). Машинный каркас: `phase3_protocol_draft.json`.  
**Пакет регистрации:** `PHASE3_OSF_REGISTRATION_PACK.md`.

Читать вместе с `PHASE0_FOUNDATION.md`, `PHASE1_FOUNDATION.md`, `PHASE2_FOUNDATION.md`, `../CONTROLLED_ATTRIBUTION_RESEARCH_PROGRAM.md` §4–10, `../PROTOCOL_FREEZE_TIMELINE.md`.

---

## 1. Короткий ответ

Две публичные confirmatory-задачи **называются** здесь (не в P2.0 и не задним числом в P2.5):

1. **NAS-Bench-201** (lookup NATS TSS simple, поиск `cifar10-valid` / hp 200).
2. **PCG Benchmark `sokoban-v0`** с именованным и **matched** `repair=structural_counts`. Zelda не вторая семья и не confirmatory-задача.

Holm, n, H, модель, клетки Blocks A–C, state-aware правило, cap — **выписаны в черновике протокола**. Они не действуют, пока нет **git tag + внешний pre-data stamp** (OSF или AsPredicted) на конкретный SHA-256 `PHASE3_PROTOCOL.md`. Git SHA и mtime — integrity aids, не регистрация (урок Q1).

Feasibility P2.1–P2.5r **не** пулится с confirmatory. CA и maze — historical / motivating, не Holm-семьи этой программы. P2.6 нет. Sidecar-адаптеры NAS/PCG — P3.1 **готовы**. Confirmatory LLM не запускается до freeze.

---

## 2. Чем это не является

- Не stamp. Файл можно менять до tag. После stamp правки только dated amendment с dual-report.
- Не Phase 4. Ни одного confirmatory run.
- Не reproduction OpenELM / LLMatic / in-context QD (программа §11, optional after core).
- Не выбор `sokoban-v0` + `zelda-v0` как двух публичных семей.
- Не перенос Holm/n из feasibility-цифр «потому что G3=50/50». Пороги G1–G8 остались GO-критериями Phase 2; confirmatory-семьи — новые, внутрипротокольные.
- Не paper-wide Holm. Не TOST без заранее названного SESOI (его нет — equivalence не заявляется).
- Не смена BC NAS на op-histogram: сетка P2.1 остаётся; coverage NAS **не** Holm (потолок 21/400).
- Не тихий identity-repair на PCG и не maze `solvability_repair`.
- Не commit/tag из этого слайса, пока не запрошен.

---

## 3. Чем это является

Ответ программы §10 Phase 3: датированный протокол с доменами, клетками, моделью, сидами, горизонтами, гейтами, Holm-семьями, release/privacy, exclusions.

P3.0 выписывает это как **draft**. Исполнение freeze:

1. P3.1 — native runner + `ReadOnlyNormalizationAdapter` для `nas201` и `pcg_sokoban`; live-промпты с хэшами; allocation `early` / `state_aware_median` в capabilities; job manifests без launch.
2. P3.2 — записать SHA-256 протокола, **внешний stamp**, потом git tag. OSF ID вписывается одной reporting-строкой, если stamp уже указывает на тот же текст.
3. Phase 4 — прогоны. First-inspection datetime логируется; не выдумывается.

§18 спеки, которые этот протокол закрывает для публичных задач: currency = USD, dated provider list price (таблица **на stamp**, не из feasibility); execution = in-process lookup/A* + существующий remote LLM caller. Sphere float-hash и dungeon adapters не нужны.

---

## 4. Слайс-план

| Слайс | Что | LLM confirmatory | Freeze |
|---|---|---|---|
| **P3.0** | foundation + `PHASE3_PROTOCOL.md` | нет | draft |
| **P3.1** | адаптеры, live-промпты, Block C kinds, job builder | нет | нет |
| **P3.2** | price table + claim lock + OSF pack; **затем** внешний stamp + tag | нет | **в силе после registry+tag** |
| **Phase 4** | Blocks A–C на двух задачах; D descriptive NAS; second-model robustness | да | amendments only |

Не начинать Phase 4 до stamp. Не калибровать genetic Hamming по P2.2/P2.4. Не писать dungeon/sphere adapters.

---

## 5. Почему так, а не иначе

| Решение | Почему |
|---|---|
| Две задачи = NAS + sokoban+counts | Единственные GO-пути Phase 2; zelda-repair провалил G5 |
| n=20 обе | программа §9: lookup ≥20; PCG eval ~67 ms, узкое место всё ещё LLM |
| H=200 | тот же горизонт, на котором читали G5/G6; не 500 «на всякий» |
| Genetic = один шаг (ребро / тайл) | идентифицированный оператор smoke; не fit к LLM Hamming. Uncalibrated = primary |
| Repair PCG matched на всех руках | фактор назван; LLM vs genetic не смешан с «починили только LLM» |
| Coverage NAS не Holm | полная таблица 21/400 бинов; QD-score anytime AUC — primary |
| Holm m=2 на семью | два простых эффекта; interaction 2×2 заранее descriptive |
| Live channel = parent fitness/quality, не archive/QD | Block A про prompt information, не про state-aware allocation (это Block C) |
| CA/maze не в Holm | Q1 — другая программа; не paper-wide, не dual-lock |
| Cap в calls/tokens | monetary таблица датируется на stamp; G7 Phase 2 cap не объявлял |

---

## 6. Практическое правило отсюда

- Не запускать confirmatory LLM «проверить протокол».
- Не пулить `nas201_isolated.json` / `pcg_isolated*.json` в анализ Phase 4.
- Не двигать n/H/Holm, глядя на P3.1 parse-smoke.
- Q1 YAML и archived nightly summaries — byte-unchanged.
- Следующий шаг оператора — **внешний stamp** по `PHASE3_OSF_REGISTRATION_PACK.md`, затем tag `phase3-protocol-v0`. Не Phase 4 из чата.

Если коротко: **пакет к stamp готов; freeze честно ещё выключен, пока нет OSF/AsPredicted + tag.**
