# Dashboard parity checklist

Manual verification for MAP-Elites dashboard after CVT archive support (E6).

**Fixtures**

| Archive | Path |
|---------|------|
| Grid smoke | `artifacts/map_elites_smoke/map_elites_archive.jsonl` |
| CVT smoke | `artifacts/map_elites_smoke_cvt/map_elites_archive.jsonl` |

Run dashboard: `cd dashboard && streamlit run Home.py`

---

## Grid regression (`map_elites_smoke`)

- [ ] **Home** — run card shows `archive_type=grid`, niches = 2500 (or summary value)
- [ ] **Archive Explorer** — heatmap 50×50, metric selector works
- [ ] **Archive Explorer** — bin selection + diagnostic panel render
- [ ] **LLM Prompt Tester** — grid system prompt (N×N grid wording), bin/cell select works

---

## CVT archive (`map_elites_smoke_cvt`)

- [ ] **Home** — run card or archive stats show `archive_type=cvt`, niches = 25
- [ ] **Archive Explorer** — scatter (not heatmap); filled + hollow empty niches
- [ ] **Archive Explorer** — cell label format `cell N (s=…, d=…)`
- [ ] **Archive Explorer** — diagnostic title matches selected cell
- [ ] **LLM Prompt Tester** — CVT system prompt mentions Voronoi / `n_centroids=25`
- [ ] **Centroids missing** — rename `cvt_centroids.json` temporarily → warning + degraded scatter

---

## Archive switching (sidebar JSONL)

- [ ] Select grid smoke → heatmap
- [ ] Select CVT smoke → scatter
- [ ] Switch back to grid → heatmap again (no stale CVT chart)

---

## Sign-off

| Date | Tester | Notes |
|------|--------|-------|
| | | |
