#!/usr/bin/env bash
# Build a curated Zenodo snapshot (protocols + paper-relevant experiment tiers).
# Does NOT upload; produces a .tar.gz under dist/zenodo/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GIT_SHORT="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
GIT_FULL="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
OUT_DIR="${ROOT}/dist/zenodo"
STAGE="${OUT_DIR}/LifeManifold-zenodo-${STAMP}-${GIT_SHORT}"
ARCHIVE="${OUT_DIR}/LifeManifold-zenodo-${STAMP}-${GIT_SHORT}.tar.gz"

mkdir -p "${STAGE}"/{protocols,experiments,checkpoints,surrogate,manuscript,meta,mazes,map_elites_nightly}

# --- license (same as GitHub root) ---
if [[ -f LICENSE ]]; then
  cp -a LICENSE "${STAGE}/"
else
  echo "WARN: missing LICENSE" >&2
fi

stage_dir() {
  local src="$1"
  local dst_parent="$2"
  if [[ -d "${src}" ]]; then
    echo "staging ${src}"
    mkdir -p "${dst_parent}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --exclude '__pycache__' --exclude '.pytest_cache' \
        "${src}" "${dst_parent}/"
    else
      cp -a "${src}" "${dst_parent}/"
    fi
  else
    echo "WARN: missing ${src}" >&2
  fi
}

# --- protocols ---
cp -a artifacts/EXPERIMENT_PROTOCOL_Q1.md \
      artifacts/EXPERIMENT_PROTOCOL_Q1_v2.md \
      artifacts/EXPERIMENT_PROTOCOL_Q1_v3.md \
      artifacts/EXPERIMENT_PROTOCOL_Q1_v4.md \
      artifacts/EXPERIMENT_PROTOCOL_Q1_v5.md \
      "${STAGE}/protocols/" 2>/dev/null || true
# Task lists that lock amended confirmatory paths
for f in artifacts/Q1_H3_GRAY_ZONE_CONFIRMATORY.md artifacts/Q1_H2_RANKING_CONTROLS.md; do
  if [[ -f "${f}" ]]; then
    cp -a "${f}" "${STAGE}/protocols/"
  fi
done

# --- experiment tiers (main claims + appendix packages with reported numbers) ---
TIERS=(
  # Main confirmatory / matched contrasts
  q1-full
  q1-v3-vanilla
  q1-v3-genetic-me
  q1-v3-genetic-me-uniform
  q1-v3-genetic-me-filter
  q1-v3-pyribs
  q1-stub-uniform-sensitivity
  q1-v3-llm
  q1-v3-h3-gray-zone
  q1-v3-h3-gray-zone-pilot
  q1-v5-maze-full
  q1-v5-maze-cost-h2
  q1-v4-dungeon-rerun
  q1-v3-sphere-h2
  edit-anatomy
  h1-llm-candidate-quality
  h2-gate-diagnostics
  h1-qd-cost
  q1-h2-ranking-controls
  # Appendix / descriptive packages cited in the manuscript
  q1-v4-dungeon-full-cpu
  q1-anytime-ladder
  q1-cma-encoding-ablation
  q1-v3-pyribs-pbcma
  q1-v3-pyribs-discrete-cma
  q1-v3-pyribs-native-discrete-cma
  q1-cvt
  q1-v3-sphere
  q1-v3-rastrigin
)
for t in "${TIERS[@]}"; do
  stage_dir "artifacts/experiments/${t}" "${STAGE}/experiments"
done

# Paper-wide Holm robustness JSON (lives next to experiment tiers)
if [[ -f artifacts/experiments/paper_wide_holm_robustness.json ]]; then
  echo "staging paper_wide_holm_robustness.json"
  cp -a artifacts/experiments/paper_wide_holm_robustness.json \
    "${STAGE}/experiments/"
fi

# --- checkpoints used in paper ---
for f in nightly_v3_mc_d005.pkl nightly_v3_mc_d005.summary.json \
         maze_v1.pkl; do
  if [[ -f "artifacts/surrogate/checkpoints/${f}" ]]; then
    cp -a "artifacts/surrogate/checkpoints/${f}" "${STAGE}/checkpoints/"
  else
    echo "WARN: missing artifacts/surrogate/checkpoints/${f}" >&2
  fi
done
# Sphere H2 MLP lives beside checkpoints/, not under checkpoints/
if [[ -f artifacts/surrogate/sphere_h2_mlp.joblib ]]; then
  echo "staging sphere_h2_mlp.joblib"
  cp -a artifacts/surrogate/sphere_h2_mlp.joblib "${STAGE}/checkpoints/"
else
  echo "WARN: missing artifacts/surrogate/sphere_h2_mlp.joblib" >&2
fi

# --- offline surrogate / compose-gate / ablation JSON (H3 appendix + validity) ---
SURROGATE_JSON=(
  holdout_compose_gate_alignment.json
  holdout_bootstrap_ci.json
  compose_gate_0p5_vs_0p95.json
  compose_gate_live_0p5_vs_0p95.json
  compose_gate_live_0p5_vs_0p95_minfit0p10.json
  compose_gate_fix_candidates.json
  filter_threshold_emitter_replay.json
  gp_ucb_ablation.json
)
for f in "${SURROGATE_JSON[@]}"; do
  if [[ -f "artifacts/surrogate/${f}" ]]; then
    echo "staging surrogate/${f}"
    cp -a "artifacts/surrogate/${f}" "${STAGE}/surrogate/"
  else
    echo "WARN: missing artifacts/surrogate/${f}" >&2
  fi
done

# --- maze wall-time microbench (break-even algebra inputs) ---
if [[ -d artifacts/mazes/walltime ]]; then
  stage_dir artifacts/mazes/walltime "${STAGE}/mazes"
fi

# --- warm-start archive used by LLM / ME runners ---
if [[ -d artifacts/map_elites_nightly/baseline ]]; then
  stage_dir artifacts/map_elites_nightly/baseline "${STAGE}/map_elites_nightly"
fi

# --- manuscript snapshot (source, not PDF requirement) ---
if [[ -d artifacts/manuscript ]]; then
  rsync -a --exclude '*.aux' --exclude '*.log' --exclude '*.out' \
    artifacts/manuscript/ "${STAGE}/manuscript/" 2>/dev/null \
    || cp -a artifacts/manuscript/. "${STAGE}/manuscript/"
fi

# --- meta ---
cat > "${STAGE}/meta/TIMELINE.md" << EOF
# Freeze ↔ artifact timeline

Snapshot built: ${STAMP} (UTC)
Git HEAD: ${GIT_FULL}
Short: ${GIT_SHORT}

This deposit is an **integrity snapshot** of protocol files and paper-relevant
experiment artifacts for the LifeManifold journal revision. It is **not** a
prospective OSF/Zenodo pre-registration of runs that already completed.

| Family | Protocol freeze (doc) | Artifact tier |
|--------|----------------------|---------------|
| Bundled stub/hints | v2 matrix (pre-v3) | \`q1-full\` |
| H4 (F-RQ4) | 2026-07-12 | \`q1-v3-pyribs\` |
| H2 genetic filter | 2026-07-17 | \`q1-v3-genetic-me-uniform\` / \`filter\` |
| Ladder / genetic_me bar | descriptive | \`q1-v3-genetic-me\`, \`q1-anytime-ladder\` |
| Dungeon AUC (appendix) | 2026-07-17 (v4) | \`q1-v4-dungeon-rerun\` |
| Dungeon CPU@32.5k (app.) | supplementary | \`q1-v4-dungeon-full-cpu\` |
| Maze H5 (F-B5) | v5 before readout | \`q1-v5-maze-full\` |
| Maze cost-scaled wall H2 | supplementary | \`q1-v5-maze-cost-h2\` + \`mazes/walltime\` |
| H3-gray (F-RQ3-gray) | path 2026-07-28 | \`q1-v3-h3-gray-zone\` (+ pilot) |
| Matched H1 TOST | reporting / post-hoc | \`q1-stub-uniform-sensitivity\` + \`q1-v3-llm\` |
| H1 LLM quality (accepted elites) | descriptive package A | \`h1-llm-candidate-quality\` |
| H1 QD / cost companions | descriptive | \`h1-qd-cost\` |
| H2 gate diagnostics | descriptive package A | \`h2-gate-diagnostics\` |
| H2 ranking controls | after mixed-2x2 | \`q1-h2-ranking-controls\` |
| Encoding appendix | descriptive | \`q1-cma-encoding-ablation\`, \`*-pbcma\`, \`*-discrete-cma\` |
| Sphere H2 transfer | supplementary | \`q1-v3-sphere-h2\` + \`checkpoints/sphere_h2_mlp.joblib\` |
| CVT sensitivity | descriptive | \`q1-cvt\` |
| Compose-gate / hold-out | 2026-07-28 | \`surrogate/*.json\` |

Amendment kinds (Lock / Extension / Reporting): see manuscript appendix
and \`protocols/EXPERIMENT_PROTOCOL_Q1_v3.md\` §12.
EOF

cat > "${STAGE}/README.md" << EOF
# LifeManifold — Zenodo integrity snapshot

**Git commit:** \`${GIT_FULL}\`  
**Built (UTC):** ${STAMP}

## What this is

Curated snapshot of:

- dated experiment protocol files (v2–v5) + H3 gray-zone task list,
- paper-relevant experiment tiers under \`experiments/\` (main claims **and**
  appendix packages that report numbers: dungeon CPU@32.5k, encoding / pbCMA,
  anytime ladder, CVT, B3 sphere/rastrigin, Sphere H2, maze cost wall),
- surrogate checkpoints (\`nightly_v3_mc_d005\`, \`maze_v1\`, \`sphere_h2_mlp.joblib\`),
- offline compose-gate / hold-out / filter-replay JSON under \`surrogate/\`,
- maze wall-time microbench (\`mazes/walltime/\`),
- warm-start archive (\`map_elites_nightly/baseline/\`),
- manuscript LaTeX sources,
- \`meta/TIMELINE.md\` (freeze ↔ artifact map).

## What this is not

- Not a claim that every historical run was prospectively registered on Zenodo.
- Not the full \`artifacts/\` tree. Still omitted (by design): exploratory prompt
  pilots (\`q1-hints-*\`, weak-LLM), smoke/gate/shadow dungeon–maze tiers,
  and the bulky \`q1-h2-threshold-sensitivity\` matrix (~0.6 GB). Those do not
  drive confirmatory pass/fail; re-run from the pinned GitHub commit if needed.
- Code and schedulers live in the public GitHub repo; pin the same commit SHA.

## Layout notes

- Experiment tier paths mirror the repo: \`experiments/<tier>/\`.
- Checkpoints that live under \`artifacts/surrogate/checkpoints/\` in the repo
  are here under \`checkpoints/\`; \`sphere_h2_mlp.joblib\` is placed beside them
  (in-repo path is \`artifacts/surrogate/sphere_h2_mlp.joblib\`).
- Compose-gate / hold-out JSON mirror \`artifacts/surrogate/\` → \`surrogate/\`.

## Reproduce analysis (high level)

See repository README and \`protocols/\`. Typical entry points:

\`\`\`bash
uv run python scripts/analyze_q1_statistics.py --help
\`\`\`

## License

MIT License — see \`LICENSE\` in this deposit (same as the GitHub repository).
EOF

cat > "${STAGE}/meta/MANIFEST.txt" << EOF
git_full=${GIT_FULL}
git_short=${GIT_SHORT}
built_utc=${STAMP}
EOF

echo "Creating ${ARCHIVE} ..."
mkdir -p "${OUT_DIR}"
tar -C "${OUT_DIR}" -czf "${ARCHIVE}" "$(basename "${STAGE}")"

echo
echo "Staged:  ${STAGE}"
echo "Archive: ${ARCHIVE}"
du -sh "${ARCHIVE}" "${STAGE}"
echo
echo "Next: upload ${ARCHIVE} to https://zenodo.org/uploads (or publish a GitHub Release for the code DOI)."
