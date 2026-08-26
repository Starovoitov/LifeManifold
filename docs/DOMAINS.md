# Parallel domains (maze, dungeon, sphere)

LifeManifold’s **primary product path** is the cellular-automata `WorldSpec` stack:

```bash
python -m worldspace --illuminator mapelites --scheduler …
```

The packages below reuse the same QD ideas (archive, emitters, optional surrogate / acquisition) on **other genotypes and evaluators**. They are **not** wired into `python -m worldspace` and are **not** required to understand the CA pipeline.

| Domain | Code | Entry | Scheduler YAML |
| --- | --- | --- | --- |
| Maze | `worldspace/mazes/` | `scripts/run_maze_qd.py` | `worldspace/specs/maze_scheduler_*.yaml` |
| Dungeon | `worldspace/dungeons/` | `scripts/run_dungeon_qd.py` | `worldspace/specs/dungeon_scheduler_*.yaml` |
| Analytic Sphere | `worldspace/benchmarks/` | `scripts/run_sphere_h2.py` (H2 gate); `scripts/run_sphere_rq1.py` (H1 2×2) | `worldspace/specs/sphere_scheduler_*.yaml` (RQ1) |

**Also out of the main CLI:** pyribs / CMA baselines under `worldspace/illuminators/pyribs_*.py`, `discrete_cma_emitter.py`, `pbcma_emitter.py`, launched via `scripts/run_pyribs_*.py`.

For CA MAP-Elites, surrogate, and dashboard docs, start at [ARCHITECTURE.md](ARCHITECTURE.md).
