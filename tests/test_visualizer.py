import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from worldspace.visualizer.plotting import (
    load_ca_step_trace_jsonl,
    plot_ca_step_metrics_timeseries,
    plot_ca_step_pca_trajectories,
    plot_ca_step_umap_trajectories,
    plot_world_metrics_pca_scatter_from_jsonl,
    plot_world_metrics_umap_scatter_from_jsonl,
    plot_dominant_metric_delta_scatter_from_jsonl,
    summarize_ca_step_trace_by_world,
)
from worldspace.visualizer.visualizer import main as visualizer_main


def _sample_row(yield_index: int, ca_step: int) -> dict:
    return {
        "yield_index": yield_index,
        "ca_step": ca_step,
        "metrics": {
            **{
                k: v
                for k, v in _METRICS_7.items()
                if k
                not in (
                    "entropy",
                    "density_mean",
                    "mo_eoc_indicator",
                )
            },
            "entropy": 0.5 + 0.01 * ca_step,
            "density_mean": 0.2 + 0.001 * yield_index,
            "mo_eoc_indicator": 0.6 + 0.02 * ca_step + 0.01 * yield_index,
        },
    }


_METRICS_7 = {
    "entropy": 0.5,
    "stability": 0.4,
    "average_lifespan": 0.3,
    "density_mean": 0.2,
    "oscillation_score": 0.35,
    "diversity": 0.25,
    "mo_eoc_indicator": 0.55,
    "topology_interface_index": 0.4,
    "topology_window_heterogeneity": 0.35,
    "compressibility_score": 0.2,
    "ecology_state_entropy_norm": 0.45,
    "ecology_resource_adjacency": 0.3,
}


class TestVisualizer(unittest.TestCase):
    def test_metrics_pca_scatter_from_jsonl(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        fd_p, png = tempfile.mkstemp(suffix=".png")
        os.close(fd_j)
        os.close(fd_p)
        try:
            rows = [
                {"metrics": dict(_METRICS_7), "cluster_id": 0},
                {
                    "metrics": {**_METRICS_7, "mo_eoc_indicator": 0.9},
                    "cluster_id": 1,
                },
            ]
            Path(jsonl).write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n",
                encoding="utf-8",
            )
            plot_world_metrics_pca_scatter_from_jsonl(jsonl, png)
            self.assertGreater(Path(png).stat().st_size, 100)
        finally:
            os.unlink(jsonl)
            os.unlink(png)

    def test_pca_scatter_runs_kmeans_without_cluster_id_column(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        fd_p, png = tempfile.mkstemp(suffix=".png")
        os.close(fd_j)
        os.close(fd_p)
        try:
            rows = [
                {"metrics": dict(_METRICS_7)},
                {"metrics": {**_METRICS_7, "mo_eoc_indicator": 0.95}},
                {"metrics": {**_METRICS_7, "entropy": 0.9, "density_mean": 0.5}},
                {"metrics": {**_METRICS_7, "stability": 0.1, "diversity": 0.9}},
            ]
            Path(jsonl).write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n",
                encoding="utf-8",
            )
            plot_world_metrics_pca_scatter_from_jsonl(jsonl, png, k_clusters=2)
            self.assertGreater(Path(png).stat().st_size, 100)
        finally:
            os.unlink(jsonl)
            os.unlink(png)

    def test_metrics_umap_scatter_from_jsonl(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        fd_p, png = tempfile.mkstemp(suffix=".png")
        os.close(fd_j)
        os.close(fd_p)
        try:
            rows = [
                {"metrics": dict(_METRICS_7), "cluster_id": 0},
                {
                    "metrics": {**_METRICS_7, "mo_eoc_indicator": 0.9},
                    "cluster_id": 1,
                },
                {
                    "metrics": {**_METRICS_7, "entropy": 0.88},
                    "cluster_id": 0,
                },
            ]
            Path(jsonl).write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n",
                encoding="utf-8",
            )
            plot_world_metrics_umap_scatter_from_jsonl(jsonl, png)
            self.assertGreater(Path(png).stat().st_size, 100)
        finally:
            os.unlink(jsonl)
            os.unlink(png)

    def test_load_and_summarize_ca_step_trace(self):
        fd, p = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            lines = [
                json.dumps(_sample_row(0, 0), ensure_ascii=True),
                json.dumps(_sample_row(0, 1), ensure_ascii=True),
                json.dumps(_sample_row(1, 0), ensure_ascii=True),
            ]
            Path(p).write_text("\n".join(lines) + "\n", encoding="utf-8")
            df = load_ca_step_trace_jsonl(p)
            self.assertEqual(len(df), 3)
            self.assertIn("mo_eoc_indicator", df.columns)
            summ = summarize_ca_step_trace_by_world(df)
            self.assertGreater(len(summ), 0)
        finally:
            os.unlink(p)

    def test_plots_write_png(self):
        fd, src = tempfile.mkstemp(suffix=".jsonl")
        fd_ts, out_ts = tempfile.mkstemp(suffix=".png")
        fd_pc, out_pc = tempfile.mkstemp(suffix=".png")
        fd_um, out_um = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        os.close(fd_ts)
        os.close(fd_pc)
        os.close(fd_um)
        try:
            rows = []
            for yi in (0, 1):
                for t in range(5):
                    rows.append(json.dumps(_sample_row(yi, t), ensure_ascii=True))
            Path(src).write_text("\n".join(rows) + "\n", encoding="utf-8")
            df = load_ca_step_trace_jsonl(src)
            plot_ca_step_metrics_timeseries(
                df,
                [0, 1],
                out_ts,
                metric_names=["mo_eoc_indicator", "entropy"],
            )
            plot_ca_step_pca_trajectories(df, [0, 1], out_pc)
            plot_ca_step_umap_trajectories(df, [0, 1], out_um)
            self.assertGreater(Path(out_ts).stat().st_size, 100)
            self.assertGreater(Path(out_pc).stat().st_size, 100)
            self.assertGreater(Path(out_um).stat().st_size, 100)
        finally:
            os.unlink(src)
            os.unlink(out_ts)
            os.unlink(out_pc)
            os.unlink(out_um)

    def test_dominant_metric_delta_scatter_from_metrics_only(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        fd_p, png = tempfile.mkstemp(suffix=".png")
        os.close(fd_j)
        os.close(fd_p)
        try:
            rows = [
                {"metrics": dict(_METRICS_7)},
                {"metrics": {**_METRICS_7, "mo_eoc_indicator": 0.95}},
                {"metrics": {**_METRICS_7, "entropy": 0.9}},
            ]
            Path(jsonl).write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n",
                encoding="utf-8",
            )
            plot_dominant_metric_delta_scatter_from_jsonl(jsonl, png, k_clusters=2)
            self.assertGreater(Path(png).stat().st_size, 100)
        finally:
            os.unlink(jsonl)
            os.unlink(png)

    def test_dominant_metric_delta_reads_legacy_jsonl_columns(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        fd_p, png = tempfile.mkstemp(suffix=".png")
        os.close(fd_j)
        os.close(fd_p)
        try:
            rows = [
                {
                    "embedding_2d": [0.1, 0.2],
                    "embedding_axes": {"x_metric": "entropy"},
                    "cluster_id": 0,
                },
                {
                    "embedding_2d": [0.3, 0.1],
                    "embedding_axes": {"x_metric": "entropy"},
                    "cluster_id": 1,
                },
            ]
            Path(jsonl).write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n",
                encoding="utf-8",
            )
            plot_dominant_metric_delta_scatter_from_jsonl(jsonl, png)
            self.assertGreater(Path(png).stat().st_size, 100)
        finally:
            os.unlink(jsonl)
            os.unlink(png)

    def test_metrics_jsonl_writes_dominant_metric_delta_pca_and_umap(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd_j)
        try:
            rows = [
                {"world": {"seed": 1}, "metrics": dict(_METRICS_7), "cluster_id": 0},
                {
                    "world": {"seed": 2},
                    "metrics": {**_METRICS_7, "mo_eoc_indicator": 0.9},
                    "cluster_id": 1,
                },
                {
                    "world": {"seed": 3},
                    "metrics": {**_METRICS_7, "entropy": 0.88},
                    "cluster_id": 0,
                },
            ]
            Path(jsonl).write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n",
                encoding="utf-8",
            )
            with tempfile.TemporaryDirectory() as d:
                visualizer_main(
                    [
                        "--output-dir",
                        d,
                        "--metrics-jsonl",
                        jsonl,
                    ]
                )
                for name in (
                    "dominant_metric_delta.png",
                    "dominant_metric_delta_norm.png",
                    "pca.png",
                    "pca_norm.png",
                    "umap.png",
                    "umap_norm.png",
                ):
                    self.assertGreater(
                        (Path(d) / name).stat().st_size,
                        100,
                        msg=name,
                    )
        finally:
            os.unlink(jsonl)

    def test_metrics_umap_scatter_differs_from_ca_trajectories(self):
        fd_m, metrics = tempfile.mkstemp(suffix=".jsonl")
        fd_ca, ca = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd_m)
        os.close(fd_ca)
        try:
            mrows = [
                {"metrics": dict(_METRICS_7), "cluster_id": i % 2} for i in range(5)
            ]
            Path(metrics).write_text(
                "\n".join(json.dumps(r, ensure_ascii=True) for r in mrows) + "\n",
                encoding="utf-8",
            )
            crows = [
                json.dumps(_sample_row(0, t), ensure_ascii=True) for t in range(6)
            ] + [json.dumps(_sample_row(10, t), ensure_ascii=True) for t in range(6)]
            Path(ca).write_text("\n".join(crows) + "\n", encoding="utf-8")
            with tempfile.TemporaryDirectory() as d:
                visualizer_main(
                    [
                        "--output-dir",
                        d,
                        "--metrics-jsonl",
                        metrics,
                        "--ca-step-jsonl",
                        ca,
                        "--ca-trace-worlds",
                        "0,10",
                    ]
                )
                umap_scatter = Path(d) / "umap.png"
                umap_traj = Path(d) / "umap_trajectories.png"
                self.assertTrue(umap_scatter.is_file())
                self.assertTrue(umap_traj.is_file())
                self.assertNotEqual(
                    umap_scatter.read_bytes(),
                    umap_traj.read_bytes(),
                    "umap.png must be per-world scatter, not CA trajectories",
                )
        finally:
            os.unlink(metrics)
            os.unlink(ca)

    def test_bad_metrics_jsonl_still_runs_ca_plots(self):
        fd_bad, bad = tempfile.mkstemp(suffix=".jsonl")
        fd_ca, ca = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd_bad)
        os.close(fd_ca)
        try:
            Path(bad).write_text('{"only": "metadata"}\n', encoding="utf-8")
            rows = [json.dumps(_sample_row(0, t), ensure_ascii=True) for t in range(4)]
            Path(ca).write_text("\n".join(rows) + "\n", encoding="utf-8")
            with tempfile.TemporaryDirectory() as d:
                rc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "worldspace.visualizer",
                        "--output-dir",
                        d,
                        "--metrics-jsonl",
                        bad,
                        "--ca-step-jsonl",
                        ca,
                    ],
                    cwd=Path(__file__).resolve().parent.parent,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(rc.returncode, 0, msg=rc.stderr + rc.stdout)
                self.assertTrue((Path(d) / "ca_step_timeseries.png").is_file())
                self.assertTrue((Path(d) / "pca_trajectories.png").is_file())
                self.assertTrue((Path(d) / "pca_trajectories_norm.png").is_file())
                self.assertTrue((Path(d) / "umap_trajectories.png").is_file())
                self.assertTrue((Path(d) / "umap_trajectories_norm.png").is_file())
        finally:
            os.unlink(bad)
            os.unlink(ca)

    def test_ca_step_jsonl_writes_standard_pngs(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd_j)
        try:
            rows = [json.dumps(_sample_row(0, t), ensure_ascii=True) for t in range(4)]
            Path(jsonl).write_text("\n".join(rows) + "\n", encoding="utf-8")
            with tempfile.TemporaryDirectory() as d:
                visualizer_main(
                    [
                        "--output-dir",
                        d,
                        "--ca-step-jsonl",
                        jsonl,
                    ]
                )
                self.assertTrue((Path(d) / "ca_step_timeseries.png").is_file())
                self.assertTrue((Path(d) / "pca_trajectories.png").is_file())
                self.assertTrue((Path(d) / "pca_trajectories_norm.png").is_file())
                self.assertTrue((Path(d) / "umap_trajectories.png").is_file())
                self.assertTrue((Path(d) / "umap_trajectories_norm.png").is_file())
        finally:
            os.unlink(jsonl)

    def test_output_dir_only_writes_dashboard_and_galleries(self):
        from worldspace.visualizer.diagnostics import GALLERY_NEW_METRICS

        with tempfile.TemporaryDirectory() as d:
            visualizer_main(["--output-dir", d])
            dash = Path(d) / "diagnostic_dashboard.png"
            self.assertTrue(dash.is_file())
            self.assertGreater(dash.stat().st_size, 5000)
            for key in GALLERY_NEW_METRICS:
                p = Path(d) / f"gallery_{key}.png"
                self.assertTrue(p.is_file(), msg=key)
                self.assertGreater(p.stat().st_size, 2000, msg=key)

    def test_world_spec_json_writes_diagnostic_dashboard(self):
        from worldspace.generators import RandomWorldGenerator

        w = RandomWorldGenerator(grid_size=14, steps=16)._make_world(seed=42)
        fd, jpath = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            Path(jpath).write_text(
                json.dumps(w.to_json_dict(), ensure_ascii=True),
                encoding="utf-8",
            )
            with tempfile.TemporaryDirectory() as d:
                visualizer_main(
                    [
                        "--output-dir",
                        d,
                        "--world-spec-json",
                        jpath,
                    ]
                )
                dash = Path(d) / "diagnostic_dashboard.png"
                self.assertTrue(dash.is_file())
                self.assertGreater(dash.stat().st_size, 5000)
        finally:
            os.unlink(jpath)

    def test_metric_tertile_gallery_smoke(self):
        from worldspace.visualizer.diagnostics import plot_metric_tertile_gallery

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "gallery_compressibility_score.png"
            plot_metric_tertile_gallery(
                "compressibility_score",
                out,
                scan_seeds=35,
                grid_size=14,
                steps=18,
                seed_offset=0,
            )
            self.assertGreater(out.stat().st_size, 3000)
