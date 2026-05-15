import json
import os
import tempfile
import unittest
from pathlib import Path

from worldspace.visualizer.plotting import (
    load_ca_step_trace_jsonl,
    plot_ca_step_metrics_timeseries,
    plot_ca_step_pca_trajectories,
    plot_ca_step_umap_trajectories,
    summarize_ca_step_trace_by_world,
)
from worldspace.visualizer.visualizer import main as visualizer_main


def _sample_row(yield_index: int, ca_step: int) -> dict:
    return {
        "yield_index": yield_index,
        "ca_step": ca_step,
        "metrics": {
            "entropy": 0.5 + 0.01 * ca_step,
            "stability": 0.4,
            "average_lifespan": 1.0,
            "density_mean": 0.2 + 0.001 * yield_index,
            "oscillation_score": 0.1,
            "diversity": 0.3,
            "interestingness": 0.6 + 0.02 * ca_step + 0.01 * yield_index,
        },
    }


class TestVisualizer(unittest.TestCase):
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
            self.assertIn("interestingness", df.columns)
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
                metric_names=["interestingness", "entropy"],
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

    def test_embedding_subcommand_writes_png(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        fd_p, png = tempfile.mkstemp(suffix=".png")
        os.close(fd_j)
        os.close(fd_p)
        try:
            row = {
                "world": {"seed": 1, "birth": [3], "survival": [2, 3]},
                "metrics": {"interestingness": 0.5},
                "embedding_2d": [0.1, 0.2],
                "embedding_axes": {"x_metric": "entropy"},
                "cluster_id": 0,
            }
            Path(jsonl).write_text(
                json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8"
            )
            visualizer_main(["embedding", jsonl, "--plot", png, "--title", "unit test"])
            self.assertGreater(Path(png).stat().st_size, 100)
        finally:
            os.unlink(jsonl)
            os.unlink(png)

    def test_ca_trace_subcommand_writes_pngs(self):
        fd_j, jsonl = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd_j)
        try:
            rows = [json.dumps(_sample_row(0, t), ensure_ascii=True) for t in range(4)]
            Path(jsonl).write_text("\n".join(rows) + "\n", encoding="utf-8")
            with tempfile.TemporaryDirectory() as d:
                visualizer_main(["ca-trace", jsonl, "--output-dir", d])
                self.assertTrue((Path(d) / "ca_timeseries.png").is_file())
                self.assertTrue((Path(d) / "ca_pca_trajectories.png").is_file())
                self.assertTrue((Path(d) / "ca_umap_trajectories.png").is_file())
        finally:
            os.unlink(jsonl)
