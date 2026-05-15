import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from worldspace.generators import (
    GeneticWorldGenerator,
    HybridGALlmWorldGenerator,
    LLMWorldGenerator,
    RandomWorldGenerator,
    call_llm,
    load_llm_generator_yaml,
)
from worldspace.metrics import METRIC_KEYS, METRICS_VECTOR_DIM
from worldspace.pipeline import (
    _fit_dominant_metric_orthogonal_pca,
    _project_dominant_metric_orthogonal,
    stream_world_space_to_jsonl,
)


class TestWorldSpaceMVP(unittest.TestCase):
    def test_stream_jsonl_line_count_and_metrics_bounds(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            generator = RandomWorldGenerator(grid_size=12, steps=20)
            stream_world_space_to_jsonl(
                generator, 5, path, k_clusters=2, echo_stdout=False
            )
            with open(path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(lines), 5)
            row = json.loads(lines[0])
            self.assertIn("world", row)
            self.assertIn("metrics", row)
            self.assertIn("embedding_2d", row)
            self.assertEqual(len(row["embedding_2d"]), 2)
            self.assertIn("embedding_axes", row)
            self.assertIn("x_metric", row["embedding_axes"])
            self.assertIn("cluster_id", row)
            st = row["metrics"]["stability"]
            self.assertGreaterEqual(st, 0.0)
            self.assertLessEqual(st, 1.0)
            self.assertIn("interestingness", row["metrics"])
        finally:
            os.unlink(path)

    def test_metrics_trace_only_no_stdout_without_echo(self):
        """No main ``--output`` and ``echo_stdout=False`` keeps stdout quiet; trace still fills."""
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "t.jsonl"
            gen = RandomWorldGenerator(grid_size=4, steps=4)
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                stream_world_space_to_jsonl(
                    gen,
                    2,
                    None,
                    k_clusters=2,
                    echo_stdout=False,
                    metrics_trace_path=trace_path,
                )
            self.assertEqual(buf.getvalue(), "")
            tlines = [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(tlines), 2)

    def test_stream_zero_worlds_metrics_trace_not_opened(self):
        """``n_worlds`` <= 0 returns before trace file open — no handle leak (regression)."""
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "trace.jsonl"
            stream_world_space_to_jsonl(
                RandomWorldGenerator(grid_size=4, steps=4),
                0,
                None,
                metrics_trace_path=trace_path,
            )
            self.assertFalse(trace_path.exists())

    def test_metrics_trace_file_closed_when_ca_trace_open_fails(self):
        """If the CA trace file cannot be opened, the metrics trace handle is still closed."""
        d = tempfile.mkdtemp()
        try:
            p1 = Path(d) / "m.jsonl"
            p2 = Path(d) / "c.jsonl"
            first_handle_closed = {"done": False}
            real_open = Path.open

            def open_wrapper(self, *args, **kwargs):
                if self.resolve() == p2.resolve():
                    raise PermissionError("simulated ca trace open failure")
                fh = real_open(self, *args, **kwargs)
                if self.resolve() == p1.resolve():
                    orig_close = fh.close

                    def close_and_mark() -> None:
                        first_handle_closed["done"] = True
                        orig_close()

                    fh.close = close_and_mark  # type: ignore[method-assign]
                return fh

            with patch.object(Path, "open", open_wrapper):
                with self.assertRaises(PermissionError):
                    stream_world_space_to_jsonl(
                        RandomWorldGenerator(grid_size=4, steps=2),
                        1,
                        None,
                        metrics_trace_path=p1,
                        ca_step_trace_path=p2,
                    )
            self.assertTrue(
                first_handle_closed["done"],
                "metrics trace handle must be closed in finally when ca trace open fails",
            )
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_run_world_ca_step_trace_one_line_per_timestep(self):
        from dataclasses import replace
        import io

        from worldspace.simulator import run_world

        w = RandomWorldGenerator(grid_size=6, steps=6).generate(1)[0]
        w = replace(w, steps=5)
        buf = io.StringIO()
        run_world(w, ca_step_trace_file=buf, ca_step_trace_yield_index=3)
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 5)
        r0 = json.loads(lines[0])
        self.assertEqual(r0["yield_index"], 3)
        self.assertEqual(r0["ca_step"], 0)
        self.assertIn("metrics", r0)

    def test_stream_ca_step_trace_line_count(self):
        fd_o, out_path = tempfile.mkstemp(suffix=".jsonl")
        fd_c, ca_path = tempfile.mkstemp(suffix=".ca.jsonl")
        os.close(fd_o)
        os.close(fd_c)
        try:
            gen = RandomWorldGenerator(grid_size=6, steps=4)
            stream_world_space_to_jsonl(
                gen,
                2,
                out_path,
                k_clusters=2,
                echo_stdout=False,
                ca_step_trace_path=ca_path,
            )
            with open(ca_path, encoding="utf-8") as f:
                ca_lines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(ca_lines), 8)
        finally:
            os.unlink(out_path)
            os.unlink(ca_path)

    def test_stream_jsonl_llm_pipeline_calls_call_llm_once_per_step(self):
        """Pass-2 must reuse cached worlds; no second full ``iter_worlds`` (halves LLM traffic)."""
        response = json.dumps(
            {
                "birth": [3, 4],
                "survival": [2, 3, 4],
                "noise": 0.07,
                "resource_regen": 0.12,
                "predation": 0.31,
                "reasoning": "ok",
            }
        )
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with patch("worldspace.generators.call_llm", return_value=response) as m_llm:
                generator = LLMWorldGenerator(grid_size=10, steps=10, seed=1)
                stream_world_space_to_jsonl(
                    generator, 4, path, k_clusters=2, echo_stdout=False
                )
            self.assertEqual(m_llm.call_count, 3)
            with open(path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(lines), 4)
        finally:
            os.unlink(path)

    def test_metrics_trace_jsonl_pipeline_any_generator(self):
        fd_o, out_path = tempfile.mkstemp(suffix=".jsonl")
        fd_t, trace_path = tempfile.mkstemp(suffix=".trace.jsonl")
        os.close(fd_o)
        os.close(fd_t)
        try:
            gen = RandomWorldGenerator(grid_size=8, steps=8)
            stream_world_space_to_jsonl(
                gen,
                3,
                out_path,
                k_clusters=2,
                echo_stdout=False,
                metrics_trace_path=trace_path,
            )
            with open(trace_path, encoding="utf-8") as f:
                tlines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(tlines), 3)
            for i, ln in enumerate(tlines):
                row = json.loads(ln)
                self.assertEqual(row["yield_index"], i)
                self.assertIn("world", row)
                self.assertIn("birth", row["world"])
                self.assertIn("metrics", row)
                self.assertIn("interestingness", row["metrics"])
        finally:
            os.unlink(out_path)
            os.unlink(trace_path)

    def test_stream_llm_metrics_trace_line_count_matches_worlds(self):
        response = json.dumps(
            {
                "birth": [3, 4],
                "survival": [2, 3, 4],
                "noise": 0.07,
                "resource_regen": 0.12,
                "predation": 0.31,
                "reasoning": "ok",
            }
        )
        fd_o, out_path = tempfile.mkstemp(suffix=".jsonl")
        fd_t, trace_path = tempfile.mkstemp(suffix=".trace.jsonl")
        os.close(fd_o)
        os.close(fd_t)
        try:
            with patch("worldspace.generators.call_llm", return_value=response):
                gen = LLMWorldGenerator(grid_size=8, steps=8, seed=0)
                stream_world_space_to_jsonl(
                    gen,
                    4,
                    out_path,
                    k_clusters=2,
                    echo_stdout=False,
                    metrics_trace_path=trace_path,
                )
            with open(trace_path, encoding="utf-8") as f:
                tlines = [ln for ln in f if ln.strip()]
            self.assertEqual(len(tlines), 4)
        finally:
            os.unlink(out_path)
            os.unlink(trace_path)

    def test_dominant_metric_embedding_axes(self):
        rng = np.random.default_rng(0)
        n, d = 50, METRICS_VECTOR_DIM
        x = rng.standard_normal((n, d))
        x[:, 4] *= 30.0
        j = 4
        mean_exp = x.mean(axis=0)
        mean, j_fit, axis_name, pca = _fit_dominant_metric_orthogonal_pca(x)
        self.assertEqual(j_fit, j)
        self.assertEqual(axis_name, METRIC_KEYS[j])
        np.testing.assert_allclose(mean, mean_exp)
        self.assertIsNotNone(pca)
        np.testing.assert_allclose(pca.mean_, np.delete(mean_exp, j), rtol=0, atol=1e-9)
        vec = x[3]
        emb = _project_dominant_metric_orthogonal(vec, mean, j, pca)
        self.assertAlmostEqual(emb[0], vec[j] - mean[j])
        self.assertFalse(np.isnan(emb[1]))

    def test_genetic_generator_bounds_and_count(self):
        generator = GeneticWorldGenerator(
            grid_size=10,
            steps=10,
            population_size=6,
            elite_count=2,
            mutation_scale=0.03,
            seed=7,
        )
        worlds = generator.generate(4)
        self.assertEqual(len(worlds), 4)
        for w in worlds:
            self.assertGreaterEqual(w.noise, 0.0)
            self.assertLessEqual(w.noise, 0.2)
            self.assertGreaterEqual(w.resource_regen, 0.0)
            self.assertLessEqual(w.resource_regen, 0.5)
            self.assertGreaterEqual(w.predation, 0.0)
            self.assertLessEqual(w.predation, 1.0)
            self.assertGreaterEqual(len(w.birth), 1)
            self.assertGreaterEqual(len(w.survival), 1)
            self.assertTrue(all(0 <= v <= 8 for v in w.birth))
            self.assertTrue(all(0 <= v <= 8 for v in w.survival))

    def test_genetic_generator_preserves_diversity(self):
        generator = GeneticWorldGenerator(
            grid_size=10,
            steps=10,
            population_size=8,
            elite_count=3,
            mutation_scale=0.02,
            seed=0,
        )
        worlds = generator.generate(20)
        self.assertEqual(len(worlds), 20)
        signatures = {
            (
                tuple(w.birth),
                tuple(w.survival),
                round(w.noise, 4),
                round(w.resource_regen, 4),
                round(w.predation, 4),
            )
            for w in worlds
        }
        self.assertGreaterEqual(len(signatures), 8)

    def test_genetic_solution_seed_depends_on_generation(self):
        generator = GeneticWorldGenerator(
            grid_size=10,
            steps=10,
            population_size=6,
            elite_count=2,
            mutation_scale=0.03,
            seed=11,
        )
        world = RandomWorldGenerator(grid_size=10, steps=10).generate(1)[0]
        solution = generator._encode_world(world)
        seed_g0 = generator._solution_seed(solution, 0)
        seed_g1 = generator._solution_seed(solution, 1)
        seed_g5 = generator._solution_seed(solution, 5)
        self.assertNotEqual(seed_g0, seed_g1)
        self.assertNotEqual(seed_g1, seed_g5)

    def test_llm_generator_iterative_updates_from_llm_response(self):
        response = json.dumps(
            {
                "birth": [3, 4],
                "survival": [2, 3, 4],
                "noise": 0.07,
                "resource_regen": 0.12,
                "predation": 0.31,
                "reasoning": "increase structured growth",
            }
        )
        with patch("worldspace.generators.call_llm", return_value=response):
            generator = LLMWorldGenerator(
                grid_size=10,
                steps=10,
                seed=5,
            )
            worlds = generator.generate(3)
        self.assertEqual(len(worlds), 3)
        self.assertEqual(worlds[1].birth, [3, 4])
        self.assertEqual(worlds[1].survival, [2, 3, 4])
        self.assertAlmostEqual(worlds[1].noise, 0.07)
        self.assertAlmostEqual(worlds[1].resource_regen, 0.12)
        self.assertAlmostEqual(worlds[1].predation, 0.31)

    def test_hybrid_generator_emits_world_per_generation(self):
        response = json.dumps(
            {
                "birth": [3, 4],
                "survival": [2, 3, 4],
                "noise": 0.06,
                "resource_regen": 0.11,
                "predation": 0.25,
            }
        )
        with patch("worldspace.generators.call_llm", return_value=response):
            generator = HybridGALlmWorldGenerator(grid_size=10, steps=10, seed=2)
            worlds = generator.generate(4)
        self.assertEqual(len(worlds), 4)
        for w in worlds:
            self.assertGreaterEqual(w.noise, 0.0)
            self.assertLessEqual(w.noise, 0.2)
            self.assertGreaterEqual(w.resource_regen, 0.0)
            self.assertLessEqual(w.resource_regen, 0.5)
            self.assertGreaterEqual(w.predation, 0.0)
            self.assertLessEqual(w.predation, 1.0)

    def test_hybrid_initial_population_depends_on_seed(self):
        with patch("worldspace.generators.call_llm", return_value="{}"):
            g0 = HybridGALlmWorldGenerator(grid_size=10, steps=10, seed=0)
            g1 = HybridGALlmWorldGenerator(grid_size=10, steps=10, seed=101)
            w0 = g0.generate(1)[0]
            w1 = g1.generate(1)[0]
        sig0 = (
            tuple(w0.birth),
            tuple(w0.survival),
            round(w0.noise, 6),
            round(w0.resource_regen, 6),
            round(w0.predation, 6),
        )
        sig1 = (
            tuple(w1.birth),
            tuple(w1.survival),
            round(w1.noise, 6),
            round(w1.resource_regen, 6),
            round(w1.predation, 6),
        )
        self.assertNotEqual(sig0, sig1)

    def test_bundled_llm_spec_defines_qwen_provider(self):
        spec_path = (
            Path(__file__).resolve().parent.parent
            / "worldspace"
            / "specs"
            / "llm_world_generator.yaml"
        )
        cfg = load_llm_generator_yaml(spec_path)
        llm = cfg["llm"]
        qwen = llm["providers"]["qwen"]
        self.assertEqual(qwen.get("provider"), "openai")
        self.assertEqual(qwen.get("model"), "qwen-plus")
        self.assertEqual(qwen.get("api_key_env"), "QWEN_API_KEY")
        self.assertIn("dashscope-intl.aliyuncs.com", str(qwen.get("api_base", "")))

    def test_call_llm_qwen_from_bundled_spec_posts_remote_request(self):
        spec_path = (
            Path(__file__).resolve().parent.parent
            / "worldspace"
            / "specs"
            / "llm_world_generator.yaml"
        )
        providers = load_llm_generator_yaml(spec_path)["llm"]["providers"]
        llm_body = {"choices": [{"message": {"content": "{}"}}]}
        fake_cm = MagicMock()
        fake_cm.__enter__.return_value.read.return_value = json.dumps(
            llm_body, ensure_ascii=True
        ).encode("utf-8")

        with patch.dict(os.environ, {"QWEN_API_KEY": "test-qwen-key"}):
            with patch("worldspace.generators.request.urlopen", return_value=fake_cm) as m_open:
                out = call_llm(
                    mode="remote",
                    provider_name="qwen",
                    providers=providers,
                    prompt="ping",
                    temperature=0.2,
                    max_tokens=350,
                )

        self.assertEqual(out, "{}")
        m_open.assert_called_once()
        req = m_open.call_args[0][0]
        hdrs = dict(req.header_items())
        self.assertEqual(hdrs.get("Authorization"), "Bearer test-qwen-key")
        self.assertEqual(hdrs.get("Content-type"), "application/json")
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen-plus")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 350)


if __name__ == "__main__":
    unittest.main()
