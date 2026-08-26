"""Sphere RQ1 / H1 runner, policy, parser, and mock-LLM smoke."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ribs.archives import GridArchive

from worldspace.benchmarks.qd_sphere import (
    DEFAULT_ARCHIVE_DIMS,
    DEFAULT_SOLUTION_DIM,
    archive_ranges,
    clip_solution,
    linear_projection_measures,
    sphere_objective,
)
from worldspace.benchmarks.sphere_llm import (
    MockSphereLlmEmitter,
    SphereLlmEmitter,
    apply_sphere_deltas,
    parse_sphere_deltas,
)
from worldspace.benchmarks.sphere_rq1 import (
    SphereSchedulerConfig,
    emit_genetic,
    load_sphere_scheduler,
    run_sphere_qd,
    save_sphere_h1_surrogate,
    select_target_cell,
    train_sphere_h1_surrogate,
)


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "worldspace/specs"


class _Caller:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        return self.response


class _ErrorCaller(_Caller):
    def __init__(self, message: str) -> None:
        super().__init__("")
        self.message = message

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        raise RuntimeError(self.message)


class _SequenceCaller(_Caller):
    def __init__(self, responses: list[object]) -> None:
        super().__init__("")
        self.responses = iter(responses)

    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return str(item)


class _EnvelopeCaller(_Caller):
    def __call__(self, **kwargs: object) -> str:
        self.prompts.append(str(kwargs["prompt"]))
        raise AttributeError("'str' object has no attribute 'get'")


def _deltas_json() -> str:
    deltas = [0.0] * 20
    for i in range(4):
        deltas[i] = 0.2
    return json.dumps({"deltas": deltas})


def _seeded_archive() -> GridArchive:
    archive = GridArchive(
        solution_dim=DEFAULT_SOLUTION_DIM,
        dims=(10, 10),
        ranges=archive_ranges(DEFAULT_SOLUTION_DIM),
        seed=0,
        learning_rate=1.0,
    )
    rng = np.random.default_rng(0)
    for _ in range(30):
        child = clip_solution(rng.uniform(-5.12, 5.12, size=DEFAULT_SOLUTION_DIM))
        archive.add(
            child[np.newaxis, :],
            np.asarray([float(sphere_objective(child))]),
            linear_projection_measures(child)[np.newaxis, :],
        )
    return archive


class TestSphereRq1Schedulers(unittest.TestCase):
    def test_locked_schedulers_load(self) -> None:
        genetic = load_sphere_scheduler(SPECS / "sphere_scheduler_genetic.yaml")
        minfit = load_sphere_scheduler(SPECS / "sphere_scheduler_genetic_minfit.yaml")
        self.assertEqual(genetic.target_selection, "uniform_frontier")
        self.assertEqual(minfit.target_selection, "min_fitness_frontier")
        self.assertEqual(genetic.emitters.count("genetic"), 30)
        self.assertEqual(minfit.emitters.count("genetic"), 30)
        self.assertEqual(genetic.iterations, 100)
        factorial = {
            "llm_stub_minfit": ("stub", "min_fitness_frontier", None),
            "llm_stub_uniform": ("stub", "uniform_frontier", None),
            "llm_hints_minfit": (
                "hints",
                "min_fitness_frontier",
                "artifacts/surrogate/sphere_h1_mlp.joblib",
            ),
            "llm_hints_uniform": (
                "hints",
                "uniform_frontier",
                "artifacts/surrogate/sphere_h1_mlp.joblib",
            ),
        }
        for condition, (mode, policy, checkpoint) in factorial.items():
            with self.subTest(condition=condition):
                config = load_sphere_scheduler(
                    SPECS / f"sphere_scheduler_{condition}.yaml"
                )
                self.assertEqual(config.llm_prompt_mode, mode)
                self.assertEqual(config.target_selection, policy)
                self.assertEqual(config.surrogate_checkpoint, checkpoint)
                self.assertEqual(config.emitters.count("llm"), 30)
                self.assertEqual(config.archive_dims, DEFAULT_ARCHIVE_DIMS)


class TestSphereTargetPolicy(unittest.TestCase):
    def test_minfit_picks_lowest_frontier_elite(self) -> None:
        archive = _seeded_archive()
        rng = np.random.default_rng(1)
        uniform = [
            select_target_cell(archive, rng, target_selection="uniform_frontier")
            for _ in range(20)
        ]
        minfit = [
            select_target_cell(archive, rng, target_selection="min_fitness_frontier")
            for _ in range(20)
        ]
        self.assertTrue(all(t.parent is not None for t in uniform + minfit))
        min_ids = {t.cell_id for t in minfit}
        uni_ids = {t.cell_id for t in uniform}
        self.assertEqual(len(min_ids), 1)
        self.assertGreater(len(uni_ids), 1)
        self.assertLess(
            minfit[0].parent.objective,  # type: ignore[union-attr]
            max(t.parent.objective for t in uniform),  # type: ignore[union-attr]
        )


class TestSphereLlmParser(unittest.TestCase):
    def test_strict_parser_accepts_fenced_json(self) -> None:
        payload = _deltas_json()
        deltas = parse_sphere_deltas(f"```json\n{payload}\n```")
        self.assertEqual(deltas.shape, (20,))
        parent = np.zeros(20)
        child = apply_sphere_deltas(parent, deltas)
        self.assertGreater(float(np.linalg.norm(child - parent)), 0.1)

    def test_rejects_wrong_length_and_too_small(self) -> None:
        with self.assertRaises(ValueError):
            parse_sphere_deltas(json.dumps({"deltas": [0.2] * 19}))
        with self.assertRaises(ValueError):
            parse_sphere_deltas(json.dumps({"deltas": [0.01] * 20}))
        with self.assertRaises(ValueError):
            apply_sphere_deltas(np.zeros(20), parse_sphere_deltas(_deltas_json()) * 0.0)

    def test_accepts_extra_keys_when_deltas_are_valid(self) -> None:
        payload = json.loads(_deltas_json())
        payload["explanation"] = "nudge toward the target cell"
        deltas = parse_sphere_deltas(json.dumps(payload))
        self.assertEqual(deltas.shape, (20,))

    def test_rejects_empty_or_non_string_content(self) -> None:
        with self.assertRaises(ValueError):
            parse_sphere_deltas("")
        with self.assertRaises(ValueError):
            parse_sphere_deltas(None)  # type: ignore[arg-type]

    def test_stub_constants_in_prompt(self) -> None:
        archive = _seeded_archive()
        target = select_target_cell(
            archive, np.random.default_rng(2), target_selection="uniform_frontier"
        )
        caller = _Caller(_deltas_json())
        emitter = SphereLlmEmitter(
            prompt_mode="stub",
            call_llm_text=caller,  # type: ignore[arg-type]
        )
        emitted = emitter.emit(
            target=target,
            rng=np.random.default_rng(3),
            prediction=None,
        )
        self.assertEqual(emitted.emitter_type, "llm")
        self.assertIn("predicted fitness: 0.5000", caller.prompts[0])
        self.assertIn("uncertainty: 1.0000", caller.prompts[0])

    def test_hints_use_live_prediction(self) -> None:
        archive = _seeded_archive()
        target = select_target_cell(
            archive, np.random.default_rng(4), target_selection="uniform_frontier"
        )
        from worldspace.benchmarks.sphere_rq1 import SpherePrediction

        caller = _Caller(_deltas_json())
        emitter = SphereLlmEmitter(
            prompt_mode="hints",
            call_llm_text=caller,  # type: ignore[arg-type]
        )
        emitter.emit(
            target=target,
            rng=np.random.default_rng(5),
            prediction=SpherePrediction(0.31, 0.07),
        )
        self.assertIn("predicted fitness: 0.3100", caller.prompts[0])
        self.assertIn("uncertainty: 0.0700", caller.prompts[0])


class TestSphereRunnerSmoke(unittest.TestCase):
    def test_genetic_short_run_writes_summary(self) -> None:
        config = SphereSchedulerConfig(
            condition="genetic",
            iterations=3,
            batch_size=50,
            archive_dims=(20, 20),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "seed_0"
            result = run_sphere_qd(config, seed=0, output_dir=out)
            self.assertEqual(result.proposals, 150)
            self.assertGreater(result.coverage, 0.0)
            summary = json.loads((out / "nightly_run_summary.json").read_text())
            self.assertEqual(summary["benchmark"], "sphere")
            self.assertEqual(summary["target_selection"], "uniform_frontier")
            self.assertTrue((out / "archive_trace.jsonl").is_file())

    def test_mock_llm_hints_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "sphere_h1.joblib"
            sur = train_sphere_h1_surrogate(seed=0, n_train=200, n_members=2)
            save_sphere_h1_surrogate(sur, ckpt)
            config = SphereSchedulerConfig(
                condition="llm_hints_uniform",
                iterations=2,
                batch_size=50,
                archive_dims=(20, 20),
                initial_random_candidates=10,
                llm_prompt_mode="hints",
                surrogate_checkpoint=str(ckpt),
                emitters=("random",) * 20 + ("llm",) * 30,
            )
            out = Path(tmp) / "llm"
            result = run_sphere_qd(
                config,
                seed=1,
                output_dir=out,
                llm_emitter=MockSphereLlmEmitter(),
            )
            self.assertEqual(result.proposals, 100)
            summary = json.loads((out / "nightly_run_summary.json").read_text())
            self.assertEqual(summary["llm_prompt_mode"], "hints")
            self.assertGreater(summary["llm_audit"]["attempts"], 0)

    def test_genetic_fallback_when_parent_missing(self) -> None:
        rng = np.random.default_rng(0)
        from worldspace.benchmarks.sphere_rq1 import SphereTarget

        empty = SphereTarget(cell_id=0, bin=(0, 0), center=(0.0, 0.0), parent=None)
        emitted = emit_genetic(empty, rng)
        self.assertEqual(emitted.emitter_type, "random")

    def test_llm_emitter_counts_no_parent_as_fallback(self) -> None:
        from worldspace.benchmarks.sphere_rq1 import SphereTarget

        empty = SphereTarget(cell_id=0, bin=(0, 0), center=(0.0, 0.0), parent=None)
        caller = _Caller(_deltas_json())
        emitter = SphereLlmEmitter(
            prompt_mode="stub",
            call_llm_text=caller,  # type: ignore[arg-type]
        )
        emitted = emitter.emit(
            target=empty,
            rng=np.random.default_rng(0),
            prediction=None,
        )
        self.assertEqual(emitted.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.attempts, 1)
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.api_calls, 0)
        self.assertEqual(caller.prompts, [])
        self.assertEqual(emitter.audit.failure_reasons.get("no_parent"), 1)


class TestSphereLlmFailureModes(unittest.TestCase):
    def _target(self):
        archive = _seeded_archive()
        return select_target_cell(
            archive, np.random.default_rng(6), target_selection="uniform_frontier"
        )

    def _emit(self, caller: object, *, max_retries: int = 2):
        emitter = SphereLlmEmitter(
            prompt_mode="stub",
            call_llm_text=caller,  # type: ignore[arg-type]
            max_retries=max_retries,
        )
        result = emitter.emit(
            target=self._target(),
            rng=np.random.default_rng(7),
            prediction=None,
        )
        return emitter, result

    def test_retry_recovers_from_malformed_content(self) -> None:
        from unittest.mock import patch

        caller = _SequenceCaller(["not-json", _deltas_json()])
        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(caller, max_retries=1)
        self.assertEqual(result.emitter_type, "llm")
        self.assertEqual(emitter.audit.retries, 1)
        self.assertEqual(emitter.audit.fallbacks, 0)

    def test_persistent_malformed_content_uses_genetic_fallback(self) -> None:
        from unittest.mock import patch

        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(_Caller("```json\n{not json}\n```"))
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertGreater(emitter.audit.api_calls, 1)
        self.assertGreater(
            emitter.audit.failure_reasons.get("json", 0)
            + emitter.audit.failure_reasons.get("invalid", 0),
            0,
        )

    def test_transient_network_error_recovers_before_fallback(self) -> None:
        from unittest.mock import patch

        caller = _SequenceCaller(
            [
                RuntimeError(
                    "LLM request failed: [SSL: UNEXPECTED_EOF_WHILE_READING]"
                ),
                _deltas_json(),
            ]
        )
        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(caller, max_retries=1)
        self.assertEqual(result.emitter_type, "llm")
        self.assertEqual(emitter.audit.retries, 1)
        self.assertEqual(emitter.audit.fallbacks, 0)
        self.assertEqual(emitter.audit.invalid_response_reasons.get("network"), 1)

    def test_persistent_network_error_uses_genetic_fallback(self) -> None:
        from unittest.mock import patch

        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(
                _ErrorCaller("LLM request failed: connection reset")
            )
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.failure_reasons.get("network"), 1)

    def test_persistent_http_520_uses_genetic_fallback(self) -> None:
        from unittest.mock import patch

        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(
                _ErrorCaller("LLM request failed: HTTP 520: <none>")
            )
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.failure_reasons.get("network"), 1)

    def test_malformed_api_envelope_does_not_kill_the_seed(self) -> None:
        from unittest.mock import patch

        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(
                _ErrorCaller("LLM response missing message.content")
            )
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.failure_reasons.get("network"), 1)

    def test_html_200_body_does_not_kill_the_seed(self) -> None:
        from unittest.mock import patch

        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(
                _ErrorCaller("LLM response is not valid JSON")
            )
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.failure_reasons.get("network"), 1)

    def test_broken_choices_envelope_uses_genetic_fallback(self) -> None:
        from unittest.mock import patch

        with patch("worldspace.benchmarks.sphere_llm.time.sleep"):
            emitter, result = self._emit(_EnvelopeCaller(""))
        self.assertEqual(result.emitter_type, "llm_fallback_genetic")
        self.assertEqual(emitter.audit.fallbacks, 1)
        self.assertEqual(emitter.audit.failure_reasons.get("envelope"), 1)

    def test_missing_api_key_still_aborts(self) -> None:
        caller = _ErrorCaller(
            "Environment variable 'OPENAI_API_KEY' is required for remote provider 'openai'"
        )
        emitter = SphereLlmEmitter(
            prompt_mode="stub",
            call_llm_text=caller,  # type: ignore[arg-type]
        )
        with self.assertRaises(RuntimeError) as caught:
            emitter.emit(
                target=self._target(),
                rng=np.random.default_rng(7),
                prediction=None,
            )
        self.assertIn("OPENAI_API_KEY", str(caught.exception))
        self.assertEqual(emitter.audit.fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
