import gc
import io
import json
import os
import pickle
import tempfile
import threading
import time
import unittest
import weakref
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bedrock_rl.sdg.policy import (
    MovementProfile as PolicyMovementProfile,
    PolicyVeto as DirectPolicyVeto,
    SurvivalBehaviors as PolicySurvivalBehaviors,
    SurvivalPolicy as PolicySurvivalPolicy,
    movement_actions as direct_movement_actions,
)
from bedrock_rl.sdg import (
    MovementProfile, PolicyVeto, SurvivalBehaviors, SurvivalPolicy,
    SnapshotQualifier, movement_actions,
)
from bedrock_rl.sdg import TaskVariantGenerator
from bedrock_rl.core.components import ComponentSpec
from bedrock_rl.sdg.generation import (Case, CaseFile, GenerationSpec,
                                           JSONLWriter,
                                           CaseRejected, ProcessExecutor,
                                           RandomCases, SerialExecutor,
                                           SeedCases, ThreadExecutor,
                                           TrialGenerator, _call_group,
                                           _environment_gpus,
                                           _pin_process_gpu,
                                           _thread_map_bounded,
                                           _worker_ceiling,
                                           assign_path,
                                           generation_jobs)
from bedrock_rl.env.journal import EVENTS, Event
from bedrock_rl.sdg.pipeline import observable_action_summary


class _DuplicateSampler:
    def __init__(self, **_):
        pass

    def sample(self, count, seed):
        del count, seed
        yield Case(0, 7, 1)
        yield Case(1, 7, 2)
        yield Case(2, 9, 3)


class _TwoDuplicateSampler:
    def __init__(self, **_):
        pass

    def sample(self, count, seed):
        del count, seed
        yield Case(0, 7, 1)
        yield Case(1, 7, 2)


class _PassGenerator:
    def __init__(self, **_):
        pass

    def __call__(self, case):
        return {"case": case.to_dict()}


class _ThreadUnsafeGenerator(_PassGenerator):
    thread_safe = False


class _RejectFirstTenGenerator(_PassGenerator):
    def __call__(self, case):
        if case.index < 10:
            raise CaseRejected("synthetic rejection", code="synthetic")
        return super().__call__(case)


class _RejectFirstGenerator(_PassGenerator):
    def __call__(self, case):
        if case.index == 0:
            raise CaseRejected(
                "first candidate rejected", code="first_candidate",
                details={"stage": "unit"})
        return super().__call__(case)


class _SlowFirstGenerator(_PassGenerator):
    def __call__(self, case):
        if case.index == 0:
            time.sleep(0.15)
        return super().__call__(case)


class _EnvironmentGenerator(_PassGenerator):
    def __call__(self, case):
        del case
        names = (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        )
        return {name: os.environ.get(name) for name in names}


class _ThreeWorldSampler:
    def __init__(self, **_):
        pass

    def sample_worlds(self, count, samples_per_world, seed):
        del seed
        for world_index, world_seed in enumerate((7, 9, 11)[:count]):
            yield tuple(Case(
                world_index * samples_per_world + sample_index,
                world_seed, world_seed * 100 + sample_index, {},
                world_index=world_index, sample_index=sample_index)
                for sample_index in range(samples_per_world))


class _RejectOneVariant:
    def __init__(self, **_):
        pass

    def __call__(self, case):
        if case.seed == 7 and case.sample_index == 1:
            raise CaseRejected(
                "unsafe route", code="unsafe_route",
                details={"plan": {"expanded_nodes": 12}})
        return {"case": case.to_dict()}


class _RejectEveryFifthWorld:
    def __init__(self, **_):
        pass

    def __call__(self, case):
        if case.seed % 5 == 0:
            raise CaseRejected("synthetic unsafe world", code="unsafe_world")
        return {"case": case.to_dict()}


class _OutOfOrderRejectEveryFifthWorld(_RejectEveryFifthWorld):
    def __call__(self, case):
        # Force later candidate worlds to complete before some earlier ones;
        # accepted seed selection must still follow sampler order.
        if case.seed % 7 == 0:
            time.sleep(0.02)
        return super().__call__(case)


class _TimeoutSecondSampleGenerator(_PassGenerator):
    def __init__(self, **_):
        self.tainted = False

    def __call__(self, case):
        if case.seed != 0 and self.tainted:
            raise CaseRejected(
                "timed-out candidate state leaked into replacement",
                code="timeout_state_leaked")
        if case.seed == 0 and case.sample_index == 0:
            self.tainted = True
        if case.seed == 0 and case.sample_index == 1:
            time.sleep(0.25)
        return super().__call__(case)


class _TimeoutFirstQualifier(_PassGenerator):
    def qualify_group(self, cases):
        if cases[0].seed == 0:
            time.sleep(0.25)


class _TimeoutFirstCaseGenerator(_PassGenerator):
    def __call__(self, case):
        if case.index == 0:
            time.sleep(0.25)
        return super().__call__(case)


class _RejectQualifiedGroup(_PassGenerator):
    def __init__(self):
        self.calls = 0

    def qualify_group(self, cases):
        self.calls += 1
        raise CaseRejected(
            f"seed {cases[0].seed} lacks a cheap prerequisite",
            code="cheap_prerequisite")

    def __call__(self, case):
        raise AssertionError("qualified rejection must precede generation")


def _qualification_probe(world, *, expected_seed, start, decision_seed):
    if world.seed != expected_seed:
        raise AssertionError("wrong world seed binding")
    if start != (13, 61, -5):
        raise AssertionError(f"wrong player-cell binding: {start}")
    if decision_seed != 123:
        raise AssertionError("wrong decision-seed binding")


class _LargePickleGenerator:
    def __init__(self, size=512 * 1024, **_):
        self.size = int(size)

    def __call__(self, case):
        # A tuple and bytes sentinel exercise arbitrary picklable results,
        # rather than relying on the normal JSON-shaped generator output.
        sample = int(case.sample_index or 0)
        return sample, bytes([sample + 1]) * self.size


class _LargeSentinel:
    def __init__(self, index, size=512 * 1024):
        self.index = int(index)
        self.blob = b"x" * int(size)


class _ReleaseCheckingGenerator:
    def __init__(self):
        self.previous = None

    def __call__(self, case):
        gc.collect()
        if self.previous is not None and self.previous() is not None:
            raise RuntimeError("previous grouped value is still retained")
        value = _LargeSentinel(case.sample_index)
        self.previous = weakref.ref(value)
        return value


class _BrokenGenerator:
    def __init__(self, **_):
        pass

    def __call__(self, case):
        raise OSError(f"engine failed for {case.seed}")


class _SequentialWorldSampler:
    def __init__(self, **_):
        pass

    def sample_worlds(self, count, samples_per_world, seed):
        del seed
        for world_index in range(count):
            yield tuple(Case(
                world_index * samples_per_world + sample_index,
                world_index, world_index * 1000 + sample_index, {},
                world_index=world_index, sample_index=sample_index)
                for sample_index in range(samples_per_world))


class _MemoryWriter:
    def __init__(self, output, **_):
        self.output = output

    def write(self, records):
        return list(records)


_CLOSE_EVENTS = []


class _ClosableGenerator(_PassGenerator):
    def close(self):
        _CLOSE_EVENTS.append("closed")


class _PrefixWriter:
    def __init__(self, output, **_):
        self.output = output

    def write(self, records):
        return [next(iter(records))]


class _BrokenWriter:
    def __init__(self, output, **_):
        raise RuntimeError(f"cannot open {output}")


class GenerationTests(unittest.TestCase):
    def test_observable_summary_names_live_click_vocabulary(self):
        summary = observable_action_summary(
            "public observation",
            [{"action": "left_click_hold", "ticks": 30}], 0)

        self.assertIn("interact with what I am aiming at", summary)
        self.assertNotIn("wait for the next observation", summary)

    def test_group_qualification_rejects_before_any_sample_executes(self):
        generator = _RejectQualifiedGroup()
        cases = (
            Case(0, 7, 101, {}, world_index=0, sample_index=0),
            Case(1, 7, 202, {}, world_index=0, sample_index=1),
        )

        outcome = _call_group(generator, cases)

        self.assertFalse(outcome["ok"])
        self.assertFalse(outcome["fatal"])
        self.assertEqual(outcome["attempted"], 0)
        self.assertEqual(outcome["discarded"], 0)
        self.assertEqual(outcome["code"], "cheap_prerequisite")
        self.assertEqual(generator.calls, 1)

    def test_snapshot_qualifier_batches_resource_and_source_counts(self):
        from tests.test_world_map import write_snapshot

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "world.bsnp"
            write_snapshot(snapshot)
            qualifier = SnapshotQualifier(
                resources={
                    "diamonds": {"blocks": ["diamond_ore"],
                                 "minimum": 3},
                    "floor": {"blocks": ["stone"], "minimum": 4},
                },
                source_fluids={
                    "water": {"blocks": ["flowing_water", "water"],
                              "minimum": 1},
                    "lava": {"blocks": ["flowing_lava", "lava"],
                             "minimum": 1},
                })

            result = qualifier.qualify(
                SimpleNamespace(), snapshot, Case(0, -37, 123, {}))
            again = qualifier.qualify(
                SimpleNamespace(), snapshot, Case(1, -37, 456, {}))

        self.assertEqual(result["resources"], {"diamonds": 3, "floor": 4})
        self.assertEqual(result["source_fluids"], {"water": 1, "lava": 1})
        self.assertEqual(result["snapshot_sha256"],
                         again["snapshot_sha256"])

    def test_snapshot_qualifier_reports_structured_missing_counts(self):
        from tests.test_world_map import write_snapshot

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "world.bsnp"
            write_snapshot(snapshot)
            qualifier = SnapshotQualifier(resources={
                "diamonds": {"blocks": ["diamond_ore"], "minimum": 4},
            })
            with self.assertRaises(CaseRejected) as raised:
                qualifier.qualify(
                    SimpleNamespace(), snapshot, Case(0, -37, 123, {}))

        self.assertEqual(raised.exception.code,
                         "qualification_resource_missing")
        self.assertEqual(raised.exception.details["missing"]["diamonds"], {
            "available": 3, "required": 4, "blocks": ["diamond_ore"],
        })

    def test_world_function_check_binds_case_and_snapshot_fields(self):
        from tests.test_world_map import write_snapshot

        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "world.bsnp"
            write_snapshot(snapshot)
            qualifier = SnapshotQualifier(checks=[{
                "type": "bedrock_rl.sdg:"
                        "WorldFunctionCheck",
                "function": f"{__name__}:_qualification_probe",
                "kwargs": {"expected_seed": -37},
                "bindings": {
                    "start": "player_cell",
                    "decision_seed": "decision_seed",
                },
            }])

            result = qualifier.qualify(
                SimpleNamespace(), snapshot, Case(0, -37, 123, {}))

        self.assertEqual(result["checks"]["0"]["accepted"], True)

    def test_generate_cli_writes_progress_to_stderr(self):
        from bedrock_rl.cli.generate import main

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "data.yaml"
            config.write_text(
                "\n".join((
                    "count: 1",
                    f"generator: {__name__}:_PassGenerator",
                    "executor: bedrock_rl.sdg:SerialExecutor",
                    f"writer: {__name__}:_MemoryWriter",
                    "output: unused",
                )))
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main([str(config)]), 0)

        self.assertIn("progress records=1/1", stderr.getvalue())
        self.assertIn("candidates=1/1 committed=1", stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["accepted"], 1)

    def test_generate_cli_formats_unknown_eta_from_json_safe_progress(self):
        from bedrock_rl.cli.generate import main

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "data.yaml"
            config.write_text(
                "\n".join((
                    "count: 1",
                    "max_attempts: 11",
                    f"generator: {__name__}:_RejectFirstTenGenerator",
                    "executor: bedrock_rl.sdg:SerialExecutor",
                    f"writer: {__name__}:_MemoryWriter",
                    "output: unused",
                )))
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main([str(config)]), 0)

        self.assertIn("records=0/1 candidates=10/11", stderr.getvalue())
        self.assertIn("eta=unknown", stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["accepted"], 1)

    def test_generation_rejects_unsupported_schema_version(self):
        with self.assertRaisesRegex(
                ValueError, "unsupported generation schema_version 2"):
            GenerationSpec.from_dict({
                "schema_version": 2,
                "count": 1,
                "generator": f"{__name__}:_PassGenerator",
                "output": "unused.jsonl",
            })

    def test_scripted_frameless_generation_disables_raster_capture(self):
        from bedrock_rl.sdg import ScriptedEpisodeGenerator

        fake_task = SimpleNamespace(
            max_turns=3, view=SimpleNamespace(type="x"), raw={}, max_lines=4)
        generator = ScriptedEpisodeGenerator.__new__(ScriptedEpisodeGenerator)
        generator.task = fake_task
        generator.task_path = Path("task.yaml")
        generator.policy = SimpleNamespace(create=lambda: SimpleNamespace(
            actions=lambda *_args: None))
        generator.netherite_home = None
        generator.frames_root = "/tmp/unused"
        generator.include_images = False
        generator.record_views = False
        generator.semantic_camera = False
        generator.recording_every = None
        generator.max_turns = 3
        generator.metadata = {}
        case = Case(0, 7, 11, {})
        with patch("bedrock_rl.sdg.pipeline.Episode.draw",
                   side_effect=RuntimeError("stop after draw")) as draw:
            with self.assertRaisesRegex(RuntimeError, "stop after draw"):
                generator(case)
        self.assertFalse(draw.call_args.kwargs["capture_frames"])

    def test_scripted_turn_state_excludes_static_episode_spec(self):
        from bedrock_rl.sdg.pipeline import _episode_state

        episode = SimpleNamespace(
            done=False, success=False, failed=False, turns=3, nlines=7,
            spec={"task_yaml": "/private/host/task.yaml", "seed": 11},
            final_reward=lambda: 0.25,
        )
        with patch(
                "bedrock_rl.adapters.netherite.probes.capture_observation",
                return_value={"tick": 19}):
            state = _episode_state(episode)

        self.assertEqual(state, {
            "episode": {
                "done": False, "success": False, "failed": False,
                "turns": 3, "actions": 7, "reward": 0.25,
            },
            "observation": {"tick": 19},
        })
        self.assertNotIn("/private/host", str(state))

    def test_scripted_snapshot_rejection_is_an_expected_world_rejection(self):
        from bedrock_rl.sdg import ScriptedEpisodeGenerator
        from bedrock_rl.env.snapshots import SnapshotRejected

        generator = ScriptedEpisodeGenerator.__new__(ScriptedEpisodeGenerator)
        generator.task = SimpleNamespace(
            max_turns=1, raw={}, max_lines=4, max_ticks=100)
        generator.netherite_home = None
        generator.frames_root = "/tmp/unused"
        generator.include_images = False
        generator.recording_every = None
        with patch(
                "bedrock_rl.sdg.pipeline.Episode.draw",
                side_effect=SnapshotRejected("nearest tree is too far")):
            with self.assertRaises(CaseRejected) as raised:
                generator(Case(0, 7, 11, {}))
        self.assertEqual(raised.exception.code, "world_constraint")
        self.assertIn("episode_draw_seconds=", str(raised.exception))

    def test_scripted_success_records_generation_stage_timings(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        fake_task = SimpleNamespace(
            name="timed", goal="wait", max_turns=1,
            view=SimpleNamespace(type="x"), raw={}, max_lines=4)
        fake_episode = SimpleNamespace(
            spec={"task_yaml": "task.yaml", "seed": 7},
            t0_png=b"", view_provenance={}, env=SimpleNamespace(
                obs={"blocks": [], "inv_counts": {}}, journal=[], ticks=0),
            done=False, success=True, failed=False, turns=1, nlines=0,
            final_reward=lambda: 1.0, close=Mock())
        fake_policy = SimpleNamespace(
            reset=lambda *_args: None, actions=lambda *_args: None,
            cost_certificate=lambda: {"planned_actions": 12})
        generator = ScriptedEpisodeGenerator.__new__(ScriptedEpisodeGenerator)
        generator.task = fake_task
        generator.task_path = Path("task.yaml")
        generator.task_template = None
        generator.policy = SimpleNamespace(
            type="tests:TimedPolicy", create=lambda: fake_policy)
        generator.netherite_home = None
        generator.frames_root = "/tmp/unused"
        generator.include_images = False
        generator.record_views = False
        generator.semantic_camera = False
        generator.recording_every = None
        generator.max_turns = 1
        generator.max_turns_override = None
        generator.metadata = {}

        with patch("bedrock_rl.sdg.pipeline.Episode.draw",
                   return_value=fake_episode), \
                patch("bedrock_rl.sdg.pipeline.cu_user_prompt",
                      return_value="wait"), \
                patch("bedrock_rl.sdg.pipeline._episode_state",
                      return_value={}):
            trajectory = generator(Case(0, 7, 11, {}))

        timing = trajectory.metadata["generation_timing_seconds"]
        self.assertEqual(set(timing), {
            "episode_draw", "policy_preflight", "live_progression",
            "total_to_verification",
        })
        self.assertTrue(all(value >= 0 for value in timing.values()))
        self.assertEqual(
            trajectory.provenance["policy_cost_certificate"],
            {"planned_actions": 12})

    def test_scripted_generator_closes_resources_when_reset_fails(self):
        from bedrock_rl.sdg import ScriptedEpisodeGenerator

        fake_task = SimpleNamespace(
            max_turns=3, view=SimpleNamespace(type="x"), raw={}, max_lines=4)
        fake_episode = SimpleNamespace(close=Mock())
        fake_policy = SimpleNamespace(
            reset=Mock(side_effect=RuntimeError("bad reset")), close=Mock())
        generator = ScriptedEpisodeGenerator.__new__(ScriptedEpisodeGenerator)
        generator.task = fake_task
        generator.task_path = Path("task.yaml")
        generator.policy = SimpleNamespace(
            type="tests:Policy", create=lambda: fake_policy)
        generator.netherite_home = None
        generator.frames_root = "/tmp/unused"
        generator.include_images = False
        generator.record_views = False
        generator.semantic_camera = False
        generator.recording_every = None
        generator.max_turns = 3
        generator.metadata = {}
        case = Case(0, 7, 11, {})

        with patch("bedrock_rl.sdg.pipeline.Episode.draw",
                   return_value=fake_episode):
            with self.assertRaisesRegex(RuntimeError, "bad reset"):
                generator(case)

        fake_policy.close.assert_called_once_with()
        fake_episode.close.assert_called_once_with()

    def test_scripted_preflight_rejection_reports_draw_and_plan_timing(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        fake_task = SimpleNamespace(
            name="timed", goal="wait", max_turns=1,
            view=SimpleNamespace(type="x"), raw={}, max_lines=4)
        fake_episode = SimpleNamespace(
            spec={"task_yaml": "task.yaml", "seed": 7},
            view_provenance={}, close=Mock())
        fake_policy = SimpleNamespace(
            reset=Mock(side_effect=CaseRejected(
                "no safe route", code="plan_no_safe_route")), close=Mock())
        generator = ScriptedEpisodeGenerator.__new__(ScriptedEpisodeGenerator)
        generator.task = fake_task
        generator.task_path = Path("task.yaml")
        generator.task_template = None
        generator.policy = SimpleNamespace(
            type="tests:Policy", create=lambda: fake_policy)
        generator.netherite_home = None
        generator.frames_root = "/tmp/unused"
        generator.include_images = False
        generator.record_views = False
        generator.semantic_camera = False
        generator.recording_every = None
        generator.max_turns = 1
        generator.max_turns_override = None
        generator.metadata = {}

        with patch("bedrock_rl.sdg.pipeline.Episode.draw",
                   return_value=fake_episode):
            with self.assertRaises(CaseRejected) as raised:
                generator(Case(0, 7, 11, {}))
        self.assertEqual(raised.exception.code, "plan_no_safe_route")
        self.assertIn("episode_draw_seconds=", str(raised.exception))
        self.assertIn("policy_preflight_seconds=", str(raised.exception))

    def test_named_jobs_share_defaults_without_becoming_a_pipeline(self):
        jobs = generation_jobs({
            "defaults": {
                "executor": "example:Executor",
                "generator": {"type": "example:Generator", "shared": 1},
            },
            "jobs": {
                "train": {"count": 4, "output": "train.jsonl"},
                "dev": {
                    "count": 2,
                    "output": "dev.jsonl",
                    "generator": {"split": "dev"},
                },
            },
        })
        self.assertEqual(tuple(jobs), ("train", "dev"))
        self.assertEqual(jobs["dev"]["generator"], {
            "type": "example:Generator", "shared": 1, "split": "dev"})

        replaced = generation_jobs({
            "defaults": {"generator": {
                "type": "example:PromptGenerator", "mode": "rl"}},
            "jobs": {"demos": {"generator": {
                "type": "example:DemoGenerator", "policy": "expert"}}},
        })
        self.assertEqual(replaced["demos"]["generator"], {
            "type": "example:DemoGenerator", "policy": "expert"})

    def test_explicit_seed_split_is_reproducible(self):
        one = list(SeedCases([3, 5, 8]).sample(6, 11))
        two = list(SeedCases([3, 5, 8]).sample(6, 11))
        self.assertEqual(one, two)
        self.assertEqual({case.seed for case in one}, {3, 5, 8})

        explicit = list(SeedCases(
            [3, 5], decision_seeds=[101, 202],
            variables={"stride": {"integer": [1, 3]}}).sample(4, 999))
        self.assertEqual([case.decision_seed for case in explicit],
                         [101, 202, 101, 202])
        self.assertEqual(
            [case.values for case in explicit],
            [case.values for case in SeedCases(
                [3, 5], decision_seeds=[101, 202],
                variables={"stride": {"integer": [1, 3]}}).sample(4, 1)])

    def test_case_file_replays_exact_scout_cases_without_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.jsonl"
            rows = [
                {"case": {"index": 19, "seed": 71,
                          "decision_seed": 901,
                          "values": {"stride": 2.25},
                          "world_values": {"ore": "near"},
                          "world_index": 19, "sample_index": 0}},
                {"index": 23, "seed": 83, "decision_seed": 902,
                 "values": {}, "world_index": 23, "sample_index": 0},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows))

            sampler = CaseFile("accepted.jsonl", base_dir=directory)
            cases = list(sampler.sample(2, seed=999))
            groups = list(sampler.sample_worlds(2, 1, seed=123))

        self.assertEqual([case.to_dict() for case in cases], [
            rows[0]["case"], rows[1]])
        self.assertEqual([group[0] for group in groups], cases)

    def test_case_file_fails_loudly_when_the_manifest_is_short(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted.jsonl"
            path.write_text(json.dumps({
                "index": 0, "seed": 7, "decision_seed": 11,
                "world_index": 0, "sample_index": 0,
            }))
            sampler = CaseFile(path)
            with self.assertRaisesRegex(ValueError, "contains only 1 cases"):
                list(sampler.sample_worlds(2, 1, seed=0))

    def test_grouped_worlds_do_not_change_when_augmentation_changes(self):
        one = list(RandomCases().sample_worlds(8, 1, 91))
        four = list(RandomCases().sample_worlds(8, 4, 91))
        repeated = list(RandomCases().sample_worlds(8, 4, 91))

        self.assertEqual([group[0].seed for group in one],
                         [group[0].seed for group in four])
        self.assertEqual([group[0].decision_seed for group in one],
                         [group[0].decision_seed for group in four])
        self.assertEqual(four, repeated)
        self.assertTrue(all(
            len({case.decision_seed for case in group}) == 4
            for group in four))

    def test_world_variables_are_shared_and_sample_variables_are_independent(self):
        sampler = RandomCases(
            world_variables={"tree_distance": {"integer": [4, 12]}},
            variables={"stride": {"uniform": [1.5, 3.0]}})
        groups = list(sampler.sample_worlds(5, 3, 91))

        self.assertTrue(all(
            len({case.values["tree_distance"] for case in group}) == 1
            for group in groups))
        self.assertTrue(all(
            case.world_values == {
                "tree_distance": case.values["tree_distance"]}
            for group in groups for case in group))
        self.assertTrue(all(
            len({case.values["stride"] for case in group}) == 3
            for group in groups))
        self.assertEqual(groups,
                         list(sampler.sample_worlds(5, 3, 91)))
        with self.assertRaisesRegex(ValueError, "overlap: duplicate"):
            RandomCases(
                world_variables={"duplicate": {"constant": 1}},
                variables={"duplicate": {"constant": 2}})

        mutable = list(RandomCases(world_variables={
            "setup": {"constant": {"distance": [4, 12]}},
        }).sample_worlds(1, 2, 7))[0]
        mutable[0].world_values["setup"]["distance"][0] = 99
        mutable[0].values["setup"]["distance"][1] = 88
        self.assertEqual(mutable[1].world_values["setup"]["distance"],
                         [4, 12])
        self.assertEqual(mutable[1].values["setup"]["distance"], [4, 12])

    def test_task_bindings_support_world_values_and_list_indexes(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        root = Path(__file__).resolve().parents[1]
        generator = ScriptedEpisodeGenerator(
            root / "examples" / "farm_one_log_sdg" / "task.yaml",
            "bedrock_rl.sdg:ExpertPolicy",
            include_images=False,
            task_bindings={
                "initial_state.spawn.constraints.0.min_distance":
                    "world_values.tree_min",
            })
        groups = list(RandomCases(
            world_variables={"tree_min": {"integer": [2, 3]}},
            variables={"stride": {"uniform": [1.5, 3.0]}},
        ).sample_worlds(1, 3, 44))

        identities = [generator.group_identity(case) for case in groups[0]]
        tasks = [generator._task_document(case) for case in groups[0]]
        self.assertEqual(len({item["task_digest"]
                              for item in identities}), 1)
        self.assertEqual(
            [task["initial_state"]["spawn"]["constraints"][0]
             ["min_distance"] for task in tasks],
            [groups[0][0].world_values["tree_min"]] * 3)

        document = {"rows": [{"distance": 4}]}
        assign_path(document, "rows.0.distance", 9)
        self.assertEqual(document, {"rows": [{"distance": 9}]})

    def test_headless_qualification_precedes_deterministic_image_replay(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        action_message = {
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "computer",
                    "arguments": json.dumps({
                        "actions": [{"action": "wait", "ticks": 1}],
                    }),
                },
            }],
        }
        headless = SimpleNamespace(
            messages=[action_message],
            metadata={"generation_timing_seconds": {"live": 1.25}})
        rendered = SimpleNamespace(
            messages=[action_message], metadata={})
        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.include_images = True
        generator.headless_qualification = True
        generator.recording_every = 7
        generator.record_views = True
        generator._run_case = Mock(side_effect=[headless, rendered])
        case = Case(0, 7, 11, {})

        result = generator(case)

        self.assertIs(result, rendered)
        self.assertEqual(generator._run_case.call_args_list, [
            unittest.mock.call(
                case, include_images=False, recording_every=None,
                record_views=False),
            unittest.mock.call(
                case, include_images=True, recording_every=7,
                record_views=True),
        ])
        self.assertEqual(
            result.metadata["headless_qualification"]["turns"], 1)
        self.assertEqual(
            result.metadata["headless_qualification"]["timing_seconds"],
            {"live": 1.25})

    def test_dense_recording_transfers_to_writer_owner_before_worker_exit(self):
        from bedrock_rl.env import episode as episode_module
        from bedrock_rl.sdg import ScriptedEpisodeGenerator

        class Policy:
            def actions(self, task, episode, case, turn):
                del task, episode, case, turn
                return None

            def demonstration_grounding(self):
                return {"eligible": True,
                        "target_selection": "public_observation",
                        "execution": "public_interface"}

        with tempfile.TemporaryDirectory() as directory:
            frames = Path(episode_module._owned_frames_dir(directory))
            episode = SimpleNamespace(
                _frames_dir=str(frames), spec={"task_yaml": "task.yaml"},
                view_provenance={"type": "minecraft-official"},
                t0_png=None, env=SimpleNamespace(journal=[]), success=True,
                done=True, failed=False, turns=0, nlines=0,
                final_reward=lambda: 1.0, close=lambda: None)
            task = SimpleNamespace(
                name="recording", goal="record", max_turns=1,
                view=SimpleNamespace(type="minecraft-official"))
            generator = ScriptedEpisodeGenerator.__new__(
                ScriptedEpisodeGenerator)
            generator._task_and_digest = lambda case: (task, "digest")
            generator.qualifier = None
            generator.max_turns_override = None
            generator.netherite_home = None
            generator.frames_root = directory
            generator.policy = SimpleNamespace(
                type="test:Policy", create=Policy)
            generator.task_path = Path("task.yaml")
            generator.semantic_camera = False
            generator.assistant_content = None
            generator.metadata = {}
            generator._artifact_owner_pid = os.getpid()

            with (patch("bedrock_rl.sdg.pipeline.Episode.draw",
                        return_value=episode),
                  patch("bedrock_rl.sdg.pipeline.cu_parser",
                        return_value=lambda *args: None),
                  patch("bedrock_rl.sdg.pipeline.cu_user_prompt",
                        return_value="<image>\nTASK: record"),
                  patch("bedrock_rl.sdg.pipeline._episode_state",
                        return_value={}),
                  patch("bedrock_rl.sdg.pipeline._record_new_events",
                        return_value=0)):
                trajectory = generator._run_case(
                    Case(0, 7, 11, {}), include_images=False,
                    recording_every=1, record_views=False)

            preserved = Path(
                trajectory.metadata["_recording_frames_dir"])
            self.assertTrue(preserved.is_dir())
            self.assertEqual(
                (preserved / episode_module._FRAME_OWNER)
                .read_text().splitlines()[0], str(os.getpid()))
            self.assertIsNone(episode._frames_dir)

    def test_scripted_generator_retries_policy_seed_for_same_world(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        trajectory = SimpleNamespace(metadata={})
        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.decision_attempts = 3
        generator._materialize_case = Mock(side_effect=[
            CaseRejected("unsafe ordering", code="unsafe_route"),
            trajectory,
        ])
        case = Case(
            8, 41, 73, {"stride": 2.0},
            world_index=4, sample_index=0,
            world_values={"tree_min": 3})

        result = generator(case)

        self.assertIs(result, trajectory)
        first, second = [call.args[0]
                         for call in generator._materialize_case.call_args_list]
        self.assertEqual(first, case)
        self.assertEqual(second.seed, case.seed)
        self.assertEqual(second.index, case.index)
        self.assertEqual(second.values, case.values)
        self.assertEqual(second.world_values, case.world_values)
        self.assertNotEqual(second.decision_seed, case.decision_seed)
        self.assertEqual(result.metadata["decision_selection"], {
            "original_decision_seed": 73,
            "selected_decision_seed": second.decision_seed,
            "selected_attempt": 1,
            "rejected_attempts": [{
                "attempt": 0,
                "decision_seed": 73,
                "code": "unsafe_route",
            }],
            "configured_attempts": 3,
        })

    def test_scripted_generator_does_not_retry_world_rejection(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.decision_attempts = 4
        generator._materialize_case = Mock(side_effect=CaseRejected(
            "no qualifying lava", code="world_constraint",
            details={"constraint": "lava"}))

        with self.assertRaises(CaseRejected) as raised:
            generator(Case(0, 41, 73, {}))

        self.assertEqual(generator._materialize_case.call_count, 1)
        self.assertEqual(raised.exception.code, "world_constraint")
        self.assertEqual(raised.exception.details, {
            "constraint": "lava",
            "decision_selection": {
                "attempts": [{
                    "attempt": 0,
                    "decision_seed": 73,
                    "code": "world_constraint",
                }],
                "configured_attempts": 4,
            },
        })

    def test_scripted_generator_classifies_world_level_rejections(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        for code in (
                "world_constraint", "world_plan_invalid_start",
                "world_plan_bounds",
                "world_plan_resource_unreachable",
                "world_plan_search_budget",
                "qualification_resource_missing", "engine_timeout"):
            with self.subTest(code=code):
                self.assertFalse(
                    ScriptedEpisodeGenerator._decision_retryable(
                        CaseRejected("unusable world", code=code)))
        self.assertTrue(ScriptedEpisodeGenerator._decision_retryable(
            CaseRejected(
                "different ordering may find a site",
                code="progression_fluid_cast_site_missing")))

    def test_scripted_generator_filters_snapshot_perimeter_case(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )
        from bedrock_rl.sdg.policy.world import SnapshotBoundsError

        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.decision_attempts = 4
        generator._materialize_case = Mock(side_effect=SnapshotBoundsError(
            "block (168, 24, -34) is outside Bounds(min_x=-152)"))

        with self.assertRaises(CaseRejected) as raised:
            generator(Case(1355, 975685147, 8907199094094683820, {}))

        self.assertEqual(generator._materialize_case.call_count, 1)
        self.assertEqual(raised.exception.code, "world_plan_bounds")
        self.assertEqual(raised.exception.details["attempt"], 0)

    def test_scripted_generator_does_not_hide_unrelated_index_errors(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.decision_attempts = 1
        generator._materialize_case = Mock(side_effect=IndexError(
            "block (168, 24, -34) is outside Bounds(but not the map)"))

        with self.assertRaises(IndexError):
            generator(Case(1, 2, 3, {}))

    def test_scripted_generator_isolates_engine_timeout_to_one_case(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )
        from bedrock_rl.env.engine import EngineTimeout

        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.decision_attempts = 4
        generator._materialize_case = Mock(side_effect=EngineTimeout(
            "engine turn exceeded 120 seconds"))

        with self.assertRaises(CaseRejected) as raised:
            generator(Case(0, 41, 73, {}))

        self.assertEqual(generator._materialize_case.call_count, 1)
        self.assertEqual(raised.exception.code, "engine_timeout")
        self.assertEqual(raised.exception.details, {
            "attempt": 0,
            "decision_seed": 73,
        })

    def test_scripted_qualification_isolates_engine_timeout(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )
        from bedrock_rl.env.engine import EngineTimeout

        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.qualifier = Mock()
        generator._qualified_world = None
        generator._qualified_cases = {}
        generator._qualification_snapshot = None
        generator.netherite_home = None
        task = SimpleNamespace(snapshot_for=Mock(side_effect=EngineTimeout(
            "snapshot build exceeded its deadline")))

        with self.assertRaises(CaseRejected) as raised:
            generator._qualify_case(task, "digest", Case(0, 41, 73, {}))

        self.assertEqual(raised.exception.code, "engine_timeout")
        self.assertEqual(raised.exception.details, {
            "phase": "qualification",
        })

    def test_scripted_generator_validates_decision_retries(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        root = Path(__file__).resolve().parents[1]
        task = root / "examples" / "farm_one_log_sdg" / "task.yaml"
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ScriptedEpisodeGenerator(
                task, "unused:Policy", include_images=False,
                decision_attempts=0)
        with self.assertRaisesRegex(ValueError, "bound into the task"):
            ScriptedEpisodeGenerator(
                task, "unused:Policy", include_images=False,
                decision_attempts=2,
                task_bindings={"metadata.policy_seed": "decision_seed"})

    def test_task_binding_cannot_change_inside_one_world_group(self):
        from bedrock_rl.sdg import (
            ScriptedEpisodeGenerator,
        )

        root = Path(__file__).resolve().parents[1]
        generator = ScriptedEpisodeGenerator(
            root / "examples" / "farm_one_log_sdg" / "task.yaml",
            "bedrock_rl.sdg:ExpertPolicy",
            include_images=False,
            task_bindings={"max_turns": "values.turn_limit"})
        group = (
            Case(0, 7, 101, {"turn_limit": 20},
                 world_index=0, sample_index=0),
            Case(1, 7, 202, {"turn_limit": 30},
                 world_index=0, sample_index=1),
        )

        outcome = _call_group(generator, group)

        self.assertFalse(outcome["ok"])
        self.assertTrue(outcome["fatal"])
        self.assertEqual(outcome["code"], "group_identity_mismatch")

    def test_explicit_grouped_decisions_are_paired_by_sample_index(self):
        groups = list(SeedCases(
            [3, 5], decision_seeds=[101, 202]).sample_worlds(2, 2, 11))
        self.assertEqual(
            [[case.decision_seed for case in group] for group in groups],
            [[101, 202], [101, 202]])
        with self.assertRaisesRegex(ValueError, "distinct decision_seeds"):
            list(SeedCases(
                [3], decision_seeds=[101, 101, 202]).sample_worlds(1, 2, 11))

    def test_unique_world_generation_never_counts_duplicate_success(self):
        module = __name__
        document = {
            "count": 2,
            "max_attempts": 3,
            "unique_worlds": True,
            "sampler": f"{module}:_DuplicateSampler",
            "generator": f"{module}:_PassGenerator",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        }
        result = GenerationSpec.from_dict(document).execute()
        self.assertEqual(
            [record["case"]["seed"] for record in result["output"]],
            [7, 9])
        self.assertEqual(result["attempted"], 3)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["rejection_reasons"][0]["code"],
                         "duplicate_world")

    def test_rejection_ledger_keeps_exact_cases_and_structured_details(self):
        module = __name__
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "data.yaml"
            ledger = Path(directory) / "rejections.jsonl"
            spec = GenerationSpec.from_dict({
                "count": 1,
                "max_attempts": 2,
                "generator": f"{module}:_RejectFirstGenerator",
                "executor": "bedrock_rl.sdg:SerialExecutor",
                "writer": f"{module}:_MemoryWriter",
                "output": "unused.jsonl",
                "rejections_output": "rejections.jsonl",
            }, path=str(config))

            result = spec.execute()
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]

        self.assertEqual(
            Path(result["rejections_output"]), ledger.resolve())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema"], "bedrock.rejection.v1")
        self.assertEqual(rows[0]["case"]["index"], 0)
        self.assertEqual(rows[0]["code"], "first_candidate")
        self.assertEqual(rows[0]["details"], {"stage": "unit"})

    def test_unique_world_generation_fails_before_running_insufficient_pool(self):
        module = __name__
        spec = GenerationSpec.from_dict({
            "count": 2,
            "max_attempts": 2,
            "unique_worlds": True,
            "sampler": f"{module}:_TwoDuplicateSampler",
            "generator": "does.not.matter:Generator",
            "output": "unused",
        })
        with self.assertRaisesRegex(ValueError, "only 1 distinct"):
            spec.execute()

    def test_world_groups_emit_exact_worlds_times_samples(self):
        module = __name__
        spec = GenerationSpec.from_dict({
            "worlds": 2,
            "samples_per_world": 3,
            "sampler": f"{module}:_ThreeWorldSampler",
            "generator": f"{module}:_PassGenerator",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        })
        result = spec.execute()

        self.assertEqual(len(result["output"]), 6)
        self.assertEqual(
            [(row["case"]["seed"], row["case"]["sample_index"])
             for row in result["output"]],
            [(7, 0), (7, 1), (7, 2),
             (9, 0), (9, 1), (9, 2)])
        self.assertEqual(result["worlds_accepted"], 2)
        self.assertEqual(result["samples_per_world"], 3)
        self.assertEqual((result["accepted"], result["rejected"]), (6, 0))

    def test_library_progress_is_opt_in_bounded_and_deterministic(self):
        module = __name__
        events = []
        spec = GenerationSpec.from_dict({
            "worlds": 2,
            "samples_per_world": 1,
            "max_world_attempts": 3,
            "sampler": f"{module}:_ThreeWorldSampler",
            "generator": f"{module}:_RejectOneVariant",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        })
        with patch("bedrock_rl.sdg.generation.time.monotonic",
                   side_effect=[0, 11, 22, 33, 44, 55]):
            result = spec.execute(progress=events.append)

        self.assertEqual(result["worlds_accepted"], 2)
        self.assertLessEqual(len(events), result["worlds_attempted"] + 1)
        self.assertEqual(events[-1]["accepted"], 2)
        self.assertEqual(events[-1]["wanted"], 2)
        self.assertEqual(events[-1]["attempt_limit"], 3)
        self.assertEqual(events[-1]["top_rejections"], "none")
        self.assertEqual(events[-1]["rejection_counts"], {})
        for event in events:
            json.dumps(event, allow_nan=False)
        # No callback means the library stays quiet and pays no formatting
        # surface; this second run is the ordinary programmatic API.
        self.assertEqual(spec.execute()["worlds_accepted"], 2)

        unknown_eta = []
        GenerationSpec.from_dict({
            "count": 1,
            "max_attempts": 11,
            "generator": f"{module}:_RejectFirstTenGenerator",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        }).execute(progress=unknown_eta.append)
        self.assertIsNone(unknown_eta[0]["eta_seconds"])
        self.assertEqual(
            unknown_eta[0]["rejection_counts"], {"synthetic": 10})
        json.dumps(unknown_eta[0], allow_nan=False)

    def test_group_values_are_small_references_loaded_lazily_in_order(self):
        group = next(_ThreeWorldSampler().sample_worlds(1, 3, 0))
        with tempfile.TemporaryDirectory() as directory:
            outcome = _call_group(
                _LargePickleGenerator(), group, directory)

            self.assertTrue(outcome["ok"])
            self.assertNotIn("values", outcome)
            # Three 512-KiB values are on disk; the outcome is only one trusted
            # internal reference, which is also the process-IPC payload.
            self.assertLess(len(pickle.dumps(outcome)), 4096)
            spool = outcome["spool"]
            staged = Path(spool.directory)
            self.assertEqual(
                [path.name for path in sorted(staged.iterdir())],
                [f"{index:08d}.pickle" for index in range(3)])

            values = spool.values()
            first = next(values)
            self.assertEqual((first[0], len(first[1])), (0, 512 * 1024))
            self.assertFalse((staged / "00000000.pickle").exists())
            self.assertTrue((staged / "00000001.pickle").exists())
            self.assertEqual([value[0] for value in values], [1, 2])
            self.assertFalse(staged.exists())

    def test_each_large_group_value_is_released_after_immediate_staging(self):
        group = next(_ThreeWorldSampler().sample_worlds(1, 3, 0))
        generator = _ReleaseCheckingGenerator()
        with tempfile.TemporaryDirectory() as directory:
            outcome = _call_group(generator, group, directory)

            self.assertTrue(outcome["ok"])
            gc.collect()
            self.assertIsNone(generator.previous())
            self.assertEqual(
                [value.index for value in outcome["spool"].values()],
                [0, 1, 2])

    def test_failed_group_discards_already_staged_large_siblings(self):
        class RejectSecond(_LargePickleGenerator):
            def __call__(self, case):
                if case.sample_index == 1:
                    raise CaseRejected("unsafe sibling", code="unsafe_sibling")
                return super().__call__(case)

        group = next(_ThreeWorldSampler().sample_worlds(1, 3, 0))
        with tempfile.TemporaryDirectory() as directory:
            outcome = _call_group(RejectSecond(), group, directory)

            self.assertFalse(outcome["ok"])
            self.assertEqual(outcome["attempted"], 2)
            self.assertEqual(outcome["discarded"], 1)
            self.assertEqual(outcome["code"], "unsafe_sibling")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_group_executor_close_removes_every_unconsumed_stage(self):
        generator = ComponentSpec.parse(f"{__name__}:_LargePickleGenerator")
        groups = list(_ThreeWorldSampler().sample_worlds(3, 3, 0))
        executors = (
            SerialExecutor(), ThreadExecutor(workers=2),
            ProcessExecutor(workers=1),
        )
        for executor in executors:
            with self.subTest(executor=type(executor).__name__):
                outcomes = executor.run_groups(
                    iter(groups), generator, base_dir=".", metadata={})
                first = next(outcomes)
                spool_root = Path(first["spool"].directory).parent
                self.assertTrue(spool_root.exists())
                # Leave this group, plus any bounded thread-prefetch groups,
                # unread. Closing the run owns and removes all of them.
                outcomes.close()
                self.assertFalse(spool_root.exists())

    @unittest.skipIf(os.name == "nt", "forkserver is a POSIX contract")
    def test_process_group_ipc_returns_spool_reference_not_large_values(self):
        generator = ComponentSpec.parse(f"{__name__}:_LargePickleGenerator")
        groups = list(_ThreeWorldSampler().sample_worlds(2, 3, 0))
        outcomes = ProcessExecutor(workers=2).run_groups(
            groups, generator, base_dir=".", metadata={})
        outcome = next(outcomes)
        spool_root = Path(outcome["spool"].directory).parent
        try:
            self.assertNotIn("values", outcome)
            self.assertLess(len(pickle.dumps(outcome)), 4096)
            self.assertEqual(
                [value[0] for value in outcome["spool"].values()],
                [0, 1, 2])
        finally:
            outcomes.close()
        self.assertFalse(spool_root.exists())

    def test_one_failed_sample_discards_its_complete_world(self):
        module = __name__
        spec = GenerationSpec.from_dict({
            "worlds": 2,
            "samples_per_world": 3,
            "max_world_attempts": 3,
            "sampler": f"{module}:_ThreeWorldSampler",
            "generator": f"{module}:_RejectOneVariant",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        })
        result = spec.execute()

        self.assertEqual(len(result["output"]), 6)
        self.assertNotIn(7, {row["case"]["seed"]
                             for row in result["output"]})
        self.assertEqual(
            {row["case"]["seed"] for row in result["output"]}, {9, 11})
        self.assertEqual(
            (result["worlds_attempted"], result["worlds_accepted"],
             result["worlds_rejected"]), (3, 2, 1))
        self.assertEqual(
            (result["attempted"], result["accepted"], result["rejected"]),
            (9, 6, 3))
        self.assertEqual(result["samples_executed"], 8)
        self.assertEqual(result["rejection_reasons"][0]["code"],
                         "unsafe_route")
        self.assertEqual(
            result["rejection_reasons"][0]["details"],
            {"plan": {"expanded_nodes": 12}})
        self.assertEqual(result["world_acceptance_rate"], 2 / 3)

    def test_five_hundred_world_batch_is_exact_and_group_atomic(self):
        module = __name__
        result = GenerationSpec.from_dict({
            "worlds": 500,
            "samples_per_world": 3,
            "max_world_attempts": 625,
            "sampler": f"{module}:_SequentialWorldSampler",
            "generator": f"{module}:_RejectEveryFifthWorld",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        }).execute()

        rows = result["output"]
        self.assertEqual(len(rows), 1500)
        self.assertEqual(len({row["case"]["seed"] for row in rows}), 500)
        self.assertTrue(all(row["case"]["seed"] % 5 for row in rows))
        self.assertEqual(result["worlds_attempted"], 625)
        self.assertEqual(result["worlds_rejected"], 125)
        self.assertEqual(result["samples_executed"], 1625)
        self.assertEqual(result["rejection_reasons"], [{
            "code": "unsafe_world", "count": 125,
            "reason": "sample 0 rejected: synthetic unsafe world",
        }])

    @unittest.skipIf(os.name == "nt", "forkserver is a POSIX contract")
    def test_parallel_replacements_choose_same_worlds_as_serial(self):
        module = __name__

        def run(executor):
            result = GenerationSpec.from_dict({
                "worlds": 20,
                "samples_per_world": 2,
                "max_world_attempts": 30,
                "sampler": f"{module}:_SequentialWorldSampler",
                "generator": (
                    f"{module}:_OutOfOrderRejectEveryFifthWorld"),
                "executor": executor,
                "writer": f"{module}:_MemoryWriter",
                "output": "unused",
            }).execute()
            rows = result["output"]
            return ([row["case"]["seed"] for row in rows
                     if row["case"]["sample_index"] == 0], result)

        serial_worlds, serial = run(
            "bedrock_rl.sdg:SerialExecutor")
        process_worlds, process = run({
            "type": "bedrock_rl.sdg:ProcessExecutor",
            "workers": 4,
            "chunksize": 2,
        })

        self.assertEqual(process_worlds, serial_worlds)
        self.assertEqual(len(process_worlds), 20)
        self.assertEqual(len(set(process_worlds)), 20)
        self.assertEqual(process["worlds_attempted"],
                         serial["worlds_attempted"])
        self.assertEqual(process["rejection_reasons"],
                         serial["rejection_reasons"])

    @unittest.skipUnless(
        hasattr(__import__("signal"), "setitimer"),
        "candidate deadlines require POSIX interval timers")
    def test_process_candidate_timeout_replaces_group_in_clean_worker(self):
        module = __name__
        result = GenerationSpec.from_dict({
            "worlds": 1,
            "samples_per_world": 2,
            "max_world_attempts": 2,
            "sampler": f"{module}:_SequentialWorldSampler",
            "generator": f"{module}:_TimeoutSecondSampleGenerator",
            "executor": {
                "type": "bedrock_rl.sdg:ProcessExecutor",
                # A timeout must still use isolated child processes with one
                # active slot; it may never install SIGALRM in the parent.
                "workers": 1,
                "candidate_timeout_seconds": 0.05,
            },
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        }).execute()

        self.assertEqual(
            [row["case"]["seed"] for row in result["output"]], [1, 1])
        self.assertEqual(result["worlds_attempted"], 2)
        self.assertEqual(result["worlds_rejected"], 1)
        self.assertEqual(result["samples_executed"], 3)
        self.assertEqual(result["rejection_reasons"], [{
            "code": "candidate_timeout",
            "count": 1,
            "reason": "candidate exceeded 0.05 seconds",
            "details": {
                "timeout_seconds": 0.05,
                "scope": "world_group",
            },
        }])

    @unittest.skipUnless(
        hasattr(__import__("signal"), "setitimer"),
        "candidate deadlines require POSIX interval timers")
    def test_process_candidate_timeout_also_bounds_qualification(self):
        module = __name__
        result = GenerationSpec.from_dict({
            "worlds": 1,
            "samples_per_world": 2,
            "max_world_attempts": 2,
            "sampler": f"{module}:_SequentialWorldSampler",
            "generator": f"{module}:_TimeoutFirstQualifier",
            "executor": {
                "type": "bedrock_rl.sdg:ProcessExecutor",
                "workers": 1,
                "candidate_timeout_seconds": 0.05,
            },
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        }).execute()

        self.assertEqual(
            [row["case"]["seed"] for row in result["output"]], [1, 1])
        self.assertEqual(result["samples_executed"], 2)
        self.assertEqual(
            result["rejection_reasons"][0]["code"], "candidate_timeout")

    @unittest.skipUnless(
        hasattr(__import__("signal"), "setitimer"),
        "candidate deadlines require POSIX interval timers")
    def test_process_candidate_timeout_bounds_independent_case(self):
        module = __name__
        result = GenerationSpec.from_dict({
            "count": 1,
            "max_attempts": 2,
            "generator": f"{module}:_TimeoutFirstCaseGenerator",
            "executor": {
                "type": "bedrock_rl.sdg:ProcessExecutor",
                "workers": 1,
                "candidate_timeout_seconds": 0.05,
            },
            "writer": f"{module}:_MemoryWriter",
            "output": "unused",
        }).execute()

        self.assertEqual(result["output"][0]["case"]["index"], 1)
        self.assertEqual(result["rejection_reasons"][0]["details"], {
            "timeout_seconds": 0.05,
            "scope": "case",
        })

    def test_candidate_timeout_must_be_finite_and_positive(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "candidate_timeout_seconds must be positive"):
                ProcessExecutor(candidate_timeout_seconds=value)

    def test_unexpected_generator_errors_fail_fast(self):
        module = __name__
        spec = GenerationSpec.from_dict({
            "worlds": 2,
            "samples_per_world": 2,
            "max_world_attempts": 3,
            "sampler": f"{module}:_ThreeWorldSampler",
            "generator": f"{module}:_BrokenGenerator",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": "bedrock_rl.sdg:JSONLWriter",
            "output": "unused.jsonl",
        })
        with self.assertRaisesRegex(
                RuntimeError, "generator_error for world seed 7.*OSError"):
            spec.execute()

    def test_sink_must_consume_every_record_before_reporting_success(self):
        module = __name__
        spec = GenerationSpec.from_dict({
            "worlds": 1,
            "samples_per_world": 3,
            "sampler": f"{module}:_ThreeWorldSampler",
            "generator": f"{module}:_PassGenerator",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_PrefixWriter",
            "output": "unused",
        })
        with self.assertRaisesRegex(RuntimeError, "consuming 1 of 3"):
            spec.execute()

    def test_sink_constructor_failure_closes_executor_generator(self):
        module = __name__
        _CLOSE_EVENTS.clear()
        spec = GenerationSpec.from_dict({
            "count": 1,
            "sampler": "bedrock_rl.sdg:RandomCases",
            "generator": f"{module}:_ClosableGenerator",
            "executor": "bedrock_rl.sdg:SerialExecutor",
            "writer": f"{module}:_BrokenWriter",
            "output": "unused",
        })
        with self.assertRaisesRegex(RuntimeError, "cannot open"):
            spec.execute()
        # Executor construction is lazy, so a bad writer never constructs the
        # generator and therefore has no engine/API resource to close.
        self.assertEqual(_CLOSE_EVENTS, [])

    def test_legacy_case_stream_stops_after_exact_count(self):
        sampled = []

        class LazySampler:
            def __init__(self, **_):
                pass

            def sample(self, count, seed):
                del seed
                for index in range(count):
                    sampled.append(index)
                    yield Case(index, index, index + 100)

        module = __name__
        with patch(f"{module}._DuplicateSampler", LazySampler):
            result = GenerationSpec.from_dict({
                "count": 1,
                "max_attempts": 1000,
                "sampler": f"{module}:_DuplicateSampler",
                "generator": f"{module}:_PassGenerator",
                "executor": "bedrock_rl.sdg:SerialExecutor",
                "writer": f"{module}:_MemoryWriter",
                "output": "unused",
            }).execute()
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(sampled, [0])

    def test_world_and_count_forms_are_unambiguous(self):
        common = {"generator": f"{__name__}:_PassGenerator",
                  "output": "unused"}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            GenerationSpec.from_dict(common)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            GenerationSpec.from_dict(
                {**common, "count": 1, "worlds": 1})
        with self.assertRaisesRegex(ValueError,
                                    "samples_per_world requires worlds"):
            GenerationSpec.from_dict(
                {**common, "count": 2, "samples_per_world": 2})
        with self.assertRaisesRegex(ValueError,
                                    "max_world_attempts requires worlds"):
            GenerationSpec.from_dict(
                {**common, "count": 2, "max_world_attempts": 3})
        with self.assertRaisesRegex(ValueError,
                                    "use max_world_attempts with worlds"):
            GenerationSpec.from_dict(
                {**common, "worlds": 2, "max_attempts": 3})

    def test_case_rejection_codes_are_aggregated_separately_from_details(self):
        from bedrock_rl.sdg.generation import CaseRejected, _call

        class Reject:
            def __call__(self, case):
                raise CaseRejected(f"seed {case.seed} has no safe route",
                                   code="plan_no_safe_route",
                                   details={"planned_nodes": 23})

        outcome = _call(Reject(), Case(0, 17, 2))
        self.assertEqual(outcome["code"], "plan_no_safe_route")
        self.assertIn("seed 17", outcome["reason"])
        self.assertEqual(outcome["details"], {"planned_nodes": 23})

    def test_world_and_decision_randomness_are_independent_and_reproducible(self):
        plain = list(RandomCases().sample(5, 9))
        varied = list(RandomCases(variables={
            "branch_length": {"integer": [6, 14]},
        }).sample(5, 9))
        self.assertEqual([case.seed for case in plain],
                         [case.seed for case in varied])
        self.assertEqual([case.decision_seed for case in plain],
                         [case.decision_seed for case in varied])

    def test_unique_seed_range_cannot_loop_forever(self):
        with self.assertRaisesRegex(ValueError, "unique world seeds"):
            list(RandomCases(world_seed_min=0,
                             world_seed_max=2).sample(3, 0))

    def test_trial_generator_runs_an_import_path_harness(self):
        root = Path(__file__).resolve().parents[1]
        run = root / "tests" / "fixtures" / "tiny_task" / "run.yaml"
        trajectory = TrialGenerator(str(run))(Case(0, 7, 11))
        self.assertEqual(trajectory.reward.total, 1.0)
        self.assertEqual(trajectory.metadata["generation_case"]["seed"], 7)

    def test_jsonl_sink_does_not_leave_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"

            def records():
                yield {"ok": 1}
                raise RuntimeError("generation failed")

            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                JSONLWriter(output).write(records())
            self.assertFalse(output.exists())

    def test_jsonl_sink_preserves_existing_output_on_late_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"
            output.write_text("previous-complete-output\n")

            def records():
                yield {"ok": 1}
                raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                JSONLWriter(output).write(records())
            self.assertEqual(output.read_text(), "previous-complete-output\n")
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_task_variant_generator_writes_exact_requested_count(self):
        root = Path(__file__).resolve().parents[1]
        template = root / "examples" / "farm_one_log_sdg" / "task.yaml"
        with tempfile.TemporaryDirectory() as directory:
            document = {
                "count": 3,
                "seed": 2026,
                "sampler": {
                    "type": "bedrock_rl.sdg:RandomCases",
                    "variables": {"max_turns": {"integer": [7, 9]}},
                },
                "generator": {
                    "type": ("bedrock_rl.sdg:"
                             "TaskVariantGenerator"),
                    "template": str(template),
                    "bindings": {"max_turns": "values.max_turns"},
                },
                "executor": {
                    "type": "bedrock_rl.sdg:SerialExecutor"},
                "writer": "bedrock_rl.sdg:JSONLWriter",
                "output": str(Path(directory) / "tasks.jsonl"),
            }
            spec = GenerationSpec.from_dict(document, path=str(template))
            result = spec.execute()
            rows = [json.loads(line) for line in
                    Path(result["output"]).read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(result["acceptance_rate"], 1.0)
        self.assertEqual(result["rejection_reasons"], [])
        self.assertEqual(len({row["case"]["seed"] for row in rows}), 3)
        self.assertTrue(all(row["task"]["max_turns"] ==
                            row["case"]["values"]["max_turns"]
                            for row in rows))
        self.assertTrue(all(row["task"]["success"]["type"] ==
                            "item_gained" for row in rows))

    def test_movement_profile_caps_each_closed_loop_step(self):
        profile = MovementProfile(
            min_blocks=2.0, max_blocks=3.0,
            min_pause_ticks=1, max_pause_ticks=1)
        actions = profile.actions(__import__("random").Random(7))
        self.assertEqual(actions[-1], {"action": "wait", "ticks": 1})
        movement = actions[0]
        self.assertLessEqual(
            movement["ticks"] / profile.ticks_per_block, 3.0)
        self.assertGreaterEqual(
            movement["ticks"] / profile.ticks_per_block, 2.0 - 0.1)

    def test_movement_profile_allows_explicit_fine_positioning(self):
        profile = MovementProfile(min_blocks=2.0, max_blocks=3.0)
        actions = profile.actions(
            __import__("random").Random(7), blocks=0.2)
        self.assertEqual(actions[0], {
            "action": "key", "k": "w", "ticks": 1})

    def test_policy_controls_have_direct_and_compatible_imports(self):
        self.assertIs(MovementProfile, PolicyMovementProfile)
        self.assertIs(SurvivalBehaviors, PolicySurvivalBehaviors)
        self.assertIs(SurvivalPolicy, PolicySurvivalPolicy)
        self.assertIs(PolicyVeto, DirectPolicyVeto)
        self.assertIs(movement_actions, direct_movement_actions)

    def test_policy_movement_vetoes_an_observed_hazard(self):
        policy = SurvivalPolicy(movement={
            "min_blocks": 2.0, "max_blocks": 2.0,
            "min_pause_ticks": 0, "max_pause_ticks": 0,
        })
        policy.reset(None, None, Case(0, 7, 11, {}))
        env = SimpleNamespace(obs={
            "x": 0.0, "y": 64.0, "z": 0.0,
            "blocks": [(11, 1, 64, 0)],
        })

        result = policy.movement_actions(env, 4.0, 0.0)

        self.assertIsInstance(result, PolicyVeto)
        self.assertEqual(result.code, "observed_hazard")
        self.assertEqual(result.observed, ((1, 64, 0),))
        self.assertEqual(result.to_dict()["target"], [4.0, 0.0])

    def test_policy_movement_caps_to_target_and_supports_edge_sneak(self):
        policy = SurvivalPolicy(movement={
            "min_blocks": 2.0, "max_blocks": 2.0,
            "min_pause_ticks": 0, "max_pause_ticks": 0,
        }, behaviors={"hazards": False})
        policy.reset(None, None, Case(0, 7, 11, {}))
        env = SimpleNamespace(obs={
            "x": 0.0, "y": 64.0, "z": 0.0, "blocks": [],
        })

        actions = policy.movement_actions(env, 1.0, 0.0, sneak=True)

        self.assertEqual(actions, [{
            "action": "key", "k": ["w", "sneak"], "ticks": 6,
        }])
        env.obs["x"] = 0.98
        self.assertIsNone(policy.movement_actions(env, 1.0, 0.0))

    def test_movement_spacing_and_timing_can_vary_per_case(self):
        profile = MovementProfile().vary({
            "movement_min_blocks": 1.25,
            "movement_max_blocks": 2.5,
            "movement_ticks_per_block": 7.0,
            "movement_min_pause_ticks": 2,
            "movement_max_pause_ticks": 5,
        })
        self.assertEqual(
            (profile.min_blocks, profile.max_blocks,
             profile.ticks_per_block, profile.min_pause_ticks,
             profile.max_pause_ticks),
            (1.25, 2.5, 7.0, 2, 5))

    def test_survival_behaviors_are_composable_and_strict(self):
        behaviors = SurvivalBehaviors.parse({
            "hazards": {"blocks": ["lava"], "clearance_blocks": 0.75},
            "descent": False,
            "pickups": {"attempts": 7, "wait_ticks": 5,
                        "step_blocks": 0.4},
            "lighting": {"every_blocks": 12, "item": "torch",
                         "pause_ticks": 3},
        })
        self.assertEqual(behaviors.hazard_blocks, ("lava",))
        self.assertEqual(behaviors.hazard_clearance_blocks, 0.75)
        self.assertFalse(behaviors.safe_descent)
        self.assertEqual((behaviors.pickup_attempts,
                          behaviors.pickup_wait_ticks), (7, 5))
        self.assertEqual(behaviors.pickup_step_blocks, 0.4)
        self.assertEqual(behaviors.light_every_blocks, 12)
        self.assertEqual(behaviors.light_pause_ticks, 3)
        with self.assertRaisesRegex(TypeError, "unknown survival"):
            SurvivalBehaviors.parse({"combat": {"enabled": True}})

    def test_survival_policy_applies_case_variance_and_tracks_lighting(self):
        policy = SurvivalPolicy(behaviors={
            "lighting": {"every_blocks": 10},
            "pickups": {"attempts": 3, "wait_ticks": 8},
        })
        policy.reset(None, None, Case(0, 7, 11, {
            "lighting_every_blocks": 6,
            "pickup_wait_ticks": 4,
            "pickup_step_blocks": 0.25,
        }))
        self.assertTrue(policy.lighting_due(0))
        policy.mark_lit(0)
        self.assertFalse(policy.lighting_due(0))
        self.assertTrue(policy.lighting_due(6))
        self.assertEqual(policy.pickup_wait(0), [
            {"action": "wait", "ticks": 4}])
        self.assertEqual(policy.behaviors.pickup_step_blocks, 0.25)
        self.assertIsNone(policy.pickup_wait(3))
        self.assertFalse(policy.is_hazard(1))
        self.assertTrue(policy.is_hazard(10))

    def test_survival_policy_tracks_real_travel_after_lights_exist(self):
        policy = SurvivalPolicy(behaviors={
            "lighting": {"every_blocks": 10},
        })
        policy.reset(None, None, Case(0, 7, 11, {}))
        env = SimpleNamespace(obs={"x": 0.0, "y": 20.0, "z": 0.0})
        policy.start_route_lighting(env)
        self.assertTrue(policy.route_light_due())
        policy.route_light_placed()
        self.assertFalse(policy.route_light_due())

        env.obs.update(x=6.0)
        policy.observe_route_travel(env)
        env.obs.update(z=4.0)
        policy.observe_route_travel(env)
        self.assertTrue(policy.route_light_due())

        policy.route_light_placed()
        env.obs.update(x=100.0)
        policy.observe_route_travel(env)
        self.assertFalse(policy.route_light_due())

    def test_survival_policy_finds_only_current_public_inventory_slot(self):
        env = SimpleNamespace(journal=[
            Event("slot_changed", EVENTS["slot_changed"], 10,
                  (14, 318, 1, 0, 0, 0)),
            Event("slot_changed", EVENTS["slot_changed"], 11,
                  (14, 0, 0, 318, 1, 0)),
            Event("slot_changed", EVENTS["slot_changed"], 12,
                  (21, 318, 2, 0, 0, 0)),
        ])
        self.assertEqual(
            SurvivalPolicy.observed_inventory_slot(env, (318,)), 21)

    def test_task_variants_can_replace_arbitrary_initial_state_sections(self):
        root = Path(__file__).resolve().parents[1]
        template = root / "examples" / "farm_one_log_sdg" / "task.yaml"
        generator = TaskVariantGenerator(
            str(template),
            bindings={
                "initial_state.inventory": "values.inventory",
                "initial_state.player.health": "values.health",
                "initial_state.spawn.constraints": "values.spawn",
                "initial_state.world.time": "values.time",
            })
        case = Case(3, 17, 23, {
            "inventory": [{"item": "torch", "count": 12, "slot": 8}],
            "health": 14,
            "spawn": [{"block": "iron_ore", "min_distance": 2,
                       "max_distance": 10}],
            "time": 6000,
        })
        task = generator(case)["task"]
        state = task["initial_state"]
        self.assertEqual(state["inventory"], case.values["inventory"])
        self.assertEqual(state["player"]["health"], 14)
        self.assertEqual(state["spawn"]["constraints"], case.values["spawn"])
        self.assertEqual(state["world"]["time"], 6000)

    def test_process_executor_preflights_before_starting_workers(self):
        root = Path(__file__).resolve().parents[1]
        bad = ComponentSpec.parse({
            "type": "does.not.exist:Generator",
        })
        with patch("bedrock_rl.sdg.generation.get_context") as context:
            with self.assertRaises(ImportError):
                ProcessExecutor(workers=2).run(
                    [Case(0, 1, 2), Case(1, 3, 4)], bad,
                    base_dir=str(root), metadata={})
        context.assert_not_called()

    def test_auto_workers_respect_cpu_and_engine_memory_budgets(self):
        with patch("bedrock_rl.sdg.generation._available_cpu_count",
                   return_value=24), patch(
                       "bedrock_rl.sdg.generation._memory_worker_ceiling",
                       return_value=7):
            self.assertEqual(_worker_ceiling("auto"), 7)
            # An explicit measurement-based choice remains an override.
            self.assertEqual(_worker_ceiling(40), 40)

        # Large generation hosts are not silently stranded at 32 workers.
        with patch("bedrock_rl.sdg.generation._available_cpu_count",
                   return_value=96), patch(
                       "bedrock_rl.sdg.generation._memory_worker_ceiling",
                       return_value=128):
            self.assertEqual(_worker_ceiling("auto"), 96)

        # If the platform cannot report memory, CPU availability remains a
        # finite bound rather than falling back to an arbitrary machine-wide
        # process cap.
        with patch("bedrock_rl.sdg.generation._memory_limit_bytes",
                   return_value=None), patch(
                       "bedrock_rl.sdg.generation._available_cpu_count",
                       return_value=72):
            from bedrock_rl.sdg.generation import _memory_worker_ceiling
            self.assertEqual(_memory_worker_ceiling(), 72)

    def test_environment_gpu_list_is_validated_and_pinned_round_robin(self):
        class Counter:
            def __init__(self):
                self.value = 0
                self.lock = threading.Lock()

            def get_lock(self):
                return self.lock

        with patch.dict(os.environ, {"NETHERITE_ENV_GPUS": "2, 4,7"}):
            devices = _environment_gpus()
            self.assertEqual(devices, ("2", "4", "7"))
            counter = Counter()
            assigned = []
            for _ in range(5):
                _pin_process_gpu(devices, counter)
                assigned.append(os.environ["NETHERITE_ENV_GPUS"])
            self.assertEqual(assigned, ["2", "4", "7", "2", "4"])
        with patch.dict(os.environ, {"NETHERITE_ENV_GPUS": "1,1"}):
            with self.assertRaisesRegex(ValueError, "must not repeat"):
                _environment_gpus()

    def test_auto_process_workers_are_capped_to_environment_gpus(self):
        pools = []

        class Counter:
            def __init__(self):
                self.value = 0
                self.lock = threading.Lock()

            def get_lock(self):
                return self.lock

        class Pool:
            def __init__(self, workers, initializer, initargs):
                self.workers = workers
                self.closed = self.terminated = False
                pools.append(self)
                initializer(*initargs)

            def apply_async(self, fn, args, callback, error_callback):
                try:
                    callback(fn(*args))
                except BaseException as exc:
                    error_callback(exc)

            def close(self):
                self.closed = True

            def terminate(self):
                self.terminated = True

            def join(self):
                pass

        context = SimpleNamespace(
            Pool=Pool, Value=lambda *_args: Counter())
        cases = [Case(index, index, index + 100) for index in range(8)]
        generator = ComponentSpec.parse(f"{__name__}:_PassGenerator")
        with patch.dict(os.environ, {"NETHERITE_ENV_GPUS": "2,4,7"}), \
                patch("bedrock_rl.sdg.generation.get_context",
                      return_value=context), patch(
                          "bedrock_rl.sdg.generation._worker_ceiling",
                          return_value=16):
            outcomes = list(ProcessExecutor(workers="auto").run(
                cases, generator, base_dir=".", metadata={}))
        self.assertEqual(pools[0].workers, 3)
        self.assertTrue(pools[0].closed)
        self.assertFalse(pools[0].terminated)
        self.assertEqual(len(outcomes), len(cases))

    @unittest.skipIf(os.name == "nt", "forkserver is a POSIX contract")
    def test_process_executor_runs_real_bounded_pool(self):
        generator = ComponentSpec.parse(f"{__name__}:_PassGenerator")
        outcomes = list(ProcessExecutor(workers=2).run(
            [Case(index, index + 10, index + 100) for index in range(6)],
            generator, base_dir=".", metadata={}))
        self.assertEqual([value["case"]["seed"] for value in outcomes],
                         list(range(10, 16)))
        self.assertTrue(all(value["ok"] for value in outcomes))

    @unittest.skipIf(os.name == "nt", "forkserver is a POSIX contract")
    def test_process_workers_own_parallelism_instead_of_blas(self):
        generator = ComponentSpec.parse(
            f"{__name__}:_EnvironmentGenerator")
        outcomes = list(ProcessExecutor(workers=2).run(
            [Case(0, 1, 2), Case(1, 3, 4)], generator,
            base_dir=".", metadata={}))
        self.assertTrue(all(outcome["ok"] for outcome in outcomes))
        self.assertTrue(all(
            set(outcome["value"].values()) == {"1"}
            for outcome in outcomes))

    def test_thread_order_window_counts_completed_later_results(self):
        release_first = threading.Event()
        later_finished = threading.Event()
        consumed = []
        received = []

        def values():
            for index in range(100):
                consumed.append(index)
                yield index

        def work(index):
            if index == 0:
                if not release_first.wait(2):
                    raise RuntimeError("test did not release first result")
            elif index == 3:
                later_finished.set()
            return index

        outcomes = _thread_map_bounded(
            values(), work, workers=2, prefetch=2)
        consumer = threading.Thread(
            target=lambda: received.append(next(outcomes)), daemon=True)
        consumer.start()
        self.assertTrue(later_finished.wait(2))
        # Results 1-3 are complete, but they still occupy the four-result
        # ordered window while result 0 is blocked. No fifth value may be
        # consumed/submitted merely because a later future completed.
        self.assertEqual(consumed, [0, 1, 2, 3])

        release_first.set()
        consumer.join(2)
        self.assertFalse(consumer.is_alive())
        self.assertEqual(received, [0])
        # Emitting result 0 frees exactly one window slot before yielding.
        self.assertEqual(consumed, [0, 1, 2, 3, 4])
        outcomes.close()

    def test_process_order_window_counts_completed_later_results(self):
        submitted, pools = [], []
        all_submitted = threading.Event()

        class Pool:
            def __init__(self, workers, initializer, initargs):
                self.closed = False
                self.terminated = False
                pools.append(self)
                initializer(*initargs)

            def apply_async(self, fn, args, callback, error_callback):
                submitted.append((args[0].index, fn, args, callback,
                                  error_callback))
                if len(submitted) == 8:
                    all_submitted.set()

            def close(self):
                self.closed = True

            def terminate(self):
                self.terminated = True

            def join(self):
                pass

        def finish(index):
            _, fn, args, callback, error_callback = submitted[index]
            try:
                callback(fn(*args))
            except BaseException as exc:
                error_callback(exc)

        context = SimpleNamespace(Pool=Pool)
        cases = (Case(index, index, index + 100)
                 for index in range(100))
        generator = ComponentSpec.parse(f"{__name__}:_PassGenerator")
        with patch("bedrock_rl.sdg.generation.get_context",
                   return_value=context):
            outcomes = ProcessExecutor(
                workers=2, chunksize=4).run(
                    cases, generator, base_dir=".", metadata={})
            received = []
            consumer = threading.Thread(
                target=lambda: received.append(next(outcomes)), daemon=True)
            consumer.start()
            self.assertTrue(all_submitted.wait(2))
            for index in range(1, 8):
                finish(index)
            # Seven completed image-heavy results remain bounded behind slow
            # ordinal 0; completion cannot refill the eight-result window.
            self.assertEqual(len(submitted), 8)

            finish(0)
            consumer.join(2)
            self.assertFalse(consumer.is_alive())
            self.assertEqual(received[0]["case"]["index"], 0)
            self.assertEqual(len(submitted), 9)
            outcomes.close()
        self.assertTrue(pools[0].terminated)
        self.assertFalse(pools[0].closed)

    def test_serial_and_thread_executors_close_shared_generator(self):
        generator = ComponentSpec.parse(f"{__name__}:_ClosableGenerator")
        for executor in (SerialExecutor(), ThreadExecutor(workers=2)):
            with self.subTest(executor=type(executor).__name__):
                _CLOSE_EVENTS.clear()
                outcomes = executor.run(
                    (Case(index, index, index + 100)
                     for index in range(10_000)),
                    generator, base_dir=".", metadata={})
                self.assertTrue(next(outcomes)["ok"])
                outcomes.close()
                self.assertEqual(_CLOSE_EVENTS, ["closed"])

    def test_thread_executor_refuses_a_stateful_generator(self):
        generator = ComponentSpec.parse(f"{__name__}:_ThreadUnsafeGenerator")
        outcomes = ThreadExecutor(workers=2).run(
            [Case(0, 1, 2)], generator, base_dir=".", metadata={})

        with self.assertRaisesRegex(TypeError, "stateful.*ProcessExecutor"):
            next(outcomes)

    def test_process_executor_prefetch_is_bounded_and_terminates_tail(self):
        submitted, pools = [], []

        class Pool:
            def __init__(self, workers, initializer, initargs):
                self.workers = workers
                self.closed = False
                self.terminated = False
                pools.append(self)
                initializer(*initargs)

            def apply_async(self, fn, args, callback, error_callback):
                submitted.append(args[0].index)
                try:
                    callback(fn(*args))
                except BaseException as exc:
                    error_callback(exc)

            def close(self):
                self.closed = True

            def terminate(self):
                self.terminated = True

            def join(self):
                pass

        context = SimpleNamespace(Pool=Pool)
        cases = (Case(index, index, index + 100)
                 for index in range(10_000))
        generator = ComponentSpec.parse(f"{__name__}:_PassGenerator")
        with patch("bedrock_rl.sdg.generation.get_context",
                   return_value=context):
            outcomes = ProcessExecutor(
                workers=2, chunksize=1).run(
                    cases, generator, base_dir=".", metadata={})
            first = next(outcomes)
            self.assertTrue(first["ok"])
            # Two workers means a two-case active window. One replacement is
            # submitted before the first result is yielded, but not 10,000
            # materialized cases. Closing exact-count generation kills it.
            self.assertEqual(submitted, [0, 1, 2])
            outcomes.close()
        self.assertEqual(len(pools), 1)
        self.assertTrue(pools[0].terminated)
        self.assertFalse(pools[0].closed)

    @unittest.skipIf(os.name == "nt", "forkserver is a POSIX contract")
    def test_process_heartbeat_reports_while_ordered_head_is_running(self):
        generator = ComponentSpec.parse(
            f"{__name__}:_SlowFirstGenerator")
        heartbeats = []
        outcomes = ProcessExecutor(
            workers=2, heartbeat_seconds=0.01).run(
                [Case(0, 1, 2), Case(1, 3, 4)], generator,
                base_dir=".", metadata={}, heartbeat=heartbeats.append)
        try:
            self.assertTrue(next(outcomes)["ok"])
        finally:
            outcomes.close()
        self.assertTrue(heartbeats)
        self.assertEqual(
            set(heartbeats[0]),
            {"submitted", "completed", "committed", "running"})
        self.assertGreaterEqual(heartbeats[-1]["submitted"], 2)

    def test_process_executor_fails_when_worker_dies_without_callback(self):
        pools = []

        class Pool:
            def __init__(self, workers, initializer, initargs):
                del workers
                initializer(*initargs)
                self._pool = [SimpleNamespace(pid=4321, exitcode=-9)]
                self.terminated = False
                pools.append(self)

            def apply_async(self, fn, args, callback, error_callback):
                del fn, args, callback, error_callback

            def close(self):
                raise AssertionError("dead worker pool cannot close cleanly")

            def terminate(self):
                self.terminated = True

            def join(self):
                pass

        generator = ComponentSpec.parse(f"{__name__}:_PassGenerator")
        with patch("bedrock_rl.sdg.generation.get_context",
                   return_value=SimpleNamespace(Pool=Pool)):
            outcomes = ProcessExecutor(
                workers=2, heartbeat_seconds=0.001).run(
                    [Case(0, 1, 2), Case(1, 3, 4)], generator,
                    base_dir=".", metadata={})
            with self.assertRaisesRegex(RuntimeError,
                                        "worker exited.*-9"):
                next(outcomes)
        self.assertTrue(pools[0].terminated)

    def test_process_executor_terminates_and_joins_on_keyboard_interrupt(self):
        pools = []

        class Pool:
            def __init__(self, workers, initializer, initargs):
                self.terminated = False
                self.joined = False
                pools.append(self)
                initializer(*initargs)

            def apply_async(self, fn, args, callback, error_callback):
                error_callback(KeyboardInterrupt())

            def close(self):
                raise AssertionError("interrupted pool must not close cleanly")

            def terminate(self):
                self.terminated = True

            def join(self):
                self.joined = True

        generator = ComponentSpec.parse(f"{__name__}:_PassGenerator")
        with patch("bedrock_rl.sdg.generation.get_context",
                   return_value=SimpleNamespace(Pool=Pool)):
            outcomes = ProcessExecutor(workers=2).run(
                [Case(0, 1, 2), Case(1, 3, 4)], generator,
                base_dir=".", metadata={})
            with self.assertRaises(KeyboardInterrupt):
                next(outcomes)
        self.assertTrue(pools[0].terminated)
        self.assertTrue(pools[0].joined)
