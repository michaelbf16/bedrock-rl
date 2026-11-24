import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from bedrock_rl.env.episode import Episode
from bedrock_rl.env.task import Task
from bedrock_rl.core.trajectory import Trajectory
from bedrock_rl.sdg.generation import Case
from bedrock_rl.sdg.media import _load_recorded_task
from bedrock_rl.sdg.pipeline import (
    EpisodeDatasetGenerator, ScriptedEpisodeGenerator, load_episode_specs,
)


DOCUMENT = {
    "name": "recorded", "goal": "select the pickaxe", "view": "semantic",
    "initial_state": {"world": {"time": 1000}},
    "snapshots_dir": "cache",
    "success": {"type": "selected_item", "items": ["iron_pickaxe"]},
}


class TaskDefinitionTests(unittest.TestCase):
    def test_generated_bound_task_replays_after_template_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objective.yaml"
            path.write_text(yaml.safe_dump(DOCUMENT))
            generator = ScriptedEpisodeGenerator(
                str(path), "bedrock_rl.sdg:ExpertPolicy", include_images=False,
                task_bindings={"initial_state.world.time": "values.time",
                               "success.items": "values.items"})
            generator.policy = SimpleNamespace(
                type="test:NoOp", create=lambda: SimpleNamespace(
                    actions=lambda *_: None))
            episode = SimpleNamespace(
                spec={"task_yaml": str(path), "seed": 1,
                      "init_yaw": 0, "init_pitch": 0},
                t0_png=None, view_provenance={},
                env=SimpleNamespace(journal=[]), success=True,
                final_reward=lambda: 1.0, turns=1, close=Mock())
            with patch("bedrock_rl.sdg.pipeline.Episode.draw",
                       return_value=episode), patch(
                    "bedrock_rl.sdg.pipeline.cu_user_prompt", return_value="go"), \
                    patch("bedrock_rl.sdg.pipeline._episode_state",
                          return_value={}):
                trajectory = generator(Case(
                    0, 1, 2, {"time": 13000, "items": ["stone_pickaxe"]}))
            spec = trajectory.provenance["episode_spec"]
            self.assertEqual(spec["task_definition"]["sha256"],
                             trajectory.provenance["task_digest"])
            path.unlink()
            replay = _load_recorded_task(trajectory, spec)
            self.assertEqual(replay.initial_state.world["time"], 13000)
            self.assertEqual(replay.success.spec["items"], ["stone_pickaxe"])
            self.assertEqual(replay.snapshots_dir,
                             str(Path(directory).resolve() / "cache"))

    def test_resource_base_relocates_with_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old, new = root / "old", root / "new"
            with patch("bedrock_rl.env.task.resources.repo_root", return_value=old):
                task = Task(copy.deepcopy(DOCUMENT),
                            path=str(old / "custom/objective.yaml"))
                definition = task.definition()
            self.assertEqual(definition["resource_base"], "custom")
            with patch("bedrock_rl.env.task.resources.repo_root", return_value=new):
                restored = Task.from_definition(definition)
            self.assertEqual(restored.path, str(new / "custom/objective.yaml"))
            self.assertEqual(restored.snapshots_dir, str(new / "custom/cache"))

    def test_rl_prompt_row_records_the_same_definition_for_tool_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(yaml.safe_dump(DOCUMENT))
            generator = EpisodeDatasetGenerator(str(path))
            episode = SimpleNamespace(
                spec={"task_yaml": str(path), "seed": 1,
                      "init_yaw": 0, "init_pitch": 0},
                env=SimpleNamespace(), t0_png=b"image", close=Mock())
            with patch("bedrock_rl.sdg.pipeline.Episode.draw",
                       return_value=episode), patch(
                    "bedrock_rl.sdg.pipeline.cu_user_prompt", return_value="go"):
                row = generator(Case(0, 1, 2))
            spec = row["extra_info"]
            tool_spec = spec["tools_kwargs"]["computer"]["create_kwargs"]
            self.assertEqual(spec["task_definition"],
                             tool_spec["task_definition"])
            self.assertIsInstance(spec["task_definition"], str)
            self.assertEqual(spec["task_digest"],
                             json.loads(spec["task_definition"])["sha256"])
            self.assertEqual(Task.from_definition(
                spec["task_definition"]).initial_state.world["time"], 1000)

    def test_task_definition_survives_pandas_parquet_arrays(self):
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pandas and pyarrow are optional")
        task = Task(copy.deepcopy(DOCUMENT), path="objective.yaml")
        definition = task.definition()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.parquet"
            pq.write_table(pa.Table.from_pylist([{"extra_info": {
                "task_definition": definition}}]), path)
            restored_definition = pd.read_parquet(path).iloc[0][
                "extra_info"]["task_definition"]
            restored = Task.from_definition(restored_definition)
        self.assertEqual(restored.success.spec["items"], ["iron_pickaxe"])
        self.assertEqual(restored.definition()["sha256"], definition["sha256"])

    def test_json_definitions_preserve_distinct_task_fields_in_parquet(self):
        try:
            import datasets  # noqa: F401
            import pandas as pd
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("datasets, pandas and pyarrow are optional")
        from bedrock_rl.data import write_rows
        first = Task(copy.deepcopy(DOCUMENT), path="first.yaml")
        other = copy.deepcopy(DOCUMENT)
        other["max_turns"] = 8
        other["initial_state"]["inventory"] = [{"item": "log", "count": 4}]
        second = Task(other, path="second.yaml")
        definitions = [task.definition() for task in (first, second)]
        rows = [{"extra_info": {
            "task_yaml": "missing.yaml", "seed": 7,
            "init_yaw": 0, "init_pitch": 0,
            "task_definition": json.dumps(definition),
            "task_digest": definition["sha256"],
        }} for definition in definitions]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.parquet"
            write_rows(rows, path, batch_size=1)
            saved = pd.read_parquet(path)
            specs = load_episode_specs(path)
        for row, definition in zip(saved["extra_info"], definitions):
            self.assertEqual(Task.from_definition(
                row["task_definition"]).definition()["document"],
                definition["document"])
        self.assertEqual([spec["task_digest"] for spec in specs],
                         [item["sha256"] for item in definitions])
        self.assertEqual([Task.from_definition(spec["task_definition"])
                          .definition()["document"] for spec in specs],
                         [item["document"] for item in definitions])

    def test_parquet_specs_keep_nested_frozen_view_and_support_legacy_rows(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is optional")
        view = {"type": "semantic", "engine_frame_size": [428, 240],
                "engine_assets": {"kind": "procedural"}}
        info = {"task_yaml": "missing.yaml", "seed": 5, "init_yaw": 0,
                "init_pitch": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.parquet"
            for extra in ({}, {"view": view}):
                pq.write_table(pa.Table.from_pylist([
                    {"extra_info": {**info, **extra}}]), path)
                self.assertEqual(load_episode_specs(path)[0].get("view"),
                                 extra.get("view"))

    def test_environment_adapter_constructs_parser_from_recorded_task(self):
        from bedrock_rl.adapters.netherite.environment import NetheriteEnvironment
        task = Task(copy.deepcopy(DOCUMENT), path="missing.yaml")
        spec = {"task_yaml": task.path, "seed": 7,
                "init_yaw": 0, "init_pitch": 0,
                "task_definition": json.dumps(task.definition())}
        environment = NetheriteEnvironment(task.path, episode=spec)
        with patch("bedrock_rl.adapters.netherite.environment.cu_parser",
                   return_value="parser") as parser, patch(
                "bedrock_rl.adapters.netherite.environment.Episode.from_spec",
                return_value=object()) as open_episode, patch.object(
                Task, "load", side_effect=AssertionError("not needed")):
            session = environment._open_blocking(task.path)
        self.assertEqual(session.task.definition()["sha256"],
                         task.definition()["sha256"])
        self.assertIs(parser.call_args.args[0], session.task)
        self.assertIs(open_episode.call_args.kwargs["task"], session.task)

    def test_tampered_definition_and_conflicting_digest_are_rejected(self):
        task = Task(copy.deepcopy(DOCUMENT), path="objective.yaml")
        definition = task.definition()
        changed = copy.deepcopy(definition)
        changed["document"]["initial_state"]["world"]["time"] = 13000
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            Task.from_definition(changed)
        with self.assertRaisesRegex(ValueError, "disagrees with definition"):
            _load_recorded_task(Trajectory(), {
                "task_definition": definition, "task_digest": "0" * 64})

    def test_legacy_replay_refuses_changed_task_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.yaml"
            path.write_text(yaml.safe_dump(DOCUMENT))
            digest = Task.load(path).definition()["sha256"]
            document = copy.deepcopy(DOCUMENT)
            document["initial_state"]["world"]["time"] = 13000
            path.write_text(yaml.safe_dump(document))
            with self.assertRaisesRegex(ValueError, "differs from the recorded"):
                _load_recorded_task(Trajectory(), {
                    "task_yaml": str(path), "task_digest": digest})

    def test_episode_from_spec_uses_definition_without_loading_template(self):
        class RecordedEpisode(Episode):
            def __init__(self, task, seed, yaw, pitch, **kwargs):
                self.task = task
                self.spec = {"view": {}}
                self.identity = (seed, yaw, pitch)

        task = Task(copy.deepcopy(DOCUMENT), path="missing-objective.yaml")
        definition = task.definition()
        spec = {"task_yaml": task.path, "seed": 7,
                "init_yaw": 3, "init_pitch": 4,
                "task_definition": definition,
                "task_digest": definition["sha256"]}
        with patch.object(Task, "load", side_effect=AssertionError("not needed")):
            replay = RecordedEpisode.from_spec(spec, parser=None)
            json_replay = RecordedEpisode.from_spec(
                {**spec, "task_definition": json.dumps(definition)}, parser=None)
        self.assertEqual(replay.identity, (7, 3.0, 4.0))
        self.assertEqual(json_replay.identity, replay.identity)
        self.assertEqual(replay.task.definition()["sha256"], definition["sha256"])
        other = copy.deepcopy(DOCUMENT)
        other["initial_state"]["world"]["time"] = 13000
        with self.assertRaisesRegex(ValueError, "supplied task differs"):
            RecordedEpisode.from_spec(spec, parser=None, task=Task(other))


if __name__ == "__main__":
    unittest.main()
