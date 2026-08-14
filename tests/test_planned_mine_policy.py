import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bedrock_rl.sdg.policy.planned_mine import PlannedMinePolicy
from bedrock_rl.sdg.policy import MinecraftController
from bedrock_rl.sdg import ScriptedEpisodeGenerator
from bedrock_rl.sdg.generation import Case, CaseRejected
from bedrock_rl.env.task import Task
from tests.test_world_planning import write_planning_snapshot


class Event(dict):
    def __init__(self, name, **values):
        super().__init__(values)
        self.name = name


class FakeEnv:
    def __init__(self, snapshot, shift=0):
        self.snapshot_in = str(snapshot)
        self.obs = {"x": shift + 0.5, "y": 1.0, "z": 0.5}
        self.journal = []


class FakeController:
    def __init__(self):
        self.moves = []
        self.mines = []

    def reset(self, task, episode, case):
        self.reset_values = task, episode, case

    def face_and_move_to(self, env, x, z, **kwargs):
        self.moves.append((x, z, kwargs))
        env.obs["x"], env.obs["z"] = x, z
        return [{"action": "key", "k": "w", "ticks": 2}]

    @staticmethod
    def observed_block(env, cell):
        del env, cell
        return None

    @staticmethod
    def placed(env, cell, item):
        del env, cell, item
        return False

    @staticmethod
    def drop_is_safe(current_y, landing_y):
        return current_y - landing_y <= 1

    @staticmethod
    def _block_ids(value):
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(17 if item == "log" else int(item)
                     for item in values)

    @staticmethod
    def broken(env, cell, block=None):
        del block
        return any(event.name == "block_broken"
                   and tuple(event[key] for key in ("x", "y", "z")) == cell
                   for event in env.journal)

    def mine_exact(self, env, cell, **kwargs):
        self.mines.append((cell, kwargs))
        return [{"action": "left_click_hold", "ticks": kwargs["ticks"]}]


def task_file(directory):
    path = Path(directory) / "task.yaml"
    path.write_text("""\
name: planned_tree_test
goal: Break two logs.
seeds: [0]
initial_state:
  spawn:
    constraints:
      - {block: log, min_distance: 2, max_distance: 4}
success: {type: block_broken, block: log, count: 2, reward: 1}
""")
    return path


def execute(policy, task, episode, case, limit=100):
    outputs = []
    for turn in range(limit):
        value = policy.actions(task, episode, case, turn)
        if value is None:
            break
        outputs.append(value)
        if value[0]["action"] == "left_click_hold":
            cell = policy.controller.mines[-1][0]
            episode.env.journal.append(Event(
                "block_broken", block=17,
                x=cell[0], y=cell[1], z=cell[2]))
    else:
        raise AssertionError("policy did not finish")
    return outputs


class PlannedMinePolicyTests(unittest.TestCase):
    def test_default_uses_public_minecraft_controller(self):
        self.assertIsInstance(PlannedMinePolicy().controller,
                              MinecraftController)

    def test_task_drives_plan_and_journal_alone_drives_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "world.bsnp"
            write_planning_snapshot(snapshot)
            task = Task.load(task_file(directory))
            episode = SimpleNamespace(env=FakeEnv(snapshot))
            case = Case(0, 77, 991, {})
            controller = FakeController()
            policy = PlannedMinePolicy(controller=controller)
            policy.reset(task, episode, case)

            outputs = execute(policy, task, episode, case)

        self.assertTrue(outputs)
        self.assertEqual(policy.stage, "complete")
        self.assertEqual(policy.diagnostics(episode.env)["verified_breaks"], 2)
        self.assertEqual(len(controller.mines), 2)
        # A valid plan may mine multiple visible trunk blocks from the start
        # stance without fabricating a movement action. Route continuity,
        # exact target clicks, and journal-proven progress are the invariant.
        targets = [target for resource in policy.plan.resources
                   for cluster in resource.clusters
                   for target in cluster.targets]
        self.assertEqual(targets[0].route.stands[0], policy.plan.start)
        self.assertTrue(all(target.route.stands[-1]
                            == target.mining_stance
                            for target in targets))
        self.assertEqual(
            {cell for cell, _kwargs in controller.mines},
            {target.cell for target in targets})
        self.assertEqual(policy.provenance()["decision_seed"], 991)
        self.assertNotIn("policy_plan", str(outputs))

    def test_translated_snapshots_translate_actions_without_constants(self):
        plans = []
        with tempfile.TemporaryDirectory() as directory:
            task = Task.load(task_file(directory))
            for index, shift in enumerate((0, 40)):
                snapshot = Path(directory) / f"world-{shift}.bsnp"
                write_planning_snapshot(
                    snapshot, shift=shift, seed=100 + index)
                episode = SimpleNamespace(env=FakeEnv(snapshot, shift))
                controller = FakeController()
                policy = PlannedMinePolicy(controller=controller)
                case = Case(index, 100 + index, 1234, {})
                policy.reset(task, episode, case)
                execute(policy, task, episode, case)
                plans.append((policy, controller))

        left, right = plans
        self.assertEqual(
            [(cell[0] + 40, cell[1], cell[2])
             for cell, _kwargs in left[1].mines],
            [cell for cell, _kwargs in right[1].mines])
        self.assertEqual(
            [(x + 40, z) for x, z, _kwargs in left[1].moves],
            [(x, z) for x, z, _kwargs in right[1].moves])
        self.assertNotEqual(left[0].provenance()["snapshot_sha256"],
                            right[0].provenance()["snapshot_sha256"])

    def test_rejects_non_block_break_objectives(self):
        task = SimpleNamespace(success=SimpleNamespace(
            type="selected_item", spec={}))
        with self.assertRaises(CaseRejected) as raised:
            PlannedMinePolicy()._objective(task)
        self.assertEqual(raised.exception.code,
                         "planned_mine_unsupported_success")

    def test_item_goal_mines_and_collects_the_matching_natural_block(self):
        task = SimpleNamespace(success=SimpleNamespace(
            type="item_gained", spec={"item": "log", "count": 1}))
        policy = PlannedMinePolicy(controller=FakeController())

        label, blocks, count, drop_item = policy._objective(task)

        self.assertEqual((label, blocks, count, drop_item),
                         ("log", ("log",), 1, "log"))

    def test_generator_saves_plan_in_provenance_not_messages(self):
        plan = {
            "schema": "bedrock.world_plan.v1",
            "snapshot_sha256": "secret-planning-evidence",
        }
        policy = SimpleNamespace(
            reset=Mock(), actions=Mock(return_value=None),
            provenance=Mock(return_value=plan), close=Mock())
        episode = SimpleNamespace(
            env=SimpleNamespace(journal=[]), t0_png=None,
            view_provenance={"type": "test"}, success=True,
            turns=0, final_reward=lambda: 1.0, close=Mock(),
            spec={
                "task_yaml": str(
                    (Path(__file__).resolve().parents[1] / "examples" /
                     "farm_one_log_sdg" / "task.yaml")),
                "seed": 77, "init_yaw": 1.5, "init_pitch": -2.5,
                "view": {"type": "test"},
            })
        generator = ScriptedEpisodeGenerator.__new__(
            ScriptedEpisodeGenerator)
        generator.task = SimpleNamespace(
            name="tree", goal="break logs", max_lines=4,
            max_ticks=100, raw={}, view=SimpleNamespace(type="test"))
        generator.task_path = Path("task.yaml")
        generator.policy = SimpleNamespace(
            type="tests:Policy", create=lambda: policy)
        generator.netherite_home = None
        generator.frames_root = "/tmp/unused"
        generator.include_images = False
        generator.record_views = False
        generator.semantic_camera = False
        generator.recording_every = None
        generator.max_turns = 1
        generator.metadata = {}

        with (patch(
                "bedrock_rl.sdg.pipeline.Episode.draw",
                return_value=episode),
              patch(
                "bedrock_rl.sdg.pipeline.cu_user_prompt",
                return_value="break logs"),
              patch(
                "bedrock_rl.sdg.pipeline._episode_state",
                return_value={"episode": {
                    "done": True, "success": True, "failed": False,
                    "turns": 0, "actions": 0, "reward": 1.0,
                }})):
            trajectory = generator(Case(0, 77, 991, {}))

        self.assertEqual(trajectory.provenance["policy_plan"], plan)
        episode_spec = trajectory.provenance["episode_spec"]
        self.assertEqual(episode_spec["task_yaml"],
                         "examples/farm_one_log_sdg/task.yaml")
        self.assertEqual(episode_spec["seed"], 77)
        self.assertNotIn(
            "spec", trajectory.state.samples[0].values["episode"])
        self.assertNotIn(str(Path(__file__).resolve().parents[1]),
                         str(trajectory.to_dict()))
        self.assertNotIn("secret-planning-evidence",
                         str(trajectory.messages))


if __name__ == "__main__":
    unittest.main()
