from types import SimpleNamespace
import json
from pathlib import Path
import struct
import unittest

import yaml

from bedrock_rl.adapters.netherite.chat import tool_schema
from bedrock_rl.adapters.netherite.computer import (
    compile_actions, cu_help, cu_suffix, validate_cu_example)
from bedrock_rl.adapters.netherite.probes import ObservationProbe
from bedrock_rl.env.task import Task


ROOT = Path(__file__).resolve().parents[1]


class MinecraftFrameworkTests(unittest.TestCase):
    def test_computer_example_respects_the_task_action_budget(self):
        suffix = cu_suffix(1)
        payload = suffix.split("<tool_call>\n", 1)[1].split(
            "\n</tool_call>", 1)[0]
        actions = json.loads(payload)["arguments"]["actions"]
        self.assertEqual(len(actions), 1)
        _, count = compile_actions(
            actions, pitch=0, max_actions=1, max_ticks=1)
        self.assertEqual(count, 1)

    def test_every_shipped_prompt_example_respects_its_real_task_parser(self):
        task_files = sorted((ROOT / "examples").glob("*/task.yaml"))
        self.assertTrue(task_files)
        for path in task_files:
            with self.subTest(task=path.parent.name):
                validate_cu_example(Task.load(path))

    def test_cu_help_states_task_action_and_tick_budgets(self):
        prompt = cu_help(2, 7)
        self.assertIn(
            "max 2 per call, max 7 total game ticks per call", prompt)

    def test_rl_launcher_derives_verl_ceiling_from_task_turn_budget(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "bin" / "rl.sh").read_text()
        self.assertIn('BRL_VERL_TURNS="$((MAX_TURNS + 1))"', launcher)
        self.assertIn('if [ -z "${RAY_ADDRESS:-}" ]; then', launcher)
        self.assertIn(
            'BRL_RAY_INIT_ARGS+=(ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS")',
            launcher,
        )
        self.assertIn(
            '"${BRL_RAY_INIT_ARGS[@]+"${BRL_RAY_INIT_ARGS[@]}"}"',
            launcher,
        )
        self.assertIn(
            '"BRL_KEEP_LATEST_IMAGES": "0"',
            launcher,
        )
        self.assertIn("BRL_RAY_ENV_JSON=", launcher)
        self.assertIn('"${BRL_RAY_ENV_ARGS[@]}"', launcher)
        self.assertIn(
            'data.dataloader_num_workers="${DATALOADER_WORKERS:-0}"',
            launcher,
        )
        self.assertIn(
            'actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}"',
            launcher,
        )
        self.assertNotIn('BRL_LOGGERS="$BRL_LOGGERS,\\"wandb\\""',
                         launcher)
        self.assertNotIn(
            'VAL_DATA="${VAL_DATA:-$DATA_DIR/rl_prompts.parquet}"', launcher)
        self.assertIn("validation_disabled.parquet", launcher)
        self.assertIn(
            'brl_run python - "$TRAIN_FILES" "$BRL_DISABLED_VAL_DATA"',
            launcher)
        self.assertIn(
            "pq.write_table(disabled_validation_table(source), path)",
            launcher,
        )

    def test_observation_probe_detaches_zero_copy_arrays_for_json(self):
        raw = bytearray((1, 2, 3))
        observation = {"x": 4.5, "cam": memoryview(raw)}
        session = SimpleNamespace(
            episode=SimpleNamespace(env=SimpleNamespace(obs=observation)))
        captured = ObservationProbe(fields=("x", "cam")).capture(session)
        raw[0] = 9
        self.assertEqual(captured, {"x": 4.5, "cam": [1, 2, 3]})
        self.assertEqual(observation["cam"][0], 9)

    def test_default_observation_probe_omits_large_pixel_buffers(self):
        observation = {
            "tick": 1, "x": 2, "y": 3, "z": 4, "yaw": 5, "pitch": 6,
            "dead": 0, "dimension": 0,
            "health": 20, "food": 20, "saturation": 5, "exhaustion": 0,
            "hotbar_ids": (), "hotbar_counts": (), "hotbar_meta": (),
            "hotbar_sel": 0, "container": 0, "inv_counts": {},
            "blocks": [], "logs": [], "coal": [],
            "cam": memoryview(bytearray((1, 2))),
            "depth": memoryview(bytearray((3, 4))),
            "edge": memoryview(bytearray((5, 6))),
        }
        session = SimpleNamespace(
            episode=SimpleNamespace(env=SimpleNamespace(obs=observation)))
        captured = ObservationProbe().capture(session)
        self.assertNotIn("cam", captured)
        self.assertNotIn("depth", captured)
        self.assertNotIn("edge", captured)
        self.assertEqual(captured["health"], 20)
        self.assertEqual(captured["food"], 20)

    def test_observation_probe_derives_vitals_from_public_journal(self):
        class Event(dict):
            name = "vitals_changed"

        observation = {
            "tick": 1, "x": 2, "y": 3, "z": 4, "yaw": 5, "pitch": 6,
            "dead": 0, "dimension": 0,
            "hotbar_ids": (), "hotbar_counts": (), "hotbar_meta": (),
            "hotbar_sel": 0, "container": 0, "inv_counts": {},
            "blocks": [], "logs": [], "coal": [],
        }
        task = SimpleNamespace(initial_state=SimpleNamespace(
            player={"health": 20, "food": 20}))
        env = SimpleNamespace(
            obs=observation, journal=[Event(health=17, food=18)])
        session = SimpleNamespace(episode=SimpleNamespace(
            task=task, env=env))
        captured = ObservationProbe().capture(session)
        self.assertEqual(captured["health"], 17.0)
        self.assertEqual(captured["food"], 18)

    def test_computer_tool_exposes_wasd_and_mouse_controls(self):
        text = yaml.safe_dump(tool_schema())
        self.assertIn("w=forward s=back a/d=strafe", text)
        self.assertIn("mouse_move", text)
        self.assertIn("left_click", text)

    def test_world_click_can_hold_the_sneak_modifier(self):
        schedule, count = compile_actions(
            [{"action": "right_click", "sneak": True}], pitch=0)
        self.assertEqual(count, 1)
        self.assertEqual(schedule, [({"use": 1, "sneak": 1}, 1)])
        with self.assertRaisesRegex(Exception, "true or false"):
            compile_actions(
                [{"action": "right_click", "sneak": 1}], pitch=0)

    def test_sneak_is_a_public_combinable_movement_key(self):
        schedule, count = compile_actions(
            [{"action": "key", "k": ["s", "sneak"], "ticks": 8}],
            pitch=0)
        self.assertEqual(count, 1)
        self.assertEqual(schedule, [({"forward": -1, "sneak": 1}, 8)])
        self.assertIn("sneak=crouch", yaml.safe_dump(tool_schema()))
        self.assertIn("sprint=run", yaml.safe_dump(tool_schema()))

        schedule, count = compile_actions(
            [{"action": "key", "k": ["w", "sprint"], "ticks": 3}],
            pitch=0)
        self.assertEqual(count, 1)
        self.assertEqual(schedule, [({"forward": 1, "sprint": 1}, 3)])

    def test_exact_click_target_event_decodes_engine_coordinates(self):
        from bedrock_rl.env.journal import (
            HEAD_BYTES, MAGIC, REC_BYTES, VERSION, JournalReader,
        )

        header = struct.pack("<IHH", MAGIC, VERSION, REC_BYTES)
        record = struct.pack("<HHi6i", 18, 2, 42, 49, -3, 65, 11, 1, 0)
        events = JournalReader().feed(header + record)
        self.assertEqual(HEAD_BYTES, len(header))
        self.assertEqual(events[0].as_dict(), {
            "event": "click_target", "tick": 42, "block": 49,
            "x": -3, "y": 65, "z": 11, "was_block": 1, "face": 2,
        })

    def test_bucket_target_adds_a_typed_event_without_a_wire_version_bump(self):
        from bedrock_rl.env.journal import (
            HEAD_BYTES, MAGIC, REC_BYTES, VERSION, EventMatch, JournalReader,
        )

        self.assertEqual(VERSION, 1)
        header = struct.pack("<IHH", MAGIC, VERSION, REC_BYTES)
        source = struct.pack(
            "<HHi6i", 20, 0, 43, 9, -4, 62, 12, 0, 3)
        inactive = struct.pack(
            "<HHi6i", 20, 0, 44, 0, 0, 0, 0, 0, -1)

        events = JournalReader().feed(header + source + inactive)

        self.assertEqual(HEAD_BYTES, len(header))
        self.assertEqual(events[0].as_dict(), {
            "event": "bucket_target", "tick": 43, "block": 9,
            "x": -4, "y": 62, "z": 12, "meta": 0, "slot": 3,
        })
        self.assertEqual(events[1].as_dict(), {
            "event": "bucket_target", "tick": 44, "block": 0,
            "x": 0, "y": 0, "z": 0, "meta": 0, "slot": -1,
        })
        self.assertTrue(EventMatch(
            "bucket_target",
            {"type": "bucket_target", "block": "water", "meta": 0,
             "slot": 3},
        ).matches(events[0]))

    def test_journal_flags_expose_broken_placed_and_fluid_state(self):
        from bedrock_rl.env.journal import (
            FLUID_BODY, FLUID_OVERLAY, FLUID_VIEWPOINT,
            MAGIC, REC_BYTES, VERSION, EventMatch, JournalReader,
        )

        header = struct.pack("<IHH", MAGIC, VERSION, REC_BYTES)
        broken = struct.pack(
            "<HHi6i", 2, 1, 49, 1, -4, 64, 9, 270, 1)
        placed = struct.pack(
            "<HHi6i", 3, 2, 50, 50, -3, 65, 11, 50, 8)
        fluid_flags = FLUID_BODY | FLUID_VIEWPOINT | FLUID_OVERLAY
        fluid = struct.pack(
            "<HHi6i", 19, fluid_flags, 51, 10, -3, 65, 11, 0, 0)
        events = JournalReader().feed(header + broken + placed + fluid)

        self.assertEqual(events[0].as_dict(), {
            "event": "block_broken", "tick": 49, "block": 1,
            "x": -4, "y": 64, "z": 9, "tool": 270, "drop": 1,
            "meta": 1,
        })
        self.assertEqual(events[1].as_dict(), {
            "event": "block_placed", "tick": 50, "block": 50,
            "x": -3, "y": 65, "z": 11, "item": 50, "slot": 8,
            "meta": 2,
        })
        self.assertEqual(events[2].as_dict(), {
            "event": "player_fluid_state", "tick": 51, "block": 10,
            "x": -3, "y": 65, "z": 11, "was_block": 0,
            "was_flags": 0, "flags": fluid_flags,
        })
        self.assertTrue(EventMatch(
            "block_placed", {"type": "block_placed", "meta": 2}
        ).matches(events[1]))
        self.assertTrue(EventMatch(
            "player_fluid_state",
            {"type": "player_fluid_state", "block": "lava",
             "min_flags": FLUID_BODY},
        ).matches(events[2]))

    def test_all_composes_unordered_verified_milestones(self):
        from bedrock_rl.env.journal import Event, EVENTS
        from bedrock_rl.env.task import Check

        check = Check({"type": "all", "of": [
            {"type": "block_placed", "block": "torch", "count": 2},
            {"type": "dimension_changed", "dimension": -1},
        ]})
        # The dimension event intentionally appears first: unlike sequence,
        # all is a conjunction and does not invent an ordering requirement.
        env = SimpleNamespace(journal=[
            Event("dimension_changed", EVENTS["dimension_changed"], 1,
                  (-1, 0, 0, 0, 0, 0)),
            Event("block_placed", EVENTS["block_placed"], 2,
                  (50, 1, 60, 1, 50, 0), flags=2),
            Event("block_placed", EVENTS["block_placed"], 3,
                  (50, 2, 60, 1, 50, 0), flags=2),
        ])
        self.assertTrue(check.holds(env, {}))
        env.journal.pop()
        self.assertFalse(check.holds(env, {}))

    def test_binary_observation_preserves_hotbar_metadata(self):
        from bedrock_rl.env.engine import (
            BIN_MAGIC, BIN_SIZE, HEAD_FMT, parse_obs,
        )

        record = bytearray(BIN_SIZE)
        record[:4] = BIN_MAGIC
        struct.pack_into(HEAD_FMT, record, 4, 1, 0, 65, 0, 0, 0, 0)
        offset = 4 + struct.calcsize(HEAD_FMT)
        main_ids = [280] + [0] * 26
        main_counts = [2] + [0] * 26
        main_meta = [4] + [0] * 26
        values = ([17] + [0] * 8 + [1] + [0] * 8
                  + [3] + [0] * 8 + main_ids + main_counts + main_meta
                  + [0, 0] + [0] * 9)
        struct.pack_into(f"<{len(values)}i", record, offset, *values)
        observation = parse_obs(record)
        self.assertEqual(observation["hotbar_ids"][0], 17)
        self.assertEqual(observation["hotbar_counts"][0], 1)
        self.assertEqual(observation["hotbar_meta"][0], 3)
        self.assertEqual(observation["inventory_ids"][9], 280)
        self.assertEqual(observation["inventory_counts"][9], 2)
        self.assertEqual(observation["inventory_meta"][9], 4)


if __name__ == "__main__":
    unittest.main()
