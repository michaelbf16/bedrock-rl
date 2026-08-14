import unittest
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bedrock_rl import models
from bedrock_rl.adapters.netherite.chat import (
    Coder, prepare_sft_multiturn, tool_schema,
)
from bedrock_rl.adapters.netherite.computer import tool_call_attempts
from bedrock_rl.cli.main import cmd_models_resolve
from bedrock_rl.core.messages import validate_message
from bedrock_rl.core.tools import ToolCall
from bedrock_rl.env.actions import ProgramError
from bedrock_rl.env.episode import (NO_TOOL_CALL_REWARD,
                                      TOOL_USE_REWARD_FLOOR, Episode)
from bedrock_rl.env.task import Check, Task
from bedrock_rl.env.journal import Event, EventMatch, EVENTS


class ConsistencyRegressionTests(unittest.TestCase):
    def test_frame_scratch_reaps_sigkill_leftovers(self):
        from bedrock_rl.env import episode as episode_module
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "123-dead"
            stale.mkdir()
            (stale / episode_module._FRAME_OWNER).write_text("123")

            with patch.object(episode_module, "_process_alive",
                              return_value=False):
                current = Path(episode_module._owned_frames_dir(directory))

            self.assertFalse(stale.exists())
            self.assertTrue(current.is_dir())
            self.assertEqual((current / episode_module._FRAME_OWNER)
                             .read_text().splitlines()[0], str(os.getpid()))

    def test_frame_scratch_detects_pid_reuse(self):
        from bedrock_rl.env import episode as episode_module
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "123-reused"
            stale.mkdir()
            (stale / episode_module._FRAME_OWNER).write_text("123\nold")

            with (patch.object(episode_module, "_process_alive",
                               return_value=True),
                  patch.object(episode_module, "_process_identities",
                               return_value={123: "new"})):
                episode_module._owned_frames_dir(directory)

            self.assertFalse(stale.exists())

    def test_recording_transfers_ownership_to_the_writer(self):
        from bedrock_rl.env import episode as episode_module
        with tempfile.TemporaryDirectory() as directory:
            retained = Path(episode_module._owned_frames_dir(directory))
            episode_module.retain_frames_dir(retained)
            root_key = os.fspath(Path(directory).resolve())
            episode_module._REAPED_FRAME_ROOTS.discard(root_key)

            with patch.object(
                    episode_module, "_process_alive",
                    side_effect=lambda pid: int(pid) == os.getpid()):
                current = Path(episode_module._owned_frames_dir(directory))

            self.assertTrue(retained.is_dir())
            self.assertEqual(
                (retained / episode_module._FRAME_OWNER)
                .read_text().splitlines()[0], str(os.getpid()))
            self.assertTrue(current.is_dir())

    def test_post_open_initialization_failure_closes_engine_and_frames(self):
        from bedrock_rl.env import episode as episode_module

        class Engine:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        engine = Engine()
        task = SimpleNamespace(
            view="semantic", raw={}, journal=True,
            initial_state=SimpleNamespace(world={}))
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(
                    episode_module, "open_view",
                    return_value=(object(), Path(directory), {})),
                  patch.object(episode_module, "_open_env",
                               return_value=engine),
                  patch.object(
                      episode_module.Episode, "_initialize_state",
                      side_effect=RuntimeError("reset failed"))):
                with self.assertRaisesRegex(RuntimeError, "reset failed"):
                    episode_module.Episode(
                        task, 1, 0, 0, lambda *args: None,
                        frames_root=directory)

            self.assertTrue(engine.closed)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_deep_tool_json_is_charged_as_malformed(self):
        body = "[" * 20000 + "]" * 20000
        attempts = tool_call_attempts(
            f"<tool_call>{body}</tool_call>")

        self.assertEqual(len(attempts), 1)
        self.assertIn("malformed tool envelope", attempts[0][2])

    def test_malformed_turn_still_honors_episode_tick_limit(self):
        episode = Episode.__new__(Episode)
        episode.task = SimpleNamespace(
            max_ticks=200, shaping=[], converge=None, fail=None,
            success=SimpleNamespace(holds=lambda *_args: False))
        episode.max_turns = 10
        episode.env = SimpleNamespace(ticks=200, obs={"pitch": 0.0})
        episode.turns = 0
        episode.done = False
        episode.penalty = 0.0
        episode.start_obs = {}
        episode.gui_open = False
        episode._sync_screen = lambda: None
        episode._parse = lambda *_: (_ for _ in ()).throw(
            ProgramError("bad call"))
        episode.frame = lambda: None

        _text, _frame, reward, done = episode.step("malformed")

        self.assertEqual(reward, -0.2)
        self.assertTrue(done)

    def test_mining_hold_releases_on_exact_first_block_break(self):
        class Env:
            def __init__(self):
                self.obs = {"pitch": 0.0}
                self.journal = []
                self.received_stop = None
                self.ticks = 0
                self.items = None
                self.journal_on = True

            def act_batch(self, items, observer=None, stop_factory=None):
                del observer
                self.items = items
                for keys, _ticks in items:
                    if "dyaw" in keys:
                        self.journal.append(Event(
                            "click_target", EVENTS["click_target"], 1,
                            (2, 4, 81, 2, 0, 0)))
                    stop = stop_factory(self, keys)
                    if stop is not None:
                        self.received_stop = stop

            def crosshair(self):
                return {"id": 2, "dist": 1.0}

        episode = Episode.__new__(Episode)
        episode.task = SimpleNamespace(
            max_ticks=200, settle_ticks=0, max_turns=10,
            shaping=(), success=SimpleNamespace(holds=lambda *_: False),
            fail=None, converge=None, journal=True, perception=None)
        episode.per_turn_ticks = 200
        episode.max_turns = 10
        episode.env = Env()
        episode.turns = 0
        episode.done = episode.success = episode.failed = False
        episode.nlines = 0
        episode._parse = lambda *_: (
            [({"dyaw": 3.0}, 1), ({"attack": 1}, 28), ({}, 1)], 2)
        episode._sync_screen = lambda: None
        episode.gui_open = False
        episode._tick_observer = None
        episode.start_obs = {}
        episode.fired = []
        episode.shaped = episode.converged = 0.0
        episode.frame = lambda: None

        with patch("bedrock_rl.env.episode.render_obs", return_value=""):
            episode.step("hold")

        stop = episode.env.received_stop
        self.assertIsNotNone(stop)
        self.assertEqual(episode.env.items[0][0], {"dyaw": 3.0})
        episode.env.journal.append(Event(
            "block_broken", EVENTS["block_broken"], 27,
            (2, 4, 81, 2, 270, 3)))
        self.assertTrue(stop(episode.env))
        episode.env.journal.append(Event(
            "block_broken", EVENTS["block_broken"], 28,
            (31, 4, 81, 1, 270, 0)))
        self.assertTrue(stop(episode.env))

    def test_explicitly_disabled_journal_does_not_arm_mining_release(self):
        class Env:
            obs = {"pitch": 0.0}
            ticks = 0
            journal = []
            journal_on = False
            received_factory = "unset"

            def act_batch(self, items, observer=None, stop_factory=None):
                del items, observer
                self.received_factory = stop_factory

        episode = Episode.__new__(Episode)
        episode.task = SimpleNamespace(
            max_ticks=200, settle_ticks=0, max_turns=10,
            shaping=(), success=SimpleNamespace(holds=lambda *_: False),
            fail=None, converge=None, journal=False, perception=None)
        episode.per_turn_ticks = 200
        episode.max_turns = 10
        episode.env = Env()
        episode.turns = 0
        episode.done = episode.success = episode.failed = False
        episode.nlines = 0
        episode._parse = lambda *_: ([({"attack": 1}, 28)], 1)
        episode._sync_screen = lambda: None
        episode.gui_open = False
        episode._tick_observer = None
        episode.start_obs = {}
        episode.fired = []
        episode.shaped = episode.converged = 0.0
        episode.frame = lambda: None
        with patch("bedrock_rl.env.episode.render_obs", return_value=""):
            episode.step("hold")
        self.assertIsNone(episode.env.received_factory)

    def test_model_resolve_command_does_not_assume_fixed_image_count(self):
        self.assertEqual(cmd_models_resolve(SimpleNamespace(
            model="qwen3-vl", json=False)), 0)

    def test_registry_loader_applies_family_chat_template(self):
        tokenizer = SimpleNamespace(
            pad_token_id=None, pad_token=None, eos_token="<eos>")
        processor = SimpleNamespace(tokenizer=tokenizer, chat_template=None)

        class AutoProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                del args, kwargs
                return processor

        with patch.dict(sys.modules, {
                "transformers": SimpleNamespace(AutoProcessor=AutoProcessor)}):
            loaded = models.load_processor(
                models.resolve("qwen2-vl"),
                "Qwen/Qwen2-VL-2B-Instruct")
        self.assertIn("tool_calls", loaded.chat_template)
        self.assertEqual(tokenizer.pad_token, "<eos>")

    def test_chat_template_does_not_double_encode_tool_arguments(self):
        try:
            from jinja2 import Environment
        except ImportError:
            self.skipTest("jinja2 is not installed")
        template_path = (Path(__file__).parents[1] / "bedrock_rl" /
                         "templates" / "chat_template" /
                         "qwen_chatml_tools.jinja")
        template = Environment().from_string(template_path.read_text())
        call = ToolCall("call_1", "computer", {"actions": []}).to_openai()

        rendered = template.render(
            messages=[{"role": "assistant", "content": None,
                       "tool_calls": [call]}],
            tools=[], add_generation_prompt=False, add_vision_id=False)
        payload = rendered.split("<tool_call>\n", 1)[1].split(
            "\n</tool_call>", 1)[0]

        self.assertEqual(json.loads(payload), {
            "name": "computer", "arguments": {"actions": []}})

    def test_shipped_template_renders_sft_and_rollout_t0_identically(self):
        try:
            from jinja2 import Environment
        except ImportError:
            self.skipTest("jinja2 is not installed")
        template_path = (Path(__file__).parents[1] / "bedrock_rl" /
                         "templates" / "chat_template" /
                         "qwen_chatml_tools.jinja")
        template_source = template_path.read_text()

        class Tokenizer:
            def apply_chat_template(self, messages, *, tools, tokenize,
                                    add_generation_prompt, chat_template):
                self.assertions = (tokenize, chat_template)
                return Environment().from_string(chat_template).render(
                    messages=messages, tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    add_vision_id=False)

        tokenizer = Tokenizer()
        coder = Coder(
            SimpleNamespace(tokenizer=tokenizer, image_processor=object()),
            tools=tool_schema(), chat_template=template_source,
            spec=SimpleNamespace(
                key="template-test", image_span=("<VS>", "<IP>", "<VE>")))
        online = [{"role": "user", "content": "<image>\nTASK: farm log"}]
        sft, _ = prepare_sft_multiturn(online, [object()])

        self.assertEqual(coder.render(online, [1], True),
                         coder.render(sft, [1], True))

    def test_null_tool_calls_is_an_absent_optional_field(self):
        message = {"role": "assistant", "content": "done",
                   "tool_calls": None}
        self.assertIs(validate_message(message), message)

    def test_item_gained_rejects_items_without_aggregate_counter(self):
        with self.assertRaisesRegex(ValueError, "hotbar_gained"):
            Check({"type": "item_gained", "item": "diamond"})

    def test_journal_compositions_accept_only_flat_journal_events(self):
        for composition in ("sequence", "all", "any"):
            with self.subTest(composition=composition, bad="snapshot"):
                with self.assertRaisesRegex(ValueError,
                                            "one journal event"):
                    Check({"type": composition, "of": [
                        {"type": "selected_item",
                         "items": ["iron_pickaxe"]},
                    ]})
            with self.subTest(composition=composition, bad="nested"):
                with self.assertRaisesRegex(ValueError,
                                            "one journal event"):
                    Check({"type": composition, "of": [
                        {"type": "sequence", "of": [
                            {"type": "item_crafted", "item": "stick"},
                        ]},
                    ]})

    def test_task_journal_switch_rejects_integer_zero(self):
        raw = {
            "name": "journal-type",
            "goal": "wait",
            "journal": 0,
            "success": {"type": "item_gained", "item": "log"},
        }
        with self.assertRaisesRegex(TypeError, "journal must be true or false"):
            Task(raw)

    def test_journal_checks_ignore_events_before_episode_baseline(self):
        check = Check({"type": "aimed_at", "block": "log"})
        event = Event("aimed_at", EVENTS["aimed_at"], 1,
                      (17, 4, 0, 0, 0, 0))
        env = SimpleNamespace(journal=[event])
        self.assertTrue(check.holds(env, {}, 0))
        self.assertFalse(check.holds(env, {}, 1))

    def test_journal_backed_task_forces_stream_on(self):
        task = Task({
            "name": "journal-required",
            "goal": "break a log",
            "seeds": [1],
            "success": {"type": "block_broken", "block": "log"},
        })

        self.assertIs(task.journal, True)

    def test_journal_equality_uses_task_units_not_wire_units(self):
        match = EventMatch(
            "aimed_at", {"type": "aimed_at", "dist": 4.0})
        event = Event("aimed_at", EVENTS["aimed_at"], 1,
                      (1, 16, 0, 0, 0, 0))

        self.assertTrue(match.matches(event))

    def test_journal_equality_rejects_unrepresentable_scaled_value(self):
        with self.assertRaisesRegex(ValueError, "cannot be represented"):
            EventMatch("aimed_at", {"type": "aimed_at", "dist": 0.1})

    def test_task_init_is_a_closed_mapping(self):
        base = {"name": "closed-init", "goal": "wait", "seeds": [1],
                "success": {"type": "pitch_below", "value": -80}}
        with self.assertRaisesRegex(ValueError, "unknown task init key"):
            Task({**base, "init": {"ywa": 10}})
        with self.assertRaisesRegex(TypeError, "init must be a mapping"):
            Task({**base, "init": [0, 1]})

    def test_any_tool_using_episode_stays_above_no_call_floor(self):
        episode = Episode.__new__(Episode)
        episode.task = SimpleNamespace(
            decision_cost=10.0,
            success_reward=-100.0,
            fail=SimpleNamespace(reward=-100.0),
            no_commit=-100.0,
        )
        episode.nlines = 100
        episode.penalty = 100.0
        episode.shaped = -100.0
        episode.converged = -100.0
        for success, failed in ((True, False), (False, True), (False, False)):
            episode.success = success
            episode.failed = failed
            self.assertEqual(episode.final_reward(), TOOL_USE_REWARD_FLOOR)
            self.assertGreater(episode.final_reward(), NO_TOOL_CALL_REWARD)

    def test_draw_rejects_an_already_selected_success_item(self):
        from bedrock_rl.env.catalog import resolve_items

        target = resolve_items(["iron_pickaxe"])[0]
        other = resolve_items(["stone_pickaxe"])[0]

        class DrawEpisode(Episode):
            selections = iter((target, other))

            def __init__(self, task, seed, yaw, pitch, parser, **kwargs):
                del seed, parser, kwargs
                selected = next(self.selections)
                self.task = task
                self.env = SimpleNamespace(obs={
                    "hotbar_sel": 0,
                    "hotbar_ids": (selected,) + (0,) * 8,
                })
                self.start_obs = dict(self.env.obs)
                self.spec = {"init_yaw": yaw, "init_pitch": pitch}
                self.closed = False

            def close(self):
                self.closed = True

        task = SimpleNamespace(
            name="equip", path="task.yaml", init={},
            success=Check({"type": "selected_item",
                           "items": ["iron_pickaxe"]}),
            sample_init=lambda rng: (0.0, 0.0),
        )
        episode = DrawEpisode.draw(task, 7, object(), parser=None)
        self.assertEqual(episode.env.obs["hotbar_ids"][0], other)


if __name__ == "__main__":
    unittest.main()
