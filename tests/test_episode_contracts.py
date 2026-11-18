"""Episode budgets and rewards stay the same across tool-harness choices."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from bedrock_rl.adapters.netherite.computer import cu_parser
from bedrock_rl.adapters.netherite.probes import EpisodeProbe
from bedrock_rl.adapters.netherite.reward import EpisodeReward
from bedrock_rl.adapters.netherite.tools import NetheriteComputerTool
from bedrock_rl.core.components import ComponentSpec
from bedrock_rl.core.harness import ToolHarness
from bedrock_rl.core.task import TaskSpec
from bedrock_rl.core.tools import FunctionTool
from bedrock_rl.env.engine import ObsHelpers
from bedrock_rl.env.episode import Episode, NO_TOOL_CALL_REWARD
from bedrock_rl.env.task import Task


class TickEngine(ObsHelpers):
    """One transition per act, using the production batch scheduler."""

    def __init__(self, pickup_tick):
        self.pickup_tick = pickup_tick
        self.ticks = 0
        self.journal = []
        self.journal_on = False
        self.calls = []
        self.obs = dict(x=0, y=64, z=0, pitch=0, yaw=0,
                        inv_counts={"log": 0}, hotbar_ids=(0,) * 9,
                        hotbar_counts=(0,) * 9)

    def reset_pose(self, yaw, pitch):
        self.obs.update(yaw=yaw, pitch=pitch)
        self.ticks = 0

    def act(self, **keys):
        self.calls.append(keys)
        self.ticks += 1
        if self.ticks >= self.pickup_tick:
            self.obs["inv_counts"]["log"] = 1

    def close(self):
        pass


class TickView:
    def capture(self, episode):
        episode.env.act(snap=1)
        output = io.BytesIO()
        Image.new("RGB", (1, 1), (episode.env.ticks, 0, 0)).save(
            output, format="PNG")
        return output.getvalue()


def open_episode(*, max_ticks=10, settle_ticks=6, pickup_tick=12,
                 max_turns=20, capture_frames=False):
    task = Task({
        "name": "budget", "goal": "Pick up a log", "max_ticks": max_ticks,
        "max_ticks_per_turn": 100, "max_turns": max_turns,
        "settle_ticks": settle_ticks,
        "success": {"type": "item_gained", "item": "log", "count": 1},
    })
    engine = TickEngine(pickup_tick)
    with patch("bedrock_rl.env.episode.open_view",
               return_value=(TickView(), Path("/unused"), {})), \
            patch("bedrock_rl.env.episode._open_env", return_value=engine):
        return Episode(task, 1, 0, 0, cu_parser(task),
                       capture_frames=capture_frames)


def wait(ticks):
    return json.dumps([{"action": "wait", "ticks": ticks}])


@patch("bedrock_rl.env.episode.render_obs", return_value="observation")
class EpisodeBudgetTests(unittest.TestCase):
    def test_late_pickup_cannot_succeed_outside_total_tick_budget(self, _render):
        for visual in (False, True):
            with self.subTest(visual=visual):
                episode = open_episode(capture_frames=visual)
                self.addCleanup(episode.close)
                _, frame, reward, done = episode.step(wait(9))
                self.assertEqual(episode.env.ticks, 10)
                self.assertTrue(done)
                self.assertFalse(episode.success)
                self.assertEqual(reward, 0.0)
                if visual:
                    self.assertEqual(frame.getpixel((0, 0)), (10, 0, 0))

    def test_settling_and_observation_share_the_remaining_budget(self, _render):
        episode = open_episode(settle_ticks=20, pickup_tick=10)
        self.addCleanup(episode.close)
        _, _, reward, done = episode.step(wait(2))
        self.assertEqual(episode.env.ticks, 10)
        self.assertTrue(episode.success)
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)

    def test_last_tick_is_reserved_for_the_terminal_observation(self, _render):
        episode = open_episode(max_ticks=2, pickup_tick=3,
                               capture_frames=True)
        self.addCleanup(episode.close)
        _, frame, _, done = episode.step(wait(9))
        self.assertEqual(episode.env.calls, [{"snap": 1}, {"snap": 1}])
        self.assertEqual(frame.getpixel((0, 0)), (2, 0, 0))
        self.assertFalse(episode.success)
        self.assertTrue(done)

    def test_malformed_terminal_call_and_repeated_reads_do_not_overrun(self,
                                                                    _render):
        episode = open_episode(max_ticks=2, capture_frames=True)
        self.addCleanup(episode.close)
        _, frame, reward, done = episode.step("{")
        self.assertEqual(reward, -0.2)
        self.assertTrue(done)
        self.assertEqual(episode.env.ticks, 2)
        _, repeated, reward, done = episode.step(wait(1))
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertEqual(episode.turns, 1)
        self.assertEqual(episode.env.ticks, 2)
        self.assertEqual(repeated.tobytes(), frame.tobytes())
        self.assertEqual(episode.frame().tobytes(), frame.tobytes())
        self.assertEqual(episode.env.ticks, 2)

    def test_initial_observation_can_exhaust_a_one_tick_episode(self, _render):
        episode = open_episode(max_ticks=1)
        self.addCleanup(episode.close)
        self.assertTrue(episode.done)
        episode.step(wait(1))
        self.assertEqual(episode.env.ticks, 1)
        self.assertEqual(episode.turns, 0)


def tool_call(name, arguments):
    return {"id": name, "type": "function", "function": {
        "name": name, "arguments": arguments}}


class RepeatingModel:
    def __init__(self, calls):
        self.calls = calls

    def complete(self, *_args):
        return {"role": "assistant", "content": "",
                "tool_calls": self.calls}


class HarnessEpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def run_harness(self, calls, *, max_turns=4, episode_turns=2):
        episode = open_episode(max_ticks=1000, settle_ticks=0,
                               pickup_tick=2000, max_turns=episode_turns)
        self.addCleanup(episode.close)
        session = SimpleNamespace(
            episode=episode,
            initial_messages=lambda _: [{"role": "user", "content": "act"}])
        task = TaskSpec("budget", "Pick up a log", ComponentSpec("unused:x"))
        harness = ToolHarness(
            RepeatingModel(calls), environment=lambda _: session,
            tools=[FunctionTool("recipe", "look up a recipe",
                                {"type": "object"}, lambda *_: "recipe"),
                   NetheriteComputerTool()],
            probes=[EpisodeProbe()], verifier=EpisodeReward(),
            max_turns=max_turns)
        with patch("bedrock_rl.env.episode.render_obs",
                   return_value="observation"):
            trajectory = await harness.run(task)
        return episode, trajectory

    async def test_lookup_before_computer_cannot_bypass_episode_turn_budget(self):
        computer = tool_call("computer", {
            "actions": [{"action": "wait", "ticks": 1}]})
        episode, trajectory = await self.run_harness([
            tool_call("recipe", {}), computer, computer])
        self.assertEqual(episode.turns, 2)
        self.assertTrue(episode.done)
        self.assertEqual(len(trajectory.views), 2)
        # Two calls in turn one, one in turn two; its last call is skipped.
        self.assertEqual(episode.nlines, 3)
        self.assertEqual(trajectory.metadata["tool_results"][-1]["metadata"],
                         {"skipped": True, "reason": "episode_done"})

    async def test_malformed_and_unknown_calls_share_computer_turn_accounting(self):
        for name in ("computer", "invented"):
            with self.subTest(name=name):
                episode, _ = await self.run_harness([
                    tool_call("recipe", {}), tool_call(name, "{"),
                    tool_call("computer", {
                        "actions": [{"action": "wait", "ticks": 1}]}),
                ], max_turns=1)
                self.assertEqual(episode.turns, 1)
                self.assertEqual(episode.penalty, 0.2)
                self.assertEqual(episode.nlines, 1)

    async def test_plain_refusal_has_the_shared_no_tool_reward(self):
        episode, trajectory = await self.run_harness([])
        self.assertEqual(episode.turns, 0)
        self.assertEqual(trajectory.reward.total, NO_TOOL_CALL_REWARD)

    async def test_extension_and_malformed_attempts_keep_live_episode_reward(self):
        for call in (tool_call("recipe", {}), tool_call("computer", "{"),
                     tool_call("invented", {})):
            with self.subTest(call=call):
                episode, trajectory = await self.run_harness([call], max_turns=1)
                self.assertEqual(trajectory.reward.total, episode.final_reward())
                self.assertGreater(trajectory.reward.total, NO_TOOL_CALL_REWARD)


if __name__ == "__main__":
    unittest.main()
