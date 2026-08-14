from types import SimpleNamespace
import unittest

from bedrock_rl.env.journal import EVENTS, FLUID_BODY, Event
from bedrock_rl.env.task import Check


class PlayerTouchingBlockTests(unittest.TestCase):
    def setUp(self):
        self.check = Check({
            "type": "player_touching_block",
            "blocks": ["lava"],
            "margin": 0.25,
        })

    @staticmethod
    def env(x, y, z, blocks):
        return SimpleNamespace(obs={"x": x, "y": y, "z": z,
                                    "blocks": blocks})

    def test_detects_collision_or_requested_clearance(self):
        lava = [(11, 0, 10, 15)]
        self.assertTrue(self.check.holds(
            self.env(0.5, 11.0, 16.49, lava), {}))

    def test_accepts_a_real_detour(self):
        lava = [(11, 0, 10, 15)]
        self.assertFalse(self.check.holds(
            self.env(2.5, 11.0, 15.5, lava), {}))

    def test_ignores_other_blocks(self):
        stone = [(1, 0, 10, 15)]
        self.assertFalse(self.check.holds(
            self.env(0.5, 11.0, 15.5, stone), {}))

    def test_live_body_lava_event_is_an_irrevocable_terminal_match(self):
        check = Check({"type": "player_fluid_state", "block": "lava"})
        lava = Event(
            "player_fluid_state", EVENTS["player_fluid_state"], 17,
            (10, -8, 8, 22, 0, 0), FLUID_BODY,
        )
        dry_again = Event(
            "player_fluid_state", EVENTS["player_fluid_state"], 18,
            (0, -8, 8, 22, 10, FLUID_BODY), 0,
        )
        self.assertEqual(lava.as_dict(), {
            "event": "player_fluid_state", "tick": 17, "block": 10,
            "x": -8, "y": 8, "z": 22, "was_block": 0,
            "was_flags": 0, "flags": FLUID_BODY,
        })
        # The current observation and the latest event can both be dry. The
        # turn still fails because a journal check scans the whole episode;
        # leaving lava later in the same batched tool call cannot erase it.
        env = SimpleNamespace(
            obs={"x": 8.5, "y": 11.0, "z": 9.5, "blocks": []},
            journal=[lava, dry_again],
        )
        self.assertTrue(check.holds(env, {}))

    def test_any_journal_failure_matches_lava_or_lost_health(self):
        check = Check({
            "type": "any",
            "of": [
                {"type": "player_fluid_state", "block": "lava"},
                {"type": "vitals_changed", "max_health": 19},
                {"type": "overflow", "min_dropped": 1},
            ],
        })
        dry = Event(
            "player_fluid_state", EVENTS["player_fluid_state"], 1,
            (0, 0, 64, 0, -1, 0), 0,
        )
        hurt = Event(
            "vitals_changed", EVENTS["vitals_changed"], 2,
            (19, 20, 0, 20, 20, 0),
        )
        env = SimpleNamespace(journal=[dry])
        self.assertFalse(check.holds(env, {}))
        env.journal.append(hurt)
        self.assertTrue(check.holds(env, {}))

        overflow = Event(
            "overflow", EVENTS["overflow"], 3,
            (1, 0, 0, 0, 0, 0),
        )
        env.journal[:] = [dry, overflow]
        self.assertTrue(check.holds(env, {}))


if __name__ == "__main__":
    unittest.main()
