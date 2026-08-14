"""Expose the Episode's deterministic reward through the generic verifier API."""

from bedrock_rl.core.reward import RewardResult, state_value


class EpisodeReward:
    name = "bedrock"

    def evaluate(self, trajectory, state):
        score = float(state_value(state, "episode.reward"))
        success = bool(state_value(state, "episode.success"))
        return RewardResult(score, {self.name: score},
                            {"success": success,
                             "turns": int(state_value(
                                 state, "episode.turns"))})
