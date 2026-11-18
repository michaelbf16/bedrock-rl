"""Expose the Episode's deterministic reward through the generic verifier API."""

from bedrock_rl.core.reward import RewardResult, state_value
from bedrock_rl.env.episode import NO_TOOL_CALL_REWARD


class EpisodeReward:
    name = "bedrock"

    def evaluate(self, trajectory, state):
        # Match the standalone/verl format rule: an attempted tool call,
        # including a malformed or extension call, outranks a plain refusal.
        attempted = any(message.get("tool_calls")
                        for message in trajectory.messages
                        if message["role"] == "assistant")
        score = (float(state_value(state, "episode.reward")) if attempted
                 else float(NO_TOOL_CALL_REWARD))
        success = bool(state_value(state, "episode.success"))
        return RewardResult(score, {self.name: score},
                            {"success": success,
                             "turns": int(state_value(
                                 state, "episode.turns"))})
