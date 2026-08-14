"""Sampling defaults shared by training, harvesting, and evaluation."""

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
MAX_TOOL_CALLS_PER_TURN = 4
EPISODE_DONE_TOOL_MESSAGE = (
    "Tool call was not executed because the episode had already ended."
)
TOOL_CALL_LIMIT_MESSAGE = (
    "Tool call was not executed because one assistant turn is limited to "
    f"{MAX_TOOL_CALLS_PER_TURN} calls."
)


def multimodal_image_limit(
        max_turns, max_tool_calls=MAX_TOOL_CALLS_PER_TURN,
        keep_latest_images=0):
    """Safe vLLM image-item cap for a multi-call rollout.

    One initial observation is followed by one frame per executed call. The
    harness executes at most :data:`MAX_TOOL_CALLS_PER_TURN` calls from
    one assistant message and resolves any excess as skipped. This keeps the
    declared vLLM profiling bound realistic as well as sufficient. It is an
    acceptance limit, not an image-window policy. A positive image window is
    also a hard upper bound and avoids profiling discarded history.
    """
    turns = int(max_turns)
    calls = int(max_tool_calls)
    keep = int(keep_latest_images)
    if turns < 1 or calls < 1 or keep < 0:
        raise ValueError(
            "turn/tool-call budgets must be positive and image window "
            "must be nonnegative")
    unwindowed = 1 + turns * calls
    return min(keep, unwindowed) if keep else unwindowed


def sampling_knobs(temperature, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """One distribution, translated by each model-serving adapter.

    Hugging Face and vLLM both use top-k zero for no top-k filtering and
    temperature zero for greedy decoding. The only spelling difference is
    ``max_new_tokens`` versus vLLM's ``max_tokens``.
    """
    return {
        "temperature": float(temperature),
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "max_new_tokens": int(max_new_tokens),
    }
