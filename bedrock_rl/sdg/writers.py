"""Streaming writers for SDG prompt and demonstration artifacts."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from pathlib import Path
from typing import Mapping

from bedrock_rl.adapters.netherite.chat import SFT_IMAGE_TAG
from bedrock_rl.core.messages import decode_data_image_url, validate_transcript
from bedrock_rl.core.trajectory import Trajectory
from bedrock_rl.data import write_rows


class ParquetWriter:
    """Write records to Hugging Face-compatible parquet incrementally."""

    def __init__(self, output, batch_size=32, **_):
        self.output = str(output)
        self.batch_size = int(batch_size)

    def write(self, records):
        write_rows(records, self.output, batch_size=self.batch_size)
        return self.output


def _embedded_image_url(part, where):
    """The data URL carried by one canonical OpenAI image part."""
    if part.get("type") != "image_url":
        raise ValueError(
            f"{where} is not canonical OpenAI image_url content")
    value = part.get("image_url")
    if isinstance(value, Mapping):
        value = value.get("url")
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ValueError(
            f"{where} must contain an embedded data-URL image; external "
            "URLs and runtime image objects are not portable SFT data")
    return value


def _success_markers(trajectory):
    markers = []
    reward = trajectory.reward
    if reward is not None and "success" in reward.metrics:
        markers.append(bool(reward.metrics["success"]))
    if trajectory.state.samples:
        latest = trajectory.state.latest
        episode = latest.get("episode")
        if isinstance(episode, Mapping) and "success" in episode:
            markers.append(bool(episode["success"]))
    return markers


def _episode_metadata(trajectory):
    """Seed and turn count from the explicit generation/episode record."""
    metadata = trajectory.metadata
    generation_case = metadata.get("generation_case")
    seed = (generation_case.get("seed")
            if isinstance(generation_case, Mapping) else metadata.get("seed"))
    turns = (trajectory.reward.metrics.get("turns")
             if trajectory.reward is not None else None)
    if trajectory.state.samples:
        episode = trajectory.state.latest.get("episode")
        if isinstance(episode, Mapping):
            spec = episode.get("spec")
            if seed is None and isinstance(spec, Mapping):
                seed = spec.get("seed")
            if turns is None:
                turns = episode.get("turns")
    if (isinstance(seed, bool) or not isinstance(seed, (int, float))
            or not math.isfinite(seed)):
        raise ValueError("successful trajectory has no numeric world seed")
    if int(seed) != seed:
        raise ValueError(f"trajectory world seed {seed!r} is not an integer")
    if (isinstance(turns, bool) or not isinstance(turns, (int, float))
            or not math.isfinite(turns)):
        raise ValueError("successful trajectory has no numeric turn count")
    if int(turns) != turns or int(turns) < 1:
        raise ValueError(f"trajectory turn count {turns!r} is not positive")
    return int(seed), int(turns)


def trajectory_sft_row(value):
    """Convert one successful canonical trajectory into one SFT row.

    Canonical history keeps images embedded in OpenAI content parts so the
    artifact is portable. Hugging Face parquet keeps image bytes in a separate
    column. This conversion walks messages once, moves those bytes to that
    column in encounter order, and leaves an exact ``<image>`` text marker in
    each original content-part position. Tool calls and their matching OpenAI
    tool-result messages are copied without flattening.
    """
    if isinstance(value, Mapping):
        value = Trajectory.from_dict(value)
    if not isinstance(value, Trajectory):
        raise TypeError("TrajectorySFTWriter accepts canonical Trajectory values")
    if value.finished_at is None:
        raise ValueError("cannot train on an unfinished trajectory")
    if value.reward is None:
        raise ValueError("cannot train on a trajectory without reward")
    if not math.isfinite(value.reward.total):
        raise ValueError("trajectory reward must be finite")
    markers = _success_markers(value)
    if not markers:
        raise ValueError("trajectory has no explicit success marker")
    if not all(markers):
        raise ValueError("cannot train on a non-success trajectory")

    # to_dict performs the canonical per-message validation before any data is
    # moved out of the artifact. Work on that copy so the trajectory itself is
    # never rewritten by a dataset export.
    messages = value.to_dict()["messages"]
    validate_transcript(messages)
    if any((message.get("metadata") or {}).get("malformed_tool_calls")
           for message in messages if message.get("role") == "assistant"):
        raise ValueError(
            "cannot train on a trajectory containing malformed tool calls")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("SFT trajectory has no assistant turn")
    images = []
    for message_index, message in enumerate(messages):
        # OpenAI permits null assistant content when ``tool_calls`` carries the
        # turn. Qwen3-VL's template iterates content first, so render an
        # equivalent empty string without changing the saved trajectory.
        if (message.get("role") == "assistant"
                and message.get("content") is None
                and message.get("tool_calls")):
            message["content"] = ""
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        replaced = []
        for part_index, part in enumerate(content):
            where = f"message {message_index}.content[{part_index}]"
            if part.get("type") != "image_url":
                replaced.append(copy.deepcopy(part))
                continue
            url = _embedded_image_url(part, where)
            images.append({"bytes": decode_data_image_url(url, where),
                           "path": None})
            replaced.append({"type": "text", "text": SFT_IMAGE_TAG})
        message["content"] = replaced

    seed, turns = _episode_metadata(value)
    compact = {"ensure_ascii": False, "sort_keys": True,
               "allow_nan": False,
               "separators": (",", ":")}
    return {
        "messages": json.dumps(messages, **compact),
        "images": images,
        "reward": float(value.reward.total),
        "reward_json": json.dumps(value.reward.to_dict(), **compact),
        "seed": seed,
        "turns": turns,
        "trajectory_id": value.id,
        "source_trajectory_id": value.id,
        "curriculum_stage": turns,
        "loss_last_assistant_only": True,
        # JSON text tolerates different provenance/metadata keys per policy
        # and still round-trips exactly; an Arrow struct would force every
        # generated policy into the schema inferred from the first row.
        "provenance": json.dumps(value.provenance, **compact),
        "trajectory_metadata": json.dumps(value.metadata, **compact),
    }


def trajectory_sft_rows(value, keep_latest_images=1,
                        require_grounded=False):
    """Yield one causal, bounded SFT example per eligible assistant action."""
    from bedrock_rl.adapters.netherite.chat import prepare_sft_multiturn

    if isinstance(value, Mapping):
        value = Trajectory.from_dict(value)
    base = trajectory_sft_row(value)
    messages = json.loads(base["messages"])
    images = base["images"]
    assistant_indices = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    compact = {"ensure_ascii": False, "sort_keys": True,
               "allow_nan": False, "separators": (",", ":")}
    for stage, message_index in enumerate(assistant_indices, 1):
        message = messages[message_index]
        grounding = (message.get("metadata") or {}).get(
            "demonstration_grounding")
        if grounding is None:
            grounding = (value.provenance or {}).get(
                "demonstration_grounding") or {}
        if require_grounded and grounding.get("eligible") is not True:
            continue
        prefix = messages[:message_index + 1]
        image_count = sum(
            part.get("text") == SFT_IMAGE_TAG
            for message in prefix
            for part in (message.get("content")
                         if isinstance(message.get("content"), list) else ())
            if isinstance(part, Mapping)
        )
        causal_messages, causal_images = prepare_sft_multiturn(
            prefix, images[:image_count],
            last_assistant_only=True,
            keep_latest_images=keep_latest_images)
        row = dict(base)
        row.update({
            "messages": json.dumps(causal_messages, **compact),
            "images": causal_images,
            "curriculum_stage": stage,
            "loss_last_assistant_only": True,
        })
        yield row


class TrajectorySFTWriter:
    """Write each successful assistant action as a causal SFT row."""

    def __init__(self, output, batch_size=8, keep_latest_images=1,
                 require_grounded=True, **_):
        self.output = str(output)
        self.batch_size = int(batch_size)
        self.keep_latest_images = int(keep_latest_images)
        if not isinstance(require_grounded, bool):
            raise TypeError("require_grounded must be true or false")
        self.require_grounded = require_grounded
        if self.batch_size < 1:
            raise ValueError("trajectory SFT batch_size must be at least one")
        if self.keep_latest_images < 0:
            raise ValueError("keep_latest_images cannot be negative")

    def write(self, records):
        def rows():
            for record in records:
                value = (Trajectory.from_dict(record)
                         if isinstance(record, Mapping) else record)
                exported = False
                for row in trajectory_sft_rows(
                        value,
                        keep_latest_images=self.keep_latest_images,
                        require_grounded=self.require_grounded):
                    exported = True
                    yield row
                if self.require_grounded and not exported:
                    raise ValueError(
                        "trajectory has no assistant turn declared "
                        "model-grounded; set require_grounded: false only "
                        "for an intentional privileged-planner experiment")

        write_rows(rows(), self.output, batch_size=self.batch_size)
        return self.output
class TrajectoryArtifactWriter:
    """Save accepted trajectories and their exact model-view preview together.

    A single-record ``.json`` output is pretty-printed by default. Set
    ``compact: true`` for large visual trajectories; a one-line JSON object is
    still ordinary JSON and avoids doubling every embedded image and state
    sample with indentation. Multiple records always use JSON Lines. ``gif``
    is optional and intentionally previews the first accepted trajectory.
    ``redact_images`` is only for small, human-readable repository examples:
    it replaces embedded image bytes in the saved copy with an explicit
    reserved URL while the GIF still uses the real frames. Training artifacts
    must keep the default so their OpenAI messages remain self-contained.
    """

    def __init__(self, output, gif=None, gif_width=800, duration_ms=55,
                 final_hold_ms=900, compact=False, redact_images=False,
                 replay=None, *, base_dir=".", **_):
        self.output = Path(output)
        self.gif = None if gif is None else Path(gif)
        if self.gif is not None and not self.gif.is_absolute():
            self.gif = (Path(base_dir) / self.gif).resolve()
        self.gif_width = int(gif_width)
        self.duration_ms = int(duration_ms)
        self.final_hold_ms = int(final_hold_ms)
        self.compact = bool(compact)
        self.redact_images = bool(redact_images)
        self.replay = None
        if replay is not None:
            if not isinstance(replay, Mapping):
                raise TypeError("trajectory writer replay must be a mapping")
            replay = dict(replay)
            known = {
                "gif", "views", "labels", "recording_every", "panel_width",
                "renderer_homes", "duration_ms", "final_hold_ms",
                "replay_workers",
            }
            unknown = sorted(set(replay) - known)
            if unknown:
                raise ValueError("unknown trajectory writer replay key(s): "
                                 + ", ".join(unknown))
            if not replay.get("gif"):
                raise ValueError("trajectory writer replay needs gif")
            replay_gif = Path(replay.pop("gif"))
            if not replay_gif.is_absolute():
                replay_gif = (Path(base_dir) / replay_gif).resolve()
            replay["gif"] = replay_gif
            self.replay = replay

    @staticmethod
    def _redact_images(trajectory):
        placeholder = "https://bedrock-rl.invalid/base64_redacted"
        messages = [trajectory.messages]
        messages.extend(view.messages for view in trajectory.views)
        replaced = 0
        for transcript in messages:
            for message in transcript:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if part.get("type") != "image_url":
                        continue
                    image = part.get("image_url")
                    if isinstance(image, Mapping):
                        image["url"] = placeholder
                        replaced += 1
        trajectory.metadata["artifact_images"] = {
            "redacted": True, "count": replaced,
            "placeholder": placeholder,
        }

    def write(self, records):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(
            f".{self.output.name}.{os.getpid()}.tmp")
        first = None
        first_saved = None
        first_recording = None
        count = 0
        try:
            with temporary.open("w") as stream:
                for record in records:
                    value = (record if isinstance(record, Trajectory) else
                             Trajectory.from_dict(record))
                    recording = value.metadata.get("_recording_frames_dir")
                    keep_recording = False
                    try:
                        if first is None:
                            # Media needs at most one trajectory. Keeping one
                            # accepted record is bounded even when the writer is
                            # fed hundreds of image-heavy worlds.
                            first = value
                            first_recording = recording
                            keep_recording = bool(recording)
                        saved = copy.deepcopy(value)
                        saved.metadata.pop("_recording_frames_dir", None)
                        if self.redact_images:
                            self._redact_images(saved)
                        if count == 0:
                            first_saved = saved
                        elif count == 1:
                            # Only a singleton can use the readable JSON form.
                            # Release that extra copy as soon as the stream is
                            # known to contain more than one record.
                            first_saved = None
                        # Match JSONLWriter's established readable one-line
                        # encoding while writing the record incrementally.
                        json.dump(saved.to_dict(), stream, ensure_ascii=False,
                                  sort_keys=True)
                        stream.write("\n")
                        count += 1
                    finally:
                        if recording and not keep_recording:
                            shutil.rmtree(recording, ignore_errors=True)
            if count == 0:
                raise ValueError("cannot write an empty trajectory artifact")
            if (self.output.suffix == ".json" and count == 1
                    and not self.compact):
                with temporary.open("w") as stream:
                    json.dump(first_saved.to_dict(), stream,
                              ensure_ascii=False, indent=2)
                    stream.write("\n")
            os.replace(temporary, self.output)
            if self.gif is not None:
                from bedrock_rl.sdg.media import (
                    recording_frames, save_frames_gif, save_trajectory_gif,
                )
                if first_recording:
                    save_frames_gif(
                        recording_frames(first_recording,
                                         width=self.gif_width),
                        self.gif, duration_ms=self.duration_ms,
                        final_hold_ms=self.final_hold_ms)
                else:
                    save_trajectory_gif(
                        first, self.gif, width=self.gif_width,
                        duration_ms=self.duration_ms,
                        final_hold_ms=self.final_hold_ms)
            replay_result = None
            if self.replay is not None:
                from bedrock_rl.sdg.media import (
                    save_replayed_views_gif,
                )
                config = dict(self.replay)
                path = config.pop("gif")
                config.setdefault("duration_ms", self.duration_ms)
                config.setdefault("final_hold_ms", self.final_hold_ms)
                replay_result = save_replayed_views_gif(
                    first, path, **config)
            return {"trajectory": str(self.output),
                    "gif": (None if self.gif is None else str(self.gif)),
                    "replay": replay_result}
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if first_recording:
                shutil.rmtree(first_recording, ignore_errors=True)

__all__ = (
    "ParquetWriter",
    "TrajectoryArtifactWriter",
    "TrajectorySFTWriter",
    "trajectory_sft_row",
    "trajectory_sft_rows",
)
