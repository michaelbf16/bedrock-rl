"""Token-exact Netherite agent loop for the pinned verl training adapter.

This replaces the patched upstream ToolAgentLoop.  Context is represented as
token segments: model-generated token IDs are retained verbatim, while
user/tool segments have model-rendered variants with and without their image.
An image window therefore changes the next model request before generation
without Qwen token constants, site-package edits, or decode/re-encode of
sampled tokens.

When a dynamic view removes an image that an earlier assistant turn sampled
against, that earlier turn remains causal history but is masked out of the
loss.  A single verl sequence cannot represent two different historical
contexts; training it anyway would optimize a prompt the policy never saw.
"""

from __future__ import annotations

import copy
import io
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (AgentLoopBase,
                                                      AgentLoopMetrics,
                                                      AgentLoopOutput)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

from bedrock_rl.adapters.netherite.environment import NetheriteEnvironment
from bedrock_rl.adapters.netherite.probes import (EpisodeProbe,
                                                    ObservationProbe)
from bedrock_rl.adapters.netherite.tools import NetheriteComputerTool
from bedrock_rl.adapters.verl.segments import (TokenSegment,
                                                 finalize_segments,
                                                 render_segments)
from bedrock_rl.core.components import ComponentSpec
from bedrock_rl.core.context import (ContextRequest, KeepLastImages,
                                       apply_context)
from bedrock_rl.core.messages import (copy_messages, data_image_part,
                                        decode_data_image_url, is_image_part)
from bedrock_rl.core.reward import RewardResult, evaluate
from bedrock_rl.core.sampling import (EPISODE_DONE_TOOL_MESSAGE,
                                      MAX_TOOL_CALLS_PER_TURN,
                                      TOOL_CALL_LIMIT_MESSAGE)
from bedrock_rl.core.state import ProbeSet
from bedrock_rl.core.tools import (ToolCall, ToolContext, ToolRegistry,
                                     ToolResult)
from bedrock_rl.core.trajectory import TrajectoryRecorder
from bedrock_rl.data import png_bytes
from bedrock_rl.env.episode import (NO_TOOL_CALL_REWARD,
                                    TOOL_USE_REWARD_FLOOR)


def _frames_root(required: bool):
    """A stable parent for process-owned frame dirs unless configured."""
    configured = (os.environ.get("BRL_FRAMES_ROOT") or "").strip()
    if configured:
        return configured
    if not required:
        return None
    root = Path(tempfile.gettempdir()) / "bedrock-rl-verl-frames"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _keep_last(value) -> int | None:
    if value is None:
        value = os.environ.get("BRL_KEEP_LATEST_IMAGES", "0")
    value = int(value)
    if value < 0:
        raise ValueError("keep_last_images must be non-negative")
    return None if value == 0 else value


def _component(value, what):
    if isinstance(value, (str, Mapping)):
        return ComponentSpec.parse(value, what=what).create()
    return value


def _components(values, what):
    return [_component(value, f"{what} item") for value in values]


def _strip_images(messages):
    out = copy_messages(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [part for part in content
                                  if not (isinstance(part, dict)
                                          and part.get("type") in (
                                              "image", "image_url",
                                              "input_image"))]
    return out


def _image_suffix(messages, start):
    """Retain image parts at ordinal ``start`` and later."""
    out = copy_messages(messages)
    at = 0
    for message in out:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept = []
        for part in content:
            if is_image_part(part):
                if at >= start:
                    kept.append(part)
                at += 1
            else:
                kept.append(part)
        message["content"] = kept
    return out


def _canonicalize(messages, images, *, prefix="initial"):
    """Replace verl/PIL image parts with portable OpenAI data URLs."""
    out = copy_messages(messages)
    images = list(images or ())
    image_ids = []
    at = 0
    for message in out:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        replaced = []
        for part in content:
            if is_image_part(part):
                if at >= len(images):
                    raise ValueError("message has more image parts than the "
                                     "processor returned")
                image = images[at]
                replaced.append(data_image_part(png_bytes(image)))
                image_ids.append(f"{prefix}:{at}")
                at += 1
            else:
                replaced.append(copy.deepcopy(part))
        message["content"] = replaced
    if at != len(images):
        raise ValueError(f"processor returned {len(images)} images but only "
                         f"{at} message parts referenced them")
    return out, tuple(image_ids)


def _placeholder_tool_messages(messages):
    out = copy_messages(messages)
    for message in out:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                ({"type": "image"} if is_image_part(part)
                 else copy.deepcopy(part)) for part in content]
        # verl's shipped templates predate tool_call_id. The canonical tool
        # message retains it; only the renderer's compatibility view drops it.
        if message.get("role") == "tool":
            message.pop("tool_call_id", None)
            message.pop("name", None)
    return out


def _runtime_result_images(messages, attachment=None):
    """Runtime pixels matching canonical result image parts in order."""
    urls = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if is_image_part(part):
                image = part.get("image_url")
                urls.append(image.get("url") if isinstance(image, Mapping)
                            else None)
    if not urls:
        return []
    if attachment is not None and len(urls) == 1:
        return [attachment]
    from PIL import Image
    images = []
    for index, url in enumerate(urls):
        data = decode_data_image_url(url, f"tool result image {index}")
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            images.append(source.convert("RGB"))
    return images


def bounded_sampling_params(sampling_params, used_tokens, total_tokens):
    """Cap one model turn to the rollout's remaining response budget."""
    remaining = int(total_tokens) - int(used_tokens)
    if remaining <= 0:
        return None
    params = dict(sampling_params)
    requested = params.get("max_tokens")
    params["max_tokens"] = (remaining if requested is None else min(
        int(requested), remaining))
    return params if params["max_tokens"] >= 1 else None


def _multimodal_control_tokens(processor, tokenizer):
    """Tokens that represent attached media and must not be generated."""
    if processor is None:
        return ()
    tokens = []
    for name in ("image", "video", "audio", "vision_start", "vision_end",
                 "vision_pad"):
        token = getattr(processor, f"{name}_token", None)
        token_id = getattr(processor, f"{name}_token_id", None)
        if token is None and token_id is not None:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token, str) and token and token not in tokens:
            tokens.append(token)

    # Some processor classes keep the wrapper tokens only on their tokenizer.
    for token in getattr(tokenizer, "additional_special_tokens", ()):
        lowered = str(token).lower()
        if (any(label in lowered for label in ("image", "video", "audio",
                                                "vision"))
                and token not in tokens):
            tokens.append(token)
    return tuple(tokens)


def forbid_multimodal_output(sampling_params, forbidden_tokens):
    """Merge media control tokens into vLLM's generation-time denylist."""
    if not forbidden_tokens:
        return sampling_params
    params = dict(sampling_params)
    configured = params.get("bad_words") or ()
    if isinstance(configured, str):
        configured = (configured,)
    params["bad_words"] = list(dict.fromkeys(
        (*configured, *forbidden_tokens)))
    return params


def _trajectory_root():
    raw = (os.environ.get("BRL_TRAJECTORY_DIR") or "").strip()
    if not raw:
        return None
    if raw.lower() in ("1", "true", "on", "yes"):
        raw = os.path.join("outputs", f"{os.environ.get('BRL_SLUG', 'run')}"
                           "_trajectories")
    return Path(os.path.expanduser(raw))


async def _rollout_reward(verifier, recorder, session, *, tool_calls,
                          computer_calls, format_penalty=0.0):
    """Score one rollout while preserving the no-format ordering floor."""
    metrics = {"success": bool(session.episode.success),
               "turns": int(session.episode.turns),
               "tool_calls": int(tool_calls),
               "computer_calls": int(computer_calls)}
    if not tool_calls:
        score = float(NO_TOOL_CALL_REWARD)
        return RewardResult(score, {"bedrock": score}, metrics)
    if verifier is not None:
        base = await evaluate(verifier, recorder.trajectory, recorder.state)
    else:
        score = float(session.episode.final_reward())
        base = RewardResult(score, {"bedrock": score})
    score = max(
        base.total - float(format_penalty), TOOL_USE_REWARD_FLOOR)
    components = dict(base.components)
    if score != base.total:
        components["bedrock_format"] = score - base.total
    # Adapter counters describe the live rollout and cannot be overwritten by
    # a verifier returning stale or unrelated fields with the same names.
    metrics = {**base.metrics, **metrics}
    metrics["unaccounted_malformed_penalty"] = float(format_penalty)
    return RewardResult(score, components, metrics, base.details)


def _counts_computer_turn(tools, call):
    """Whether this call will spend the Minecraft turn budget."""
    return (call.name == "computer"
            or (call.name not in tools
                and tools.malformed_fallback_name == "computer"))


class NetheriteAgentLoop(AgentLoopBase):
    def __init__(self, *args, keep_last_images=None,
                 save_trajectories=None, core_tools=None, probes=None,
                 verifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_last_images = _keep_last(keep_last_images)
        self.save_trajectories = (Path(save_trajectories)
                                  if save_trajectories else None)
        # ``core_tools`` are extensions, not a replacement list. A custom
        # Minecraft tool should not force every config to restate the built-in
        # computer tool just to keep the environment drivable. A deliberately
        # custom computer implementation still wins by naming itself
        # ``computer``.
        selected_tools = ([] if core_tools is None else
                          _components(core_tools, "core_tools"))
        if not any(getattr(tool, "name", None) == "computer"
                   for tool in selected_tools):
            selected_tools.insert(0, NetheriteComputerTool())
        self.tools = ToolRegistry(selected_tools)
        if self.tools.malformed_fallback_name != "computer":
            raise ValueError(
                "the Netherite computer tool must expose callable malformed "
                "accounting and malformed_fallback=True so invented tool "
                "names cannot bypass the episode reward path")
        self.tool_schemas = [OpenAIFunctionToolSchema(**schema)
                             for schema in self.tools.schemas]
        self.tool_parser = ToolParser.get_tool_parser(
            self.rollout_config.multi_turn.format, self.tokenizer)
        self.response_length = int(self.rollout_config.response_length)
        self.forbidden_output_tokens = _multimodal_control_tokens(
            getattr(self, "processor", None), self.tokenizer)
        self.probes = (ProbeSet((EpisodeProbe(), ObservationProbe()))
                       if probes is None else
                       ProbeSet(_components(probes, "probes")))
        self.verifier = (None if verifier is None else
                         _component(verifier, "verifier"))

    async def _capture(self, recorder, session, turn, phase):
        values = await self.probes.capture(session)
        recorder.record_state(turn, phase, values)
        events = session.drain_events()
        if events:
            recorder.record_events(events)
        return values

    async def _view(self, recorder, turn, task, state):
        request = ContextRequest(turn, task, state, "policy")
        policy = (None if self.keep_last_images is None else
                  KeepLastImages(self.keep_last_images))
        view = await apply_context(policy, recorder.messages, request)
        recorder.record_view("policy", turn, view)
        return view

    async def _initial_segment(self, raw_messages, images, image_ids):
        with_images = await self.apply_chat_template(
            raw_messages, tools=self.tools.schemas, images=images)
        without_images = None
        variants = {}
        if image_ids:
            without_images = await self.apply_chat_template(
                _strip_images(raw_messages), tools=self.tools.schemas)
            # A global last-N window can cut through a message containing
            # several images.  Keep an exact rendering for every non-empty
            # suffix rather than forcing the whole message in or out.
            for start in range(1, len(image_ids)):
                retained = tuple(image_ids[start:])
                variants[retained] = await self.apply_chat_template(
                    _image_suffix(raw_messages, start),
                    tools=self.tools.schemas, images=images[start:])
        return TokenSegment("initial", with_images, image_ids,
                            tuple(images or ()), without_images,
                            image_variants=variants)

    async def _tool_segment(self, messages, images, image_ids):
        rendered = _placeholder_tool_messages(messages)
        with_image = await self.apply_chat_template(
            rendered, images=images or None,
            remove_system_prompt=True)
        without_image = None
        variants = {}
        if images:
            without_image = await self.apply_chat_template(
                _strip_images(rendered), remove_system_prompt=True)
            for start in range(1, len(image_ids)):
                retained = tuple(image_ids[start:])
                variants[retained] = await self.apply_chat_template(
                    _image_suffix(rendered, start),
                    images=images[start:], remove_system_prompt=True)
        return TokenSegment("tool", with_image, tuple(image_ids),
                            tuple(images),
                            without_image, image_variants=variants)

    async def _save(self, trajectory):
        root = self.save_trajectories or _trajectory_root()
        if root is None:
            return None
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{trajectory.id}.json"
        await self.loop.run_in_executor(None, trajectory.save, path)
        return str(path)

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        raw_messages = copy_messages(kwargs["raw_prompt"])
        multi_modal = await self.process_multi_modal_info(raw_messages)
        images = list(multi_modal.get("images") or ())
        if len(images) != 1:
            raise ValueError(
                "NetheriteAgentLoop requires exactly one initial frame; "
                f"the rollout prompt supplied {len(images)}")
        canonical, image_ids = _canonicalize(raw_messages, images)

        tools_kwargs = kwargs.get("tools_kwargs") or {}
        create = ((tools_kwargs.get("computer") or {})
                  .get("create_kwargs") or {})
        if "task_yaml" not in create:
            raise ValueError("NetheriteAgentLoop needs tools_kwargs.computer."
                             "create_kwargs.task_yaml in every dataset row")
        environment = NetheriteEnvironment(
            create["task_yaml"], episode=create,
            frames_root=_frames_root(bool(images)))
        generic_task = SimpleNamespace(
            name=Path(str(create["task_yaml"])).stem,
            instruction="", path=None, metadata={})
        session = await environment.open(generic_task)
        task = session.task
        recorder = TrajectoryRecorder(
            task={"name": task.name, "path": task.path,
                  "instruction": task.goal},
            provenance={"adapter": "verl", "request": uuid4().hex,
                        "keep_last_images": self.keep_last_images,
                        "view": session.episode.view_provenance},
            metadata={"assistant_raw": []})
        recorder.extend(canonical)
        segments = [await self._initial_segment(raw_messages, images,
                                                image_ids)]
        state = await self._capture(recorder, session, 0, "reset")
        request_id = uuid4().hex
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0,
                   "compute_score": 0.0, "num_preempted": -1}
        extra_fields = {}
        executed = 0
        computer_executed = 0
        generic_format_penalty = 0.0
        expect_frames = bool(images)
        assistant_turns = 0
        response_tokens = 0
        has_logprobs = None
        try:
            while assistant_turns < task.max_turns:
                await self._view(recorder, assistant_turns, task, state)
                prompt_ids, active_images, active_ids = render_segments(
                    segments, self.keep_last_images)
                params = bounded_sampling_params(
                    sampling_params, response_tokens, self.response_length)
                if params is None:
                    break
                params = forbid_multimodal_output(
                    params, getattr(self, "forbidden_output_tokens", ()))
                if self.tool_parser.stop_token_ids:
                    params["stop_token_ids"] = list(set(
                        (params.get("stop_token_ids") or [])
                        + self.tool_parser.stop_token_ids))
                before = time.monotonic()
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids,
                    sampling_params=params,
                    image_data=active_images or None,
                    video_data=None, audio_data=None,
                    mm_processor_kwargs=self._get_mm_processor_kwargs())
                metrics["generate_sequences"] += time.monotonic() - before
                preempted = output.num_preempted
                if metrics["num_preempted"] < 0:
                    metrics["num_preempted"] = preempted or 0
                else:
                    metrics["num_preempted"] += preempted or 0
                if output.routed_experts is not None:
                    raise ValueError("NetheriteAgentLoop context segments do "
                                     "not support routed_experts traces")
                if not extra_fields:
                    extra_fields.update(output.extra_fields)
                turn_has_logprobs = output.log_probs is not None
                if has_logprobs is None:
                    has_logprobs = turn_has_logprobs
                elif has_logprobs != turn_has_logprobs:
                    raise ValueError(
                        "rollout server returned logprobs for only some "
                        "assistant turns")
                if turn_has_logprobs and len(output.log_probs) != len(
                        output.token_ids):
                    raise ValueError(
                        "rollout server logprobs do not align with token ids")
                assistant = TokenSegment(
                    "assistant", output.token_ids,
                    sampled_image_ids=active_ids,
                    logprobs=output.log_probs or [])
                segments.append(assistant)
                assistant_turns += 1
                response_tokens += len(output.token_ids)

                raw = await self.loop.run_in_executor(
                    None, self.tokenizer.decode, output.token_ids)
                recorder.trajectory.metadata["assistant_raw"].append(raw)
                parse_failure = None
                try:
                    content, calls = await self.tool_parser.extract_tool_calls(
                        output.token_ids, self.tool_schemas)
                except Exception as exc:
                    content, calls = raw, []
                    parse_failure = exc

                parsed_calls = []
                malformed = []
                malformed_metadata = {}
                for index, call in enumerate(calls):
                    call_id = f"call_{assistant_turns}_{index}"
                    wire = {
                        "id": call_id, "type": "function",
                        "function": {"name": call.name,
                                     "arguments": call.arguments},
                    }
                    try:
                        parsed = ToolCall.from_openai(wire)
                        reason = None
                    except Exception as exc:
                        # Canonical chat remains valid OpenAI JSON. The exact
                        # sampled bytes remain in assistant_raw/token segments.
                        parsed = ToolCall(call_id,
                                          str(call.name or "computer"), {})
                        reason = exc
                        malformed_metadata[call_id] = str(exc)
                    parsed_calls.append(parsed)
                    malformed.append(reason)
                if parse_failure is not None:
                    parsed_calls = [ToolCall(
                        f"call_{assistant_turns}_0", "computer", {})]
                    malformed = [parse_failure]
                    malformed_metadata = {
                        parsed_calls[0].id: str(parse_failure)}

                openai_calls = [call.to_openai() for call in parsed_calls]
                assistant_message = {"role": "assistant",
                                     "content": content or None}
                if openai_calls:
                    assistant_message["tool_calls"] = openai_calls
                if malformed_metadata:
                    assistant_message["metadata"] = {
                        "malformed_tool_calls": malformed_metadata}
                recorder.append(assistant_message)
                if not openai_calls:
                    break
                stop = False
                resolved = 0
                turn_messages = []
                observations = []
                turn_images = []
                turn_image_ids = []
                computer_turn_charged = False
                for index, (parsed, reason) in enumerate(zip(
                        parsed_calls[:MAX_TOOL_CALLS_PER_TURN],
                        malformed[:MAX_TOOL_CALLS_PER_TURN])):
                    before = time.monotonic()
                    will_count_computer = _counts_computer_turn(
                        self.tools, parsed)
                    context = ToolContext(
                        task, session, recorder.trajectory,
                        assistant_turns - 1, state,
                        {"count_turn": (will_count_computer
                                        and not computer_turn_charged)})
                    accounted_as_computer = False
                    if reason is not None:
                        result, accounted_by = (
                            await self.tools.account_malformed(
                                parsed, reason, context))
                        accounted_as_computer = accounted_by == "computer"
                        if not accounted_as_computer:
                            # Stateless extension tools cannot mutate the
                            # Minecraft episode's penalty accumulator. Charge
                            # their malformed syntax at the adapter boundary,
                            # whether or not they provide a diagnostic handler.
                            generic_format_penalty += 0.2
                        if result is None:
                            result = ToolResult(
                                f"Malformed tool call: {reason}",
                                {"malformed": str(reason)})
                    elif parsed.name not in self.tools:
                        error = ValueError(
                            f"unknown tool {parsed.name!r}; available: "
                            f"{sorted(self.tools.names) or 'none'}")
                        result, accounted_by = (
                            await self.tools.account_malformed(
                                parsed, error, context))
                        accounted_as_computer = accounted_by == "computer"
                        if result is None:
                            result = ToolResult(
                                f"Tool call failed: {error}",
                                {"malformed": str(error)})
                    else:
                        # Execution failures are infrastructure/tool failures,
                        # not malformed model output. Let them fail the rollout
                        # rather than spending a Minecraft turn and charging
                        # the policy for a transient engine error.
                        result = await self.tools.execute(parsed, context)
                    metrics["tool_calls"] += time.monotonic() - before
                    executed += 1
                    computer_executed += (
                        parsed.name == "computer" or accounted_as_computer)
                    if parsed.name == "computer" or accounted_as_computer:
                        computer_turn_charged = True
                    recorder.record_tool_result(
                        assistant_turns - 1, parsed, result)
                    result_messages = result.messages(
                        parsed.id, name=parsed.name)
                    recorder.append(result_messages[0])
                    turn_messages.append(result_messages[0])
                    observations.extend(result_messages[1:])
                    resolved = index + 1

                    # Runtime attachment, not a second screenshot. Taking
                    # another frame here would spend another game tick.
                    frame = result.attachments.get("image")
                    if (expect_frames
                            and (parsed.name == "computer"
                                 or accounted_as_computer)
                            and frame is None):
                        raise RuntimeError(
                            "visual rollout executed a computer turn without "
                            "a frame; rendering must remain enabled for every "
                            "policy observation")
                    result_images = _runtime_result_images(
                        result_messages[1:], frame)
                    if len(result_images) > 1:
                        raise RuntimeError(
                            f"tool {parsed.name!r} returned "
                            f"{len(result_images)} images; the rollout "
                            "contract permits at most one image per call")
                    turn_images.extend(result_images)
                    turn_image_ids.extend(
                        f"tool:{assistant_turns}:{index}:{image_index}"
                        for image_index in range(len(result_images)))
                    if result.done or session.done:
                        stop = True
                        break
                # The model may emit multiple calls in one assistant turn.
                # Ending the episode or exhausting the response budget after
                # the first does not erase the others from that OpenAI turn;
                # close each one with an explicit non-executed result so the
                # saved trajectory remains structurally valid.
                for parsed in parsed_calls[resolved:]:
                    content = (
                        EPISODE_DONE_TOOL_MESSAGE
                        if stop else TOOL_CALL_LIMIT_MESSAGE)
                    skipped = ToolResult(
                        content,
                        {"skipped": True}, done=bool(session.done))
                    recorder.record_tool_result(
                        assistant_turns - 1, parsed, skipped)
                    skipped_messages = skipped.messages(
                        parsed.id, name=parsed.name)
                    recorder.append(skipped_messages[0])
                    turn_messages.append(skipped_messages[0])
                recorder.extend(observations)
                turn_messages.extend(observations)
                segment = await self._tool_segment(
                    turn_messages, turn_images, turn_image_ids)
                if (response_tokens + len(segment.token_ids)
                        >= self.response_length):
                    stop = True
                else:
                    segments.append(segment)
                    response_tokens += len(segment.token_ids)
                state = await self._capture(recorder, session, assistant_turns,
                                            "after_tools")
                if stop:
                    break

            await self._capture(recorder, session, assistant_turns, "terminal")
            score_started = time.monotonic()
            reward = await _rollout_reward(
                self.verifier, recorder, session, tool_calls=executed,
                computer_calls=computer_executed,
                format_penalty=generic_format_penalty)
            score = reward.total
            trajectory = recorder.finish(reward)
            trajectory_path = await self._save(trajectory)
            (prompt_ids, response_ids, response_mask, response_logprobs,
             final_images, masked) = finalize_segments(
                 segments, self.keep_last_images)
            metrics["compute_score"] = time.monotonic() - score_started
            extra_fields.update({
                "trajectory_id": trajectory.id,
                "trajectory_path": trajectory_path,
                "context_masked_assistant_turns": masked,
                "success": bool(session.episode.success),
            })
            mm = {"images": final_images} if final_images else {}
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[:self.response_length],
                response_mask=response_mask[:self.response_length],
                response_logprobs=(
                    response_logprobs[:self.response_length]
                    if has_logprobs else None),
                multi_modal_data=mm,
                mm_processor_kwargs=self._get_mm_processor_kwargs(),
                reward_score=score, num_turns=assistant_turns + executed + 1,
                metrics=AgentLoopMetrics(**metrics),
                extra_fields=extra_fields)
        finally:
            await session.close()
