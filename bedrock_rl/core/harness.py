"""A small reference tool harness; custom harnesses implement the same run API."""

from __future__ import annotations

import copy
import inspect
from typing import Any, Mapping, Protocol

from bedrock_rl.core.context import (ContextRequest, ContextView,
                                       apply_context)
from bedrock_rl.core.guidance import GuidanceRequest, NoGuidance
from bedrock_rl.core.messages import validate_message, validate_messages
from bedrock_rl.core.reward import RewardResult, evaluate
from bedrock_rl.core.sampling import (EPISODE_DONE_TOOL_MESSAGE,
                                      MAX_TOOL_CALLS_PER_TURN,
                                      TOOL_CALL_LIMIT_MESSAGE)
from bedrock_rl.core.state import ProbeSet
from bedrock_rl.core.tools import (ToolCall, ToolContext, ToolRegistry,
                                     ToolResult)
from bedrock_rl.core.trajectory import Trajectory, TrajectoryRecorder


async def _await(value):
    return await value if inspect.isawaitable(value) else value


class Harness(Protocol):
    async def run(self, task, **kwargs) -> Trajectory: ...


class ChatModel(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict],
                       sampling: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _assistant_message(value: Any, *, call_prefix: str):
    """Normalize one provider response without letting model syntax abort it.

    Canonical trajectories cannot contain invalid OpenAI tool calls. A model
    can still emit one, so retain a valid stand-in call and return the parse
    error beside it. The harness then answers that call with an ordinary tool
    error message instead of crashing the whole trial.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise TypeError(f"model returned {type(value).__name__}; expected an "
                        "OpenAI assistant message")
    value = dict(value)
    if "choices" in value:
        choices = value["choices"]
        if not choices:
            raise ValueError("model completion contains no choices")
        choice = choices[0]
        if hasattr(choice, "model_dump"):
            choice = choice.model_dump()
        value = choice.get("message") if isinstance(choice, Mapping) else None
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if not isinstance(value, Mapping):
            raise ValueError("model completion choice contains no message")
        value = dict(value)
    value.setdefault("role", "assistant")
    value.setdefault("content", None if value.get("tool_calls") else "")
    if value["role"] != "assistant":
        raise ValueError(f"model returned role {value['role']!r}, expected "
                         "assistant")
    raw_calls = value.pop("tool_calls", None) or ()
    if not isinstance(raw_calls, (list, tuple)):
        raise TypeError("model assistant tool_calls must be a list")
    calls = []
    malformed = {}
    malformed_wire = {}
    seen = set()
    for index, raw in enumerate(raw_calls):
        fallback_id = f"{call_prefix}_{index}"
        call_id = (str(raw.get("id")) if isinstance(raw, Mapping)
                   and raw.get("id") else fallback_id)
        if call_id in seen:
            call_id = fallback_id
        suffix = 1
        while call_id in seen:
            call_id = f"{fallback_id}_{suffix}"
            suffix += 1
        seen.add(call_id)
        function = raw.get("function") if isinstance(raw, Mapping) else None
        name = (str(function.get("name")) if isinstance(function, Mapping)
                and function.get("name") else "malformed")
        try:
            candidate = copy.deepcopy(dict(raw))
            candidate["id"] = call_id
            call = ToolCall.from_openai(candidate)
        except Exception as exc:
            call = ToolCall(call_id, name, {})
            malformed[call_id] = str(exc)
            malformed_wire[call_id] = copy.deepcopy(raw)
        calls.append(call.to_openai())
    if malformed_wire:
        value.setdefault("metadata", {})["malformed_tool_calls"] = (
            malformed_wire)
    if calls:
        value["tool_calls"] = calls
    validate_message(value)
    return value, malformed


async def _session_done(session) -> bool:
    value = getattr(session, "done", False)
    if callable(value):
        value = value()
    return bool(await _await(value))


async def _drain_events(session, recorder):
    drain = getattr(session, "drain_events", None)
    if drain is None:
        return
    events = await _await(drain())
    if events:
        recorder.record_events(events)


class ToolHarness:
    """Reference multi-turn harness over OpenAI messages and arbitrary tools.

    The class is intentionally replaceable.  A custom agent harness may do
    planning, reflection, parallel tool calls, sub-agents, or anything else
    and still emit the same ``Trajectory`` artifact.
    """

    def __init__(self, model: ChatModel, *, environment=None, tools=(),
                 context=None, guidance=None, probes=(), verifier=None,
                 max_turns: int = 8, view_name: str = "policy",
                 provenance: Mapping[str, Any] | None = None):
        if int(max_turns) < 1:
            raise ValueError("max_turns must be at least one")
        self.model = model
        self.environment = environment
        self.tools = (tools if isinstance(tools, ToolRegistry)
                      else ToolRegistry(tools))
        self.context = context
        self.guidance = guidance or NoGuidance()
        self.probes = probes if isinstance(probes, ProbeSet) else ProbeSet(probes)
        self.verifier = verifier
        self.max_turns = int(max_turns)
        self.view_name = str(view_name)
        self.provenance = dict(provenance or {})

    async def _open(self, task):
        environment = self.environment
        if environment is None:
            environment = task.build_environment()
        opener = getattr(environment, "open", environment)
        return await _await(opener(task))

    async def _initial_messages(self, session, task):
        initial = getattr(session, "initial_messages", None)
        if initial is None:
            return [{"role": "user", "content": task.instruction}]
        messages = initial(task) if callable(initial) else initial
        return validate_messages(await _await(messages))

    async def _capture(self, recorder, session, turn, phase):
        values = await self.probes.capture(session)
        recorder.record_state(turn, phase, values)
        await _drain_events(session, recorder)
        return values

    async def run(self, task, *, sampling: Mapping[str, Any] | None = None,
                  trajectory_id: str | None = None,
                  metadata: Mapping[str, Any] | None = None) -> Trajectory:
        task_record = {"name": task.name, "instruction": task.instruction,
                       "path": getattr(task, "path", None),
                       "metadata": getattr(task, "metadata", {})}
        recorder = TrajectoryRecorder(
            task=task_record, provenance=self.provenance,
            metadata=metadata, trajectory_id=trajectory_id)
        session = await self._open(task)
        try:
            recorder.extend(await self._initial_messages(session, task))
            state = await self._capture(recorder, session, 0, "reset")
            for turn in range(self.max_turns):
                request = ContextRequest(turn, task, state, self.view_name,
                                         dict(metadata or {}))
                view = await apply_context(self.context, recorder.messages,
                                           request)
                guided = self.guidance.apply(
                    view, GuidanceRequest(turn, task, state, self.view_name,
                                          dict(metadata or {})))
                guided = await _await(guided)
                if not isinstance(guided, ContextView):
                    raise TypeError("guidance policy must return ContextView")
                recorder.record_view(self.view_name, turn, guided)
                response = self.model.complete(
                    copy.deepcopy(guided.messages), self.tools.schemas,
                    dict(sampling or {}))
                response, malformed = _assistant_message(
                    await _await(response), call_prefix=f"call_{turn + 1}")
                recorder.append(response)
                calls = [ToolCall.from_openai(call)
                         for call in (response.get("tool_calls") or ())]
                if not calls:
                    break
                stop = False
                observations = []
                for call_index, call in enumerate(calls):
                    if stop:
                        # An assistant may emit several calls in one message.
                        # Once an earlier call ends the session, executing the
                        # remainder mutates a terminal world.  We still owe
                        # every emitted call a matching tool result so the
                        # canonical OpenAI transcript is not left unresolved.
                        result = ToolResult(
                            EPISODE_DONE_TOOL_MESSAGE,
                            {"skipped": True, "reason": "episode_done"},
                            done=True)
                        recorder.record_tool_result(turn, call, result)
                        recorder.append(result.messages(
                            call.id, name=call.name)[0])
                        continue
                    if call_index >= MAX_TOOL_CALLS_PER_TURN:
                        result = ToolResult(
                            TOOL_CALL_LIMIT_MESSAGE,
                            {"skipped": True, "reason": "turn_call_limit"})
                        recorder.record_tool_result(turn, call, result)
                        recorder.append(result.messages(
                            call.id, name=call.name)[0])
                        continue
                    call_metadata = dict(metadata or {})
                    call_metadata["count_turn"] = call_index == 0
                    context = ToolContext(task, session, recorder.trajectory,
                                          turn, state, call_metadata)
                    reason = malformed.get(call.id)
                    if reason is not None:
                        result, _accounted_by = await self.tools.account_malformed(
                            call, reason, context)
                        if result is None:
                            result = ToolResult(
                                f"Malformed tool call: {reason}",
                                {"malformed": reason})
                    elif call.name not in self.tools:
                        reason = (f"unknown tool {call.name!r}; available: "
                                  f"{sorted(self.tools.names) or 'none'}")
                        result, _accounted_by = (
                            await self.tools.account_malformed(
                                call, reason, context))
                        if result is None:
                            result = ToolResult(
                                f"Tool call failed: {reason}",
                                {"malformed": reason})
                    else:
                        result = await self.tools.execute(call, context)
                    recorder.record_tool_result(turn, call, result)
                    messages = result.messages(call.id, name=call.name)
                    recorder.append(messages[0])
                    observations.extend(messages[1:])
                    stop = bool(result.done) or await _session_done(session)
                recorder.extend(observations)
                state = await self._capture(recorder, session, turn + 1,
                                            "after_tools")
                if stop or await _session_done(session):
                    break
            # Terminal capture is explicit even when no tool was called.  A
            # verifier can therefore distinguish refusal from an unchanged
            # environment without inferring it from message shape.
            await self._capture(recorder, session, len(recorder.trajectory.views),
                                "terminal")
            verifier = self.verifier
            if verifier is None and getattr(task, "verifier", None) is not None:
                verifier = task.build_verifier()
            reward = (RewardResult() if verifier is None else
                      await evaluate(verifier, recorder.trajectory,
                                     recorder.state))
            return recorder.finish(reward)
        finally:
            close = getattr(session, "close", None)
            if close is not None:
                await _await(close())
