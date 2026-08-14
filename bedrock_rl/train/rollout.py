"""Multi-turn rollout driver shared by SDPO and SDFT.

Episodes execute through :class:`bedrock_rl.env.episode.Episode` and are scored
from live engine state. Batches generate each active trajectory's next turn
together, then step their isolated environments concurrently.

The Hugging Face backend samples directly from the training module. The vLLM
backend adds paged KV caching, continuous batching, and cross-turn prefix
caching; LoRA weights are refreshed every step to keep SDPO on-policy. Per-turn
hints are attached here, recorded on observation messages, and never exposed to
the state-based reward path.
"""
import json
import copy
import os
import random
import shutil
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import torch

from bedrock_rl.adapters.netherite.chat import IMAGE_TAG
from bedrock_rl.adapters.netherite.computer import (NEXT_COMPUTER_MESSAGE,
                                     TOOL_NAME, cu_parser, cu_payload,
                                     cu_user_prompt, tool_call_attempts)
from bedrock_rl.sdg import guidance as hint_mod
from bedrock_rl.sdg.guidance import with_hint
from bedrock_rl.adapters.netherite.tools import malformed_computer_step
from bedrock_rl.core.sampling import (EPISODE_DONE_TOOL_MESSAGE,
                                      MAX_TOOL_CALLS_PER_TURN,
                                      TOOL_CALL_LIMIT_MESSAGE,
                                      multimodal_image_limit,
                                      sampling_knobs)
from bedrock_rl.core.messages import data_image_part, text_part
from bedrock_rl.core.reward import RewardResult
from bedrock_rl.core.state import StateEvent
from bedrock_rl.core.trajectory import Trajectory
from bedrock_rl.core.tools import ToolCall
from bedrock_rl.data import png_bytes
from bedrock_rl.env.episode import (Episode, NO_TOOL_CALL_REWARD,
                                      tool_response_text)


# The stream the per-episode HINT_P coin comes off. Its OWN generator,
# DERIVED from --seed rather than shared with it. Two things had to be
# true at once. It must not come off the same stream as the episode
# choice, because --seed also picks the EPISODES and one stream would
# make which episodes are hinted a function of which episodes were
# chosen, so an anneal from p=1 to p=0 would walk a correlated subset
# rather than a random one. And it must be REPRODUCIBLE, because a p<1
# run whose hinted set came off OS entropy cannot be replayed and its
# paired evaluation is not paired. A derived seed gets both, and the
# offset is arbitrary and fixed.
#
# The stream belongs to the RUN, not to the process. Seeding one module
# global from whichever `args` reached play() first meant a second
# trainer, a sweep cell, or an evaluation constructed in the same process
# drew from the first caller's stream: its own --seed was read and
# discarded, so which of ITS episodes got a hint stopped being a function
# of its own seed, and a p<1 arm was no longer replayable from the flags
# it logged. Keying the generator on the args object gives every run its
# own stream, derived from its own seed, advancing across that run's
# play() calls exactly as before.
#
# Keyed by the args object's identity, and released when it dies:
# argparse.Namespace compares by value and is therefore unhashable, so a
# WeakKeyDictionary cannot hold one, and a bare id() key would outlive
# the object and could be handed to whatever the allocator put at that
# address next.
_HINT_SEED_OFFSET = 0x51157
_HINT_RNGS = {}


def _leading_image_markers(content, role=None):
    """Count reserved image placeholders at the start of a message.

    The string-based trainer renderer reserves only leading ``<image>`` tokens;
    occurrences in ordinary prose remain literal text.
    """
    if role not in ("user", "tool") or not isinstance(content, str):
        return 0
    count = 0
    offset = 0
    while content.startswith(IMAGE_TAG, offset):
        count += 1
        offset += len(IMAGE_TAG)
    return count


@dataclass(frozen=True)
class SampledTurn:
    """One generation exactly as the sampler saw it.

    Distillation must train on ``response_ids`` directly.  Re-rendering a
    parsed assistant message loses free-form prose and normalizes JSON and
    whitespace, which silently turns an on-policy update into an update on a
    different sequence.  The prompt and image view are captured at the same
    boundary so every later teacher-forced forward pass describes this turn,
    not the completed trajectory.
    """

    text: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    messages: tuple[dict, ...]
    teacher_messages: tuple[dict, ...]
    images: tuple[object, ...]
    image_pads: tuple[int, ...]
    sampling_log_probs: tuple[float, ...] = ()


def _hint_rng(args):
    """The per-episode coin's generator for THIS run, derived from this
    run's own seed and created once."""
    key = id(args)
    rng = _HINT_RNGS.get(key)
    if rng is None:
        rng = random.Random(int(getattr(args, "seed", 0)) + _HINT_SEED_OFFSET)
        _HINT_RNGS[key] = rng
        try:
            weakref.finalize(args, _HINT_RNGS.pop, key, None)
        except TypeError:
            # a namespace that cannot be weakly referenced keeps its
            # entry for the life of the process, which is what the whole
            # table used to do
            pass
    return rng


class Rollout:
    def __init__(self, spec, group):
        self.spec = spec
        self.group = group
        self.ep = None
        self.messages = []
        self.images = []
        self.done = False
        self.reward = 0.0
        self.success = False
        self.turns = 0
        self.stopped = False        # ended its own turn instead of running out
        self.truncated = False      # cut off mid tool call by the token budget
        self.assistant_raw = []     # exact decoded sampler output, for audit
        self.sampled_turns = []     # exact prompt/response token boundaries
        self.state_samples = []
        # this episode's per-turn hint state, incl. the once-per-group
        # show/hide decision and the record of every suffix written
        self.hinter = None

    @property
    def hint_marks(self):
        """(message index, exact string) for every per-turn hint this
        trajectory carries, which is what makes removing them exact."""
        return tuple(self.hinter.marks) if self.hinter else ()

    @property
    def turn_hinted(self):
        """Did THIS episode's per-episode coin come up hinted?

        Recorded because `strip` erases the only other evidence. Without
        it a p<1 harvest writes a parquet in which the hinted and the
        un-hinted rows are indistinguishable, which is the one experiment
        the dial exists for."""
        return bool(self.hinter and self.hinter.on)

    def response_text(self):
        # SDPO demonstrations need the exact syntax this model family
        # sampled, not a second OpenAI-shaped serialization of parsed calls.
        return "\n".join(self.assistant_raw)

    def capture_state(self, turn, phase):
        """Keep reward-relevant game state beside, never inside, chat."""
        from bedrock_rl.adapters.netherite.probes import (
            DEFAULT_OBSERVATION_FIELDS, capture_observation,
        )
        self.state_samples.append((int(turn), str(phase), {
            "episode": {
                "done": bool(self.ep.done),
                "success": bool(self.ep.success),
                "failed": bool(self.ep.failed),
                "turns": int(self.ep.turns),
                "actions": int(self.ep.nlines),
                "reward": float(self.ep.final_reward()),
                "spec": dict(self.ep.spec),
            },
            "observation": capture_observation(
                self.ep, DEFAULT_OBSERVATION_FIELDS),
        }))

    def canonical_messages(self):
        """Embed rollout frames as portable OpenAI ``image_url`` parts."""
        images = iter(self.images)
        used = 0
        output = []
        deferred_visuals = []
        for source in self.messages:
            if source.get("role") != "tool" and deferred_visuals:
                output.append({"role": "user",
                               "content": deferred_visuals})
                deferred_visuals = []
            message = copy.deepcopy(dict(source))
            for call in message.get("tool_calls") or ():
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, dict):
                    call["function"]["arguments"] = json.dumps(
                        arguments, separators=(",", ":"))
            content = message.get("content")
            marker_count = _leading_image_markers(content,
                                                   message.get("role"))
            if marker_count:
                parts = []
                for _ in range(marker_count):
                    try:
                        frame = next(images)
                    except StopIteration as exc:
                        raise ValueError("rollout messages contain more "
                                         "image markers than frames") from exc
                    encoded = png_bytes(frame)
                    if encoded is None:
                        raise ValueError("rollout image marker references a "
                                         "missing frame")
                    parts.append(data_image_part(encoded))
                    used += 1
                remainder = content[marker_count * len(IMAGE_TAG):]
                if remainder:
                    # A later literal <image> is prose, not a frame marker.
                    parts.append(text_part(remainder))
                if message.get("role") == "tool":
                    message["content"] = [
                        part for part in parts if part["type"] == "text"]
                    visual = [part for part in parts
                              if part["type"] == "image_url"]
                    output.append(message)
                    deferred_visuals.extend(visual)
                    continue
                message["content"] = parts
            output.append(message)
        if deferred_visuals:
            output.append({"role": "user", "content": deferred_visuals})
        try:
            next(images)
        except StopIteration:
            return output
        raise ValueError(f"rollout carries more frames than its {used} image "
                         "markers")

    def exact_messages(self):
        """Return model history with every sampled assistant turn verbatim.

        ``self.messages`` is the canonical artifact: parsed calls are stored as
        structured OpenAI ``tool_calls`` so trajectories validate and tools can
        be audited.  It is not the context the policy sampled, though.  Parsed
        calls normalize JSON whitespace and cannot retain prose surrounding a
        tool envelope.  Replace only those assistant records with their exact
        decoded outputs for subsequent model turns and self-distillation.

        Hand-built Rollouts used by exporters may have no ``assistant_raw``;
        those retain their supplied canonical messages unchanged.
        """
        if not self.assistant_raw:
            return [copy.deepcopy(dict(message))
                    for message in self.messages]
        assistant_count = sum(
            message.get("role") == "assistant" for message in self.messages)
        if assistant_count != len(self.assistant_raw):
            raise ValueError(
                f"rollout has {assistant_count} canonical assistant messages "
                f"but {len(self.assistant_raw)} sampled assistant turns")
        raw = iter(self.assistant_raw)
        output = []
        for source in self.messages:
            message = copy.deepcopy(dict(source))
            if message.get("role") == "assistant":
                message = {"role": "assistant", "content": next(raw)}
            output.append(message)
        return output

    def to_trajectory(self, task):
        trajectory = Trajectory(
            messages=self.canonical_messages(),
            task={"name": task.name, "path": task.path,
                  "instruction": task.goal},
            reward=RewardResult(
                float(self.reward), {"bedrock": float(self.reward)},
                {"success": bool(self.success), "turns": int(self.turns)}),
            provenance={"adapter": "local-rollout",
                        "trainer_run": os.environ.get("BRL_SLUG"),
                        "view": self.spec.get("view")},
            metadata={"episode": dict(self.spec),
                      "assistant_raw": list(self.assistant_raw),
                      "assistant_token_ids": [
                          list(turn.response_ids) for turn in self.sampled_turns
                      ],
                      "assistant_sampling_log_probs": [
                          (list(turn.sampling_log_probs)
                           if turn.sampling_log_probs else None)
                          for turn in self.sampled_turns
                      ],
                      "turn_hints": list(self.hint_marks)},
            id=uuid4().hex)
        for turn, phase, values in self.state_samples:
            trajectory.state.record(turn, phase, values)
        if self.ep is not None:
            for event in self.ep.env.journal:
                value = event.as_dict()
                name = value.pop("event")
                tick = value.pop("tick")
                trajectory.state.extend_events(
                    [StateEvent(name, value, int(tick))])
        import time
        trajectory.finished_at = time.time()
        return trajectory

    def model_view(self):
        """Return string-based trainer messages and the configured image window.

        The framework-level policy operates on typed OpenAI parts. This is a
        bridge to ``Coder``, whose image placeholder is the literal ``<image>``
        string. Selection happens before model rendering and never mutates the
        rollout history.
        """
        try:
            keep = int(os.environ.get("BRL_KEEP_LATEST_IMAGES", "0"))
        except ValueError as exc:
            raise ValueError("BRL_KEEP_LATEST_IMAGES must be an integer") \
                from exc
        if keep < 0:
            raise ValueError("BRL_KEEP_LATEST_IMAGES cannot be negative")
        messages = self.exact_messages()
        images = list(self.images)
        if keep == 0 or len(images) <= keep:
            return messages, images
        drop = len(images) - keep
        found = sum(_leading_image_markers(message.get("content"),
                                           message.get("role"))
                    for message in messages)
        if found != len(images):
            raise ValueError(f"rollout has {found} image markers and "
                             f"{len(images)} images")
        left = drop
        for message in messages:
            content = message.get("content")
            markers = min(left, _leading_image_markers(
                content, message.get("role")))
            if markers:
                content = content[markers * len(IMAGE_TAG):]
                left -= markers
            message["content"] = content
        return messages, images[drop:]

    def sampling_view(self):
        """Freeze the student and privileged prompt views for this turn."""
        messages, images = self.model_view()
        teacher = (self.hinter.teacher_view(messages)
                   if self.hinter is not None else messages)
        return ([copy.deepcopy(message) for message in messages],
                [copy.deepcopy(message) for message in teacher],
                list(images))


def start_rollout(r, task, hint, netherite_home, frames_root):
    ep = Episode.from_spec(r.spec, cu_parser(task), task=task,
                           netherite_home=netherite_home,
                           frames_root=frames_root)
    r.ep = ep
    # specs_from_task draws a pose to ASK for; the episode records the one
    # it started at. Adopting the episode's own is what keeps this rollout
    # and the reward that scores it on one convention, whichever way the
    # spec arrived.
    r.spec.update(ep.spec)
    # The t0 screenshot belongs to the episode. Asking the env for another
    # one here would advance the live rollout before its first model view.
    r.images = [ep.t0_image()]
    # Use the shared prompt renderer so SFT and rollout see identical bytes.
    r.messages = with_hint(
        [{"role": "user", "content": cu_user_prompt(task, ep.env)}], hint)
    r.capture_state(0, "reset")
    # Turn 0's hint rides on the opening observation, which is the turn
    # the first aim is chosen on and so the one a direction hint is worth
    # most. It is appended AFTER cu_user_prompt rather than inside it, so
    # the single-sourced prompt keeps rendering the bytes every other
    # consumer renders and the suffix is a removable addition to them.
    if r.hinter is not None:
        r.hinter.attach(r.messages, ep.env, 0)


def step_rollout(r, text, max_turns):
    """Execute the bounded computer calls in one assistant message."""
    r.assistant_raw.append(text)
    attempts = tool_call_attempts(text)
    if not attempts:
        r.messages.append({"role": "assistant", "content": text})
        r.done = True
        # a turn that opened a tool call and never closed it was cut off
        # by the token budget, not ended on purpose. Counting those as
        # voluntary stops inflated the metric SDFT reads.
        r.truncated = "<tool_call>" in text and "</tool_call>" not in text
        r.stopped = not r.truncated
        return
    calls = []
    validated = []
    malformed = {}
    for index, (name, arguments, parse_error) in enumerate(attempts):
        call_id = f"local_{len(r.assistant_raw) - 1}_{index}"
        wire = {"id": call_id, "type": "function", "function": {
            "name": name, "arguments": arguments}}
        try:
            if parse_error is not None:
                raise ValueError(parse_error)
            call = ToolCall.from_openai(wire)
            error = (None if call.name == TOOL_NAME else ValueError(
                f"model called unknown tool {call.name!r}; available: "
                f"{TOOL_NAME!r}"))
        except Exception as exc:
            call = ToolCall(call_id, str(name or TOOL_NAME), {})
            error = exc
            malformed[call_id] = str(exc)
        if error is not None:
            malformed[call_id] = str(error)
        calls.append(call.to_openai())
        validated.append((call, error))
    assistant = {"role": "assistant", "content": None,
                 "tool_calls": calls}
    if malformed:
        assistant["metadata"] = {"malformed_tool_calls": malformed}
    r.messages.append(assistant)
    observations = []
    for index, ((call, error), wire) in enumerate(zip(
            validated[:MAX_TOOL_CALLS_PER_TURN],
            calls[:MAX_TOOL_CALLS_PER_TURN])):
        if r.ep.done:
            content = EPISODE_DONE_TOOL_MESSAGE
        else:
            if error is None:
                obs, frame, _, done = r.ep.step(
                    cu_payload(call.arguments), count_turn=index == 0)
                content = tool_response_text(
                    obs, done, NEXT_COMPUTER_MESSAGE)
            else:
                content, frame, _, done = malformed_computer_step(
                    r.ep, error, count_turn=index == 0)
            if frame is not None:
                r.images.append(frame)
                observations.append({"role": "user",
                                     "content": IMAGE_TAG})
            r.capture_state(r.turns + 1, "after_tool")
        r.messages.append({"role": "tool", "tool_call_id": wire["id"],
                           "name": call.name, "content": content})
    for call, wire in zip(validated[MAX_TOOL_CALLS_PER_TURN:],
                          calls[MAX_TOOL_CALLS_PER_TURN:]):
        r.messages.append({
            "role": "tool", "tool_call_id": wire["id"],
            "name": call[0].name,
            "content": (EPISODE_DONE_TOOL_MESSAGE if r.ep.done
                        else TOOL_CALL_LIMIT_MESSAGE),
        })
    r.messages.extend(observations)
    r.turns += 1
    if r.ep.done or r.turns >= max_turns:
        r.done = True
    # recomputed from the engine state this turn LEFT the world in, which is
    # what makes it a hint and not a repetition of the goal
    if not r.done and r.hinter is not None:
        r.hinter.attach(r.messages, r.ep.env, r.turns)


def _trim_generation(ids, tokenizer):
    """Drop batch padding while retaining the sampled stop token once."""
    ids = [int(token) for token in ids]
    eos = getattr(tokenizer, "eos_token_id", None)
    eos = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
    eos.discard(None)
    if eos:
        for index, token in enumerate(ids):
            if token in eos:
                return ids[:index + 1]
    pad = getattr(tokenizer, "pad_token_id", None)
    while ids and pad is not None and ids[-1] == pad:
        ids.pop()
    return ids


@torch.no_grad()
def generate_turn(model, coder, rollouts, device, max_new_tokens, temperature):
    # A sampler object stands in for the policy module in `play`, so the
    # engine choice is one branch here rather than a second rollout loop.
    # Duck-typed on a name no nn.Module carries, because the LoRA case
    # passes the training module itself and a name collision there would
    # silently take the wrong path.
    turns = getattr(model, "sample_turns", None)
    if turns is not None:
        return turns(coder, rollouts, max_new_tokens, temperature)
    knobs = sampling_knobs(temperature, max_new_tokens)
    texts, pixel, grids, contexts = [], [], [], []
    for r in rollouts:
        messages, teacher, images = r.sampling_view()
        pv, grid, npads = coder.encode_images(images)
        texts.append(coder.render(messages, npads, True))
        pixel.append(pv)
        grids.append(grid)
        contexts.append((messages, teacher, images, tuple(npads)))
    tok = coder.tok
    enc = tok(texts, return_tensors="pt", padding=True, padding_side="left",
              add_special_tokens=False)
    kw = {"input_ids": enc.input_ids.to(device),
          "attention_mask": enc.attention_mask.to(device)}
    if any(p is not None for p in pixel):
        kw["pixel_values"] = torch.cat(
            [p for p in pixel if p is not None]).to(device, model.dtype)
        kw["image_grid_thw"] = torch.cat(
            [g for g in grids if g is not None]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**kw, do_sample=knobs["temperature"] > 0,
                             temperature=max(knobs["temperature"], 1e-4),
                             top_p=knobs["top_p"], top_k=knobs["top_k"],
                             max_new_tokens=knobs["max_new_tokens"],
                             pad_token_id=tok.pad_token_id)
    prompt_width = kw["input_ids"].shape[1]
    generations = []
    for index, sequence in enumerate(out):
        prompt_ids = enc.input_ids[index][enc.attention_mask[index].bool()]
        response_ids = _trim_generation(
            sequence[prompt_width:].detach().cpu().tolist(), tok)
        messages, teacher, images, npads = contexts[index]
        generations.append(SampledTurn(
            tok.decode(response_ids, skip_special_tokens=True),
            tuple(int(token) for token in prompt_ids.tolist()),
            tuple(response_ids), tuple(messages), tuple(teacher),
            tuple(images), tuple(npads)))
    return generations


def play(model, coder, specs, task, hint, args, pool, turn_hint=None,
         turn_hint_p=None, turn_hint_visible=True):
    """Play a whole batch of episodes multi-turn, all rollouts in lockstep,
    and score them from their live engine state.

    `turn_hint` overrides the per-turn rung `args` asks for, which is how
    a hint-free evaluation stays hint-free inside a run whose ROLLOUTS are
    hinted. Pass "off" for that. None means "whatever --turn-hint said".

    `turn_hint_p` overrides the per-episode probability the same way, and
    an EVALUATION wants 1.0. Inheriting a training run's p<1 made the
    "hinted" half of a paired eval a per-episode Bernoulli MIXTURE that
    was still logged under the rung's name, which is a different number
    from the one that label claims.

    `turn_hint_visible=False` still computes feedback from each live state
    but withholds it from generation. SDPO uses that path so its student is
    on-policy and adds the recorded feedback only to the teacher view.
    """
    rollouts = [Rollout(s, s.get("group", i)) for i, s in enumerate(specs)]
    # The per-turn hint decision, taken here and once per GROUP, before
    # any turn is played. Doing it per rollout would let a group hold a
    # hinted and an un-hinted trajectory, and the SDPO demo selection and
    # any group-relative estimator would then compare across the hint.
    # It is taken before the sampler is armed, so an engine that fails to
    # come up leaves no half-hinted batch behind.
    level = turn_hint if turn_hint is not None else getattr(
        args, "turn_hint", "off")
    if level and level != "off":
        hinters = hint_mod.hinters_for_groups(
            task, [r.group for r in rollouts], level,
            float(turn_hint_p if turn_hint_p is not None
                  else getattr(args, "turn_hint_p", 1.0)),
            _hint_rng(args), visible=turn_hint_visible)
        for r, h in zip(rollouts, hinters):
            r.hinter = h
    # A batch of rollouts is the unit the policy is constant over: the
    # trainer takes its gradient step between two of these and never
    # inside one. So this is the one place a sampler that holds its own
    # copy of the weights can be refreshed, and doing it here rather than
    # in the trainer means a caller cannot forget. `after_rollouts`
    # disarms the sampler again, so a batch sampled without coming
    # through here raises instead of drawing from the last step's policy.
    before = getattr(model, "before_rollouts", None)
    if before is not None:
        before()
    try:
        _play_turns(model, coder, rollouts, task, hint, args, pool)
        for r in rollouts:
            r.reward = (float(r.ep.final_reward()) if r.turns else
                        float(NO_TOOL_CALL_REWARD))
            r.success = r.ep.success
            destination = (os.environ.get("BRL_TRAJECTORY_DIR") or "").strip()
            if destination:
                trajectory = r.to_trajectory(task)
                path = Path(destination) / f"{trajectory.id}.json"
                trajectory.save(path)
                r.trajectory_path = str(path)
    finally:
        after = getattr(model, "after_rollouts", None)
        if after is not None:
            after()
        # Every episode is a game subprocess behind a pair of pipes, and
        # the close used to sit inside the scoring loop, which a batch
        # that raised never reached. The HF sampler could not raise
        # mid-batch; vLLM can, on a request over its context or its image
        # cap, and it takes the whole generate call with it. That left a
        # batch's worth of engines running per failed step.
        for r in rollouts:
            if r.ep is not None:
                r.ep.close()
                r.ep = None
    return rollouts


def _play_turns(model, coder, rollouts, task, hint, args, pool):
    """The turn loop of `play`, split out so the sampler's arm/disarm
    pair can wrap it in a `finally` without indenting the scoring half
    into the same block. The per-turn hinters are already attached to the
    rollouts by `play` before this is called."""
    list(pool.map(lambda r: start_rollout(r, task, hint, args.netherite_home,
                                          args.frames_root), rollouts))
    max_turns = task.max_turns
    for _ in range(max_turns):
        live = [r for r in rollouts if not r.done]
        if not live:
            break
        generations = generate_turn(model, coder, live, args.device,
                                    args.max_new_tokens, args.temperature)
        for rollout, generation in zip(live, generations):
            rollout.sampled_turns.append(generation)
        list(pool.map(lambda rt: step_rollout(rt[0], rt[1].text, max_turns),
                      list(zip(live, generations))))


def specs_from_task(task, n, rng, netherite_home=None):
    """Episode specs drawn straight from the task YAML, for rungs that
    have no generated parquet yet. Same seeds and START-POSE DRAW the data
    builder uses, which is `Episode.draw` and not `task.sample_init`: the
    draw is what refuses a pose that begins with the crosshair already on
    the block the success check names, and a self-distillation run that
    harvested its own rollouts off unscreened poses would train on the
    one-swing trajectories that are exactly the thing being fixed.

    That costs one engine boot per spec here, and `start_rollout` boots
    each episode again from the spec it returns. The alternative is a
    second opinion about what a legal start is, which is what this whole
    rule exists to prevent, and the price is paid once at trainer start.
    """
    out = []
    for i in range(n):
        seed = task.seeds[i % len(task.seeds)]
        ep = Episode.draw(task, seed, rng, parser=cu_parser(task),
                          netherite_home=netherite_home)
        try:
            # the episode's OWN coordinates, which carry the pose that was
            # accepted rather than the first one drawn
            out.append(dict(ep.spec))
        finally:
            ep.close()
    return out


def model_banner(name, adapter=None):
    """What the registry made of a --model argument, as one line for the
    run header: the family it resolved to, how far this repo has trained
    that family, where the weights come from, and which attention kernel
    the load will ask for.

    Separate from load_model so the header can print before the weights
    move, which on a 32B checkpoint is the difference between a console
    that says nothing for two minutes and one that says what it is doing.
    """
    from bedrock_rl import models as reg
    return reg.banner(name, adapter=adapter)


def load_model(name, device, dtype, adapter=None, trainable=False):
    """Resolve a registry alias, HuggingFace id, or local checkpoint into
    a policy plus the family spec. Going through bedrock_rl/models.py is
    what lets an SFT or SDFT checkpoint be handed straight to the
    next stage: it resolves back to the family it came from.

    A LoRA adapter directory resolves the same way, to the base weights
    it was trained on plus the adapter, so a checkpoint from a LoRA warm
    start can be sampled from here without the caller knowing which kind
    it got. `trainable` decides whether the adapter takes gradients; the
    self-distillation harvester and every eval want it read-only.
    """
    from bedrock_rl import lora, reporting as console
    from bedrock_rl import models as reg
    spec = reg.resolve(name)
    path, adapter = reg.policy_paths(name, adapter)
    proc = reg.load_processor(spec, lora.processor_path(path, adapter))
    # transformers logs its flash-attention dtype warning once per
    # submodule, four times for one vision-language load, and a
    # self-distillation run loads twice. One copy says everything the
    # four say.
    console.quiet_transformers_repeats()
    model = reg.load_weights(spec, path, dtype, reg.attn_impl(spec))
    if adapter:
        from bedrock_rl.train import peft_util
        model = peft_util.attach(model, adapter, trainable)
    model.to(device)
    return model, proc, spec


# ── sampling with vLLM ───────────────────────────────────────────────────

ENGINES = ("hf", "vllm")
ENGINE_ENV = "BRL_ROLLOUT_ENGINE"
# The launchers spell their knobs without the prefix, and a run driven by
# `python -m` rather than by bin/ would otherwise need a second spelling
# of the same variable to get the same behaviour. Both are read.
LAUNCHER_ENGINE_ENV = "ROLLOUT_ENGINE"


class StalePolicyError(RuntimeError):
    """The rollout engine was about to sample weights that are not the
    weights being trained.

    Always fatal, and deliberately not a warning. SDPO's loss, its
    success gate and its teacher demonstrations all assume the response
    tokens came from the current policy; a batch drawn from last step's
    weights is an off-policy batch wearing an on-policy label, and it
    produces a full run of plausible numbers about a policy nobody
    trained. There is no degraded mode to fall back to.
    """


def add_sampler_args(ap):
    """The engine flags, identical in both trainers that sample."""
    ap.add_argument("--rollout-engine", default=None, choices=list(ENGINES),
                    help="how rollout turns are generated. hf is "
                         "model.generate, the default and the only one "
                         "that needs nothing but the training venv. vllm "
                         "is the engine the verl arms and "
                         "the evaluation driver samples with; on a "
                         "trainable policy it needs LoRA, because that is "
                         "the only per-step weight refresh vLLM 0.12 "
                         f"exposes as public API. ${ENGINE_ENV} and "
                         f"${LAUNCHER_ENGINE_ENV} set it too")
    ap.add_argument("--vllm-gpu-frac", type=float, default=None,
                    help="fraction of the card the rollout engine may "
                         "take. Defaults to bin/rl.sh's 0.45 when the "
                         "training model is on the same card and to "
                         "evaluation's 0.85 when nothing else "
                         "is, which is the harvest and every eval")
    ap.add_argument("--vllm-max-model-len", type=int, default=None,
                    help="context the rollout engine reserves. Defaults "
                         "to the budget bin/rl.sh computes for the other "
                         "vLLM engine in this repo. A request over it is "
                         "rejected rather than truncated")
    ap.add_argument("--vllm-lora-dir", default=None,
                    help="where the per-step adapter is written on its "
                         "way into the rollout engine; defaults to a private "
                         "temporary directory")
    return ap


def engine_choice(args):
    """Which sampling engine this run asked for.

    The environment variable is a default and the flag wins, so a
    launcher can turn it on for a whole sweep and one command can still
    opt out.
    """
    choice = getattr(args, "rollout_engine", None)
    if choice is None:
        choice = (os.environ.get(ENGINE_ENV)
                  or os.environ.get(LAUNCHER_ENGINE_ENV) or "hf")
    choice = str(choice).strip().lower()
    if choice not in ENGINES:
        raise SystemExit(
            f"unknown rollout engine {choice!r}; it is one of "
            f"{', '.join(ENGINES)}")
    return choice


def check_engine(args, trainable):
    """Everything about the engine choice that is knowable BEFORE the
    weights move, checked there.

    `VLLMSampler` is constructed after `load_model`, so a check that
    lives in it fires having already paid for the weights: a full
    fine-tune of an 8B holds fp32 master weights, which is about 35 GiB
    and several minutes to reach a message saying the flags were wrong
    before the run started, and on a card that cannot hold that the
    message is never reached at all, because the load dies first naming
    nothing. `bin/_common.sh::brl_require_flash_attn` is the same idea
    for the same reason.
    """
    if engine_choice(args) != "vllm":
        return "hf"
    import importlib.util
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            "--rollout-engine vllm, but vllm is not importable in this "
            "environment. It lives in the training venv (VERL_ENV, built "
            "by bin/setup_deps.sh), which is where the launchers run "
            "these trainers. Drop the flag to sample with HuggingFace "
            "generate instead.")
    if not trainable:
        return "vllm"
    if getattr(args, "lora_4bit", False):
        raise SystemExit(
            "--rollout-engine vllm cannot serve a --lora-4bit policy. The "
            "trainer's frozen base is NF4 and the engine would load the "
            "same checkpoint in bf16, so the adapter this refreshes every "
            "step would sit on a DIFFERENT base and the sampled policy "
            "would not be the trained one. Nothing raises when that "
            "happens, which is why it is refused here.")
    from bedrock_rl import lora
    if not lora.enabled(getattr(args, "lora_rank", 0),
                        getattr(args, "lora_adapter", None)):
        raise SystemExit(
            "--rollout-engine vllm on a trainable policy needs LoRA. "
            "vLLM 0.12 exposes add_lora/list_loras as public API and "
            "nothing equivalent for whole weights, so a full fine-tune "
            "here could only sample the weights it started with. Pass "
            "--lora-rank, or leave --rollout-engine at hf.")
    return "vllm"


def default_model_len(spec, task):
    """Return a rollout context budget consistent with ``bin/rl.sh``.

    The budget scales with the task's turn limit and model prompt allowance,
    while retaining a floor suitable for short runs. Keeping both rollout
    paths aligned avoids reserving unnecessary KV memory or truncating one path
    earlier than the other.
    """
    response = max(task.max_turns * 512, 4096)
    prompt = 0 if spec is None else spec.max_prompt_length
    return max(prompt + response, 8192)


def policy_source(name, adapter=None):
    """(base weights, adapter or None) for a --model argument.

    The same two paths `load_model` resolves, exposed on their own
    because the vLLM engine serves the base and the adapter separately
    and must be handed exactly what the HF loader would have stacked.
    """
    from bedrock_rl import models as reg
    return reg.policy_paths(name, adapter)


class VLLMSampler:
    """Rollout turns from a vLLM engine, refreshed from the policy every
    batch.

    WHY THE REFRESH IS THE HARD PART. verl syncs weights into its rollout
    engine every step and the sync costs about 4 s against a 1262 s step,
    so the question was never whether an on-policy vLLM rollout is
    possible, it was which of vLLM 0.12's surfaces this trainer can reach
    without a fork. The one used here is the whole of it:

        engine.add_lora(LoRARequest(name, id, dir)) -> bool
        engine.list_loras()                         -> set[int]

    both public on vllm.LLM's `llm_engine`. Per batch the adapter is
    written to a fresh directory and added under a fresh integer id.

    THE FRESH ID IS LOAD-BEARING, and it is the trap this class exists to
    close. vLLM's LRUCacheWorkerLoRAManager.add_adapter checks
    `lora_request.lora_int_id not in self.list_adapters()` FIRST and, for
    an id it already holds, does not re-read the directory at all: it
    bumps the LRU entry and activates what it loaded the first time. So
    re-adding under a constant id after overwriting the files on disk is
    a silent no-op, and the run samples step 1's adapter for all fifty
    steps while every metric reads as on-policy. LoRARequest also hashes
    and compares on `lora_name`, not on the id, so the name is fresh too.

    Adding before dropping is the other half. The manager loads the new
    adapter before it evicts anything, so a failed load raises with the
    previous adapter still live; if the old one were removed first, a
    failed load would leave the engine serving the BASE model, which is
    the one wrong answer that produces no error at all.

    WHY LORA ONLY on a trainable policy. Pushing whole updated weights
    into vLLM 0.12 needs `collective_rpc` into a worker extension class
    (its own examples/offline_inference/rlhf.py does exactly that), which
    means a second process, a weight-update process group and a
    model-specific name mapping. Under LoRA the adapter IS the update,
    the frozen base in the engine is the same frozen base the trainer
    holds, and the refresh is two public calls. A full fine-tune
    therefore stays on the HF path rather than getting a fork of verl's
    rollout worker written here, and asking for one raises.

    A frozen policy (self-distillation harvest, --eval-only) needs no refresh at
    all and is a plain drop-in.
    """

    def __init__(self, base, coder, *, policy=None, adapter=None,
                 lora_rank=0, max_turns=8, max_model_len=8192,
                 gpu_frac=0.45, workdir=None, say=print):
        import atexit
        from vllm import LLM
        from bedrock_rl import lora
        from bedrock_rl import models as registry
        # vLLM receives a local immutable snapshot, never a moving Hub
        # branch.  The transformers policy loader already binds registry
        # revisions; resolving here makes the independently loaded sampler
        # consume those same bytes for the whole run.
        self.base = str(registry.weights_path(base))
        self.coder = coder
        self.policy = policy
        self.generation = 0
        self.request = None
        # A frozen policy is on-policy the moment the engine is up. A
        # trainable one is not armed until the first refresh.
        self.armed = policy is None
        self._owns_workdir = workdir is None
        self.workdir = (tempfile.mkdtemp(prefix="bedrock-vllm-lora-")
                        if workdir is None else str(workdir))
        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        self._generated_paths = set()
        # A rank of 0 with an adapter to continue is still LoRA: the rank
        # is the adapter's, exactly as bedrock_rl/lora.py::enabled and
        # every launcher already treat it.
        rank = int(lora_rank or 0)
        if not rank and adapter:
            rank = int(lora.adapter_rank(adapter)[0] or 0)
        if policy is not None and rank <= 0:
            raise SystemExit(
                "--rollout-engine vllm on a trainable policy needs LoRA. "
                "vLLM 0.12 exposes add_lora/list_loras as public API and "
                "nothing equivalent for whole weights, so a full "
                "fine-tune here could only sample the weights it started "
                "with. Pass --lora-rank, or leave --rollout-engine at hf.")
        # SDPO importance correction uses policy log-probabilities before
        # temperature/top-k/top-p processing. Pin the vLLM 0.12 mode instead
        # of trusting its current default to remain unchanged.
        kw = {"logprobs_mode": "raw_logprobs"}
        if rank > 0:
            kw.update({
                "enable_lora": True,
                "max_lora_rank": lora.vllm_max_rank(rank),
                "max_loras": 1,
            })
        spec = getattr(coder, "spec", None)
        if spec is not None:
            # Everything family-specific comes from the registry entry,
            # which is the rule bedrock_rl/models.py states about
            # itself. The tower backend is the one that matters: the
            # families that declare it declare it because vLLM's default
            # vision kernel is wrong for them, and bin/rl.sh already
            # pushes the same field into the verl-side engine.
            kw["trust_remote_code"] = spec.trust_remote_code
            if getattr(spec, "vit_attn_backend", None):
                kw["mm_encoder_attn_backend"] = spec.vit_attn_backend
        if adapter and policy is None:
            self.request = self._request(adapter)
        say(f"rollout engine: vllm serving {self.base}"
            + (f" + adapter {adapter}" if adapter else "")
            + (f" (lora rank {rank})" if rank else "")
            + (", refreshed every batch" if policy is not None else ""))
        if adapter:
            self._warn_about_tower_adapters(adapter, say)
        # enforce_eager, matching the evaluation driver: the checkpoint
        # this run produces is evaluated by that script, and a policy
        # sampled under CUDA graphs here and eagerly there is two
        # slightly different numeric paths on the same weights. It also
        # keeps the engine's footprint predictable next to a training
        # model on the same card, which is the case this is built for.
        try:
            keep_images = int(os.environ.get(
                "BRL_KEEP_LATEST_IMAGES", "0"))
        except ValueError as exc:
            raise ValueError(
                "BRL_KEEP_LATEST_IMAGES must be an integer") from exc
        self.llm = LLM(model=self.base, max_model_len=int(max_model_len),
                       gpu_memory_utilization=float(gpu_frac),
                       enforce_eager=True,
                       limit_mm_per_prompt={"image":
                           multimodal_image_limit(
                               max_turns,
                               keep_latest_images=keep_images)}, **kw)
        # close() is reached on the happy path, and the paths that matter
        # are the other ones: --max-dry-steps ends an SDPO run with
        # SystemExit and is the DOCUMENTED outcome on a policy that never
        # succeeds, and a StalePolicyError ends it with two staged
        # adapters on disk rather than one. A few hundred megabytes a
        # step accumulating in temporary storage is what that costs.
        atexit.register(self.close)

    @staticmethod
    def _warn_about_tower_adapters(adapter, say):
        """The evaluation driver's warning, on the path that trains.

        vLLM serves LoRA for the language model only and drops an adapter
        on the vision tower without saying so, so an adapter carrying one
        is not the policy being sampled. Every path in this repo excludes
        the tower; an adapter that names no exclusion at all came from
        somewhere else and may not. It mattered on an eval script; it
        matters more here, because this engine is the ON-POLICY sampler.
        """
        from bedrock_rl import lora
        if (lora.read_adapter_config(adapter) or {}).get("exclude_modules"):
            return
        say(f"warning  {adapter} declares no exclude_modules, so it may "
            f"carry adapters on the vision tower. vLLM serves LoRA for "
            f"the language model only and drops those without saying so, "
            f"which would make the sampled policy a different policy from "
            f"the one that trains.")

    def _request(self, path):
        from vllm.lora.request import LoRARequest
        self.generation += 1
        return LoRARequest(f"bedrock-{self.generation}", self.generation,
                           str(path))

    # ── the on-policy guarantee ──────────────────────────────────────
    def before_rollouts(self):
        """Put the weights the trainer is holding into the engine.

        Called by `play` at every batch boundary, and unconditionally: a
        refresh that is skipped when it looks unnecessary is a refresh
        whose necessity is decided by whoever last edited the trainer.
        The adapter is small next to the step it saves, so paying for it
        on a batch that did not need it costs nothing worth naming.
        """
        if self.policy is None:
            self.armed = True
            return
        from bedrock_rl.train import peft_util
        old = self.request
        out = os.path.join(self.workdir, f"gen{self.generation + 1}")
        peft_util.save_adapter(self.policy, out, self.base)
        self._generated_paths.add(out)
        request = self._request(out)
        engine = self.llm.llm_engine
        if not engine.add_lora(request):
            raise StalePolicyError(
                f"vLLM refused the step's adapter at {out}, so the engine "
                "still holds the previous one and this batch would be "
                "off-policy.")
        live = engine.list_loras()
        if request.lora_int_id not in live:
            raise StalePolicyError(
                f"vLLM took adapter id {request.lora_int_id} and then does "
                f"not list it ({sorted(live)}), so what it would sample is "
                "not what this step trained.")
        self.request = request
        self.armed = True
        if old is not None:
            # The manager reads the directory during add_lora and keeps
            # the tensors, so the previous one is dead weight now. Fifty
            # steps of a rank-32 adapter on an 8B model is a few hundred
            # megabytes a step, which fills temporary storage on a trainer.
            shutil.rmtree(old.lora_path, ignore_errors=True)
            self._generated_paths.discard(str(old.lora_path))

    def after_rollouts(self):
        self.armed = False

    def eval(self):
        """A vLLM engine is always in inference mode.

        Answered as a no-op so that a sampler is substitutable for the
        policy module everywhere the trainer toggles modes around a batch
        of rollouts, rather than every such site growing a branch on
        which engine is in use.
        """
        return self

    # ── generation ───────────────────────────────────────────────────
    def sample_turns(self, coder, rollouts, max_new_tokens, temperature):
        from vllm import SamplingParams
        # `play` passes the run's Coder down; the one this engine was
        # built for is the same object, and is here so an evaluation
        # loop that has only the sampler still renders through it.
        coder = coder if coder is not None else self.coder
        if not self.armed:
            raise StalePolicyError(
                "this batch was sampled without a weight refresh, so the "
                "engine would draw from the policy as it was before the "
                "last gradient step. Rollouts go through "
                "bedrock_rl.train.rollout.play, which refreshes and "
                "arms the sampler around every batch.")
        knobs = sampling_knobs(temperature, max_new_tokens)
        params = SamplingParams(temperature=knobs["temperature"],
                                top_p=knobs["top_p"], top_k=knobs["top_k"],
                                max_tokens=knobs["max_new_tokens"],
                                logprobs=0)
        reqs = []
        contexts = []
        for rollout in rollouts:
            messages, teacher, images = rollout.sampling_view()
            _, _, npads = coder.encode_images(images)
            prompt = coder.render(messages, [1] * len(images), True)
            frames = self.fit_frames(coder, images)
            reqs.append({"prompt": prompt,
                         "multi_modal_data": {"image": frames}})
            contexts.append((messages, teacher, images, tuple(npads)))
        outs = self.llm.generate(reqs, params, lora_request=self.request,
                                 use_tqdm=False)
        generations = []
        for output, context in zip(outs, contexts):
            sample = output.outputs[0]
            messages, teacher, images, npads = context
            # RequestOutput.prompt_token_ids may retain vLLM's one-token
            # multimodal placeholders instead of the expanded image span.
            # The teacher-forced HF forward consumes the expanded form, so
            # derive that prompt through the same Coder/image pads it uses.
            prompt_ids = tuple(coder.ids(
                list(messages), list(npads), add_generation_prompt=True))
            self.verify_prompt_length(
                prompt_ids, getattr(output, "prompt_token_ids", None))
            response_ids = tuple(int(token) for token in sample.token_ids)
            sampling_log_probs = self.chosen_log_probs(
                response_ids, sample.logprobs)
            generations.append(SampledTurn(
                sample.text, prompt_ids, response_ids, tuple(messages),
                tuple(teacher), tuple(images), npads, sampling_log_probs))
        return generations

    @staticmethod
    def verify_prompt_length(prompt_ids, actual_prompt_ids):
        """Refuse a vLLM/HF processor split before distillation uses it."""
        if (actual_prompt_ids is None
                or len(actual_prompt_ids) != len(prompt_ids)):
            actual = ("missing" if actual_prompt_ids is None
                      else str(len(actual_prompt_ids)))
            raise StalePolicyError(
                "vLLM sampled from a prompt of length " + actual
                + f", but the teacher-forced coder produced "
                f"{len(prompt_ids)} tokens")

    @staticmethod
    def chosen_log_probs(token_ids, positions):
        """Extract each sampled token's own vLLM log-probability."""
        if positions is None or len(positions) != len(token_ids):
            raise StalePolicyError(
                "vLLM did not return one sampling log-probability per "
                "response token, so SDPO cannot importance-correct this "
                "batch")
        values = []
        for token, alternatives in zip(token_ids, positions):
            selected = alternatives.get(int(token))
            if selected is None:
                raise StalePolicyError(
                    f"vLLM omitted sampled token {token} from its own "
                    "log-probability record")
            values.append(float(selected.logprob))
        return tuple(values)

    @staticmethod
    def prompt(coder, r):
        """The rollout's prompt, rendered for vLLM.

        Through the same Coder the HF path and the training views render
        with, so the family's chat template and the family's image span
        are read in one place. The only difference is the pad count: the
        HF path expands each <image> into its real per-frame run of pad
        tokens because it hands the model pixel values it has already
        preprocessed, while vLLM is given the pictures and does that
        expansion itself, so it wants the single-pad placeholder the
        chat template would have written. Asking Coder for one pad per
        image produces exactly that placeholder, out of the same
        image_span, rather than a second opinion about what it looks
        like.
        """
        messages, images = r.model_view()
        return coder.render(messages, [1] * len(images), True)

    @staticmethod
    def frames(coder, r):
        """The frames, in the band the training view will read them in.

        bedrock_rl.models.fit_pixels is what Coder.encode_images runs
        before the processor sees a frame, so applying it here is what
        keeps the picture the policy sampled against and the picture the
        distillation forward pass sees the same picture.
        """
        _, images = r.model_view()
        return VLLMSampler.fit_frames(coder, images)

    @staticmethod
    def fit_frames(coder, images):
        """Apply the registry's pixel band to an already-selected view."""
        if coder.spec is None:
            return images
        from bedrock_rl import models as reg
        return [reg.fit_pixels(im, coder.spec) for im in images]

    def close(self):
        for path in tuple(self._generated_paths):
            shutil.rmtree(path, ignore_errors=True)
        self._generated_paths.clear()
        if self._owns_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)


def build_sampler(model, coder, args, task, *, base=None, adapter=None,
                  trainable=True, say=print):
    """What `play` should be handed to sample with.

    One factory for all three callers (SDPO training, SDPO --eval-only,
    self-distillation harvest) so that "which engine, and is it on-policy" is
    decided once. `trainable` is the SDPO training run and nothing else:
    it is what makes the sampler refresh itself every batch, it is the
    only case that requires LoRA, and it is the only case with a training
    model on the card beside the engine.
    """
    if engine_choice(args) == "hf":
        return model
    if base is None:
        base, adapter = policy_source(args.model, adapter)
    frac = getattr(args, "vllm_gpu_frac", None)
    if frac is None:
        # 0.45 is bin/rl.sh's split, and it is a split because verl holds
        # an actor on the same card. The harvest and every eval hold
        # nothing else, and leaving half the card idle there is half the
        # KV cache, which is half the batch the engine will run at once.
        # The evaluation driver, which is that same job, uses 0.85.
        frac = 0.45 if trainable else 0.85
    return VLLMSampler(
        base, coder,
        policy=model if trainable else None,
        adapter=adapter,
        lora_rank=int(getattr(args, "lora_rank", 0) or 0),
        max_turns=task.max_turns,
        max_model_len=(getattr(args, "vllm_max_model_len", None)
                       or default_model_len(getattr(coder, "spec", None),
                                            task)),
        gpu_frac=frac,
        workdir=getattr(args, "vllm_lora_dir", None),
        say=say)


def load_coder(name, adapter=None):
    """The processor and the family spec, without moving any weights.

    The self-distillation harvest and an --eval-only run on the vLLM engine never
    forward through the transformers module at all: the engine holds the
    same weights and samples from those. They still need a Coder, and
    building one used to mean a full `load_model`, which for an 8B is
    about 18 GiB of host memory and a whole checkpoint read to obtain two
    objects that need neither, and then vLLM reads the same files again.
    """
    from bedrock_rl import lora, reporting as console
    from bedrock_rl import models as reg
    spec = reg.resolve(name)
    path, adapter = reg.policy_paths(name, adapter)
    console.quiet_transformers_repeats()
    proc = reg.load_processor(spec, lora.processor_path(path, adapter))
    return proc, spec, path, adapter
