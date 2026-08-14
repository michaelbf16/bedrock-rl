"""Sample multi-turn `computer` trials from a policy with vLLM and score
them through the same Episode the trainer uses.

It measures how often a checkpoint solves the task, optionally with a task
hint. Data harvesting belongs to ``workflow: self_distill`` so evaluation has
one responsibility and does not maintain a second trajectory converter.

  uv run python -m bedrock_rl.eval.trials \
      --data data/<task>_cu/rl_prompts.parquet \
      --task examples/<example>/task.yaml \
      --model <path> [--n 64] [--samples 4] [--max-turns N] [--dp 8] \
      [--hint examples/<example>/task.yaml]

--model takes a LoRA adapter directory as readily as a checkpoint. Both
kinds a run of this repo produces resolve: an SFT or SDPO adapter,
which names its own base model, and an RL run's merged directory, which
holds the base weights with a `lora_adapter/` beside them. The second
one is the one to get right, because its base weights are the UNCHANGED
base, so serving that directory without its adapter evaluates the model
before training and reports it as a result.

ONE CARD by default, and that default is a protocol number rather than a
limitation nobody got to. `--dp N` runs N of these evaluations side by
side, a shard of the trial list each, and merges them; `--tp N` gives
ONE engine N cards, which is for a checkpoint that does not fit on one.
Neither is plumbing. --tp changes the DISTRIBUTION being sampled, in the
last bits, which more trials do not average away; --dp re-rolls which
trajectories come out of it, which is the same noise a re-run already
has, because nothing here is seeded. The --tp help explains the comparison
tradeoff.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time


# vLLM forks its workers by default, and this script has already touched
# CUDA in the parent by the time they start, so the engine dies with
# "Cannot re-initialize CUDA in forked subprocess" on every invocation.
# Set here rather than left to the caller: a driver that only works when
# you remember an environment variable is a driver that does not work.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# This one runs in the TRAINING venv, where this repo is not installed:
# it needs vllm, which the CPU project env deliberately does not
# carry. Every other script here runs under `uv run` and finds the
# package that way, so this bootstrap is not boilerplate to copy.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bedrock_rl import lora, models as reg  # noqa: E402
from bedrock_rl.adapters.netherite.chat import (IMAGE_TAG,  # noqa: E402
                                                  template_messages,
                                                  tool_schema)
from bedrock_rl.adapters.netherite.computer import (NEXT_COMPUTER_MESSAGE,  # noqa: E402
                                     TOOL_NAME, cu_parser, cu_payload,
                                     tool_call_attempts, tool_calls)
from bedrock_rl.env import journal  # noqa: E402
from bedrock_rl.sdg.guidance import hint_text, with_hint  # noqa: E402
from bedrock_rl.env.task import Task, relative_path  # noqa: E402
from bedrock_rl.env.episode import (Episode,  # noqa: E402
                                      NO_TOOL_CALL_REWARD,
                                      tool_response_text)
from bedrock_rl.core.sampling import (DEFAULT_MAX_NEW_TOKENS,  # noqa: E402
                                      EPISODE_DONE_TOOL_MESSAGE,
                                      MAX_TOOL_CALLS_PER_TURN,
                                      TOOL_CALL_LIMIT_MESSAGE,
                                      multimodal_image_limit,
                                      sampling_knobs)


def payloads(text):
    """Every `computer` payload in one assistant turn, in wire order."""
    return [cu_payload(arguments) for name, arguments in tool_calls(text)
            if name == TOOL_NAME]


def first_payload(text):
    """The first `computer` payload in an assistant turn, or None.

    The extraction lives beside the action language, so standalone
    evaluation and every trainer share one compatibility parser."""
    found = payloads(text)
    return found[0] if found else None


def without_leading_image_marker(text):
    """Move one typed t0 image out of a string without changing its suffix."""
    text = str(text)
    return text[len(IMAGE_TAG):] if text.startswith(IMAGE_TAG) else text


def execute_turn(episode, text, turn_index=0):
    """Execute one decoded turn and retain its sample-exact next context."""
    from bedrock_rl.adapters.netherite.tools import malformed_computer_step
    from bedrock_rl.core.tools import ToolCall

    parsed = tool_call_attempts(text)
    calls = [{"id": f"eval_{turn_index}_{index}", "type": "function",
              "function": {"name": name, "arguments": (
                  arguments if isinstance(arguments, (str, dict))
                  else {"malformed_arguments": arguments})}}
             for index, (name, arguments, _error) in enumerate(parsed)]
    if not calls:
        return [], [], [], bool(episode.done)
    # The next model turn conditions on the bytes it actually sampled, not a
    # normalized JSON serialization of the parsed calls. The call list remains
    # separately available for execution and audit.
    messages = [{"role": "assistant", "content": text}]
    observations = []
    frames = []
    done = bool(episode.done)
    for index, ((name, arguments, parse_error), call) in enumerate(zip(
            parsed[:MAX_TOOL_CALLS_PER_TURN],
            calls[:MAX_TOOL_CALLS_PER_TURN])):
        if episode.done:
            content = EPISODE_DONE_TOOL_MESSAGE
        else:
            error = None
            try:
                if parse_error is not None:
                    raise ValueError(parse_error)
                parsed_call = ToolCall.from_openai(call)
                if parsed_call.name != TOOL_NAME:
                    raise ValueError(
                        f"model called unknown tool {parsed_call.name!r}; "
                        f"available: {TOOL_NAME!r}")
            except (TypeError, ValueError) as exc:
                error = exc
            if error is None:
                # Engine failures remain trial failures. Only a model-format
                # error is converted into the charged malformed transition.
                observation, frame, _, done = episode.step(
                    cu_payload(parsed_call.arguments), count_turn=index == 0)
                content = tool_response_text(
                    observation, done, NEXT_COMPUTER_MESSAGE)
            else:
                content, frame, _, done = malformed_computer_step(
                    episode, error, count_turn=index == 0)
            if frame is not None:
                frames.append(frame)
                observations.append(
                    {"role": "user", "content": [{"type": "image"}]})
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "name": name, "content": content})
    for (name, _arguments, _parse_error), call in zip(
            parsed[MAX_TOOL_CALLS_PER_TURN:],
            calls[MAX_TOOL_CALLS_PER_TURN:]):
        messages.append({
            "role": "tool", "tool_call_id": call["id"], "name": name,
            "content": (EPISODE_DONE_TOOL_MESSAGE if episode.done
                        else TOOL_CALL_LIMIT_MESSAGE),
        })
    messages.extend(observations)
    return calls, messages, frames, done


def execute_payloads(episode, text):
    """Compatibility tuple for callers that only need execution results."""
    calls, messages, frames, done = execute_turn(episode, text)
    tools = [message for message in messages if message["role"] == "tool"]
    observation = tools[-1]["content"] if tools else ""
    return calls, observation, frames[-1] if frames else None, done


def evaluation_task(rows, explicit_path=None):
    """Load the one task contract every row in an evaluation names.

    A frozen prompt row is already bound to a task YAML.  ``--task`` is an
    assertion about that contract, not an override that can silently score a
    dataset with a different verifier.  Direct users of this script may omit
    it; the public CLI always supplies it.
    """
    if not rows:
        raise SystemExit("the evaluation dataset has no prompt rows")
    paths = [str(row["extra_info"]["task_yaml"]) for row in rows]
    loaded = {}
    for path in paths:
        if path not in loaded:
            loaded[path] = Task.load(path)
    identities = {os.path.realpath(task.path) for task in loaded.values()}
    if len(identities) != 1:
        raise SystemExit(
            "one evaluation cannot mix task contracts; the selected rows "
            f"name {len(identities)} different task YAML files")
    row_task = next(iter(loaded.values()))
    if explicit_path is None:
        return row_task
    explicit = Task.load(explicit_path)
    if os.path.realpath(explicit.path) != os.path.realpath(row_task.path):
        raise SystemExit(
            f"--task names {explicit.path}, but the frozen dataset rows name "
            f"{row_task.path}; refusing to score one task with another "
            "task's verifier")
    return explicit


def vllm_engine_kwargs(model_path, lora_kwargs):
    """Combine policy-specific and model-family-specific vLLM settings."""
    out = dict(lora_kwargs)
    backend = reg.resolve(model_path).vit_attn_backend
    if backend:
        out["mm_encoder_attn_backend"] = backend
    return out


def opened_a_world_container(ep):
    """Did this trial get a placed block's screen up?

    Read off the event journal, which names the container and so tells a
    crafting table apart from the player's own inventory. Reported next
    to the solve rate because on a task that needs a table the two can
    come apart in both directions: opening one and fumbling the recipe is
    a different failure from never opening one, and only the second means
    the container screens are out of reach.
    """
    # `kind` is the raw wire integer the engine writes, not the name the
    # task YAML would spell; CONTAINER_KINDS is the one table that maps
    # between them. Comparing it against the string made this true for
    # every trial that ever pressed `e`.
    player = journal.CONTAINER_KINDS["player"]
    return any(e.name == "container_opened" and e["kind"] != player
               for e in ep.env.journal)


# ── one evaluation, or N of them ─────────────────────────────────────────
# The loop below is turn-synchronous: every live episode's prompt goes
# through vLLM in one batch, and then every one of them steps the ENGINE,
# on CPU, one episode at a time. Only the first half is GPU-bound, and
# that is the whole argument for the shape of this: N whole copies of the
# script scale both halves, where anything that parallelises the engine
# alone -- tensor parallel, or vLLM's own data-parallel setting -- leaves
# the CPU half exactly as serial as it was. Tensor parallel is here for
# capacity instead: it splits one engine across cards, which is how a
# checkpoint too large for one card gets served at all, and it pays
# all-reduces per layer to do it. No measurement of that trade is in this
# repo, so neither this comment nor --tp's help quotes one.

# Prefixed onto every line when this process is one shard of several,
# because N shards share one stdout and `turn 3: 12 sampled` from an
# unnamed eighth of the batch is not a measurement. Empty on the
# single-process path, which is the default, so that path prints exactly
# what it always printed.
PREFIX = ""


def say(msg):
    """print(), with this shard's name on every line of it."""
    for line in msg.split("\n"):
        print(PREFIX + line if line else line)


def shard_bounds(total, i, n):
    """The half-open range of trials shard i of n owns.

    Contiguous, in order, and covering, so that the shards concatenated
    in i order ARE the unsharded list rather than a permutation of it.
    Most of what reads --dump-actions keys on the seed inside each
    record and does not care, but not all of it: a paired analysis that
    takes an arm's successes in DUMP ORDER and zips two arms' lists
    positionally would, on a merge that reordered anything, pair one
    trial of this arm against a different trial of that one and
    still print a p-value.
    """
    return total * i // n, total * (i + 1) // n


def parse_shard(text):
    """`i/N` as a pair. Unsharded is shard 0 of 1."""
    if not text:
        return 0, 1
    try:
        i, n = (int(part) for part in text.split("/", 1))
    except ValueError:
        raise SystemExit(f"--shard wants i/N, got {text!r}\n"
                         f"  --shard 0/8   the first eighth of the trials")
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"--shard {text}: N must be at least 1 and i must "
                         f"be one of 0..{max(n - 1, 0)}")
    return i, n


def record(r):
    """Everything the summary and the dumps need about one trial.

    A record rather than a Roll because the --dp parent has no episodes:
    its shards are other processes, and anything the summary reads off a
    live Roll has to survive as data or the merged run cannot print the
    same lines as the single-process one.

    It is also what lets a trial's ENGINE be released the moment the
    trial ends rather than at the end of the run, which is worth about
    500 MiB of host memory per finished episode on a path
    that used to hold every one of them open until the last turn.

    `broken` is None for a trial that ran. A trial whose engine failed
    has no reward and is not a zero: it is a measurement that did not
    happen, and every reader here has to keep the two apart.

    The reward is dropped on `broken` and NOT on "there is no episode
    left", which is a distinction with teeth. The failure this exists for
    is an engine that stops answering PART WAY THROUGH, and such a
    trial still has a live Episode holding a real, truncated
    `final_reward()`. Keying on the episode instead let exactly that
    case -- the only one that matters -- keep its partial number and
    land in the mean, under a line saying the mean excluded it.
    """
    ep, broken = r.ep, getattr(r, "broken", None)
    scored = ep is not None and not broken
    return {"seed": r.seed,
            "trial_id": f"{r.row_index}:{r.k}",
            "row_index": r.row_index,
            "sample_index": r.k,
            "init_yaw": r.init_yaw,
            "init_pitch": r.init_pitch,
            "success": bool(scored and ep.success),
            # DID IT ANSWER AT ALL. On a task with a `fail:` check,
            # success false with failed false is an episode that never
            # committed, and that is a different policy from one that
            # committed and missed -- the two now score within a decision
            # cost of each other by design (`no_commit`), so this field is
            # the only thing in --records-out that separates them. Always
            # false on a task with no `fail:` check.
            "failed": bool(scored and getattr(ep, "failed", False)),
            "actions": first_payload(r.turns[0] if r.turns else ""),
            "emitted_tool_call": any(
                tool_call_attempts(turn) for turn in r.turns),
            "turns": len(r.turns),
            "reward": (float(ep.final_reward()) if scored and ep.turns else
                       (float(NO_TOOL_CALL_REWARD) if scored else None)),
            "opened": bool(scored and opened_a_world_container(ep)),
            "broken": broken,
            "protocol": dict(r.protocol)}


def broken_records(recs):
    """The trials that did not run, in trial order."""
    return [x for x in recs if x.get("broken")]


def failure_reason(e):
    """What to call the thing that ended a trial.

    Two words about which half of the loop broke, because they send a
    reader to different files. An EngineError is the game process --
    dead, or alive and no longer answering -- and everything else is
    this script, the action parser or the task's own checks. The
    exception's own type and message follow it either way, so this only
    decides the headline.
    """
    from bedrock_rl.env.engine import EngineError
    if isinstance(e, EngineError):
        return "the engine stopped answering mid-episode"
    return "the episode could not be stepped"


def generate_isolated(rolls, requests, generate):
    """Sample a batch while attributing a bad request to one trial.

    vLLM validates multimodal limits inside its batched ``generate`` call.
    Bisecting only after a failure preserves normal batching while preventing
    one malformed/oversized prompt from discarding every other trial. Global
    engine failures propagate immediately so changing the batch shape cannot
    silently turn an OOM or crashed worker into a different eval protocol.
    """
    rolls, requests = list(rolls), list(requests)
    if len(rolls) != len(requests):
        raise ValueError("roll/request batch lengths differ")
    if not rolls:
        return []
    try:
        outputs = list(generate(requests))
    except Exception as exc:
        # Only vLLM's item-limit validation identifies one bad request hidden
        # inside a batch. OOMs, engine faults, and output contract bugs are
        # batch/global failures; changing batch shape for those would alter the
        # evaluation protocol and could publish a result from adaptive retries.
        message = str(exc).lower()
        request_local = (isinstance(exc, ValueError)
                         and "image(s) may be provided" in message)
        if not request_local:
            raise
        if len(rolls) == 1:
            rolls[0].fail("the model could not sample this trial", exc)
            return []
        middle = len(rolls) // 2
        return (generate_isolated(
                    rolls[:middle], requests[:middle], generate)
                + generate_isolated(
                    rolls[middle:], requests[middle:], generate))
    if len(outputs) != len(rolls):
        raise RuntimeError(
            f"model returned {len(outputs)} outputs for "
            f"{len(rolls)} requests")
    return list(zip(rolls, outputs))


def solved(r):
    """Has this trial succeeded, whether or not its engine is still up?

    A finished trial has already handed its engine back and answers from
    its record; a live one has no record yet and answers from its episode.
    The per-turn progress line asks this of every trial including the
    ones that ended several turns ago.
    """
    if r.rec is not None:
        return bool(r.rec["success"])
    return bool(r.ep is not None and r.ep.success)


def summarise(recs):
    """The two closing lines, from records, for both paths.

    One function so a sharded run and a single-process run cannot drift
    into printing two different summaries of the same trials.

    A run in which nothing broke prints exactly the same two lines. A run in
    which something
    broke prints a third block, and `refuse_partial` below is what stops
    the first two being quoted anyway.
    """
    n = len(recs)
    solved = [x for x in recs if x["success"]]
    one_turn = [x for x in solved if x["turns"] == 1]
    # The mean is over the trials that HAVE a reward. A broken trial
    # counted as 0.0 would move this line by up to a whole reward unit
    # while it still read as an evaluation of the policy.
    scored = [x["reward"] for x in recs if x.get("reward") is not None]
    mean_r = sum(scored) / max(len(scored), 1)
    called = sum(1 for x in recs if x.get(
        "emitted_tool_call", x["actions"] is not None))
    say(f"\ntrials {n}  emitted a tool call {called}  solved "
        f"{len(solved)} ({100.0 * len(solved) / max(n, 1):.1f}%)  "
        f"turn-1 solves {len(one_turn)}  mean reward {mean_r:.3f}")
    # A task with no block to open reports zero here and that is a fact
    # about the task, not a score, which is why the line says what it
    # counted rather than printing a bare percentage.
    opened = sum(1 for x in recs if x["opened"])
    say(f"trials that got a table, furnace or chest screen up: "
        f"{opened}/{n}")
    bad = broken_records(recs)
    if bad:
        say(f"\n{len(bad)} of {n} TRIALS DID NOT RUN. Every count above "
            f"excludes them and every denominator above includes them; "
            f"the mean reward is over the {len(scored)} that were scored:")
        for x in bad:
            say(f"  seed {x['seed']}: {x['broken']}")


def refuse_partial(recs):
    """Stop, loudly, if any trial did not run. Returns None otherwise.

    The same rule `wait_for` and `read_records` already apply to shards,
    applied to episodes: an evaluation that lost one is a smaller
    evaluation wearing the name of the one that was asked for, and a
    percentage printed over it is quotable against rows that were not
    measured that way. So the run reports everything it has -- the dumps
    and --records-out are written first, because those trials cost real
    GPU time -- and then exits non-zero so nothing downstream records the
    number as a result.

    This is the shape an engine that stops answering has to fail in: a
    named, non-zero exit that names the lost trials, rather than an
    unbounded wait that produces neither a number nor an error.
    """
    bad = broken_records(recs)
    if not bad:
        return
    raise SystemExit(
        f"{len(bad)} of {len(recs)} trials did not run, so no evaluation "
        f"number is claimed for this run:\n"
        + "".join(f"  seed {x['seed']}: {x['broken']}\n" for x in bad)
        + "  every trial that DID run is in the report above and in "
          "--records-out, which carries a `broken` field per trial; "
          "re-run the evaluation rather than quoting a smaller one")


def write_dump(recs, path):
    """--dump-actions: three fields per trial, in trial order."""
    out = [{"seed": x["seed"], "success": x["success"],
            "actions": x["actions"]} for x in recs]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    uniq = {x["actions"] for x in out if x["actions"]}
    say(f"first-turn action lists: {len(uniq)} distinct over "
        f"{len(out)} trials and "
        f"{len({x['seed'] for x in out})} seeds -> {path}")


def report(a, recs, whole=True):
    """Summary and dumps, off the records, wherever they came from.

    `whole` says these records are the WHOLE evaluation, and a shard's
    are not, so a shard prints a different line for them on purpose.
    A log reader greps the run log with an unanchored
    `trials (\\d+) emitted a tool call ...` and takes the FIRST match,
    and a harness that tees every shard's stdout into one log gives it
    eight candidates. Eight shards printing the summary's shape would
    hand it one eighth of the run's mean reward and tool-call
    count beside the merged solve count, in a results file that says
    nothing about it. Prefixing the line is not enough: the regex is
    unanchored, so it matches inside `[shard 3/8] trials 12 ...` too.
    """
    if whole:
        summarise(recs)
    else:
        say(f"done: {len(recs)} trials, "
            f"{sum(1 for x in recs if x['success'])} solved")
    bad = broken_records(recs)
    if a.dump_actions and bad:
        # NOT WRITTEN when a trial did not run, and that is the whole
        # of the protection. The dump carries seed, success and actions
        # and has no field for "this one never happened", so a broken
        # trial is indistinguishable in it from a trial that tried
        # and failed. A sweep driver that skips an arm because its dump
        # file already exists would then adopt a refused evaluation as a
        # result, on the next run, silently. --records-out below is
        # written either way and is the same three fields plus the ones
        # that say so.
        say(f"NOT writing {a.dump_actions}: {len(bad)} trial(s) did not "
            f"run, and the dump has no way to say which. The full records "
            f"are the file to read")
    elif a.dump_actions:
        write_dump(recs, a.dump_actions)
    if a.records_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.records_out)),
                    exist_ok=True)
        with open(a.records_out, "w") as f:
            json.dump(recs, f, indent=1)


# The flags that describe the WHOLE evaluation, so a shard must not
# inherit them: --dp is the split itself, and the three outputs are the
# merged files the parent writes once.
WHOLE_RUN_FLAGS = ("--dp", "--shard", "--dump-actions", "--records-out")


def child_argv(argv, i, n, records):
    """This command line, as the one shard i of n should run.

    The parent's own argv is REWRITTEN rather than rebuilt out of the
    parsed namespace. A rebuild is a second list of every flag this
    script takes, and it goes stale the first time somebody adds one --
    silently, because a flag missing from a shard is not an error, it is
    a shard evaluating a slightly different policy. Everything but the
    five above reaches the shard untouched, whatever it is.
    """
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok.split("=", 1)[0] in WHOLE_RUN_FLAGS:
            skip = "=" not in tok
            continue
        out.append(tok)
    out += ["--shard", f"{i}/{n}", "--records-out", records]
    return out


def wait_for(procs):
    """Every shard, or the first failure.

    A shard that dies takes the evaluation with it rather than being
    left out of the merge. The shards are contiguous slices of one
    trial list, so seven of eight is a smaller evaluation wearing the
    same protocol's name, and the percentage it prints would be quotable
    against rows that were not measured that way. Killing the survivors
    is the caller's, on EVERY way out and not only this one.
    """
    left = list(procs)
    while left:
        for p in list(left):
            rc = p.poll()
            if rc is None:
                continue
            left.remove(p)
            if rc:
                raise SystemExit(
                    f"shard {procs.index(p)} of {len(procs)} exited {rc}, "
                    f"so no merged number is printed: it would be a "
                    f"smaller evaluation than the one asked for, under "
                    f"the name of the one asked for")
        if left:
            time.sleep(0.2)


def stop(procs, grace=10.0):
    """Every shard and everything it started: TERM, then KILL.

    A shard is not one process. vLLM spawns its own workers -- this file
    sets the spawn method at the top -- and every live Episode is an
    engine process, so a SIGTERM to the shard alone leaves both holding
    cards. Eight orphan workers can make a box unusable.
    Popen puts each shard in its own SESSION so one signal reaches all
    of it, and the KILL after the grace period is because a process
    wedged in NCCL teardown does not answer the first one -- waiting on
    terminate() alone hangs the parent on exactly the failure this is
    called for.
    """
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                p.terminate()
    deadline = time.time() + grace
    for p in procs:
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.1)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                p.kill()
        p.wait()


def read_records(paths, n):
    """The shards' records, in shard order, checked for completeness.

    An exit code of 0 was the only thing standing between a shard that
    evaluated fewer trials than its slice and a percentage printed
    over fewer trials than the protocol names. The parent never reads
    the prompt parquet, so it does not know the total -- but it does not
    have to: the shards' own sizes must be the balanced split of their
    own sum, and one short shard makes the others too big for the total
    it implies.
    """
    out, sizes = [], []
    for i, path in enumerate(paths):
        try:
            with open(path) as f:
                shard = json.load(f)
        except (OSError, ValueError) as e:
            raise SystemExit(
                f"shard {i} of {n} exited 0 and its records are not "
                f"readable: {e}\n"
                f"  nothing is merged, because the number would be over "
                f"fewer trials than the run asked for")
        sizes.append(len(shard))
        out += shard
    total = sum(sizes)
    want = [hi - lo for lo, hi
            in (shard_bounds(total, i, n) for i in range(n))]
    if sizes != want:
        raise SystemExit(
            f"the shards returned {sizes} trials, which is not the "
            f"balanced split {want} of {total}\n"
            f"  a shard evaluated fewer trials than its slice, so a "
            f"merged number would be over fewer than the run asked for")
    return out


def run_shards(a, argv):
    """--dp: N of this script, one per card, merged into one report.

    The parent loads no weights, starts no vLLM and steps no episode.
    It splits the trial list, runs a shard per card, and joins the
    per-trial records back together in shard order, which is the
    order the unsharded run would have produced them in.
    """
    if a.shard:
        raise SystemExit(
            "--dp spawns the shards and --shard is what it passes each "
            "one, so naming both says two things about one split\n"
            "  --dp 8        eight shards on this box\n"
            "  --shard 0/8   run one of them, for a split across boxes")
    # vLLM takes the cards it is given, so the split has to be made of
    # CUDA_VISIBLE_DEVICES here: an unset or short list would put every
    # shard's engine on card 0, which is slower than not sharding and
    # would still print a number.
    devices = [d.strip() for d
               in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if d.strip()]
    if len(devices) < a.dp * a.tp:
        raise SystemExit(
            f"--dp {a.dp} with --tp {a.tp} needs {a.dp * a.tp} cards and "
            f"CUDA_VISIBLE_DEVICES names {len(devices)}\n"
            f"  each shard gets its own, so a short list would stack "
            f"engines on one card\n"
            f"  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ... --dp {a.dp}")
    # Shard outputs live in a private directory so an interrupted run cannot
    # leave fragments beside a completed evaluation artifact.
    tmp = tempfile.mkdtemp(prefix="brl_dp_")
    records = [os.path.join(tmp, f"shard{i}.json") for i in range(a.dp)]
    procs, keep = [], False
    say(f"--dp {a.dp} --tp {a.tp}: up to {a.dp} independent engines over "
        f"cards {','.join(devices[:a.dp * a.tp])}, one shard of the "
        f"trials each")
    try:
        for i in range(a.dp):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                devices[i * a.tp:(i + 1) * a.tp])
            # A shard's stdout is a pipe here, not a terminal, so python
            # block-buffers it and the prefixed lines of eight shards
            # arrive in 8 KiB clumps that do not respect line
            # boundaries. Every deployed caller tees this.
            env["PYTHONUNBUFFERED"] = "1"
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(__file__)]
                + child_argv(argv, i, a.dp, records[i]),
                env=env, start_new_session=True))
        wait_for(procs)
        merged = read_records(records, a.dp)
        report(a, merged)
        # The parent enforces the same rule its children do. A shard that
        # lost a trial exits non-zero and `wait_for` has already
        # stopped, so this should be unreachable -- which is exactly why
        # it is here: the merged records are the only place the whole
        # evaluation exists, and "a broken trial can only arrive one
        # way" is the kind of thing that stays true until somebody adds a
        # second way.
        refuse_partial(merged)
    except BaseException:
        # EVERY way out, not only a failed shard: a Popen that raises
        # half way through the launch loop, and a Ctrl-C or a SIGTERM
        # from whatever supervises this, both used to leave the shards
        # already started running on their cards with no parent.
        stop(procs)
        # Rows that were really sampled, at the cost of the GPUs, are
        # not deleted because the merge did not get to them. The
        # records are a few kilobytes and go with them.
        #
        keep = any(os.path.exists(p) for p in records)
        if keep:
            say(f"the shards' records are kept at {tmp}")
        raise
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


def parser():
    """The flags, as their own function.

    Separate from main() so the interface can be asked what it takes
    without a GPU, a checkpoint, or an engine in the room. Parser tests use
    this function directly rather than copying command defaults.
    """
    # No abbreviations, which argparse allows by default. --dp's shard
    # rewrite strips the flags naming the WHOLE run by name, so
    # `--dump o.json` would parse here as --dump-actions and survive
    # that strip, and all N shards would write the merged run's dump
    # file at once. Refusing the abbreviation costs a longer command
    # line and removes the class.
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default=None,
                    help="task YAML asserted by the frozen rows; the public "
                         "bedrock CLI always supplies this")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--tokenizer", default=None,
        help="processor/tokenizer to use while serving --model. Defaults "
             "to the checkpoint's processor (or its LoRA base). Set this "
             "to the original base model when evaluating an exported "
             "full-weight checkpoint so both comparison arms render the "
             "identical chat template and use the identical tokenizer")
    ap.add_argument("--n", type=int, default=64, help="prompt rows to use")
    ap.add_argument("--samples", type=int, default=1,
                    help="trials per prompt row")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="turn cap; defaults to the task YAML's own "
                         "max_turns, which is what Episode and trial.py "
                         "read. A flag that disagreed with the task cut "
                         "trajectories the episode had not finished.")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int,
                    default=DEFAULT_MAX_NEW_TOKENS,
                    help="per-turn response budget; shared with the "
                         "non-verl trainers")
    ap.add_argument("--hint", default=None,
                    help="task YAML whose hint goes in a SYSTEM message "
                         "while sampling (bedrock_rl/sdg/guidance.py)")
    ap.add_argument("--hint-level", default="procedure",
                    help="hint grade: none | coarse | procedure")
    ap.add_argument("--finish-level", default="instruct",
                    help="stop-scaffold grade: off | instruct")
    ap.add_argument("--keep-latest-images", type=int, default=0,
                    help="retain the newest N observation images in model "
                         "context; 0 keeps all. Recorded in every trial")
    ap.add_argument("--dump-actions", default=None,
                    help="write each trial's seed and first-turn action "
                         "list as JSON. Identical coordinates across "
                         "different seeds mean the policy memorised them "
                         "rather than reading the screen")
    ap.add_argument("--blank-frames", action="store_true",
                    help="replace every frame with flat grey. A policy "
                         "that keeps its score here is not looking at the "
                         "screen, it is replaying memorised coordinates")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--dp", type=int, default=1,
                    help="run N copies of this evaluation side by side, "
                         "one card each, over contiguous shards of the "
                         "same trial list, and merge them. This is the "
                         "parallelism the loop argues for: the generate "
                         "half is the only GPU-bound half and every "
                         "episode steps the engine on CPU, so N whole "
                         "copies scale both halves where --tp scales "
                         "neither. It needs CUDA_VISIBLE_DEVICES to name "
                         "N x --tp cards, one shard's worth each. "
                         "Default 1, one card; changing it re-rolls which trajectories "
                         "you get, so see --tp before comparing rows")
    ap.add_argument("--tp", type=int, default=1,
                    help="cards for ONE engine, vLLM tensor parallel. For "
                         "a checkpoint that does not fit on one card or "
                         "leaves no room for its KV cache, NOT for "
                         "throughput. It splits every matmul, so partial "
                         "sums are reduced in a different order and the "
                         "logits differ in their last bits, which at "
                         "--temp 1.0 can flip a sampled token and then "
                         "the rest of the trajectory. Nothing here is "
                         "seeded, so no two runs of one command agree "
                         "token for token in any case; what --tp changes "
                         "is the DISTRIBUTION being sampled, and that is "
                         "the part more trials do not average away. "
                         "Hold it fixed across the rows of a comparison")
    ap.add_argument("--shard", default=None,
                    help="run trials i of N only, written `i/N`. --dp "
                         "passes this to its own children; by hand it "
                         "splits one evaluation across boxes, and the "
                         "shards' records concatenate in i order into the "
                         "unsharded run's own order")
    ap.add_argument("--records-out", default=None,
                    help="every trial's seed, success, reward, turn "
                         "count and first-turn actions as JSON, in "
                         "trial order. --dump-actions is three of those "
                         "fields; this is the file --dp merges its shards "
                         "through")
    ap.add_argument("--lora", default=None,
                    help="a LoRA adapter directory to serve on top of "
                         "--model; inferred from --model when that is "
                         "itself an adapter, or a merged RL checkpoint "
                         "with a lora_adapter/ beside its weights")
    return ap


def main():
    global PREFIX
    a = parser().parse_args()
    if a.dp < 1 or a.tp < 1:
        raise SystemExit("--dp and --tp count cards, so neither is less "
                         "than 1")
    if a.keep_latest_images < 0:
        raise SystemExit("--keep-latest-images cannot be negative")
    if a.dp > 1:
        # Before any engine import: the parent of a sharded run holds no
        # weights and touches no card, so a box that is fully committed
        # to its shards has nothing else on it.
        return run_shards(a, sys.argv[1:])
    shard, shards = parse_shard(a.shard)
    if shards > 1:
        PREFIX = f"[shard {shard}/{shards}] "

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # An adapter argument is two paths: the weights to serve and the
    # adapter to serve on top of them. Sorting that out here rather than
    # in the caller is what lets every checkpoint this repo writes be
    # passed to --model unchanged.
    # The command accepts the same registry aliases as every trainer. A bare
    # alias is not a Hugging Face repository id, so handing it directly to
    # AutoProcessor made `bedrock eval ... --model qwen3-vl-2b` fail even
    # though the identical spelling trains successfully.
    policy = reg.resolve_policy(a.model, a.lora)
    model_path, adapter = policy.base, policy.adapter
    model_path = reg.weights_path(model_path)
    lora_request, lora_kwargs = None, {}
    if adapter:
        rank = policy.adapter_rank
        # vLLM serves LoRA for the language model only and drops any
        # adapter on the vision tower without saying so, so an adapter
        # carrying one is not the policy that gets sampled here. Every
        # path in this repo excludes the tower; an adapter that names no
        # exclusion at all came from somewhere else and may not.
        if not (lora.read_adapter_config(adapter) or {}).get(
                "exclude_modules"):
            say(f"warning  {adapter} declares no exclude_modules, so it "
                f"may carry adapters on the vision tower. vLLM serves "
                f"LoRA for the language model only and drops those "
                f"without saying so, which would make this evaluation a "
                f"different policy from the one that trained.")
        lora_kwargs = {"enable_lora": True,
                       "max_lora_rank": lora.vllm_max_rank(rank),
                       "max_loras": 1}
        lora_request = LoRARequest("bedrock", 1, str(adapter))
        say(f"serving {model_path} + adapter {adapter} (rank {rank})")

    raw = load_dataset("parquet", data_files=a.data)["train"]
    rows = [raw[i] for i in range(min(a.n, len(raw)))]
    eval_task = evaluation_task(rows, a.task)
    hint = (hint_text(Task.load(a.hint), a.hint_level,
                      a.finish_level) if a.hint else None)
    knobs = sampling_knobs(a.temp, a.max_new_tokens)
    protocol = {
        "task": relative_path(eval_task.path),
        "hint": bool(hint),
        "hint_level": a.hint_level if hint else "none",
        "finish_level": a.finish_level if hint else "off",
        "keep_latest_images": a.keep_latest_images,
        "blank_frames": bool(a.blank_frames),
        "sampling": dict(knobs),
        "max_tool_calls_per_turn": MAX_TOOL_CALLS_PER_TURN,
        "max_model_len": int(a.max_model_len),
        "tensor_parallel_size": int(a.tp),
    }
    # The turn budget belongs to the task, the same place Episode and
    # bedrock_rl/train/rollout.py reads it from. Every prompt row in one file
    # carries the same task_yaml, so the first row decides it.
    max_turns = a.max_turns
    if max_turns is None:
        max_turns = eval_task.max_turns
    elif max_turns != eval_task.max_turns:
        raise SystemExit(
            f"--max-turns {max_turns} disagrees with the task's "
            f"max_turns: {eval_task.max_turns}; the task owns the episode "
            "budget")
    protocol["max_turns"] = int(max_turns)
    say(f"max turns: {max_turns}"
        + ("" if a.max_turns is not None else " (from the task YAML)"))

    # The trial list as SPECS, before any of them owns an engine
    # process, because a shard has to be able to skip the ones that are
    # not its own without starting them. Slicing here and constructing
    # Rolls below also keeps the engines' start order exactly what it
    # was: after the weights are loaded, never before.
    specs = [(row_index, row, k)
             for row_index, row in enumerate(rows)
             for k in range(a.samples)]
    lo, hi = shard_bounds(len(specs), shard, shards)
    if shards > 1 and lo == hi:
        # More shards than trials. Saying so and stopping is better
        # than loading a model to sample nothing with, and the empty
        # record file is what keeps the parent's merge total-order.
        # Sharded runs only: an UNSHARDED run over an empty list still
        # goes the whole way round, because that path's output is the
        # one every published number was read off and this is not the
        # change that gets to alter it.
        say(f"no trials in this shard of {len(specs)}")
        report(a, [], whole=False)
        return

    processor_path = (reg.model_path(a.tokenizer) if a.tokenizer else
                      lora.processor_path(model_path, adapter))
    spec = reg.resolve(model_path)
    processor = reg.load_processor(spec, processor_path)
    tools = tool_schema()
    # tensor_parallel_size is passed rather than left to vLLM's own
    # default of 1, so the protocol these numbers were taken under is
    # pinned by this file and not by a default in another project.
    engine_kwargs = vllm_engine_kwargs(model_path, lora_kwargs)
    llm = LLM(model=model_path, tokenizer=processor_path,
              max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_frac, enforce_eager=True,
              tensor_parallel_size=a.tp,
              limit_mm_per_prompt={"image": multimodal_image_limit(
                  max_turns,
                  keep_latest_images=a.keep_latest_images)}, **engine_kwargs)
    sp = SamplingParams(
        temperature=knobs["temperature"], top_p=knobs["top_p"],
        top_k=knobs["top_k"], max_tokens=knobs["max_new_tokens"])

    def maybe_blank(img):
        if not a.blank_frames:
            return img
        from PIL import Image
        return Image.new("RGB", img.size, (127, 127, 127))

    class Roll:
        """One live trial: its own episode, its own message list."""

        def __init__(self, row, k, row_index):
            xi = row["extra_info"]
            self.task = eval_task
            self.seed = int(xi["seed"])
            self.user_text = without_leading_image_marker(
                row["prompt"][-1]["content"])
            text = self.user_text
            self.images = [maybe_blank(row["images"][0])]
            self.msgs = with_hint(
                [{"role": "user",
                  "content": [{"type": "image"},
                              {"type": "text", "text": text}]}], hint)
            self.k = k
            self.row_index = int(row_index)
            self.init_yaw = float(xi["init_yaw"])
            self.init_pitch = float(xi["init_pitch"])
            self.turns = []
            self.done = False
            # Everything that names this trial is set BEFORE the engine
            # is opened, because opening it is one of the things that can
            # fail and a failure nobody can attribute to a trial is a
            # failure of the whole run.
            self.ep = None
            self.broken = None
            self.rec = None
            self.protocol = protocol
            try:
                self.ep = Episode.from_spec(
                    xi, cu_parser(self.task), task=self.task,
                    frames_root=os.path.join("/tmp", "sample_frames"))
            except Exception as e:
                self.fail("could not open the episode", e)

        def fail(self, doing, e):
            """This ROLLOUT is lost. The evaluation is not.

            One engine used to be able to take the whole run with it, and
            the worse half of that was silence: a wedged engine held an
            unbounded read and the process sat at 0% CPU for ever. The
            read is bounded now (bedrock_rl/env/engine.py), so what
            reaches here is a named error, and what it costs is this
            trial. `refuse_partial` at the end is what stops the run
            reporting a number over the ones that survived.
            """
            self.broken = f"{doing}: {type(e).__name__}: {e}"
            say(f"BROKEN  seed {self.seed} sample {self.k}: {self.broken}")
            self.finish()

        def finish(self):
            """Score this trial, release its engine, never touch it again.

            Called the moment a trial ends rather than after the last
            turn, because until it was, a 96-episode evaluation held 96
            game processes open from its first turn to its last, at about
            500 MiB of host memory each, most of them
            finished and doing nothing. The record is taken first and is
            all anything downstream reads, so nothing here depends on an
            episode still being open.
            """
            if self.rec is not None:
                return self.rec
            self.done = True
            self.rec = record(self)
            ep, self.ep = self.ep, None
            if ep is not None:
                try:
                    ep.close()
                except Exception as e:
                    say(f"warning  seed {self.seed}: the episode would not "
                        f"close ({e}), so its engine may still be running")
            return self.rec

        def render(self):
            """(prompt text, frames) under the SAME window training used.

            Typed message selection comes from core.context, before this
            model-specific renderer sees the view. The custom verl loop uses
            that same contract and does not patch upstream tokenizer code.
            """
            from bedrock_rl.core.context import keep_last_images
            keep = a.keep_latest_images
            msgs, imgs = self.msgs, self.images
            if keep > 0:
                view = keep_last_images(msgs, keep)
                removed = len(self.images) - keep
                msgs, imgs = view.messages, self.images[max(removed, 0):]
            return processor.apply_chat_template(
                template_messages(msgs), tokenize=False,
                add_generation_prompt=True,
                tools=tools), imgs

        def advance(self, out_text):
            self.turns.append(out_text)
            calls, messages, frames, done = execute_turn(
                self.ep, out_text, len(self.turns) - 1)
            if not calls:
                self.done = True
                return
            self.msgs.extend(messages)
            self.images.extend(maybe_blank(frame) for frame in frames)
            self.done = done

    # The shard's own trials, in the order the unsharded run would
    # have built them, so a record keeps the episode spec it came from
    # and the merged file keeps the whole list's order.
    rolls = [Roll(row, k, row_index)
             for row_index, row, k in specs[lo:hi]]
    say(f"{len(rolls)} trials, hint="
        f"{a.hint_level if hint else 'none'}"
        + (f", trials {lo} to {hi - 1} of {len(specs)}"
           if shards > 1 else ""))
    for turn in range(max_turns):
        live = [r for r in rolls if not r.done]
        if not live:
            break
        rendered = [r.render() for r in live]
        reqs = [{"prompt": text, "multi_modal_data": {"image": imgs}}
                for text, imgs in rendered]
        sampled = generate_isolated(
            live, reqs,
            lambda batch: llm.generate(
                batch, sp, lora_request=lora_request))
        for r, o in sampled:
            try:
                r.advance(o.outputs[0].text)
            except Exception as e:
                # An engine that dies or stops answering costs its own
                # trial and nothing else. Every other episode in this
                # batch has already been sampled for and is still worth
                # finishing.
                #
                # The reason is taken from the EXCEPTION and not from
                # where the handler sits. `except Exception` here catches
                # a parser or verifier bug just as readily, and a
                # blanket "the engine stopped answering" would put that
                # diagnosis on all 96 trials and send whoever reads the
                # log to look at the engine.
                r.fail(failure_reason(e), e)
            if r.done:
                r.finish()
        say(f"  turn {turn + 1}: {len(sampled)} sampled, "
            f"{sum(1 for r in rolls if solved(r))} solved so far")

    recs = [r.finish() for r in rolls]
    report(a, recs, whole=shards == 1)
    # Last, after everything a real trial paid for has been written.
    refuse_partial(recs)


if __name__ == "__main__":
    main()
