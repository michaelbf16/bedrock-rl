"""Rationalisation hints: privileged guidance shown only while sampling,
deleted before training.

SDPO and the self-distillation workflow use this module and neither may keep its own
copy. SDPO shows feedback only to the self-teacher and withholds it from
the student. Self-distillation samples with a hint, keeps successes, then deletes it.

There are TWO channels here and they are different shapes.

The STATIC channel is a written guide out of the task YAML, the same
string on every turn of every episode. It is one SYSTEM message holding
that string and nothing else.

The PER-TURN channel is state-conditional: a provider reads the LIVE
engine state each turn and writes a sentence about where the target is
right now. It is a suffix on the OBSERVATION text, so it lands in a
`user` or `tool` message and never in an assistant one.

That channel has three separable parts and only two of them are in this
file. A LADDER of rungs says how much is said; a per-episode probability
(`--turn-hint-p`) decides whether anything is said at all; a per-task
PROVIDER recomputes the sentence from live engine geometry every turn.
The first two are task-agnostic and live here. The task selects the third
by import path, so custom feedback stays beside the task that owns it.

Where the show/hide coin is flipped is a design decision and not a
detail. Fixing the rung per prompt ROW, in the dataset, satisfies the
per-group rule below and is the obvious way to do it for a trainer whose
groups are a property of its data. This module draws the coin at run
time, keyed on the GROUP ID, which is what a trainer that forms its
groups per step can actually do.

Four properties are correctness guarantees.

1. The static hint occupies exactly one place, a SYSTEM message holding
   it and nothing else, so removal is deletion of a known message.
   `assert_hint_removable` proves at run time, on prompts the model
   family's OWN chat template rendered, that removing that message
   reproduces the hint-free prompt byte for byte, that the template
   embeds the hint verbatim, and that no line of it survives into the
   hint-free rendering. A heuristic scrub would eventually leak hint text
   into training data. What it deliberately does NOT demand is that the
   difference between the two renderings be the bare hint: every template
   wraps a system message in its own role markup, and several substitute
   a default system string when there is no system message, so a check
   that insisted on a bare removable block was a check on one template's
   habits rather than on the property. The per-turn channel is ADDITIVE
   and does not touch that proof: it never writes a system message, so a
   run with `--turn-hint off` renders the same bytes it always did, and
   `check_hint_exact` still passes with the per-turn channel on.
2. No hint is ever in a loss-carrying position; only assistant turns are
   trained on (`bedrock_rl/adapters/netherite/chat.py::encode_trajectory` builds the
   mask, `bedrock_rl/train/sft.py` turns it into labels). The per-turn
   channel rests on that property rather than on `assert_single_block`,
   which a per-turn suffix cannot use, because there is no single block
   to remove.
3. Both channels are removed EXACTLY, never heuristically. Every per-turn
   suffix the sampler wrote is recorded as (message index, exact string)
   as it is written, and `without_turn_hints` deletes those and refuses
   to guess when the recorded string is not where the record says. So the
   self-distillation training set and the SDPO student context are byte for byte the
   prompt a hint-free deployment renders, and `strip` is the one call
   that removes both channels in the one order that works.
4. Hints are graded, so privileged information is a dial rather than a
   constant to trust. The per-turn channel has a second dial,
   `--turn-hint-p`, the PER-EPISODE probability the hint is shown at all.
   The decision is taken once, before the first turn, and is HOMOGENEOUS
   ACROSS A GROUP, so a group-relative estimator never compares a hinted
   rollout against an un-hinted sibling. That is a correctness guarantee
   and not a style choice: such an estimator scores a rollout against the
   mean of its own group, so in a mixed group the hinted sibling wins the
   advantage for having been told the answer and the gradient rewards
   nothing the policy did.

The rungs are named for an AIMING task because the shipped provider is
one. `TURN_LEVELS` is a global vocabulary and a task whose useful ladder
is not directional would have to widen it; that is the known limit of
this factoring, and the alternative is to let each provider declare its
own rungs.

A hint describes procedure and direction, never a tool argument.
Task YAML keys, all optional.

  system_hint          the `procedure` level, a step-by-step guide
  system_hint_coarse   the `coarse` level, direction only
  finish_hint          when to stop, the `instruct` level

Ending the turn is hard independently of finding the target, and
trajectories with no voluntary stop cannot teach one, so that scaffold is
its own dial.

The per-turn channel is code selected by the task's
`turn_hint_provider`. No trainer changes when a task adds its own provider.
"""
import argparse
import os
import random

LEVELS = ("none", "coarse", "procedure")
FINISH_LEVELS = ("off", "instruct")

# The per-turn ladder, ordered by how much a rung gives away. `off` is
# the shipped default everywhere, so nothing in this repo changes
# behaviour until a caller asks for a rung.
TURN_LEVELS = ("off", "weak", "direction", "oracle")
# What separates the observation text from the hint suffix. It is part of
# the recorded block, so removal does not read this constant back; it is
# here so that the one writer has one answer and a reader of a rendered
# prompt can see where the suffix begins.
TURN_HINT_SEP = "\n\n"


def levels(task):
    """Available hint levels for a task, level name to text."""
    out = {}
    for key, level in (("system_hint_coarse", "coarse"),
                       ("system_hint", "procedure")):
        v = task.raw.get(key)
        if v and str(v).strip():
            out[level] = str(v).strip()
    return out


def finish_text(task, level="instruct"):
    if level == "off":
        return ""
    if level not in FINISH_LEVELS:
        raise ValueError(f"unknown finish level {level!r}, expected one of "
                         f"{FINISH_LEVELS}")
    v = task.raw.get("finish_hint")
    return str(v).strip() if v else ""


def hint_text(task, level="procedure", finish="instruct"):
    """The full hint block for a task, or None at level `none`."""
    if level in (None, "none"):
        return None
    if level not in LEVELS:
        raise ValueError(f"unknown hint level {level!r}, expected one of "
                         f"{LEVELS}")
    have = levels(task)
    if level not in have:
        raise ValueError(f"task {task.name} has no {level!r} hint; it "
                         f"defines {sorted(have) or 'none'}")
    parts = [have[level]]
    tail = finish_text(task, finish)
    if tail:
        parts.append(tail)
    return "\n\n".join(parts)


def with_hint(messages, hint):
    """Prepend the hint as its own SYSTEM message."""
    if not hint:
        return list(messages)
    return [{"role": "system", "content": hint}] + list(messages)


def without_hint(messages, hint):
    """Remove the hint message, refusing to guess.

    A silent no-op here would leak the hint into the student context or
    the self-distillation training set, so a missing hint is an error rather than a
    pass-through."""
    if not hint:
        return list(messages)
    out = [m for m in messages
           if not (m["role"] == "system" and m["content"] == hint)]
    if len(out) != len(messages) - 1:
        raise ValueError("hint system message not found exactly once, "
                         "refusing to guess")
    return out


def _trim(a, b):
    """Lengths of the longest common prefix and suffix of two strings."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    j = 0
    while j < n - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    return i, j


def _stretch_over_hint(hinted_text, plain_text, hint):
    """The differing stretch, widened to whole-hint boundaries.

    Trimming the longest common prefix and suffix cuts INTO the hint
    whenever it happens to begin or end with the same character as
    whatever the template had there instead: a hint ending in a full stop
    that replaced "You are a helpful assistant." loses its full stop, and
    a test for "the hint is in there" then fails on a rendering that is
    perfectly correct. So any occurrence of the hint that overlaps the
    stretch is absorbed into it. Nothing is widened when the hint is not
    what changed, which is the case worth failing on.
    """
    start, j = _trim(hinted_text, plain_text)
    end = len(hinted_text) - j
    block = hinted_text[start:end]
    h = hint.strip()
    k = hinted_text.find(h) if h else -1
    while k != -1:
        if k < end and k + len(h) > start:
            return hinted_text[min(start, k):max(end, k + len(h))]
        k = hinted_text.find(h, k + 1)
    return block


def hint_leaks(plain_text, hint):
    """Every line of the hint that survived into a HINT-FREE rendering.

    Line by line rather than whole-string, because a partial leak is
    still privileged text in the student's context and a template that
    reflowed or truncated the block would hide from a whole-string test.
    """
    lines = [ln.strip() for ln in hint.splitlines() if ln.strip()]
    return [ln for ln in lines if ln in plain_text]


def assert_single_block(hinted_text, plain_text, hint):
    """Prove, on two real rendered prompts, that the static hint sits in
    one place and reaches none of the hint-free one. Returns the hinted
    side of the differing stretch.

    This used to demand that the stretch be the hint and NOTHING else,
    which is a property of no shipped chat template: a template writes a
    system message inside its own role markup (`<|system|>`,
    `<|im_system|>system<|im_middle|>` ... `<|im_end|>`), and several
    substitute a default system string when the message list has no
    system message rather than emitting nothing at all. Both are ordinary
    template behaviour and neither can leak a hint, so demanding their
    absence rejected families for being a different family.

    What must hold for EVERY family is the two things below, and they are
    what the self-distillation training set and the SDPO student context rest on. The
    third property, that the template embeds the hint verbatim and treats
    it as opaque, needs a second rendering to see and is proved by
    `assert_hint_removable`.
    """
    block = _stretch_over_hint(hinted_text, plain_text, hint)
    if hint.strip() not in block:
        raise ValueError(
            "the hinted and hint-free renderings do not differ by a single "
            "stretch carrying the hint, so the hint is not what changed "
            f"between them: they differ by {block[:120]!r}")
    leaks = hint_leaks(plain_text, hint)
    if leaks:
        raise ValueError(
            f"{len(leaks)} line(s) of the hint survive into the HINT-FREE "
            f"rendering, so removing it is not exact: {leaks[0][:120]!r}")
    return block


# A system message no task hint, tool schema or observation would ever
# contain, rendered a second time so the template's handling of the hint
# can be compared against its handling of something else. Plain ASCII, so
# a reader of this file can see every character of it, and MULTI-LINE,
# because a single-line probe cannot see a template that collapses one.
_PROBE = "NETHERITE-HINT-PROBE-LINE-1\nNETHERITE-HINT-PROBE-LINE-2"


def assert_hint_removable(render, body, hint, family=None):
    """Prove the static hint channel is exact for THIS model family, on
    prompts its own chat template rendered. Runs once per run, before any
    gradient is taken. Returns the differing stretch.

    `render(messages) -> str` renders one message list the way the
    trainer will; `family` names the registry entry for the error
    messages, because a family that cannot carry this channel has to be
    named in the sentence that says so rather than left to be guessed
    from a traceback.

    Four things are checked and every one of them is a property of the
    TEMPLATE, so this is family-general by construction rather than by
    enumerating families:

    1. Removing the hint message reproduces the hint-free rendering byte
       for byte. That is the whole claim: the student sees the prompt a
       hint-free deployment renders.
    2. The template embeds the system content VERBATIM and independently
       of what that content is, proved by rendering a probe string and
       substituting it back. A template that escapes, truncates, reflows
       or reorders the block fails here, and each of those would put text
       in the teacher's context that no removal could account for.
    3. The teacher really receives it. A family whose template drops
       system messages, or drops them when tools are present, renders the
       hint nowhere; that is a silent no-op that reads as a hinted run.
    4. No line of it reaches the hint-free rendering.
    """
    who = f"model family {family!r}: " if family else ""
    plain = render(list(body))
    hinted = render(with_hint(body, hint))
    if render(without_hint(with_hint(body, hint), hint)) != plain:
        raise ValueError(
            f"{who}removing the hint system message does not reproduce the "
            "hint-free rendering, so the student would train on a prompt no "
            "hint-free deployment renders")
    probe = render(with_hint(body, _PROBE))
    if probe.replace(_PROBE, hint) != hinted:
        raise ValueError(
            f"{who}this family's chat template does not embed a system "
            "message verbatim: rendering a probe and substituting the hint "
            "back does not reproduce the hinted prompt, so what the teacher "
            "reads is not the hint this run thinks it showed")
    if hint.strip() not in hinted:
        raise ValueError(
            f"{who}the hint does not appear in the rendered prompt at all. "
            "This family's chat template drops the system message (some "
            "drop it only when tools are present), so the hint channel is "
            "a silent no-op here. Give the family a chat template that "
            "renders system messages, under "
            "bedrock_rl/templates/chat_template/, and name it in its "
            "ModelSpec.")
    return assert_single_block(hinted, plain, hint)


# ── the per-turn, state-conditional channel ──────────────────────────────
def turn_hint_provider(task):
    """Build the callable selected by ``task.turn_hint_provider``.

    A bare import path resolves a function. A component mapping may configure
    a callable object, which lets one provider expose task-specific dials
    without adding trainer flags.
    """
    configured = task.raw.get("turn_hint_provider")
    if not configured:
        return None
    from bedrock_rl.core.components import ComponentSpec
    spec = ComponentSpec.parse(configured, what="turn hint provider")
    provider = spec.create() if spec.config else spec.resolve()
    if not callable(provider):
        raise TypeError("turn_hint_provider must resolve to a callable")
    return provider


def turn_hint_env_defaults():
    """The two dials as the environment sets them, CHECKED.

    argparse validates `choices` against what the command line supplies
    and never against a default, so an env default went in unchecked: a
    run with BRL_TURN_HINT=directon printed `turn hint: directon`
    in its header, loaded the model, and only failed at the first
    rollout. And a float() on a bad BRL_TURN_HINT_P raised while
    the PARSER was being built, before --help could print. Both are
    checked here, with the variable named in the message.
    """
    level = os.environ.get("BRL_TURN_HINT", "off")
    if level not in TURN_LEVELS:
        raise SystemExit(f"BRL_TURN_HINT={level!r} is not a per-turn "
                         f"hint rung; expected one of {TURN_LEVELS}")
    raw = os.environ.get("BRL_TURN_HINT_P", "1.0")
    try:
        p = float(raw)
    except ValueError:
        raise SystemExit(f"BRL_TURN_HINT_P={raw!r} is not a number")
    check_turn_hint_p(p, "BRL_TURN_HINT_P")
    return level, p


def check_turn_hint_p(p, what="--turn-hint-p"):
    """A probability, and not a NaN.

    NaN is the one that matters: every comparison against it is False, so
    `hinters_for_groups` would fall through to `rng.random() < nan`, come
    out False for every group, and run a whole sweep cell UN-HINTED while
    its header said otherwise.
    """
    if not (p >= 0.0) or not (p <= 1.0):
        raise SystemExit(f"{what}={p!r} is not a probability in [0, 1]")
    return p


def _turn_hint_p_arg(raw):
    """argparse `type` for --turn-hint-p: parse, then check."""
    try:
        return check_turn_hint_p(float(raw))
    except SystemExit as e:
        raise argparse.ArgumentTypeError(str(e))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a number")


def add_turn_hint_args(ap):
    """The two per-turn dials, on any trainer's parser.

    Defaults come from `BRL_TURN_HINT` and `BRL_TURN_HINT_P`,
    checked by `turn_hint_env_defaults` above, so a sweep cell can set a
    rung for one arm without editing a launcher. `off` and 1.0 keep every
    existing run byte for byte what it was.
    """
    env_level, env_p = turn_hint_env_defaults()
    ap.add_argument("--turn-hint", default=env_level,
                    choices=list(TURN_LEVELS),
                    help="per-turn, state-conditional hint rung, appended "
                         "to the observation text and deleted before "
                         "training; the task selects its provider by import "
                         "path")
    ap.add_argument("--turn-hint-p", type=_turn_hint_p_arg, default=env_p,
                    help="PER-EPISODE probability the per-turn hint is "
                         "shown, so hinted and un-hinted episodes can be "
                         "mixed and the hint annealed off. The draw is "
                         "made once per GROUP, never per turn")
    return ap


class TurnHinter:
    """One episode's per-turn hint state.

    `on` is the per-episode decision and it is taken by the CALLER, once,
    before the first turn, so that every rollout in a group shares it
    (`hinters_for_groups` is what does that). Nothing here ever re-draws
    it, which is the property a group-relative estimator needs.
    """

    def __init__(self, task, level="off", on=True, visible=True):
        if level not in TURN_LEVELS:
            raise ValueError(f"unknown per-turn hint level {level!r}, "
                             f"expected one of {TURN_LEVELS}")
        self.task = task
        self.level = level
        self.fn = None
        if level != "off":
            self.fn = turn_hint_provider(task)
            if self.fn is None:
                raise ValueError(
                    f"--turn-hint {level} but no per-turn hint provider is "
                    f"configured for task {task.name!r}; set "
                    "turn_hint_provider: package.module:callable in its "
                    "task YAML.")
        self.on = bool(on) and self.fn is not None
        self.visible = bool(visible)
        # (message index, exact string) per suffix written, which is what
        # makes removal exact rather than a search for hint-shaped text
        self.marks = []
        # Every computed suffix, including teacher-only feedback that was
        # deliberately not placed in the student's generation context.
        self.feedback = []

    def text(self, env, turn_index):
        """What this rung says about the live state, or None."""
        if not self.on:
            return None
        t = self.fn(self.task, env, turn_index, self.level)
        if t is None:
            return None
        if not isinstance(t, str):
            # str() on whatever came back would happily turn a dict into
            # a hint. A provider returns a sentence or None.
            raise TypeError(
                f"the per-turn hint provider for {self.task.name!r} "
                f"returned {type(t).__name__}; it must return a string or "
                "None")
        return t.strip() or None

    def attach(self, messages, env, turn_index):
        """Append this turn's hint to the message just added, and record
        it. Returns the exact block written, or None.

        It appends to `messages[-1]`, which is always the observation the
        model is about to read and is never an assistant turn. The
        assertion below is cheap and is the structural half of property 2.
        """
        if not self.on:
            return None
        if not messages:
            raise ValueError("attach was called with no message to attach "
                             "to; the caller appends the observation and "
                             "then attaches to it")
        t = self.text(env, turn_index)
        if t is None:
            return None
        i = len(messages) - 1
        if messages[i]["role"] == "assistant":
            raise ValueError(
                "a per-turn hint was about to be appended to an assistant "
                "message, which is a loss-carrying position; refusing")
        block = TURN_HINT_SEP + t
        self.feedback.append((i, block))
        if self.visible:
            messages[i] = {**messages[i],
                           "content": messages[i]["content"] + block}
            self.marks.append((i, block))
        return block

    def teacher_view(self, messages):
        """Apply recorded state-dependent feedback to a copied trajectory."""
        out = [dict(message) for message in messages]
        for index, block in self.feedback:
            if not 0 <= index < len(out):
                raise ValueError("teacher feedback message index is stale")
            message = out[index]
            if message.get("role") == "assistant":
                raise ValueError("teacher feedback targets an assistant turn")
            # A visible hint is already present. Teacher-only feedback is
            # appended here, after the unhinted student has finished acting.
            if not str(message.get("content") or "").endswith(block):
                out[index] = {**message,
                              "content": str(message.get("content") or "")
                              + block}
        return out


def hinters_for_groups(task, groups, level="off", p=1.0, rng=None,
                       visible=True):
    """One TurnHinter per rollout, with the show/hide coin flipped ONCE
    PER GROUP.

    `groups` is the group id of each rollout, in rollout order. Every
    rollout sharing a group id gets the same decision, so a group is
    hint-homogeneous and a group-relative estimator never compares a
    hinted rollout against an un-hinted one. The decision is keyed by
    group rather than by rollout because a group, and not a trajectory,
    is the unit such an estimator compares.
    """
    check_turn_hint_p(p)
    rng = rng or random
    decided = {}
    out = []
    for g in groups:
        if g not in decided:
            if p >= 1.0:
                decided[g] = True
            elif p <= 0.0:
                decided[g] = False
            else:
                decided[g] = rng.random() < p
        out.append(TurnHinter(task, level, on=decided[g], visible=visible))
    return out


def without_turn_hints(messages, marks):
    """Remove the per-turn suffixes a TurnHinter wrote, refusing to guess.

    `marks` is TurnHinter.marks, (index, exact string). The string has to
    be exactly where the record says it is, at the END of that message,
    or this raises: two turns commonly write the SAME sentence, so a
    content search could not tell them apart and a heuristic scrub is
    what this repo does not do.

    Call this BEFORE `without_hint`, because removing the system message
    shifts every index by one. `strip` does both in that order.
    """
    if not marks:
        return list(messages)
    idx = [i for i, _ in marks]
    if len(set(idx)) != len(idx):
        # Each removal strips a SUFFIX, so two marks on one message would
        # only come off correctly innermost-last. Rather than sort and
        # hope, refuse: the one caller writes at most one suffix per
        # message, and a provider that broke that has a bug this function
        # cannot paper over.
        raise ValueError(f"two per-turn hints recorded against one "
                         f"message ({sorted(idx)}); removal would not be "
                         "exact, refusing to guess")
    out = [dict(m) for m in messages]
    for i, block in marks:
        if not 0 <= i < len(out):
            raise ValueError(f"per-turn hint recorded at message {i} but "
                             f"the trajectory has {len(out)} messages; "
                             "refusing to guess")
        m = out[i]
        if m["role"] == "assistant" or not m["content"].endswith(block):
            raise ValueError(
                f"per-turn hint not found at the end of message {i} "
                f"(role {m['role']!r}); refusing to guess")
        out[i] = {**m, "content": m["content"][:-len(block)]}
    return out


def strip(messages, hint=None, marks=()):
    """Every privileged string this module put into a trajectory, gone.

    The one call both self-distillation paths make, so neither can get
    the order wrong or forget a channel.
    """
    return without_hint(without_turn_hints(messages, marks), hint)
