"""Read verl's stdout and present a bedrock-rl run on top of it.

verl's console is not ours to rewrite. What it prints is a ray driver
plus one worker per rank, so every warning arrives four times, and its
metrics line is one `step:7 - critic/score/mean:0.031 - ...` string
several hundred characters wide that no one reads across.

bin/rl.sh pipes that stream through this filter, which

  1. tees EVERY line verbatim to the raw log, so nothing is lost and a
     failed run still has the whole thing to read,
  2. parses the metrics line and re-emits it through the same reporter
     the other trainers use, one aligned line per step, plus the same
     JSON-per-line metrics file they write,
  3. drops the per-rank duplicates of lines it has already shown, and
  4. passes errors, tracebacks and first-time warnings straight through,
     because a quiet console that swallows a stack trace is worse than a
     noisy one.

bedrock-rl's own `[tag] ...` diagnostics are not filtered. They are
matched first, always shown, and duplicate counts land in the summary. A
single environment failure can repeat once per rollout worker, so whether a
line is a project diagnostic is decided before any general log filtering.

BRL_VERL_VERBOSE=1 turns the filtering off and passes every line,
colour codes aside.

  verl ... 2>&1 | python -m bedrock_rl.adapters.verl.console \
      --raw-log logs/run.log --metrics logs/run.jsonl --run <slug> \
      --config task=... --config model=...
"""
import argparse
import os
import re
import sys

from bedrock_rl import reporting as ui_mod

# verl's own console logger: verl/utils/logger/aggregate_logger.py
_STEP = re.compile(r"^step:(\d+)\s+-\s+(.*)$")
_NUMPY_SCALAR = re.compile(
    r"^np\.(?:float(?:16|32|64)|int(?:8|16|32|64)|uint(?:8|16|32|64))"
    r"\(([^()]*)\)$"
)
# ray colours its worker prefixes, so the tag never sits at the start of
# the raw line and the escapes would land in a tee'd log
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
# ray tags every worker line with its actor and pid, which is what makes
# four identical warnings look like four different lines
_RANKTAG = re.compile(r"^\([^)]*pid=\d+[^)]*\)\s*")
# a line out of verl's hydra config dump, which is data and not an event.
# They reach the loud filter because the config contains the strings
# "error" and "warn".
_CONFIG = re.compile(r"^'[^']*':")
_TS = re.compile(r"^\[?\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d[.,\d]*\]?\s*")
# every digit in a line goes into the dedupe key as one character, so the
# same warning from two vLLM servers at 22:36:08 and 22:36:10, or from two
# ranks at two pids, is one warning
_DIGITS = re.compile(r"\d+")
# Deliberately wide and case-insensitive. The console shows the FIRST
# copy of anything that looks like trouble and drops only the per-rank
# repeats, because a filter that hides a warning it did not recognise is
# worse than one that shows a line too many. Everything, matched or not,
# is in the raw log.
# `fail` covers failed and failure, which this pattern used to miss while
# matching `fatal`. A line that says something failed is the plainest
# form of trouble there is.
_LOUD = re.compile(
    r"warn|error|exception|traceback|fatal|critical|out of memory|fail"
    r"|core dumped|killed|abort|deprecat|only supports|does not support"
    r"|not supported|falling back", re.IGNORECASE)
_FATAL = re.compile(
    r"Traceback \(most recent call last\)|^\s*\S*(Error|Exception)[:(]"
    r"|\bERROR\b|\bCRITICAL\b|out of memory|core dumped")

# The verl keys the step line is worth reading for, mapped onto this
# repo's shared field names. It comes from bedrock_rl/reporting.py, which
# already had to know both names for every field in order to log to
# W&B, so this is a view of that table and not a second one. Everything
# else in the step line still reaches the metrics file and W&B under
# verl's own key.
_PICK = ui_mod.verl_keys()


def parse_step(line):
    """verl's `step:N - k:v - k:v` into (step, {k: v})."""
    m = _STEP.match(line.strip())
    if not m:
        return None, None
    out = {}
    for part in m.group(2).split(" - "):
        k, _, v = part.rpartition(":")
        if not k:
            continue
        try:
            value = v.strip()
            wrapped = _NUMPY_SCALAR.fullmatch(value)
            out[k.strip()] = float(wrapped.group(1) if wrapped else value)
        except ValueError:
            continue
    return int(m.group(1)), out


def display_metrics(raw):
    """verl's keys plus this repo's four short names for the ones the
    step line shows, so an SDPO line and a verl line read the same."""
    out = dict(raw)
    for src, dst in _PICK.items():
        if src in raw:
            out[dst] = raw[src]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw-log", default=None)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--config", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = {}
    for kv in args.config:
        k, _, v = kv.partition("=")
        cfg[k] = v
    verbose = args.verbose or os.environ.get(
        "BRL_VERL_VERBOSE", "").lower() in ("1", "true", "yes")

    raw = open(args.raw_log, "a") if args.raw_log else None
    if raw:
        cfg["verl log"] = args.raw_log
    # This process outlives Ray's TaskRunner and is the one reliable owner for
    # network-backed reporting. The launcher keeps wandb out of verl's backend
    # list so an enabled run still creates exactly one W&B client.
    ui = ui_mod.Reporter(trainer="verl", run=args.run, total=args.steps,
                         metrics=args.metrics, config=cfg)
    ui.start()
    seen = set()
    failed = False
    last = 0
    # bedrock-rl's own diagnostics: how many arrived, how many said
    # something had gone wrong, and how many were repeats of a line
    # already on screen. The counts go in the summary because one warning
    # and the same warning from every rollout worker mean different things.
    diag = diag_bad = diag_repeat = 0
    try:
        # readline rather than iteration, so a step line reaches the
        # console when verl prints it and not when python's read-ahead
        # buffer happens to fill
        for line in iter(sys.stdin.readline, ""):
            if raw:
                raw.write(line)
                raw.flush()
            # The console never sees ray's colour codes, and the per-rank
            # prefix goes too, because it is what makes one warning from
            # four ranks look like four warnings. The raw log above has
            # both, byte for byte.
            text = _ANSI.sub("", line.rstrip("\n"))
            plain = _TS.sub("", _RANKTAG.sub("", text.strip())).strip()
            # Parse the STRIPPED line. verl's console logger runs inside
            # the TaskRunner ray actor rather than on the driver, so its
            # step line arrives tagged `(TaskRunner pid=NNNN) step:1 - ...`
            # and a matcher anchored on `step:` never sees one.
            step, metrics = parse_step(plain)
            if step is not None:
                last = step
                ui.step(step, display_metrics(metrics))
                continue
            # OURS, decided before any filtering. A pattern that says
            # what looks like trouble is a guess about somebody else's
            # output, and this repo's own diagnostics are not a guess.
            mine = ui_mod.is_diagnostic(plain)
            bad = mine and bool(ui_mod.DIAGNOSTIC_TROUBLE.search(plain))
            diag += mine
            diag_bad += bad
            if verbose:
                ui.line(text)
                continue
            if mine:
                key = _DIGITS.sub("#", plain)[:ui_mod.WARNING_PREFIX]
                if key in seen:
                    diag_repeat += 1
                    continue
                seen.add(key)
                (ui.error if bad else ui.line)(plain)
                continue
            if not plain or _CONFIG.match(plain) or not _LOUD.search(plain):
                continue
            # Same opening, same warning. transformers writes the
            # submodule's class name into the middle of its
            # flash-attention warning, so the four copies one model load
            # produces differ only after the first hundred characters,
            # and vLLM stamps a clock reading into the front of its own.
            key = _DIGITS.sub("#", plain)[:ui_mod.WARNING_PREFIX]
            if key in seen:
                continue
            seen.add(key)
            if _FATAL.search(plain):
                failed = True
                ui.error(plain)
            else:
                ui.warn(plain)
    except KeyboardInterrupt:                       # pragma: no cover
        failed = True
    finally:
        if raw:
            raw.close()
        extra = {}
        if cfg.get("out"):
            extra["checkpoint"] = cfg["out"]
        if diag:
            extra["bedrock-rl"] = (
                f"{diag} diagnostic line{'' if diag == 1 else 's'}"
                + (f", {diag_bad} reporting a failure" if diag_bad else "")
                + (f", {diag_repeat} repeat{'' if diag_repeat == 1 else 's'}"
                   " of one already shown" if diag_repeat else ""))
        ui.finish(extra, failed=failed or last == 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
