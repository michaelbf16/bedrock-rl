"""Shared console, JSONL metrics, and optional Weights & Biases reporting.

``Reporter`` gives every trainer one progress interface while preserving each
trainer's native metric keys in the JSONL artifact. Interactive terminals use
Rich; redirected output uses stable line-oriented text. Set
``BEDROCK_PLAIN=1`` to force plain output. Weights & Biases is enabled only when
requested and uses verl-compatible names for metrics shared across trainers.
"""
import json
import logging
import numbers
import os
import re
import sys
import time

try:                                                # pragma: no cover
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (BarColumn, Progress, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    from rich.table import Table
    from rich.text import Text
    _RICH = True
except ImportError:                                 # pragma: no cover
    _RICH = False

WANDB_PROJECT = "bedrock-rl"

# ── the shared field vocabulary ──────────────────────────────────────────
# label and formatting for every metric any trainer shows, so the same
# quantity is called the same thing and printed the same width whichever
# trainer produced it. A key with no entry still logs and still lands in
# the metrics file; it just does not get a column.
_FIELDS = {
    "reward":          ("reward", "signed"),
    "success":         ("succ", "pct"),
    "stopped":         ("stop", "pct"),
    "truncated":       ("trunc", "pct"),
    "turns":           ("turns", "float2"),
    "loss":            ("loss", "float4"),
    "grad_norm":       ("gnorm", "float3"),
    "lr":              ("lr", "sci"),
    "epoch":           ("epoch", "float2"),
    "sec_total":       ("time", "secs"),
    "sec_rollout":     ("roll", "secs"),
    "seen":            ("seen", "int"),
    "kept":            ("kept", "int"),
    "yield":           ("yield", "pct"),
    "n":               ("n", "int"),
    "sdpo/distilled":  ("distil", "int"),
}

_WIDTH = {"signed": 7, "pct": 6, "float4": 8, "float3": 7, "float2": 6,
          "sci": 8, "secs": 7, "int": 5}

# Use verl's names for shared metrics and Bedrock-specific namespaces for the
# rest. Unlisted fields keep their original key.
_WANDB_NAMES = {
    "reward": "critic/score/mean",
    "success": "bedrock/success_rate",
    "stopped": "bedrock/voluntary_stop_rate",
    "truncated": "bedrock/truncated_rate",
    "turns": "bedrock/turns_mean",
    # This matches the key emitted by verl and consumed by its console adapter.
    "loss": "actor/pg_loss",
    "grad_norm": "actor/grad_norm",
    "lr": "actor/lr",
    "epoch": "training/epoch",
    "sec_total": "perf/time_per_step",
    "sec_rollout": "perf/time_rollout",
    "seen": "harvest/episodes_seen",
    "kept": "harvest/rows_kept",
    "yield": "harvest/yield",
}

# The columns each trainer shows, from the vocabulary above. Small and
# nearly the same everywhere on purpose: reward, success, gradient norm
# and step time are what a person watching a run actually reads.
STEP_FIELDS = {
    "sdpo": ("reward", "success", "loss", "grad_norm", "sec_total"),
    "sft": ("loss", "grad_norm", "lr", "epoch"),
    "sdft": ("reward", "success", "loss", "grad_norm", "sec_total"),
    "self_distill": ("seen", "kept", "yield", "sec_total"),
    "verl": ("reward", "loss", "grad_norm", "sec_total"),
}


def verl_keys():
    """Map verl metric keys to the shared fields shown in progress output."""
    return {_WANDB_NAMES[f]: f for f in STEP_FIELDS["verl"]}


# ── package diagnostics ──────────────────────────────────────────────────
# Package diagnostics use ``[tag] text`` on stderr. The verl console adapter
# always retains matching lines, even when it filters verbose worker output.
# Keep tags lowercase and route new diagnostics through this module. Requiring
# a space after the tag excludes similarly shaped third-party messages.
DIAGNOSTIC = re.compile(r"^\[[a-z][a-z0-9_ ]*\]\s+\S")
# Which of them is bad news, for whether it prints as an error or a note.
DIAGNOSTIC_TROUBLE = re.compile(
    r"fail|error|exception|traceback|could not|cannot|unable|refus",
    re.IGNORECASE)


def is_diagnostic(line):
    """Return whether ``line`` is a Bedrock RL ``[tag] ...`` diagnostic."""
    return bool(DIAGNOSTIC.match(line))


# ── nothing printed here is shortened ────────────────────────────────────
# The run header and the finish summary are label/value rows, and the
# values are the run slug, the resolved model path, the metrics file and
# the checkpoint directory. rich lays a grid out to fit the console and
# its default for a value that does not is an ELLIPSIS, so the one line
# that says where the checkpoint went came out as a prefix of the answer
# on any terminal narrower than the path. Folding costs a second line and
# keeps every character, which is the trade a record has to make.

def _rows_grid():
    """The label/value grid both panels use, folding rather than cutting."""
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", justify="right", overflow="fold")
    t.add_column(overflow="fold")
    return t


# ── boolean environment knobs ────────────────────────────────────────────
# Accept common explicit boolean spellings and reject ambiguous values.
TRUE = ("1", "true", "yes", "on")
FALSE = ("", "0", "false", "no", "off")


def env_flag(name, default=False):
    """A boolean environment variable, or `default` when it is unset."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise SystemExit(
        f"{name}={raw!r} is not a yes or a no.\n"
        f"  yes: {', '.join(t for t in TRUE if t)}\n"
        f"  no:  {', '.join(f for f in FALSE if f)}, or unset")


PLAIN_ENV = "BEDROCK_PLAIN"


def plain_forced():
    """True when output must carry no escape codes whatever stdout is."""
    if env_flag(PLAIN_ENV):
        return True
    if os.environ.get("TERM", "") == "dumb":
        return True
    return False


# the name the rest of this module has always called it
_plain_forced = plain_forced


def _is_tty(stream):
    if _plain_forced():
        return False
    if os.environ.get("FORCE_COLOR", "").lower() in ("1", "true", "yes"):
        return True
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _secs(v):
    v = float(v)
    if v < 60:
        return f"{v:.1f}s"
    m, s = divmod(int(v), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _render(kind, v):
    if v is None:
        return "-"
    if isinstance(v, bool) or not isinstance(v, numbers.Number):
        return str(v)
    v = float(v)
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "signed":
        return f"{v:+.3f}" if abs(v) < 1000 else f"{v:+.3g}"
    if kind == "float4":
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.3g}"
    if kind == "float3":
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.3g}"
    if kind == "float2":
        return f"{v:.2f}"
    if kind == "sci":
        return f"{v:.2e}"
    if kind == "secs":
        return _secs(v)
    return f"{int(v)}"


def _cells(fields, metrics):
    """(label, value) per column, skipping columns this record has no
    number for, so a trainer that reports no reward does not print an
    empty reward column."""
    out = []
    for key in fields:
        if key not in metrics:
            continue
        label, kind = _FIELDS.get(key, (key, "float3"))
        out.append((label, _render(kind, metrics[key]), _WIDTH.get(kind, 8)))
    return out


WARNING_PREFIX = 80


def dedupe_warnings(*logger_names, prefix=WARNING_PREFIX):
    """Collapse repeats of a warning a library emits once per submodule.

    Loading a vision-language model logs the flash-attention dtype
    warning four times, once for Qwen3VLForConditionalGeneration and once
    each for the model, the vision tower and the text tower, and a
    self-distillation run loads twice. It is one fact about the run.

    Two warnings count as the same when their first `prefix` characters
    match, not when the whole string matches, because transformers
    formats the submodule's class name INTO the middle of that sentence:
    the four copies are four distinct strings saying one thing, and the
    part they share is the first 97 characters. Nothing else is silenced.
    The first copy of every distinct warning still prints, and only
    later copies of that same opening are dropped.

    The filter goes on the library's HANDLER, not on its logger, because
    a record from transformers.modeling_utils propagates to the
    transformers handler without ever passing the transformers logger's
    own filters.
    """
    seen = set()

    def _once(record):
        if record.levelno < logging.WARNING:
            return True
        key = (record.name, record.getMessage()[:prefix])
        if key in seen:
            return False
        seen.add(key)
        return True

    for name in logger_names:
        lg = logging.getLogger(name)
        lg.addFilter(_once)
        for h in lg.handlers:
            h.addFilter(_once)


def quiet_progress_bars():
    """Turn off the huggingface tqdm bars.

    They are carriage returns and backspaces, which is a page of noise in
    a tee'd log, and on a terminal they fight the reporter's own live
    progress for the bottom line. Neither library is a dependency of this
    package, so both imports are guarded.
    """
    try:
        import datasets
        datasets.disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def quiet_transformers_repeats():
    """dedupe_warnings on the transformers library logger.

    The import is here and not at module scope because this file is part
    of the CPU package, which does not carry transformers; only the
    trainers call it.
    """
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.get_logger()          # forces the library handler to exist
    except Exception:                    # pragma: no cover
        return
    dedupe_warnings("transformers")


# ── wandb reporting backend, opt-in ─────────────────────────────────────
def wandb_requested():
    return os.environ.get("WANDB", "").lower() in ("1", "true", "yes", "on")


def _has_wandb_key():
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE", "") in ("offline", "dryrun", "disabled"):
        return True
    try:
        import netrc
        return bool(netrc.netrc().authenticators("api.wandb.ai"))
    except Exception:
        return False


class _Wandb:
    """A wandb run, or nothing at all. Never raises into the trainer."""

    def __init__(self, run, config, report):
        self.run = None
        self.url = None
        # wandb.init prints seven lines about itself and finish() prints a
        # summary table. The one fact worth having is the run URL, which
        # the reporter prints in the header block. Set it explicitly to
        # keep the client's banner off a console this module is trying to
        # make readable; WANDB_SILENT=false in the environment wins.
        os.environ.setdefault("WANDB_SILENT", "true")
        try:
            import wandb
            # wandb writes its run data to ./wandb, and a bare directory
            # is importable as a namespace package, so `import wandb`
            # from the repo root succeeds even with nothing installed.
            # Ask for the function instead of trusting the name.
            assert hasattr(wandb, "init")
        except (ImportError, AssertionError):
            report("wandb is on but the package is not importable; "
                   "logging to the metrics file only "
                   "(pip install 'bedrock-rl[wandb]')")
            return
        if not _has_wandb_key():
            report("wandb is on but no API key was found in WANDB_API_KEY "
                   "or ~/.netrc; logging to the metrics file only "
                   "(wandb login, or WANDB_MODE=offline)")
            return
        try:
            self.run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=run, config=config)
            self.url = getattr(self.run, "url", None)
        except Exception as exc:                    # pragma: no cover
            self.run = None
            report(f"wandb did not start ({type(exc).__name__}: {exc}); "
                   "logging to the metrics file only")

    def log(self, metrics, step):
        if self.run is None:
            return
        out = {}
        for k, v in metrics.items():
            if k == "step" or not isinstance(v, numbers.Number):
                continue
            out[_WANDB_NAMES.get(k, k)] = v
        try:
            self.run.log(out, step=step)
        except Exception:                           # pragma: no cover
            pass

    def finish(self):
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:                       # pragma: no cover
                pass


# ── the reporter ─────────────────────────────────────────────────────────
class Reporter:
    """Header, one line per step, progress, summary, and the metrics file.

    `total` may be None or unknown at construction; set_total() fills it
    in once the trainer knows, which is what SFT needs because
    transformers decides the step count from the epoch count.
    """

    def __init__(self, trainer, run, total=None, metrics=None, config=None,
                 fields=None, stream=None, wandb=None):
        self.trainer = trainer
        self.run = run
        self.total = total if (total or 0) > 0 else None
        self.config = dict(config or {})
        self.metrics_path = metrics
        self.fields = tuple(fields or STEP_FIELDS.get(trainer, ()))
        self.stream = stream or sys.stdout
        # A torchrun or DeepSpeed launch runs this file once per rank, and
        # every rank printing the same step line is the per-rank
        # duplication this module exists to remove. Rank zero reports;
        # the rest are inert, down to not opening the metrics file.
        self.enabled = int(os.environ.get("RANK", "0")) == 0
        self.rich = self.enabled and _RICH and _is_tty(self.stream)
        self.console = Console(file=self.stream, highlight=False,
                               soft_wrap=True) if self.rich else None
        self._fh = None
        self._progress = None
        self._task = None
        self._t0 = time.time()
        self._steps_done = 0
        self._last = {}
        self._wandb_on = wandb_requested() if wandb is None else bool(wandb)
        self._wandb = None
        self._done = False

    # -- lifecycle -------------------------------------------------------
    def __enter__(self):
        return self.start()

    def __exit__(self, _exc_type, exc, _tb):
        self.finish(failed=exc is not None)
        return False

    def start(self):
        if not self.enabled:
            return self
        if self.metrics_path:
            d = os.path.dirname(os.path.abspath(self.metrics_path))
            if d:
                os.makedirs(d, exist_ok=True)
            self._fh = open(self.metrics_path, "a")
            self.config = dict(self.config, metrics=self.metrics_path)
        quiet_progress_bars()
        self._t0 = time.time()
        self._header()
        if self._wandb_on:
            self._wandb = _Wandb(self.run, dict(self.config,
                                                trainer=self.trainer),
                                 self.warn)
            if self._wandb.url:
                self.note(f"wandb {self._wandb.url}")
        self._start_progress()
        return self

    def _start_progress(self):
        if self._progress is not None or not (self.rich and self.total):
            return
        self._progress = Progress(
            TextColumn("[bold]{task.description}"), BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), TextColumn("elapsed,"),
            TimeRemainingColumn(), TextColumn("left"),
            console=self.console, transient=True)
        self._task = self._progress.add_task(self.trainer, total=self.total,
                                             completed=self._steps_done)
        self._progress.start()

    def set_total(self, total):
        """The step count, once the trainer knows it. transformers works
        it out from the epoch count and the dataset size, so the reporter
        has no total until training begins."""
        self.total = total if (total or 0) > 0 else None
        if self._progress is not None and self.total:
            self._progress.update(self._task, total=self.total)
        else:
            self._start_progress()

    # -- output ----------------------------------------------------------
    def _write(self, text, style=None):
        if not self.enabled:
            return
        if self.rich:
            target = self._progress.console if self._progress else self.console
            # Text, never a bare string. rich reads square brackets in a
            # string as markup, and this console prints model paths and
            # verl worker lines, which are full of them: the attention
            # kernel in "-> Qwen/Qwen3-VL-2B-Instruct [flash_attention_2]"
            # was being parsed as a style tag and dropped.
            target.print(text if isinstance(text, Text)
                         else Text(str(text), style or ""))
        else:
            print(text, file=self.stream, flush=True)

    def _header(self):
        if not self.enabled:
            return
        rows = [("run", self.run), ("trainer", self.trainer)]
        rows += [(k, v) for k, v in self.config.items()]
        if self.total:
            rows.append(("steps", str(self.total)))
        if self.rich:
            t = _rows_grid()
            for k, v in rows:
                t.add_row(k, Text(str(v)))
            self.console.print(Panel(t, title="bedrock-rl",
                                     title_align="left", border_style="cyan",
                                     expand=False))
        else:
            w = max(len(k) for k, _ in rows)
            print(f"bedrock-rl {self.trainer}", file=self.stream)
            for k, v in rows:
                print(f"  {k:<{w}}  {v}", file=self.stream)
            print("", file=self.stream, flush=True)

    def _progress_text(self, step):
        el = time.time() - self._t0
        if not self.total:
            return f"[{_secs(el)} elapsed]"
        left = (el / max(step, 1)) * max(self.total - step, 0)
        return f"[{_secs(el)} elapsed, {_secs(left)} left]"

    def step(self, step, metrics):
        """One training step. Writes the record, prints the line, logs it."""
        record = dict(metrics)
        record["step"] = step
        self.record(record, step=step)
        self._steps_done = step
        self._last = record
        cells = _cells(self.fields, record)
        n = f"{step}/{self.total}" if self.total else str(step)
        if self.rich:
            line = Text()
            line.append(f"step {n:>7}  ", style="bold")
            for label, value, w in cells:
                line.append(f"{label} ", style="dim")
                line.append(f"{value:>{w}}  ")
            line.append(self._progress_text(step), style="dim")
            self._write(line)
            if self._progress is not None:
                self._progress.update(self._task, completed=step)
        else:
            body = "  ".join(f"{label} {value:>{w}}"
                             for label, value, w in cells)
            self._write(f"step {n:>7}  {body}  {self._progress_text(step)}")

    def record(self, metrics, step=None):
        """A row for the metrics file and wandb with no step line, which
        is what the periodic evaluations are."""
        if self._fh:
            self._fh.write(json.dumps(metrics, default=str) + "\n")
            self._fh.flush()
        if self._wandb:
            self._wandb.log(metrics, step if step is not None
                            else self._steps_done)

    def line(self, text):
        """Pass a line of someone else's output through unchanged, above
        the progress bar rather than through it."""
        self._write(text)

    def note(self, text):
        self._write(f"  {text}", style="dim")

    def warn(self, text):
        if self.rich:
            self._write(Text.assemble(("warning  ", "bold yellow"),
                                      (str(text), "yellow")))
        else:
            self._write(f"warning  {text}")

    def error(self, text):
        if self.rich:
            self._write(Text.assemble(("error  ", "bold red"),
                                      (str(text), "red")))
        else:
            self._write(f"error  {text}")

    def summary(self, extra=None, title="finished", failed=False):
        if not self.enabled:
            return
        rows = [("steps", f"{self._steps_done}"
                          + (f" of {self.total}" if self.total else "")),
                ("wall", _secs(time.time() - self._t0))]
        for key in self.fields:
            if key in self._last:
                label, kind = _FIELDS.get(key, (key, "float3"))
                rows.append((label, _render(kind, self._last[key])))
        # A caller that repeats a field the step line already carries gets
        # one row, not two.
        have = {k for k, _ in rows}
        for k, v in (extra or {}).items():
            if k not in have:
                rows.append((k, str(v)))
                have.add(k)
        if self.metrics_path:
            rows.append(("metrics", self.metrics_path))
        if self._wandb and self._wandb.url:
            rows.append(("wandb", self._wandb.url))
        if self.rich:
            t = _rows_grid()
            for k, v in rows:
                t.add_row(k, Text(str(v)))
            self.console.print(Panel(
                t, title=title, title_align="left",
                border_style="red" if failed else "green", expand=False))
        else:
            w = max(len(k) for k, _ in rows)
            print(f"\n{title}", file=self.stream)
            for k, v in rows:
                print(f"  {k:<{w}}  {v}", file=self.stream)
            print("", file=self.stream, flush=True)

    def finish(self, extra=None, failed=False, summary=True):
        # Trainers call this with the checkpoint path, and the context
        # manager calls it again on the way out. The second call is a
        # no-op, so a `with` block costs nothing and a bare call still
        # closes the file.
        if self._done:
            return
        self._done = True
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        if summary:
            self.summary(extra, title="stopped" if failed else "finished",
                         failed=failed)
        if self._wandb:
            self._wandb.finish()
            self._wandb = None
        if self._fh:
            self._fh.close()
            self._fh = None
