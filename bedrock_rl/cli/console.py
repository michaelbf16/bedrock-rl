"""One console, one theme, for everything this repo prints.

Built in one place so two halves of a run cannot end up with different
colours for the same meaning. Anything that prints for a person to read
should import from here rather than build its own.

Colour, borders and table padding switch off together when stdout is not
a terminal, because rich escape codes make a tee'd log unreadable and
every long run in this repo is tee'd. BEDROCK_PLAIN=1 forces plain
output and NO_COLOR is honoured.

NOTHING PRINTED HERE IS EVER SHORTENED. rich lays a table column out to
fit and its default for a cell that does not fit is an ellipsis, so a
checkpoint path in a narrow terminal came out as `/home/u/ckpts/run...`,
which is a wrong answer given confidently. Half of what this module
prints is a path, a command line or an environment assignment, and those
are values a person copies or a script parses. So `table` and `grid`
below fold rather than ellipsise, and `assign`, `command` and `paste`
soft wrap, which leaves the value on ONE line and lets the terminal do
the folding. A caller does not have to remember any of that.

BEDROCK_PLAIN is bedrock_rl/reporting.py's knob too, parsed by its
env_flag, so the CLI and the trainer reporter cannot disagree about what
a plain console is or about which spelling of no means no.
"""
import atexit
import io
import os
import shlex
import sys

from bedrock_rl import reporting as ui_mod
from rich.box import ASCII, ROUNDED, SIMPLE_HEAD
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Width a tee'd log is laid out at. A terminal decides its own.
LOG_WIDTH = 100

THEME = Theme({
    "brl.ok": "bold green",
    "brl.fail": "bold red",
    "brl.warn": "bold yellow",
    "brl.head": "bold",
    "brl.dim": "dim",
    "brl.cmd": "bold cyan",
    "brl.path": "cyan",
    "brl.val": "magenta",
})

_consoles = {}


class _Trim(io.TextIOBase):
    """Strip trailing spaces off every line on the way out.

    A table pads each cell to its column width, which is invisible on a
    screen and is nine trailing spaces per row in a file. Logs get
    grepped, diffed and pasted into issues, so the padding is stripped
    once, here, rather than by every caller remembering to.
    """

    def __init__(self, wrapped):
        self._w = wrapped
        self._pending = ""

    @property
    def encoding(self):
        return getattr(self._w, "encoding", "utf-8")

    def writable(self):
        return True

    def isatty(self):
        return False

    def fileno(self):
        # pytest's capture and any StringIO have no descriptor, and the
        # documented answer to that is the exception, not an AttributeError
        # from an attribute that is simply not there.
        try:
            return self._w.fileno()
        except AttributeError:
            raise io.UnsupportedOperation("fileno")

    def write(self, s):
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._w.write(line.rstrip() + "\n")
        return len(s)

    def flush(self):
        if self._pending:
            self._w.write(self._pending)
            self._pending = ""
        self._w.flush()


def flush():
    """Empty every console buffer, before a child process writes to the
    same file descriptor."""
    for con in _consoles.values():
        try:
            con.file.flush()
        except (OSError, ValueError):
            # interpreter shutdown can close the descriptor first, and a
            # failure to flush is not worth an exception on the way out
            pass


atexit.register(flush)


def plain():
    """True when output must carry no escape codes.

    A pipe, a file, a tmux log, NO_COLOR, or BEDROCK_PLAIN=1.

    stdout decides for stderr as well. A run whose stdout is redirected
    and whose stderr is not is a run someone is capturing, and plain text
    on both halves is the answer that keeps the capture readable.
    """
    if ui_mod.plain_forced():
        return True
    if os.environ.get("NO_COLOR"):
        return True
    return not sys.stdout.isatty()


def get_console(stderr=False):
    """The console. Cached, because rich measures the terminal on build."""
    key = bool(stderr), plain()
    if key not in _consoles:
        flat = key[1]
        raw = sys.stderr if stderr else sys.stdout
        _consoles[key] = Console(
            file=_Trim(raw) if flat else raw,
            theme=THEME,
            # `no_color` strips styles from a forced-plain terminal too,
            # which `is_terminal` alone would not do.
            no_color=flat,
            # markup OFF. Almost everything printed here is data that
            # came from somewhere else, and square brackets are ordinary
            # in it: `trainer.logger=[console]` is a hydra override this
            # repo's own bin/rl.sh passes, `[rank0]` opens most torch
            # errors. With markup on, rich reads those as style tags and
            # deletes them, so a --dry-run echo showed a command line
            # that was not the one it was about to run. Style is applied
            # by passing Text or a style= argument, never by embedding
            # tags in a string.
            markup=False,
            highlight=False,
            emoji=False,
            width=LOG_WIDTH if flat else None,
        )
    return _consoles[key]


def box():
    """The border set for anything with a frame around it.

    Box-drawing characters survive a tee, but they make a log file
    non-ASCII for decoration alone, so a plain console draws in ASCII.
    """
    return ASCII if plain() else ROUNDED


# rich's own default for a cell wider than its column, and the reason
# this module names one of its own. `ellipsis` puts a … where the rest of
# the value was, and most of what this repo tabulates is a path, an id or
# a command, where the tail is the part that identifies it. `fold` keeps
# every character and costs a second line.
OVERFLOW = "fold"


def table(*columns, title=None, **kw):
    """A table that reads the same in a terminal and in a log file.

    A terminal gets one rule under the header, which is enough separation
    when the columns are aligned. A log gets full borders, because a table
    pasted into an issue has nothing around it to sit inside.

    Every column folds unless the caller says otherwise, so no cell can
    be silently replaced by a prefix of itself.
    """
    flat = plain()
    t = Table(title=title, box=ASCII if flat else SIMPLE_HEAD,
              title_style="brl.head", header_style="brl.head",
              pad_edge=False, expand=False, **kw)
    for c in columns:
        spec = dict(c) if isinstance(c, dict) else {"header": c}
        spec.setdefault("overflow", OVERFLOW)
        t.add_column(**spec)
    return t


def grid(*column_styles, indent=2, gap=2):
    """A borderless listing, indented, for help screens.

    A bordered table around a list of command names is decoration, and
    decoration is the thing this console does not do.
    """
    t = Table.grid(padding=(0, gap))
    t.pad_edge = False
    for i, style in enumerate(column_styles):
        t.add_column(style=style, no_wrap=(i == 0), overflow=OVERFLOW)
    t.indent = indent
    return t


def print_grid(t):
    """Print an indented grid with no trailing padding.

    expand=False matters: a Padding that fills the console pads every row
    out to the terminal width, and a log full of trailing spaces is a log
    that every diff and every editor flags.
    """
    from rich.padding import Padding
    get_console().print(Padding(t, (0, 0, 0, getattr(t, "indent", 2)),
                                expand=False))


# ── the three verdicts ───────────────────────────────────────────────────
# One spelling each, so a check that passed looks the same in `doctor` as
# in `models check`. The word is carried as well as the colour, because
# half the places this prints have no colour at all.

def mark(state):
    """A pass/fail/warn cell."""
    return {"ok": Text("PASS", style="brl.ok"),
            "fail": Text("FAIL", style="brl.fail"),
            "warn": Text("WARN", style="brl.warn"),
            "skip": Text("SKIP", style="brl.dim")}[state]


def say(msg, style=None):
    get_console().print(msg, style=style)


def paste(msg, style="brl.cmd", indent=2):
    """A line meant to be copied into a shell.

    soft_wrap, so the terminal folds it and the paste buffer still holds
    one command. rich's own wrapping puts a real newline in the middle,
    and the shell then reads two commands, the second of which is a flag.
    """
    get_console().print(Text(" " * indent + str(msg), style=style),
                        soft_wrap=True)


def assign(key, value):
    """Echo one `KEY=value` environment assignment.

    On ONE line. This is the half of a --dry-run that a reader cannot see
    in the command line, and it exists so somebody can check what a child
    process will get before spending GPUs on it. Folded across two lines
    by a width-aware layout it is neither copyable nor parseable, and the
    value that gets folded is always the long one, which is always the
    path.
    """
    get_console().print(Text(f"{key}=", style="brl.dim")
                        + Text(str(value), style="brl.val"),
                        soft_wrap=True)


def warn(msg):
    """A warning, on stderr, in the same shape as `error`.

    Separate from `error` because it does not stop anything. It is for
    the facts a --dry-run cannot show by echoing assignments, the shell's
    own environment chief among them, and a run that goes ahead is the
    right behaviour for every one of them.
    """
    con = get_console(stderr=True)
    con.print(Text("warning", style="brl.warn"), Text(str(msg)))


def error(msg, hint=None):
    """An error, on stderr, in the one shape every command uses."""
    con = get_console(stderr=True)
    con.print(Text("error", style="brl.fail"), Text(str(msg)))
    if hint:
        for line in str(hint).splitlines():
            con.print("  " + line, style="brl.dim")


def command(argv):
    """Echo a command line the way a user could paste it back.

    shlex.join, because an argument with a space in it is one argument
    here and two in the shell, and a rendered task path can hold one.
    soft_wrap, because a rewrapped command line carries a real newline
    into the paste buffer and the shell reads that as two commands.
    """
    get_console().print(Text("+ ", style="brl.dim")
                        + Text(shlex.join(str(x) for x in argv),
                               style="brl.cmd"),
                        soft_wrap=True)
