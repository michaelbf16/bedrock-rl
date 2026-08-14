"""The non-python files this package reads.

Four modules read data that is not python: the `computer` tool schema
(adapters/netherite/chat.py), the per-family chat templates (models.py), the
DeepSpeed presets (train/sft.py), and the engine pin plus its patch set
(env/engine.py).

A path resolved as `__file__/../..` is right in a checkout and wrong in
an installed wheel, where the parent of the package is site-packages and
none of those trees exist. That failure is easy to miss, because `uv
sync` installs this project editable, so the checkout layout is the only
one normally exercised.

The first three are package data under `bedrock_rl/templates/`, so they
resolve beside `__file__` in both layouts. `template_path()` is the one
way in.

`patches/` cannot move, because it is the engine build input read by
bin/setup_deps.sh. The wheel therefore ships a copy, force-included under
`_data/` (see pyproject.toml), and `data_path()` is the one place that knows
it can be in either layout. The checkout wins when both exist, so editing a
patch takes effect.

An installed wheel provides the Python framework and can validate user-owned
task, run, data, and training configs. The source launchers and runnable
examples are checkout assets: setup, actual training, evaluation, rendering,
and bare example-name discovery therefore still require a clone. Keeping that
boundary explicit avoids writing outputs into site-packages or pretending the
separate C engine is wheel data.
"""
from pathlib import Path

_PKG = Path(__file__).resolve().parent
# a checkout: the trees the package does not own sit beside it, at the
# repo root
_CHECKOUT = _PKG.parent
# a wheel: the same trees, force-included under the package
_PACKAGED = _PKG / "_data"


def repo_root():
    """The checkout this package was imported from, or None when it was
    installed from a wheel. Commands which execute `bin/` or discover the
    included examples need this; explicit user config paths do not."""
    return _CHECKOUT if (_CHECKOUT / "pyproject.toml").exists() else None


def template_path(*parts):
    """A path under `bedrock_rl/templates/`: the tool schema, the chat
    templates, the DeepSpeed presets.

    Package data, so this is the same path in a checkout and in an
    installed wheel and there is no layout to choose between.
    """
    return _PKG.joinpath("templates", *parts)


# ── finding an example task by name ─────────────────────────────────────
# Complete examples own their task beside their generation and training
# manifests. Generated task instances are output data, not package source
# silently added to this lookup path.
TASK_DIRS = ("examples",)


def task_dirs():
    """Where a bare example name is looked up, in order, absolute.

    A wheel has no examples checkout, so it returns no directories.
    """
    root = repo_root()
    if root is None:
        return ()
    return tuple(root.joinpath(*d.split("/")) for d in TASK_DIRS)


def find_task(name):
    """A bare example directory name -> its task file, or None.

    A name only. Anything path-shaped is the caller's to resolve, because
    a path means what it says relative to where the user typed it and
    this function has no way to know where that was.
    """
    for directory in task_dirs():
        for filename in ("task.yaml", "task.yml"):
            path = directory / name / filename
            if path.is_file():
                return str(path)
    return None


def task_files():
    """Every task manifest owned by an included example."""
    out = []
    for directory in task_dirs():
        out += sorted(str(path) for path in directory.rglob("task.yaml"))
        out += sorted(str(path) for path in directory.rglob("task.yml"))
    return out


def data_path(*parts):
    """A path under the packaged or checkout-owned data trees.

    Returns the checkout path when neither exists, so the caller's own
    "no such file" message names the place a developer would look.
    """
    for base in (_CHECKOUT, _PACKAGED):
        p = base.joinpath(*parts)
        if p.exists():
            return p
    return _CHECKOUT.joinpath(*parts)
