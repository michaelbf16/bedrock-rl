"""Task and model path resolution shared by commands and launchers."""

from __future__ import annotations

import os

from bedrock_rl import resources


DEFAULT_TASK = "equip_pickaxe_grpo"
DEFAULT_MODEL = "qwen3-vl"
TASK_HELP = "example name, or a path to a task YAML"


def absolute_path(path):
    return None if path is None else os.path.abspath(
        os.path.expanduser(str(path))
    )


def config_relative(path, base_dir):
    """Resolve a non-empty path written inside a config beside that config."""
    if path in (None, ""):
        return path
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def task_yaml(name, base_dir=None):
    """Resolve an example name or explicit task file to an existing path."""
    name = str(name)
    if (
        name.endswith((".yaml", ".yml"))
        or os.sep in name
        or name.startswith("~")
    ):
        path = (
            config_relative(name, base_dir)
            if base_dir is not None
            else absolute_path(name)
        )
        if not os.path.isfile(path):
            raise SystemExit(f"no task file at {path}\n  bedrock task list")
        return path
    found = resources.find_task(name)
    if found:
        return found
    where = resources.task_dirs()
    searched = (
        " or ".join(str(directory) for directory in where)
        if where
        else "any task directory this install carries"
    )
    raise SystemExit(
        f"no example named {name!r} under {searched}\n  bedrock task list"
    )


def model_arg(model, base_dir=None):
    """Resolve a model path while leaving registry and Hugging Face ids intact."""
    from bedrock_rl import models

    resolver = (
        (lambda value: config_relative(value, base_dir))
        if base_dir is not None
        else absolute_path
    )
    return models.resolve_model_location(model, resolver)


def task_files():
    return resources.task_files()


__all__ = (
    "DEFAULT_MODEL",
    "DEFAULT_TASK",
    "TASK_HELP",
    "absolute_path",
    "config_relative",
    "model_arg",
    "task_files",
    "task_yaml",
)
