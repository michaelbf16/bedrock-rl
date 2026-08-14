"""Dotted ``key=value`` overrides shared by local and remote generation."""

from __future__ import annotations

import yaml


def parse_value(text):
    return yaml.safe_load(text)


def apply_override(document, expression):
    if "=" not in expression:
        raise ValueError(f"--set wants key=value, got {expression!r}")
    path, text = expression.split("=", 1)
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError(f"--set has no key in {expression!r}")
    target = document
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"--set {path}: {part} is not a mapping")
        target = child
    target[parts[-1]] = parse_value(text)


__all__ = ("apply_override", "parse_value")
