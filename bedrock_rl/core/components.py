"""Import-path components: the extension mechanism shared by the core."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping


def load_object(reference: str) -> Any:
    """Load ``package.module:attribute`` without maintaining a registry."""
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError(
            f"component reference must be 'package.module:attribute', got "
            f"{reference!r}")
    module_name, attribute = reference.split(":", 1)
    if not module_name or not attribute or ":" in attribute:
        raise ValueError(
            f"component reference must be 'package.module:attribute', got "
            f"{reference!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"cannot import component module {module_name!r} "
                          f"from {reference!r}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"component module {module_name!r} has no "
                          f"attribute {attribute!r}") from exc


@dataclass(frozen=True)
class ComponentSpec:
    """An import target and constructor configuration.

    A string is shorthand for a component with no configuration.  A mapping
    uses ``type`` for the import path; every other key is constructor input.
    ``config`` may be used as an explicit nested mapping, but mixing it with
    additional constructor keys is rejected so the effective configuration
    cannot be ambiguous.
    """

    type: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: str | Mapping[str, Any], *, what: str = "component"):
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise TypeError(f"{what} must be an import path or mapping, got "
                            f"{type(value).__name__}")
        raw = dict(value)
        target = raw.pop("type", None)
        if not target:
            raise ValueError(f"{what} mapping needs a 'type' import path")
        nested = raw.pop("config", None)
        if nested is not None:
            if raw:
                raise ValueError(
                    f"{what} puts constructor values both beside 'type' and "
                    "inside 'config'; use one form")
            if not isinstance(nested, Mapping):
                raise TypeError(f"{what}.config must be a mapping")
            raw = dict(nested)
        return cls(str(target), raw)

    def resolve(self) -> Any:
        return load_object(self.type)

    def create(self, **injected: Any) -> Any:
        """Instantiate the target, with explicit injected values winning."""
        target = self.resolve()
        if not callable(target):
            if self.config or injected:
                raise TypeError(f"component {self.type!r} is not callable but "
                                "was given constructor configuration")
            return target
        values = {**self.config, **injected}
        try:
            return target(**values)
        except TypeError as exc:
            name = getattr(target, "__qualname__", repr(target))
            raise TypeError(f"cannot construct {name} from component "
                            f"{self.type!r}: {exc}") from exc


def object_reference(value: Any) -> str:
    """Best-effort stable name for provenance and transform records."""
    if inspect.isfunction(value) or inspect.isclass(value):
        return f"{value.__module__}:{value.__qualname__}"
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def resolve_callable(value: Any, *, what: str = "callable"):
    """Resolve a callable object, import path, or configured component."""
    if callable(value):
        return value
    if isinstance(value, (str, Mapping)):
        spec = ComponentSpec.parse(value, what=what)
        value = spec.create() if spec.config else spec.resolve()
    if not callable(value):
        raise TypeError(f"{what} must resolve to a callable, got "
                        f"{type(value).__name__}")
    return value
