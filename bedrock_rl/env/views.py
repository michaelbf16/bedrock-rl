"""Task-selected model views for Netherite episodes.

The simulator state and reward never depend on this module. A view owns only
the pixels shown to a model: the semantic camera, Netherite's deterministic
procedural textures, Mojang textures generated from a client jar the user
owns, or a custom import-path component.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bedrock_rl.core.components import ComponentSpec, object_reference
from bedrock_rl.env.engine import engine_home


ASSET_MANIFEST_NAME = ".bedrock-rl-assets"
BUILTIN_VIEWS = {
    "semantic": "semantic",
    "procedural": "netherite-procedural",
    "netherite-procedural": "netherite-procedural",
    "jar": "minecraft-official",
    "minecraft-official": "minecraft-official",
}
DEFAULT_VIEW = "netherite-procedural"


def read_asset_manifest(netherite_home: str | os.PathLike[str]) -> dict[str, str]:
    """Read the small renderer-asset identity file from an engine checkout.

    Old procedural checkouts are recognized by their existing STUB_ATLASES
    stamp. Old jar builds are deliberately ``unknown``: guessing that any
    unstamped headers are Mojang art would silently certify arbitrary pixels.
    """
    root = Path(netherite_home).expanduser().resolve()
    path = root / ASSET_MANIFEST_NAME
    values: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if sep and key.strip():
                values[key.strip()] = value.strip()
    except OSError:
        stub = root / "magma" / "assets" / "STUB_ATLASES"
        if stub.is_file():
            return {"kind": "procedural", "source": "legacy-STUB_ATLASES"}
        return {"kind": "unknown", "source": "unstamped"}
    values.setdefault("kind", "unknown")
    values.setdefault("source", "manifest")
    return values


def _specific_home(kind: str, explicit: str | None) -> Path:
    if explicit:
        return engine_home(explicit).resolve()
    variable = {
        "procedural": "NETHERITE_HOME_PROCEDURAL",
        "jar": "NETHERITE_HOME_JAR",
    }.get(kind)
    if variable and os.environ.get(variable):
        return engine_home(os.environ[variable]).resolve()
    if kind == "semantic" and not os.environ.get("NETHERITE_HOME"):
        for candidate in ("NETHERITE_HOME_PROCEDURAL", "NETHERITE_HOME_JAR"):
            if os.environ.get(candidate):
                return engine_home(os.environ[candidate]).resolve()
    return engine_home().resolve()


def _png_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    save = getattr(value, "save", None)
    if save is None:
        raise TypeError("view.capture(episode) must return PNG bytes, a PIL "
                        "image, or None")
    output = io.BytesIO()
    save(output, format="PNG")
    return output.getvalue()


class EnginePixelsView:
    """The engine rasterizer using one explicitly identified asset pack."""

    def __init__(self, *, name: str, asset_kind: str,
                 texture_pack: str | None = None,
                 texture_pack_sha256: str | None = None):
        self.name = name
        self.asset_kind = asset_kind
        self.texture_pack = (None if texture_pack is None else
                             Path(str(texture_pack)).name)
        self.texture_pack_sha256 = (None if texture_pack_sha256 is None else
                                    str(texture_pack_sha256).lower())
        if self.texture_pack_sha256 is not None and self.texture_pack is None:
            raise ValueError("texture_pack_sha256 needs texture_pack")

    def resolve_engine_home(self, explicit: str | None = None) -> Path:
        return _specific_home(self.asset_kind, explicit)

    def validate(self, home: Path) -> None:
        actual = read_asset_manifest(home)
        if actual["kind"] != self.asset_kind:
            variables = ("NETHERITE_HOME_PROCEDURAL" if self.asset_kind ==
                         "procedural" else "NETHERITE_HOME_JAR")
            raise RuntimeError(
                f"task view {self.name!r} requires a {self.asset_kind} "
                f"renderer, but {home} is {actual['kind']!r} "
                f"({actual.get('source', 'no source')}). Point {variables} "
                "at a matching built Netherite checkout; pixels are never "
                "silently substituted.")
        if self.asset_kind == "jar":
            built_pack = actual.get("texture_pack")
            if self.texture_pack is None and built_pack:
                raise RuntimeError(
                    f"task requests vanilla minecraft-official pixels, but "
                    f"{home} was built with texture pack {built_pack!r}; "
                    "declare texture_pack in the task or select a vanilla "
                    "NETHERITE_HOME_JAR")
            if self.texture_pack is not None and built_pack != self.texture_pack:
                raise RuntimeError(
                    f"task requires texture pack {self.texture_pack!r}, but "
                    f"{home} was built with {built_pack or 'vanilla assets'!r}")
            if (self.texture_pack_sha256 is not None
                    and actual.get("texture_pack_sha256", "").lower()
                    != self.texture_pack_sha256):
                raise RuntimeError(
                    f"task requires texture pack sha256 "
                    f"{self.texture_pack_sha256}, but {home} records "
                    f"{actual.get('texture_pack_sha256', 'no digest')}")

    def capture(self, episode) -> bytes | None:
        return episode.env.frame()

    def provenance(self, home: Path) -> dict[str, Any]:
        return {"type": self.name, "component": object_reference(self),
                "texture_pack": self.texture_pack,
                "assets": read_asset_manifest(home)}


class SemanticView:
    """Colorized 64x36 block/depth/edge camera with exact GUI fallback.

    Netherite's semantic observation has no inventory/container surface.
    While a GUI is open the default therefore uses the engine frame, so a
    policy can still click real slots. Set ``gui_fallback: false`` only for a
    task that never needs a GUI. Both branches spend exactly one tick.
    """

    name = "semantic"

    def __init__(self, *, gui_fallback: bool = True,
                 width: int | None = None, height: int | None = None):
        self.gui_fallback = bool(gui_fallback)
        self.width = None if width is None else int(width)
        self.height = None if height is None else int(height)

    def resolve_engine_home(self, explicit: str | None = None) -> Path:
        return _specific_home("semantic", explicit)

    def validate(self, home: Path) -> None:
        del home

    def capture(self, episode) -> bytes | None:
        if episode.gui_open and self.gui_fallback:
            return episode.env.frame()
        episode.env.act()
        observation = episode.env.obs
        from bedrock_rl.env.semantic import semantic_image
        width = self.width or episode.frame_width
        height = self.height or episode.frame_height
        image = semantic_image(observation["cam"], observation["depth"],
                               observation["edge"], width=width,
                               height=height)
        return _png_bytes(image)

    def provenance(self, home: Path) -> dict[str, Any]:
        return {
            "type": self.name,
            "component": object_reference(self),
            "size": [self.width, self.height],
            "gui_fallback": self.gui_fallback,
            "engine_assets": read_asset_manifest(home),
        }


@dataclass(frozen=True)
class ViewSpec:
    """Parsed task ``view`` declaration."""

    type: str
    config: dict[str, Any]

    @classmethod
    def parse(cls, value: str | Mapping[str, Any] | None) -> "ViewSpec":
        if value is None:
            return cls(DEFAULT_VIEW, {})
        if isinstance(value, str):
            target, config = value, {}
        elif isinstance(value, Mapping):
            raw = dict(value)
            target = raw.pop("type", None)
            if not target:
                raise ValueError("task view mapping needs a 'type'")
            nested = raw.pop("config", None)
            if nested is not None:
                if raw:
                    raise ValueError("task view puts settings both beside "
                                     "'type' and inside 'config'")
                if not isinstance(nested, Mapping):
                    raise TypeError("task view.config must be a mapping")
                raw = dict(nested)
            config = raw
        else:
            raise TypeError("task view must be a name or mapping")
        canonical = BUILTIN_VIEWS.get(str(target), str(target))
        if canonical not in BUILTIN_VIEWS.values() and ":" not in canonical:
            raise ValueError(
                f"unknown task view {target!r}; use semantic, "
                "netherite-procedural, minecraft-official, or a "
                "package.module:Object import path")
        return cls(canonical, dict(config))

    def create(self):
        if self.type == "semantic":
            return SemanticView(**self.config)
        if self.type == "netherite-procedural":
            if self.config:
                raise ValueError("netherite-procedural view has no settings")
            return EnginePixelsView(name=self.type, asset_kind="procedural")
        if self.type == "minecraft-official":
            return EnginePixelsView(name=self.type, asset_kind="jar",
                                    **self.config)
        return ComponentSpec(self.type, self.config).create()


def open_view(spec: ViewSpec, explicit_home: str | None = None):
    """Construct a view, choose its engine, validate it, and describe it."""
    view = spec.create()
    capture = getattr(view, "capture", None)
    if not callable(capture):
        raise TypeError(f"task view {spec.type!r} must expose "
                        "capture(episode)")
    resolver = getattr(view, "resolve_engine_home", None)
    home = (Path(resolver(explicit_home)).expanduser().resolve()
            if callable(resolver) else engine_home(explicit_home).resolve())
    validate = getattr(view, "validate", None)
    if callable(validate):
        validate(home)
    describe = getattr(view, "provenance", None)
    provenance = (describe(home) if callable(describe) else {
        "type": spec.type, "component": object_reference(view),
        "config": dict(spec.config),
        "engine_assets": read_asset_manifest(home),
    })
    if not isinstance(provenance, Mapping):
        raise TypeError("view.provenance(home) must return a mapping")
    return view, home, dict(provenance)


def capture_png(view, episode) -> bytes | None:
    return _png_bytes(view.capture(episode))
