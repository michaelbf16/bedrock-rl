#!/usr/bin/env python3
"""Overlay a Minecraft resource pack onto a client jar without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath


def _safe_name(name: str) -> str:
    raw = name.replace("\\", "/")
    if raw.startswith(("/", "./", "../")):
        raise ValueError(f"unsafe resource-pack entry {name!r}")
    value = raw
    path = PurePosixPath(value)
    if (not value or value.startswith("/") or ".." in path.parts
            or path.parts[0] not in {"assets", "pack.mcmeta", "pack.png"}):
        raise ValueError(
            f"unsafe or non-resource-pack entry {name!r}; packs may contain "
            "assets/, pack.mcmeta, and pack.png")
    return value


def _pack_entries(pack: Path):
    """Yield normalized name, bytes, and unix mode from zip or directory."""
    if pack.is_dir():
        for path in sorted(pack.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"resource pack contains symlink {path}")
            if path.is_file():
                name = _safe_name(path.relative_to(pack).as_posix())
                yield name, path.read_bytes(), path.stat().st_mode
        return
    if not pack.is_file() or not zipfile.is_zipfile(pack):
        raise ValueError(f"resource pack {pack} is not a zip file or directory")
    with zipfile.ZipFile(pack) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"resource pack contains symlink {info.filename}")
            yield _safe_name(info.filename), archive.read(info), mode


def compose(base: Path, pack: Path, output: Path) -> None:
    if not base.is_file() or not zipfile.is_zipfile(base):
        raise ValueError(f"base client jar {base} is not a zip file")
    overlay = {}
    for name, data, mode in _pack_entries(pack):
        if name in overlay:
            raise ValueError(f"resource pack contains duplicate entry {name!r}")
        overlay[name] = (data, mode)
    if not any(name.startswith("assets/minecraft/") for name in overlay):
        raise ValueError("resource pack contains no assets/minecraft files")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(base) as source, zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED) as target:
            seen = set()
            for info in source.infolist():
                name = info.filename.replace("\\", "/")
                if name in seen or name in overlay:
                    continue
                seen.add(name)
                target.writestr(info, source.read(info))
            for name, (data, mode) in sorted(overlay.items()):
                info = zipfile.ZipInfo(name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (mode or 0o100644) << 16
                target.writestr(info, data)
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def pack_digest(pack: Path) -> str:
    """Stable digest of normalized paths and bytes (also works for a dir)."""
    digest = hashlib.sha256()
    entries = sorted((name, data) for name, data, _ in _pack_entries(pack))
    for name, data in entries:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="overlay a resource pack onto a Minecraft client jar")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--digest", action="store_true",
                        help="print the canonical pack-content sha256")
    args = parser.parse_args(argv)
    try:
        if args.digest:
            print(pack_digest(args.pack))
        else:
            if args.base is None or args.out is None:
                parser.error("--base and --out are required unless --digest is used")
            compose(args.base, args.pack, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
