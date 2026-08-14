"""Lazy access to the generated, version-pinned Minecraft registry."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def minecraft_data():
    path = Path(__file__).with_name("minecraft_1_11.json")
    with path.open() as stream:
        return json.load(stream)
