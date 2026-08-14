"""Conservative Minecraft block predicates shared by setup and planners."""

from __future__ import annotations

from dataclasses import dataclass, field


def _block_key(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    block_id = getattr(value, "block_id", None)
    if isinstance(block_id, int) and not isinstance(block_id, bool):
        return block_id
    name = str(value).lower().split("[", 1)[0]
    name = name.rsplit(":", 1)[-1]
    return name or None


@dataclass(frozen=True)
class BlockRules:
    """Configurable conservative predicates for movement and spawning."""

    passable: frozenset[str] = field(default_factory=lambda: frozenset({
        "air", "cave_air", "deadbush", "detector_rail", "fire",
        "ladder", "powered_rail", "rail", "red_flower",
        "redstone_torch", "redstone_wire", "sapling", "snow_layer",
        "standing_sign", "tallgrass", "torch", "unlit_redstone_torch",
        "vine", "void_air", "wall_sign", "wheat", "yellow_flower",
    }))
    fluids: frozenset[str] = field(default_factory=lambda: frozenset({
        "water", "flowing_water", "lava", "flowing_lava",
    }))
    hazards: frozenset[str] = field(default_factory=lambda: frozenset({
        "cactus", "campfire", "fire", "lava", "flowing_lava",
        "magma", "magma_block", "soul_fire", "sweet_berry_bush",
    }))
    unbreakable: frozenset[str] = field(default_factory=lambda: frozenset({
        "barrier", "bedrock", "end_portal", "end_portal_frame",
    }))
    passable_ids: frozenset[int] = field(default_factory=lambda: frozenset({
        0, 6, 27, 28, 31, 32, 37, 38, 39, 40, 50, 51, 55, 59, 63,
        65, 66, 68, 69, 70, 72, 75, 76, 77, 78, 83, 90, 93, 94,
        104, 105, 106, 111, 115, 127, 131, 132, 141, 142, 143,
        147, 148, 149, 150, 151, 157, 171, 175, 176, 177, 198, 207,
    }))
    fluid_ids: frozenset[int] = field(
        default_factory=lambda: frozenset({8, 9, 10, 11}))
    hazard_ids: frozenset[int] = field(
        default_factory=lambda: frozenset({10, 11, 51, 81, 213}))
    unbreakable_ids: frozenset[int] = field(
        default_factory=lambda: frozenset({7, 90, 119, 120, 166}))

    def name(self, value):
        key = _block_key(value)
        return "" if key is None else str(key)

    def is_passable(self, value):
        key = _block_key(value)
        return (key in self.passable_ids if isinstance(key, int)
                else key in self.passable)

    def is_fluid(self, value):
        key = _block_key(value)
        return (key in self.fluid_ids if isinstance(key, int)
                else key in self.fluids)

    def is_hazard(self, value):
        key = _block_key(value)
        return (key in self.hazard_ids if isinstance(key, int)
                else key in self.hazards)

    def is_unbreakable(self, value):
        key = _block_key(value)
        return (key in self.unbreakable_ids if isinstance(key, int)
                else key in self.unbreakable)

    def is_solid(self, value):
        key = _block_key(value)
        return key is not None and not self.is_passable(value) \
            and not self.is_fluid(value)

    def is_excavatable(self, value):
        key = _block_key(value)
        return key is not None and not self.is_fluid(value) \
            and not self.is_hazard(value) \
            and not self.is_unbreakable(value)


DEFAULT_BLOCK_RULES = BlockRules()

