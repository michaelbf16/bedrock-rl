"""Single source of calibrated block-breaking durations for SDG policies.

The scripted task oracle sometimes mines bare-handed and therefore uses a
different conservative profile from the progression controller, which knows
the selected tool. Both profiles and the underwater physics calculation live
here so block hardness or tool-speed changes cannot drift across policies.
"""

from __future__ import annotations

import math


SOFT_BLOCK_TICKS = 26
LOG_TOOL_TICKS = 65
PICK_TOOL_TICKS = 28
WOODEN_PICK_COAL_TICKS = 50
OBSIDIAN_DIAMOND_TICKS = 188
UNKNOWN_TUNNEL_TICKS = 70

_SOFT_TICKS_BY_ID = {2: 26, 3: 26, 12: 26, 13: 26, 18: 20, 78: 10}
_LOG_IDS = frozenset((17, 162))
_PICK_BLOCK_IDS = frozenset((1, 4, 14, 15, 16, 21, 49, 56,
                             73, 74, 129))
_ORE_AND_STONE_IDS = frozenset((1, 4, 14, 15, 16, 21, 56, 73, 74, 129))
_PICK_TOOLS = frozenset((
    "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "diamond_pickaxe",
))

BLOCK_HARDNESS = {
    1: 1.5, 2: 0.6, 3: 0.5, 4: 2.0, 12: 0.5, 13: 0.6,
    14: 3.0, 15: 3.0, 16: 3.0, 17: 2.0, 18: 0.2,
    21: 3.0, 49: 50.0, 56: 3.0, 73: 3.0, 74: 3.0,
    78: 0.1, 129: 3.0, 162: 2.0,
}
PICK_SPEED = {
    "wooden_pickaxe": 2.0, "stone_pickaxe": 4.0,
    "iron_pickaxe": 6.0, "diamond_pickaxe": 8.0,
}

ORACLE_MINE_TICKS = {
    "log": 110, "leaves": 20, "grass_block": 26, "dirt": 26,
    "sand": 26, "gravel": 26, "stone": 70, "cobblestone": 70,
    "coal_ore": 70, "iron_ore": 130,
    "diamond_ore": 110, "lapis_ore": 110, "redstone_ore": 110,
    "obsidian": OBSIDIAN_DIAMOND_TICKS,
}
ORACLE_MINE_TICKS_DEFAULT = 90


def oracle_mining_ticks(block_name):
    """Conservative hold for the task-level oracle's unknown tool state."""
    return ORACLE_MINE_TICKS.get(str(block_name), ORACLE_MINE_TICKS_DEFAULT)


def dry_mining_ticks(block_id, tool=None, fallback=80):
    """Calibrated dry hold for an exact public block id and selected tool."""
    block_id = int(block_id)
    if block_id in _SOFT_TICKS_BY_ID:
        return _SOFT_TICKS_BY_ID[block_id]
    if block_id in _LOG_IDS:
        return LOG_TOOL_TICKS
    if block_id == 49 and tool == "diamond_pickaxe":
        return OBSIDIAN_DIAMOND_TICKS
    if block_id == 16 and tool == "wooden_pickaxe":
        return WOODEN_PICK_COAL_TICKS
    if block_id in _ORE_AND_STONE_IDS and tool in _PICK_TOOLS:
        return PICK_TOOL_TICKS
    return int(fallback)


def submerged_mining_ticks(block_id, tool, dry_ticks):
    """Bounded vanilla underwater hold from shared hardness/tool physics."""
    block_id = int(block_id)
    hardness = BLOCK_HARDNESS.get(block_id)
    if hardness is None:
        return int(dry_ticks) * 5
    speed = (PICK_SPEED.get(tool, 1.0)
             if block_id in _PICK_BLOCK_IDS else 1.0)
    return max(int(dry_ticks),
               math.ceil(30.0 * hardness * 5.0 / speed) + 15)
