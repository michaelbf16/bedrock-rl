"""Minecraft 1.11.2 names exposed by the Netherite observation protocol."""

from functools import lru_cache

INV_ORDER = (
    "log", "planks", "stick", "cobblestone", "crafting_table",
    "wooden_pickaxe", "stone_pickaxe", "coal", "torch",
)

# Block metadata is not exposed, so a name may resolve to multiple ids.
BLOCK_IDS = {
    "air": (0,), "stone": (1,), "grass_block": (2,), "dirt": (3,),
    "cobblestone": (4,), "planks": (5,), "water": (8, 9),
    "flowing_lava": (10,), "stationary_lava": (11,), "lava": (10, 11),
    "sand": (12,), "gravel": (13,), "gold_ore": (14,),
    "iron_ore": (15,), "coal_ore": (16,), "log": (17, 162),
    "oak_log": (17,), "acacia_log": (162,),
    "leaves": (18, 161), "lapis_ore": (21,), "torch": (50,),
    "obsidian": (49,), "diamond_ore": (56,), "crafting_table": (58,),
    "furnace": (61, 62), "portal": (90,), "netherrack": (87,),
    "redstone_ore": (73, 74), "snow": (78, 80), "emerald_ore": (129,),
}

# Items are a separate namespace. Only INV_ORDER has aggregate counters;
# every item remains observable while it is in the hotbar.
ITEM_IDS = {
    "stone": (1,), "dirt": (3,), "cobblestone": (4,), "planks": (5,),
    "sand": (12,), "gravel": (13,), "gold_ore": (14,), "iron_ore": (15,),
    "log": (17, 162), "oak_log": (17,), "acacia_log": (162,),
    "obsidian": (49,), "torch": (50,), "chest": (54,),
    "crafting_table": (58,), "furnace": (61,), "iron_pickaxe": (257,),
    "flint_and_steel": (259,), "diamond_pickaxe": (278,),
    "coal": (263,), "diamond": (264,), "iron_ingot": (265,),
    "wooden_pickaxe": (270,), "stone_pickaxe": (274,), "stick": (280,),
    "flint": (318,),
}


def _resolve(names, catalog, kind):
    values = names if isinstance(names, (list, tuple)) else [names]
    resolved = []
    for value in values:
        if isinstance(value, int):
            resolved.append(value)
            continue
        name = str(value).split(":")[-1].strip()
        mapping = catalog.get(name)
        if mapping is None:
            mapping = _complete_catalog(kind).get(name)
        if mapping is None:
            raise KeyError(f"no Minecraft 1.11.2 {kind} mapping for {name!r}")
        resolved.extend(mapping)
    return tuple(dict.fromkeys(resolved))


@lru_cache(maxsize=2)
def _complete_catalog(kind):
    # Imported lazily so the lightweight package root never pays for a JSON
    # registry unless task validation actually resolves a Minecraft name.
    from bedrock_rl.env.minecraft_data import minecraft_data
    key = "blocks" if kind == "block" else "items"
    return {name: (int(item),)
            for name, item in minecraft_data()[key].items()}


def resolve_blocks(names):
    return _resolve(names, BLOCK_IDS, "block")


def resolve_items(names):
    return _resolve(names, ITEM_IDS, "item")
