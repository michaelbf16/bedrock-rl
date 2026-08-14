"""Minecraft 1.11.2 recipe and name registry for scripted policies.

The compact JSON beside this module is generated from one pinned
minecraft-data version. It drives both these GUI layouts and the C header in
the Netherite patch, eliminating hand-maintained Python recipe subsets.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from bedrock_rl.env.minecraft_data import minecraft_data


WILDCARD = 32767


@lru_cache(maxsize=None)
def recipe_variants(output, output_meta=None):
    name = str(output).split(":")[-1].strip()
    return tuple(
        row for row in minecraft_data()["recipes"]
        if row["name"] == name
        and (output_meta is None or row["output"][2] == int(output_meta)))


def _ingredients(recipe):
    if recipe["shaped"]:
        return [value for row in recipe["shape"] for value in row
                if value is not None]
    return list(recipe["ingredients"])


def _fits(recipe, grid_size):
    if recipe["shaped"]:
        shape = recipe["shape"]
        return len(shape) <= grid_size and len(shape[0]) <= grid_size
    return len(recipe["ingredients"]) <= grid_size * grid_size


def has_recipe(output, grid_size=None):
    recipes = recipe_variants(output)
    return bool(recipes if grid_size is None else [
        row for row in recipes if _fits(row, int(grid_size))])


def item_stack_size(item):
    """Vanilla maximum stack size for a registry item name."""
    name = str(item).split(":")[-1].strip()
    data = minecraft_data()
    item_id = data["items"].get(name)
    if item_id is None:
        raise KeyError(f"no Minecraft 1.11.2 item mapping for {name!r}")
    return int(data["stack_sizes"].get(str(item_id), 64))


def choose_recipe(output, hotbar_ids, hotbar_counts, grid_size,
                  hotbar_meta=None, output_meta=None):
    """First vanilla-ordered layout that fits the grid and held ingredients.

    Exact metadata is used when the observation supplies it. Wildcard recipe
    ingredients consume any remaining variant of that item ID.
    """
    metas = hotbar_meta if hotbar_meta is not None else (0,) * len(hotbar_ids)
    available = Counter()
    for item, count, meta in zip(hotbar_ids, hotbar_counts, metas):
        if int(item) and int(count) > 0:
            available[(int(item), int(meta))] += int(count)
    for recipe in recipe_variants(output, output_meta):
        if not _fits(recipe, int(grid_size)):
            continue
        remaining = available.copy()
        exact = Counter((int(value[0]), int(value[1]))
                        for value in _ingredients(recipe)
                        if int(value[1]) != WILDCARD)
        if any(remaining[key] < count for key, count in exact.items()):
            continue
        remaining.subtract(exact)
        wildcards = Counter(int(value[0]) for value in _ingredients(recipe)
                            if int(value[1]) == WILDCARD)
        if all(sum(count for (found, _meta), count in remaining.items()
                   if found == item and count > 0) >= needed
               for item, needed in wildcards.items()):
            return recipe
    return None


def ingredient_cells(recipe, grid_size):
    """Group a chosen layout into ``(item id, meta) -> grid cells``."""
    groups = {}
    if recipe["shaped"]:
        for row_index, row in enumerate(recipe["shape"]):
            for column_index, value in enumerate(row):
                if value is None:
                    continue
                key = (int(value[0]), int(value[1]))
                groups.setdefault(key, []).append(
                    row_index * int(grid_size) + column_index)
    else:
        for index, value in enumerate(recipe["ingredients"]):
            key = (int(value[0]), int(value[1]))
            groups.setdefault(key, []).append(index)
    return tuple((item, meta, tuple(cells))
                 for (item, meta), cells in groups.items())
