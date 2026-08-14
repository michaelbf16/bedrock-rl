"""Scripted Netherite oracles for supported task families.

expert_cu emits `computer`-tool action lists, the shipped interface and
the only one. Used for SFT data ("what would a competent turn
look like from this obs"). The env is deterministic given (seed, init
pose), so an expert sequence recorded closed-loop replays identically
when scored open-loop.
"""
import random

from bedrock_rl.env import gui_layout
from bedrock_rl.env.catalog import resolve_blocks, resolve_items
from bedrock_rl.env.engine import DEPTH_SCALE
from bedrock_rl.env.observation import CAM_H, CAM_W, TANX, TANY
from bedrock_rl.env.projection import ndc_to_angles
from bedrock_rl.sdg.policy.recipes import (
    choose_recipe, has_recipe, ingredient_cells, item_stack_size,
)
from bedrock_rl.sdg.policy.mining import oracle_mining_ticks

# Held ticks a bare-hand or correct-tool break costs. A continuous attack can
# start damaging the block behind the target after a break, so long holds are
# appropriate only where that continuation is intentional or harmless.
#
# A task whose block is missing gets the default, which is long enough for
# anything softer than stone.
def mine_ticks(block_name):
    return oracle_mining_ticks(block_name)


# Pitches a block placement is tried at, in order. Placing needs a free
# cell against a face the ray can reach, and where that is depends on
# what the player is standing in: a steep angle aims at their own feet,
# which their cube blocks, and a shallow one wants floor a block or two
# ahead. Every built-in placement oracle sweeps this same list.
PLACE_PITCHES = (55.0, 40.0, 70.0, 25.0, 85.0)
# How far off a coarse aim may be before the semantic camera takes over.
# Every oracle uses the same number for this handover.
AIM_TOL = 9.0
# How far away a block still counts as one to walk to, rather than one to
# put another of down. The solvers use the same reach, so the two cannot
# disagree about whether a table is already standing nearby.
NEARBY_BLOCK = 16.0


def pixel_to_angles(px, py, pitch):
    """Semantic-camera pixel -> (dyaw, dpitch) degrees from the current
    view centre.

    The two lines below are `oc_pixel`'s own normalization, written the
    way that file writes it: nx is 2*(px+0.5)/OC_W - 1 rearranged, and ny
    is the same with the sign flipped because this returns a pitch DELTA,
    which is positive downward, while the engine's image plane is y-up.
    The inverse itself is env/projection.py, which is the exact one; this
    used to be two independent arctangents, which is separable and the
    engine's ray is not.

    `pitch` is the camera's pitch now, and it is what makes this exact
    rather than only exact on the centre axes.
    """
    nx = (px - CAM_W / 2 + 0.5) / (CAM_W / 2)
    ny = (py - CAM_H / 2 + 0.5) / (CAM_H / 2)
    return ndc_to_angles(nx, -ny, pitch, TANX, TANY)


def find_pixel(env, ids, nearest_depth=False, max_depth=None):
    """Pixel of a target block: nearest the crosshair by default, or the
    physically closest surface (min depth, ties to center) when
    nearest_depth - the right choice when the plan is to mine it."""
    cam, dep = env.obs["cam"], env.obs["depth"]
    best, best_d = None, None
    for i, v in enumerate(cam):
        if v in ids:
            if (max_depth is not None
                    and float(dep[i]) / DEPTH_SCALE > float(max_depth)):
                continue
            py, px = divmod(i, CAM_W)
            center = (px - CAM_W / 2) ** 2 + (py - CAM_H / 2) ** 2
            # A tuple makes "nearest depth, then nearest centre" exact. A
            # weighted scalar let a large screen offset outweigh several
            # blocks of depth, contradicting nearest_depth's contract.
            d = (dep[i], center) if nearest_depth else center
            if best_d is None or d < best_d:
                best, best_d = (px, py), d
    return best


def gui_craft_2x2(env, item, output_replace=()):
    """Mouse-only oracle for one 2x2 craft: open the inventory, pick up
    the ingredient stack, right-click ONE ingredient into each grid cell,
    take the result, close (closing is what moves the result from the
    cursor into the inventory, so the success check can see it).

    Every click comes from bedrock_rl/env/gui_layout.py, which is the one
    place that knows both where a slot is and which action lands on it.
    Returns `computer` action dicts, or None when the
    ingredients are not reachable from the hotbar - the obs protocol
    exposes hotbar slots only, so a stack sitting in the main inventory is
    invisible here."""
    recipe = choose_recipe(item, env.obs["hotbar_ids"],
                           env.obs["hotbar_counts"], 2,
                           env.obs.get("hotbar_meta"))
    if recipe is None:
        return None
    actions = gui_layout.open_screen()
    for item_id, meta, cells in ingredient_cells(recipe, 2):
        metas = None if meta == 32767 else (meta,)
        src = gui_layout.find_stack(env, (item_id,), len(cells), metas)
        if src is None:
            return None
        have = env.obs["hotbar_counts"][src]
        grid = [gui_layout.CELLS_2X2[cell] for cell in cells]
        actions += gui_layout.fill_cells(
            src, grid, put_back=have > len(cells))
    return (actions + _take_result_to_hotbar(
                env, item, replace=output_replace)
            + gui_layout.close_screen())


def _take_result_to_hotbar(env, item, screen="player", replace=()):
    """Take a craft result and put it in an observable hotbar slot.

    Closing a screen while carrying the output uses vanilla's generic
    inventory insertion order, which may put it in backpack slot 9. The
    binary observation intentionally exposes only the hotbar plus a small
    set of aggregate counters, so a furnace or pickaxe placed there becomes
    invisible to every closed-loop policy. Explicitly click an existing
    matching hotbar stack or an empty hotbar cell before closing.
    """
    # Clicking a new unstackable item onto an old one swaps them instead of
    # merging. The generated item registry covers tools, armor, buckets,
    # boats, and every other max-stack-one result—not only pickaxes.
    target = (None if item_stack_size(item) == 1 else
              gui_layout.find_stack(env, resolve_items([item]), 1))
    if target is None:
        target = next((slot for slot, block in
                       enumerate(env.obs["hotbar_ids"]) if block == 0), None)
    # Long survival scripts can fill all nine cells with mined stone
    # variants before crafting their next tier. Swap an obsolete lower-tier
    # tool or obsolete stack onto the cursor; closing moves that displaced
    # stack to the backpack while the new progression item stays observable.
    replacements = {
        "stone_pickaxe": ("wooden_pickaxe", "coal", "log"),
        "furnace": ("wooden_pickaxe", "stone_pickaxe"),
        "iron_pickaxe": ("wooden_pickaxe", "stone_pickaxe", "cobblestone"),
        "diamond_pickaxe": ("wooden_pickaxe", "stone_pickaxe",
                             "iron_pickaxe", "cobblestone"),
    }
    if isinstance(replace, (str, int)):
        replace = (replace,)
    if target is None:
        for old in (*tuple(replace), *replacements.get(item, ())):
            target = gui_layout.find_stack(env, resolve_items([old]), 1)
            if target is not None:
                break
    if target is None:
        return gui_layout.take_result(screen)
    return (gui_layout.take_result(screen)
            + gui_layout.click_slot(target, "left", screen))


def _nearest_table(env, max_dist=NEARBY_BLOCK):
    """Nearest crafting table standing in the world, or None.

    obs["blocks"] is the 256 nearest non-air cells, which is what the
    model reads too, so this is not privileged knowledge."""
    best = None
    for bid, wx, wy, wz in env.obs["blocks"]:
        if bid != gui_layout.CRAFTING_TABLE_BLOCK:
            continue
        _, _, d = env.rel_angles_to(wx, wy, wz)
        if d <= max_dist and (best is None or d < best[1]):
            best = ((wx, wy, wz), d)
    return None if best is None else best


def gui_craft_3x3(env, item, rng, output_replace=()):
    """Mouse-only oracle for ONE turn of a craft that needs a table.

    A wooden pickaxe does not fit the player's 2x2 grid, so the table has
    to be standing in the world and open. That is three states and this
    returns the next turn's worth of clicks for whichever one the
    observation reports, because there is no action that does all of it:

      no table in reach   select it in the hotbar, look down, place it
      table in reach      aim at it, then right-click it open
      table screen open   fill the recipe shape, take the result, close

    Returns None when the ingredients are not reachable from the hotbar,
    the same limit gui_craft_2x2 has and for the same reason.
    """
    from bedrock_rl.adapters.netherite.computer import moves_for_angles
    recipe = choose_recipe(item, env.obs["hotbar_ids"],
                           env.obs["hotbar_counts"], 3,
                           env.obs.get("hotbar_meta"))
    if recipe is None:
        return None
    if env.obs["container"] == gui_layout.CONTAINER_TABLE:
        acts = []
        for item_id, meta, where in ingredient_cells(recipe, 3):
            metas = None if meta == 32767 else (meta,)
            src = gui_layout.find_stack(env, (item_id,), len(where), metas)
            if src is None:
                return None
            have = env.obs["hotbar_counts"][src]
            acts += gui_layout.fill_cells(
                src, [gui_layout.CELLS_3X3[c] for c in where], "table",
                put_back=have > len(where))
        return (acts + _take_result_to_hotbar(
                    env, item, "table", replace=output_replace)
                + gui_layout.close_screen())
    ch = env.crosshair()
    if (ch["id"] == gui_layout.CRAFTING_TABLE_BLOCK
            and (ch["dist"] or 99) <= 4.0):
        return [{"action": "right_click"}]
    found = _nearest_table(env)
    if found is not None:
        # The table is down and the crosshair is not on it. Aim this turn
        # and click the next, so a miss is corrected by the next
        # observation rather than by arithmetic.
        #
        # Navigation goes by world CELL and clicking goes by CROSSHAIR,
        # and those two disagree: aiming at the cell centre can leave the
        # crosshair on whatever stands in front of it, and an aim that is
        # already correct then produces no action at all and the oracle
        # spins for the whole episode. So the coarse aim hands over to the
        # semantic camera, which is a raycast and reports a surface that
        # is really reachable, and a table that is faced, in range and
        # still not under the crosshair is a reason to move, not to aim
        # again.
        p, dist = found
        ry, rp, _ = env.rel_angles_to(*p)
        if abs(ry) > AIM_TOL or abs(rp) > AIM_TOL:
            return moves_for_angles(ry, rp, env.obs["pitch"])
        if dist > 3.5:
            # a table placed at the far end of the look-down sweep can
            # land outside the six blocks that hold its screen open
            return [{"action": "key", "k": "w", "ticks": 8}]
        px = find_pixel(env, (gui_layout.CRAFTING_TABLE_BLOCK,))
        if px is not None:
            dy, dp = pixel_to_angles(*px, env.obs["pitch"])
            if abs(dy) > 1.0 or abs(dp) > 1.0:
                return moves_for_angles(dy, dp, env.obs["pitch"])
        return [{"action": "key", "k": rng.choice(["a", "d"]), "ticks": 6}]
    src = gui_layout.find_stack(
        env, resolve_items(["crafting_table"]), 1)
    if src is None:
        return None
    # the angle a block places at depends on what the player is standing
    # in, so the oracle tries a different one each turn rather than
    # repeating a look that has already failed
    want = rng.choice(PLACE_PITCHES)
    return (gui_layout.select_hotbar(src)
            + moves_for_angles(0.0, want - env.obs["pitch"],
                               env.obs["pitch"])
            + [{"action": "right_click"}])


def _hold(key, ticks):
    """Movement hold split into legal chunks. compile_actions rejects a
    single movement hold over MAX_MOVE_HOLD ticks, so the oracle has to
    cross distance the same way the model does, over several actions."""
    from bedrock_rl.adapters.netherite.computer import MAX_MOVE_HOLD
    out = []
    left = max(1, int(ticks))
    while left > 0:
        n = min(left, MAX_MOVE_HOLD)
        out.append({"action": "key", "k": key, "ticks": n})
        left -= n
    return out


# ── the oracle registry ──────────────────────────────────────────────────
# An oracle is keyed by `success.type`, the SAME string
# bedrock_rl/env/task.py::CHECKS keys its checks by. These were if/elif
# chains over that string in this file, next to a third one in tasks.py,
# so adding a task family meant editing three switches and forgetting one
# of them produced a task with no oracle and no complaint.
#
# A registered oracle takes (task, env, rng) and returns a `computer`
# action list, or None when it cannot solve this instance (target not in
# the world, ingredients not reachable).
ORACLES = {}


def oracle(*check_types):
    """Decorator: register an oracle for one or more check types."""
    def deco(fn):
        for t in check_types:
            ORACLES[t] = fn
        return fn
    return deco


@oracle("pitch_below", "pitch_above")
def _cu_pitch(task, env, rng):
    from bedrock_rl.adapters.netherite.computer import moves_for_angles
    t, spec = task.success.type, task.success.spec
    target = float(spec["value"]) + (-12 if t == "pitch_below" else 12)
    dp = max(-170, min(170, target - env.obs["pitch"]))
    return (moves_for_angles(0.0, dp, env.obs["pitch"])
            or [{"action": "wait", "ticks": 1}])


@oracle("crosshair_block")
def _cu_crosshair(task, env, rng):
    from bedrock_rl.adapters.netherite.computer import moves_for_angles
    del rng
    ids = resolve_blocks(task.success.spec["blocks"])
    # Use the nearest target cell in the public block scan for the coarse
    # search. Random yaw sweeps could revisit the same heading and made a
    # supposedly expert trajectory fail nearly half the time. The block scan
    # is part of the observation (and is already what mining oracles use), so
    # this remains an observation-grounded closed-loop demonstrator.
    max_dist = float(task.success.spec.get("max_dist", 1e18))
    px = find_pixel(env, ids, nearest_depth=True, max_depth=max_dist)
    if px is not None:
        dy, dp = pixel_to_angles(*px, env.obs["pitch"])
        moves = moves_for_angles(dy, dp, env.obs["pitch"], tol=0.1)
        if moves:
            return moves
    targets = []
    for block, x, y, z in env.obs["blocks"]:
        if block not in ids:
            continue
        dy, dp, distance = env.rel_angles_to(x, y, z)
        if distance <= max_dist:
            targets.append((distance, abs(dy) + abs(dp), dy, dp))
    if targets:
        _, _, dy, dp = min(targets)
    else:
        # A saturated block scan can omit the target. Sweep deterministically
        # so six successive calls cover the horizon instead of gambling on
        # repeated random choices.
        dy, dp = 60.0, 8.0 - float(env.obs["pitch"])
    moves = moves_for_angles(dy, dp, env.obs["pitch"], tol=0.1)
    # Cell-centre aiming can be geometrically centred while an intervening
    # surface still owns the crosshair. Nudge instead of waiting forever;
    # the next observation either exposes a target pixel or permits another
    # closed-loop correction.
    return (moves or moves_for_angles(3.0, 0.0, env.obs["pitch"], tol=0.1)
            or [{"action": "wait", "ticks": 1}])


@oracle("item_gained")
def _cu_item_gained(task, env, rng):
    from bedrock_rl.adapters.netherite.computer import moves_for_angles
    item = task.success.spec["item"]
    if (has_recipe(item, 2)
            and choose_recipe(item, env.obs["hotbar_ids"],
                              env.obs["hotbar_counts"], 2,
                              env.obs.get("hotbar_meta")) is not None):
        return gui_craft_2x2(env, item)
    acts = gui_craft_3x3(env, item, rng)
    if acts is not None:
        return acts
    p = _nearest_for(env, item)
    if p is None:
        return None
    ry, rp, dist = env.rel_angles_to(*p)
    acts = moves_for_angles(ry, rp, env.obs["pitch"])
    if dist > 2.5:
        steps = max(1, round((dist - 1.5) / 0.65))
        acts += _hold("w", 4 * steps)
        acts += moves_for_angles(0.0, max(-60, min(60, -rp * 0.6)),
                                 env.obs["pitch"] + rp)
    acts.append({"action": "left_click_hold", "ticks": 160})
    acts += _hold("w", 8)
    return acts


@oracle("hotbar_gained")
def _cu_hotbar_gained(task, env, rng):
    from bedrock_rl.adapters.netherite.computer import moves_for_angles
    px = find_pixel(env, _target_block_ids(), nearest_depth=True)
    if px is None:
        return moves_for_angles(0.0, 40, env.obs["pitch"])
    dy, dp = pixel_to_angles(*px, env.obs["pitch"])
    return (moves_for_angles(dy, dp, env.obs["pitch"])
            + [{"action": "left_click_hold", "ticks": 32},
               {"action": "key", "k": "w", "ticks": 8}])


@oracle("selected_item")
def _cu_selected_item(task, env, rng):
    del rng
    ids = set(resolve_items(task.success.spec["items"]))
    selected = int(env.obs["hotbar_sel"])
    hotbar = tuple(env.obs["hotbar_ids"])
    if 0 <= selected < len(hotbar) and hotbar[selected] in ids:
        return [{"action": "wait", "ticks": 1}]
    slot = next((index for index, item in enumerate(hotbar)
                 if item in ids), None)
    return None if slot is None else gui_layout.select_hotbar(slot)


@oracle("block_placed")
def _cu_block_placed(task, env, rng):
    """Place the requested inventory block on reachable ground."""
    del rng
    from bedrock_rl.adapters.netherite.computer import moves_for_angles
    block = task.success.spec["block"]
    block = block[0] if isinstance(block, (list, tuple)) else block
    src = gui_layout.find_stack(env, resolve_items([block]), 1)
    if src is None:
        return None
    return (gui_layout.select_hotbar(src)
            + moves_for_angles(0.0, PLACE_PITCHES[0] - env.obs["pitch"],
                               env.obs["pitch"])
            + [{"action": "right_click"}])


@oracle("block_broken")
def _cu_block_broken(task, env, rng):
    """Aim at the block named by a block_broken success check and break it.

    The oracle aims at the cell center and returns ``None`` if no matching block
    is visible. Aim and commit are returned in one call; generators that need
    an observation between those actions should provide a custom oracle.
    """
    from bedrock_rl.adapters.netherite.computer import moves_for_angles
    spec = task.success.spec
    ids = resolve_blocks(spec.get("block", []))
    if not ids:
        return None
    p = None
    best = 1e18
    o = env.obs
    for bid, wx, wy, wz in o["blocks"]:
        if bid not in ids:
            continue
        d = ((wx + 0.5 - o["x"]) ** 2 + (wy + 0.5 - o["y"] - 1.62) ** 2
             + (wz + 0.5 - o["z"]) ** 2)
        if d < best:
            p, best = (wx, wy, wz), d
    if p is None:
        return None
    ry, rp, _ = env.rel_angles_to(*p)
    moves = moves_for_angles(ry, rp, env.obs["pitch"])
    name = spec["block"]
    name = name[0] if isinstance(name, (list, tuple)) else name
    return (moves or []) + [{"action": "left_click_hold",
                             "ticks": mine_ticks(name)}]


@oracle("moved_distance")
def _cu_moved(task, env, rng):
    # jump-walk (w+space): 4 ticks of forward+jump per 0.65-block step
    ticks = 4 * (int(float(task.success.spec["value"]) / 0.65) + 4)
    return _hold(["w", "space"], ticks)


def _nearest_for(env, item):
    """Where to walk for an item the world holds rather than crafts."""
    if item == "log":
        return env.nearest("logs")
    if item == "coal":
        return env.nearest("coal")
    return None


# Default targets for the built-in ground-harvest ``hotbar_gained`` oracle.
GROUND_BLOCKS = ("grass_block", "dirt")


def _target_block_ids():
    return resolve_blocks(list(GROUND_BLOCKS))


def expert_cu(task, env, rng=None):
    """Return one turn of computer-use actions for a supported task family.

    Returns ``None`` when no registered oracle supports the task. Look targets use
    inverse camera mapping with edge clamping, and crafting uses visible GUI
    interactions rather than semantic actions. Callers request another turn
    after each resulting observation until the episode ends.
    """
    fn = ORACLES.get(task.success.type)
    return None if fn is None else fn(task, env, rng or random)
