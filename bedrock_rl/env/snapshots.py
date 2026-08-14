"""Build cached Netherite snapshots from declarative initial state.

Task authors describe inventory, player vitals, spawn points, proximity
constraints, and supported world settings in ``initial_state``. This module is
the one translation boundary from that data to Netherite's binary snapshot
format. There is deliberately no registry of task-specific setup scripts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import threading
from array import array
from collections import defaultdict
from functools import lru_cache

from bedrock_rl.env import engine
from bedrock_rl.env.block_rules import DEFAULT_BLOCK_RULES
from bedrock_rl.env.engine import MagmaEnv
from bedrock_rl.env.episode import FRAME_H, FRAME_W
from bedrock_rl.env.initial_state import SpawnPoint


# RlSnapHead is packed and fixed-size (blaze/env/blaze_snapshot.h). Keep the
# layout in one place and validate it against the file before changing bytes.
_HEAD_PRE = "<" + "".join([
    "4s", "I", "q", "q", "i", "i",   # magic version seed tick ox oz
    "3d", "6d", "2f", "3d",          # pos box yaw/pitch motion
    "4i", "f", "4i", "f", "i",       # ground/collide fall sprint/jump prev
    "f", "i", "2f", "i",             # health food saturation/exhaustion timer
    "f", "3i", "5i", "i",            # dig statics + rc/use edges
    "2d",                              # server motion
    "4i", "i", "i",                  # container xyz, world_dirty, hotbar_sel
])
INV_OFF = struct.calcsize(_HEAD_PRE)
INV_SLOTS = 37
HEAD_SIZE = INV_OFF + INV_SLOTS * 3 * 4 + 4 + 6 * 4
ITEM_SIZE = 6 * 8 + 7 * 4
POS_OFF = struct.calcsize("<4sIqqii")
BOX_OFF = POS_OFF + struct.calcsize("<3d")
ANGLE_OFF = BOX_OFF + struct.calcsize("<6d")
VITALS_OFF = struct.calcsize("<4sIqqii3d6d2f3d4if4ifi")
SETTLE_TICKS = 400
# Increment whenever snapshot construction/acceptance semantics change.  A
# stale file must never bypass stronger world validation after an upgrade.
SNAPSHOT_CACHE_VERSION = 3


class SnapshotRejected(ValueError):
    """A natural world that cannot satisfy its declared start contract."""


def _check_layout(buf):
    """Refuse a snapshot whose binary layout is not the one we patch."""
    magic, version = struct.unpack_from("<4sI", buf, 0)
    if magic != b"BSNP" or version != 1:
        raise RuntimeError(f"not a v1 .bsnp ({magic!r} v{version})")
    if len(buf) < HEAD_SIZE:
        raise RuntimeError(f"file is {len(buf)} bytes, shorter than a header")
    container = struct.unpack_from("<i", buf, INV_OFF - 6 * 4)[0]
    hotbar = struct.unpack_from("<i", buf, INV_OFF - 4)[0]
    if not (0 <= container <= 3 and 0 <= hotbar <= 8):
        raise RuntimeError(
            f"container={container} hotbar_sel={hotbar} are out of range; "
            "RlSnapHead layout changed")
    n_items = struct.unpack_from("<I", buf,
                                 INV_OFF + INV_SLOTS * 3 * 4)[0]
    rnx, rny, rnz = struct.unpack_from(
        "<3i", buf, INV_OFF + INV_SLOTS * 3 * 4 + 4 + 3 * 4)
    # Mirror the C loaders' volume bound.  Ten-chunk captures are
    # 320x128x320, so a per-axis <=256 check incorrectly rejects a valid
    # native snapshot even though it remains below the 2^24-cell cap.
    if not (rnx > 0 and rny > 0 and rnz > 0 and
            rnx * rny * rnz <= 1 << 24):
        raise RuntimeError(
            f"implausible region {rnx}x{rny}x{rnz}; RlSnapHead changed")
    end = HEAD_SIZE + n_items * ITEM_SIZE + rnx * rny * rnz * 2
    if end + 4 > len(buf):
        raise RuntimeError(
            f"head+items+region = {end} exceeds the {len(buf)} byte file")
    ncoal = struct.unpack_from("<I", buf, end)[0]
    if end + 4 + ncoal * 12 != len(buf):
        raise RuntimeError(
            f"snapshot size mismatch: expected {end + 4 + ncoal * 12}, "
            f"found {len(buf)}; RlSnapHead changed")


def _snapshot_region(buf):
    """Decoded world-coordinate region and packed-cell accessor."""

    _check_layout(buf)
    origin_x, origin_z = struct.unpack_from("<ii", buf, POS_OFF - 8)
    local_x, player_y, local_z = struct.unpack_from("<3d", buf, POS_OFF)
    item_count_off = INV_OFF + INV_SLOTS * 3 * 4
    item_count = struct.unpack_from("<I", buf, item_count_off)[0]
    region_off = item_count_off + struct.calcsize("<I")
    rx, ry, rz, nx, ny, nz = struct.unpack_from("<6i", buf, region_off)
    cells_off = HEAD_SIZE + item_count * ITEM_SIZE
    cells = array("H")
    cells.frombytes(buf[cells_off:cells_off + nx * ny * nz * 2])
    if sys.byteorder != "little":
        cells.byteswap()

    def block_id(x, y, z):
        index = ((x - rx) * ny + (y - ry)) * nz + (z - rz)
        return cells[index] >> 4

    return {
        "origin": (origin_x, origin_z),
        "player": (local_x + origin_x, player_y, local_z + origin_z),
        "bounds": (rx, ry, rz, rx + nx - 1, ry + ny - 1, rz + nz - 1),
        "shape": (nx, ny, nz),
        "cells": cells,
        "block_id": block_id,
    }


def _constraint_matches(region, spec):
    """One complete captured coordinate list per declarative constraint."""

    rx, ry, rz, _, _, _ = region["bounds"]
    by_id = defaultdict(list)
    wanted = frozenset(
        block_id for constraint in spec.constraints
        for block_id in constraint.ids)
    _, ny, nz = region["shape"]
    try:
        import numpy as np
    except ImportError:  # NumPy is an acceleration, not a runtime contract.
        hits = ((index, state >> 4)
                for index, state in enumerate(region["cells"])
                if state >> 4 in wanted)
    else:
        states = np.frombuffer(region["cells"], dtype=np.uint16)
        ids = states >> 4
        indexes = np.flatnonzero(np.isin(ids, tuple(wanted)))
        hits = ((int(index), int(ids[index])) for index in indexes)
    for index, value in hits:
        ix, remainder = divmod(index, ny * nz)
        iy, iz = divmod(remainder, nz)
        by_id[value].append((rx + ix, ry + iy, rz + iz))
    matches = []
    for constraint in spec.constraints:
        cells = tuple(
            cell for value in constraint.ids for cell in by_id[value])
        if not cells:
            bounds = region["bounds"]
            raise SnapshotRejected(
                f"no constraint block {constraint.block!r} was captured "
                f"inside snapshot bounds {bounds}")
        matches.append(cells)
    return tuple(matches)


def _constraint_index(cells, bucket_size=8):
    buckets = defaultdict(list)
    for cell in cells:
        buckets[tuple(value // bucket_size for value in cell)].append(cell)
    return buckets


@lru_cache(maxsize=2048)
def _ordered_bucket_offsets(rx, ry, rz, span, bucket_size):
    """Bucket offsets ordered by exact lower-bound distance to a point."""

    offsets = []

    def axis_distance(value, offset):
        low = offset * bucket_size
        high = low + bucket_size - 1
        return (low - value if value < low else
                value - high if value > high else 0.0)

    for dx in range(-span, span + 1):
        x2 = axis_distance(rx, dx) ** 2
        for dy in range(-span, span + 1):
            xy2 = x2 + axis_distance(ry, dy) ** 2
            for dz in range(-span, span + 1):
                lower = xy2 + axis_distance(rz, dz) ** 2
                offsets.append((lower, dx, dy, dz))
    return tuple(sorted(offsets))


def _nearest_in_band(point, constraint, buckets, bucket_size=8):
    """Whether the nearest captured match lies in the requested 3-D band."""

    px, py, pz = point
    # Compare the player's position with integer block coordinates: adding a
    # half to both later gives the block-center distance used by the contract.
    qx, qy, qz = px - 0.5, py - 0.5, pz - 0.5
    base = tuple(math.floor(value / bucket_size)
                 for value in (qx, qy, qz))
    residual = tuple(value - key * bucket_size
                     for value, key in zip((qx, qy, qz), base))
    high = constraint.max_distance
    high_squared = high ** 2
    span = math.ceil(high / bucket_size) + 1
    nearest = math.inf
    for lower, dx, dy, dz in _ordered_bucket_offsets(
            *residual, span, bucket_size):
        if lower > high_squared or lower >= nearest:
            break
        key = (base[0] + dx, base[1] + dy, base[2] + dz)
        for x, y, z in buckets.get(key, ()):
            distance = ((x + 0.5 - px) ** 2
                        + (y + 0.5 - py) ** 2
                        + (z + 0.5 - pz) ** 2)
            nearest = min(nearest, distance)
    return (constraint.min_distance ** 2 <= nearest
            <= high_squared)


def _safe_surface_cells(region, hazard_clearance=0):
    """Safe highest standable cell per interior x/z snapshot column.

    NumPy is already present in common training/data environments, but it is
    intentionally optional. The fallback implements the identical public
    contract for minimal task-authoring installs.
    """

    rx, ry, rz, max_x, max_y, max_z = region["bounds"]
    nx, ny, nz = region["shape"]
    rules = DEFAULT_BLOCK_RULES
    clearance = int(hazard_clearance)
    if clearance < 0:
        raise ValueError("spawn hazard_clearance cannot be negative")
    try:
        import numpy as np
    except ImportError:
        block_id = region["block_id"]
        horizontal_margin = max(1, clearance)
        first_y = ry + max(1, clearance)
        last_y = max_y - 1 - clearance
        for x in range(rx + horizontal_margin,
                       max_x - horizontal_margin + 1):
            for z in range(rz + horizontal_margin,
                           max_z - horizontal_margin + 1):
                for y in range(last_y, first_y - 1, -1):
                    support = block_id(x, y - 1, z)
                    feet = block_id(x, y, z)
                    head = block_id(x, y + 1, z)
                    if (rules.is_solid(support)
                            and not rules.is_hazard(support)
                            and rules.is_passable(feet)
                            and not rules.is_fluid(feet)
                            and not rules.is_hazard(feet)
                            and rules.is_passable(head)
                            and not rules.is_fluid(head)
                            and not rules.is_hazard(head)
                            and all(
                                not rules.is_fluid(block_id(
                                    x + dx, y + dy, z + dz))
                                and not rules.is_hazard(block_id(
                                    x + dx, y + dy, z + dz))
                                for dx in range(-clearance, clearance + 1)
                                for dy in range(
                                    -clearance, clearance + 2)
                                for dz in range(
                                    -clearance, clearance + 1))):
                        yield x, y, z
                        break
        return

    ids = (np.frombuffer(region["cells"], dtype=np.uint16) >> 4).reshape(
        nx, ny, nz)
    size = max(4096, int(ids.max(initial=0)) + 1)
    passable = np.zeros(size, dtype=np.bool_)
    fluid = np.zeros(size, dtype=np.bool_)
    hazard = np.zeros(size, dtype=np.bool_)
    passable[list(rules.passable_ids)] = True
    fluid[list(rules.fluid_ids)] = True
    hazard[list(rules.hazard_ids)] = True
    support = (~passable[ids[:, :-2, :]]
               & ~fluid[ids[:, :-2, :]]
               & ~hazard[ids[:, :-2, :]])
    clear = (passable[ids] & ~fluid[ids] & ~hazard[ids])
    safe = support & clear[:, 1:-1, :] & clear[:, 2:, :]
    if clearance:
        neighborhood = np.zeros_like(safe)
        x_slice = slice(clearance, nx - clearance)
        z_slice = slice(clearance, nz - clearance)
        first_y = max(1, clearance)
        last_y = min(ny - 2, ny - 2 - clearance)
        if first_y <= last_y:
            y_slice = slice(first_y - 1, last_y)
            local = np.ones(
                (nx - 2 * clearance,
                 last_y - first_y + 1,
                 nz - 2 * clearance),
                dtype=np.bool_)
            unsafe = fluid[ids] | hazard[ids]
            for dx in range(-clearance, clearance + 1):
                for dy in range(-clearance, clearance + 2):
                    for dz in range(-clearance, clearance + 1):
                        local &= ~unsafe[
                            clearance + dx:nx - clearance + dx,
                            first_y + dy:last_y + dy + 1,
                            clearance + dz:nz - clearance + dz]
            neighborhood[x_slice, y_slice, z_slice] = local
        safe &= neighborhood
    interior = safe[1:-1, :, 1:-1]
    any_safe = interior.any(axis=1)
    # argmax on a reversed y axis returns the highest safe feet cell. It is
    # only read where any_safe is true, so its all-false value is irrelevant.
    reversed_y = interior[:, ::-1, :].argmax(axis=1)
    feet_y = (ny - 3 - reversed_y) + 1
    for ix, iz in np.argwhere(any_safe):
        yield (rx + int(ix) + 1,
               ry + int(feet_y[ix, iz]),
               rz + int(iz) + 1)


def _constraint_spawn(buf, spec, seed, *, return_matches=False):
    """Choose a deterministic safe surface spawn satisfying every band."""

    region = _snapshot_region(buf)
    matches = _constraint_matches(region, spec)
    indexes = tuple(_constraint_index(cells) for cells in matches)
    original = region["player"]
    candidates = []

    # Keep one complete block between the AABB and the captured x/z boundary.
    # The snapshot remains authoritative for support, feet, and head after a
    # restore instead of asking the engine to collide against unknown cells.
    for x, y, z in _safe_surface_cells(
            region, spec.spawn_hazard_clearance):
        point = (x + 0.5, float(y), z + 0.5)
        distance = sum((a - b) ** 2 for a, b in zip(point, original))
        candidates.append((distance, x, y, z))
    candidates.sort()
    # The objective is nearest safe spawn, then a seeded tie-break. Evaluate
    # bands in precisely that order and stop on the first valid point. This is
    # equivalent to collecting every valid point and taking min(), but normal
    # worlds no longer query tens of thousands of farther surface columns.
    position = 0
    while position < len(candidates):
        end = position + 1
        distance = candidates[position][0]
        while end < len(candidates) and candidates[end][0] == distance:
            end += 1
        tied = []
        for _, x, y, z in candidates[position:end]:
            tie = hashlib.sha256(
                f"netherite-spawn-v1:{int(seed)}:{x}:{y}:{z}"
                .encode()).digest()
            tied.append((tie, x, y, z))
        for _, x, y, z in sorted(tied):
            point = (x + 0.5, float(y), z + 0.5)
            if all(_nearest_in_band(point, constraint, index)
                   for constraint, index in zip(spec.constraints, indexes)):
                selected = SpawnPoint(*point)
                return ((selected, matches) if return_matches
                        else selected)
        position = end
    bands = ", ".join(
        f"{item.block}={item.min_distance:g}.."
        f"{item.max_distance:g}" for item in spec.constraints)
    raise SnapshotRejected(
        "no safe surface spawn satisfies all constraint bands "
        f"({bands}) inside snapshot bounds {region['bounds']} "
        f"(captured radius "
        f"{max(region['shape'][0], region['shape'][2]) // 2})")


def _validate_constraint_matches(spec, matches, player):
    """Validate one pose against already decoded complete block matches."""

    player_x, player_y, player_z = map(float, player)
    for constraint, cells in zip(spec.constraints, matches):
        squared = min(
            (x + 0.5 - player_x) ** 2
            + (y + 0.5 - player_y) ** 2
            + (z + 0.5 - player_z) ** 2
            for x, y, z in cells)
        nearest = math.sqrt(squared)
        if not (constraint.min_distance <= nearest
                <= constraint.max_distance):
            raise SnapshotRejected(
                f"spawn's nearest {constraint.block} is "
                f"{nearest:.3f} blocks; expected between "
                f"{constraint.min_distance:g} and "
                f"{constraint.max_distance:g} blocks")


def _snapshot_constraint_matches(path, spec):
    with open(path, "rb") as stream:
        region = _snapshot_region(stream.read())
    return region, _constraint_matches(region, spec)


def _validate_snapshot_constraints(path, spec, player=None):
    """Check spawn bands against the complete captured BSNP block region."""
    if not spec.constraints:
        return
    region, matches = _snapshot_constraint_matches(path, spec)
    _validate_constraint_matches(
        spec, matches, region["player"] if player is None else player)


def _select_spawn_point_with_matches(path, spec, seed):
    """Resolve a point and retain any complete scan used to select it."""

    point = spec.point_for(seed)
    if point is not None or not spec.constraints:
        return point, None
    with open(path, "rb") as stream:
        return _constraint_spawn(
            stream.read(), spec, seed, return_matches=True)


def select_spawn_point(path, spec, seed):
    """Resolve an explicit or constraint-selected point without mutation."""
    point, _ = _select_spawn_point_with_matches(path, spec, seed)
    return point


def patch_initial_state(path, spec, seed, *, selected_point=None):
    """Apply inventory, vitals, and a designated spawn atomically."""
    with open(path, "rb") as stream:
        buf = bytearray(stream.read())
    _check_layout(buf)

    if spec.inventory:
        slots = spec.inventory_slots(seed)
        for index in range(INV_SLOTS):
            struct.pack_into("<3i", buf, INV_OFF + index * 12,
                             *slots.get(index, (0, 0, 0)))

    selected_item = spec.player.get("selected_item")
    if selected_item is not None:
        from bedrock_rl.env.catalog import resolve_items
        item_id = resolve_items([selected_item])[0]
        selected = next((index for index in range(9)
                         if struct.unpack_from(
                             "<i", buf, INV_OFF + index * 12)[0] == item_id),
                        None)
        if selected is None:
            raise ValueError(
                f"initial player selected_item {selected_item!r} is not "
                "present in a hotbar slot")
        struct.pack_into("<i", buf, INV_OFF - 4, selected)

    if spec.player:
        health, food, saturation, exhaustion, timer = struct.unpack_from(
            "<fi2fi", buf, VITALS_OFF)
        values = spec.player
        struct.pack_into(
            "<fi2fi", buf, VITALS_OFF,
            float(values.get("health", health)),
            int(values.get("food", food)),
            float(values.get("saturation", saturation)),
            float(values.get("exhaustion", exhaustion)), timer)

    point = (select_spawn_point(path, spec, seed)
             if selected_point is None else selected_point)
    if point is not None:
        origin_x, origin_z = struct.unpack_from("<ii", buf,
                                                POS_OFF - 2 * 4)
        old = struct.unpack_from("<3d", buf, POS_OFF)
        # The task surface names world coordinates; BSNP stores player pose
        # and collision bounds relative to the physics-window origin.  Keep
        # that implementation detail inside this binary translation layer.
        local = (point.x - origin_x, point.y, point.z - origin_z)
        delta = tuple(local[index] - old[index] for index in range(3))
        box = list(struct.unpack_from("<6d", buf, BOX_OFF))
        box = [value + delta[index % 3]
               for index, value in enumerate(box)]
        struct.pack_into("<3d", buf, POS_OFF, *local)
        struct.pack_into("<6d", buf, BOX_OFF, *box)
        yaw, pitch = struct.unpack_from("<2f", buf, ANGLE_OFF)
        struct.pack_into("<2f", buf, ANGLE_OFF,
                         yaw if point.yaw is None else point.yaw,
                         pitch if point.pitch is None else point.pitch)

    if spec.blocks:
        n_items = struct.unpack_from(
            "<I", buf, INV_OFF + INV_SLOTS * 3 * 4)[0]
        region_header = INV_OFF + INV_SLOTS * 3 * 4 + 4
        rx0, ry0, rz0, rnx, rny, rnz = struct.unpack_from(
            "<6i", buf, region_header)
        cells_at = HEAD_SIZE + n_items * ITEM_SIZE
        region_end = cells_at + rnx * rny * rnz * 2
        for volume in spec.blocks:
            for x, y, z, block_id, meta in volume.cells():
                ix, iy, iz = x - rx0, y - ry0, z - rz0
                if not (0 <= ix < rnx and 0 <= iy < rny and 0 <= iz < rnz):
                    raise ValueError(
                        f"initial block ({x}, {y}, {z}) is outside snapshot "
                        f"region x={rx0}..{rx0 + rnx - 1}, "
                        f"y={ry0}..{ry0 + rny - 1}, "
                        f"z={rz0}..{rz0 + rnz - 1}")
                index = (ix * rny + iy) * rnz + iz
                struct.pack_into("<H", buf, cells_at + index * 2,
                                 (int(block_id) << 4) | int(meta))

        # The snapshot's trailing coal coordinates are a derived search
        # index. Rebuild them after arbitrary edits instead of leaving a
        # stale index that only the batched loader would observe.
        coal = []
        for ix in range(rnx):
            for iy in range(rny):
                for iz in range(rnz):
                    index = (ix * rny + iy) * rnz + iz
                    state = struct.unpack_from(
                        "<H", buf, cells_at + index * 2)[0]
                    if state >> 4 == 16:
                        coal.append((rx0 + ix, ry0 + iy, rz0 + iz))
        tail = bytearray(struct.pack("<I", len(coal)))
        for cell in coal:
            tail.extend(struct.pack("<3i", *cell))
        buf[region_end:] = tail

    temporary = str(path) + f".patch.{os.getpid()}.{threading.get_ident()}"
    with open(temporary, "wb") as stream:
        stream.write(buf)
    os.replace(temporary, path)


def _settle(env, still=10, max_ticks=SETTLE_TICKS):
    """Wait for a worldgen spawn to stop falling before capturing it."""
    y, held = env.obs["y"], 0
    for _ in range(max_ticks):
        env.act()
        if abs(env.obs["y"] - y) < 1e-6:
            held += 1
            if held >= still:
                return True
        else:
            y, held = env.obs["y"], 0
    return False


def engine_cache_identity(netherite_home=None):
    """Stable identity of the engine a cached snapshot describes.

    The actual checkout and its recorded patch stack are kept separate from
    the revision and patch contents this repository expects. That distinction
    matters while a new patch is staged: neither side may silently reuse a
    snapshot produced by the other. Older checkouts without a manifest use
    the engine helper's inferred applied prefix, explicitly marked as such.
    """
    expected_manifest = {
        patch.name: engine.patch_digest(patch)
        for patch in engine.shipped_patches()
    }
    try:
        recorded = engine.read_patch_manifest(netherite_home)
    except ValueError as exc:
        # A malformed manifest is already rejected by engine preflight. It
        # still needs a deterministic, distinct cache identity so an old
        # valid cache is never selected before that rejection is reported.
        try:
            raw_digest = engine.patch_digest(
                engine.patch_manifest_path(netherite_home))
        except OSError:
            raw_digest = None
        actual_manifest = {"source": "invalid", "error": str(exc),
                           "sha256": raw_digest}
    else:
        if recorded is not None:
            actual_manifest = {"source": "recorded",
                               "patches": dict(sorted(recorded.items()))}
        else:
            status = engine.patch_status(netherite_home)
            actual_manifest = {
                "source": "inferred",
                "patches": {
                    name: expected_manifest.get(name, "unavailable")
                    for name in sorted(status["adopted"])
                },
                "verified": status.get("verified"),
                "problems": list(status["problems"]),
            }
    return {
        "actual": {
            "head": engine.head_sha(netherite_home),
            "manifest": actual_manifest,
        },
        "expected": {
            "head": engine.pinned_sha(),
            "manifest": expected_manifest,
        },
    }


def engine_cache_key(netherite_home=None):
    body = json.dumps(engine_cache_identity(netherite_home), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def snapshot_path(task, seed, netherite_home=None):
    """Cache path keyed by state, radius, and complete engine identity."""
    return os.path.join(
        task.snapshots_dir,
        f"v{SNAPSHOT_CACHE_VERSION}.s{int(seed)}."
        f"{task.initial_state.cache_key}."
        f"r{task.snapshot_radius}.{engine_cache_key(netherite_home)}.bsnp")


def build_snapshot(task, seed, netherite_home=None):
    """Build and live-validate one declarative start-state snapshot."""
    if not task.initial_state.enabled:
        raise ValueError(f"task {task.name!r} has no initial_state to build")
    out = snapshot_path(task, seed, netherite_home=netherite_home)
    if os.path.exists(out):
        return out
    os.makedirs(task.snapshots_dir, exist_ok=True)
    temporary = out + f".build.{os.getpid()}.{threading.get_ident()}"
    world = task.initial_state.world
    env = MagmaEnv(
        seed=int(seed), netherite_home=netherite_home,
        frame_w=FRAME_W, frame_h=FRAME_H,
        mobs=world.get("mobs", False), world_time=world.get("time"))
    try:
        if not _settle(env):
            raise RuntimeError(f"seed {seed} did not settle before snapshot")
        env.snapshot(temporary, radius=task.snapshot_radius)
    finally:
        env.close()

    try:
        selected_point, constraint_matches = _select_spawn_point_with_matches(
            temporary, task.initial_state, seed)
        patch_initial_state(
            temporary, task.initial_state, seed,
            selected_point=selected_point)
        # The engine observation deliberately returns only a bounded nearest
        # block list.  World selection must inspect the complete captured
        # region or crowded terrain can hide a valid/invalid spawn.
        if task.initial_state.constraints:
            # Constraint-only selection already decoded every matching block.
            # Inventory/player patches cannot change that proof. Declarative
            # block edits and explicit points do require one post-patch scan.
            if constraint_matches is None or task.initial_state.blocks:
                _, constraint_matches = _snapshot_constraint_matches(
                    temporary, task.initial_state)
            _validate_constraint_matches(
                task.initial_state, constraint_matches,
                (selected_point.x, selected_point.y, selected_point.z))
        check = MagmaEnv(
            seed=int(seed), netherite_home=netherite_home,
            frame_w=FRAME_W, frame_h=FRAME_H, snapshot_in=temporary,
            mobs=world.get("mobs", False), world_time=world.get("time"))
        try:
            for _ in range(3):
                check.act()
            if selected_point is not None and max(
                    abs(check.obs[key] - getattr(selected_point, key))
                    for key in ("x", "y", "z")) > 0.5:
                raise SnapshotRejected(
                    "selected spawn moved after restore; choose a safe point "
                    "inside the snapshot region and above solid ground")
            if task.initial_state.constraints:
                _validate_constraint_matches(
                    task.initial_state, constraint_matches,
                    (check.obs["x"], check.obs["y"], check.obs["z"]))
        finally:
            check.close()
        os.replace(temporary, out)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return out


def ensure_seed(task, seed, netherite_home=None):
    """Return a cached snapshot, building it deterministically if absent."""
    out = snapshot_path(task, seed, netherite_home=netherite_home)
    return out if os.path.exists(out) else build_snapshot(
        task, seed, netherite_home=netherite_home)
