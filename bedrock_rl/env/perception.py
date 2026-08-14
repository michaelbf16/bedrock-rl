"""Structured Minecraft perception derived from an engine observation.

``perception_of`` returns the targeted block, player contact blocks, and a
bounded nearby sample. ``format_perception`` renders that data as the optional
``<perception>`` prompt section. Derivation and presentation stay separate so
custom harnesses can replace either layer.

All fields come from ``env.obs``. The semantic depth buffer does not identify
the struck face, so ``side`` is omitted. Target positions inherit the depth
buffer's quarter-block quantization. Nearby counts may undercount in dense
terrain because the observation contains only the 256 nearest non-air cells.
"""
import math
from collections import Counter

from bedrock_rl.env.engine import (BREAK_REACH, CAM_H, CAM_W, CROSSHAIR_PX,
                                     EYE_HEIGHT, N_BLOCKS)
from bedrock_rl.env.observation import ID_NAMES
from bedrock_rl.env.projection import ndc_to_angles, tangents

# The semantic camera's tangents. env/observation.py has the same pair for
# the same reason and derives it the same way; both go through
# projection.tangents so neither writes the engine's ray out again.
_TANX, _TANY = tangents(CAM_W, CAM_H)

# The crosshair pixel as a normalized image point, which is `oc_pixel`'s
# own normalization of CROSSHAIR_PX. ny is negated on the way in because
# the engine's image plane is y-up and pixel rows count down.
# adapters/netherite/oracle.py::pixel_to_angles does this for any pixel, so
# this is the same ray and not a second opinion about the crosshair.
_NX = (CROSSHAIR_PX[0] - CAM_W / 2 + 0.5) / (CAM_W / 2)
_NY = -(CROSSHAIR_PX[1] - CAM_H / 2 + 0.5) / (CAM_H / 2)

# How far PAST the reported surface distance the cell is sampled. The
# depth is the range to the face and is quantized DOWN to a quarter
# block, so stepping exactly `dist` lands on or just short of the face
# and floors into the cell IN FRONT of the block.
_PAST_FACE = 0.2

# Perception uses ``air`` for block id 0. The ASCII camera uses ``air/sky`` as
# a visual legend because it cannot distinguish those pixels.
_PERCEPTION_NAMES = {**ID_NAMES, 0: "air"}

AIR = 0


def block_name(bid):
    """One of our block ids as the namespaced id their recorder emits.

    `_short_id` strips the namespace again, so an id this table does not
    know renders as its number rather than as a wrong name.
    """
    bid = int(bid)
    name = _PERCEPTION_NAMES.get(bid)
    return f"minecraft:{name}" if name else str(bid)


def crosshair_ray(obs):
    """(dx, dy, dz) unit direction of the CROSSHAIR PIXEL's ray.

    Not the look axis. The camera is even in both dimensions so no pixel
    is on the axis, and the pixel the crosshair reports sits
    about 1.11 degrees off it (see CROSSHAIR_PX in env/engine.py).
    The id on the `crosshair_target:` line came off that pixel, so the
    cell on the same line has to come off the same ray or the one line
    carries two rays' opinions.
    """
    dyaw, dpitch = ndc_to_angles(_NX, _NY, obs["pitch"], _TANX, _TANY)
    yaw = math.radians(obs["yaw"] + dyaw)
    pitch = math.radians(obs["pitch"] + dpitch)
    cp = math.cos(pitch)
    return -math.sin(yaw) * cp, -math.sin(pitch), math.cos(yaw) * cp


# ── the ray a left click actually uses ───────────────────────────────────
# The engine walks voxels from the player's eye along the look axis. The
# semantic crosshair samples a camera pixel slightly off that axis, so it can
# identify a different block near voxel boundaries. ``break_ray`` reproduces
# the engine's action ray from the observation instead of using that sample.
# Two bounded approximations remain:
#
#   1. The engine uses a sine lookup table; Python uses ``libm``. Results can
#      differ when a ray passes very close to a cell boundary.
#   2. ``obs["blocks"]`` contains at most 256 non-air cells. A missing cell is
#      treated as air only inside the radius the sample can certify.

# Vanilla face names, keyed by (axis, step). The face is the one the ray
# ENTERED through, which is the face pointing back at the eye: moving in
# +z you cross into a cell through its -z side, and -z is NORTH.
_FACES = {(0, 1): "west", (0, -1): "east",
          (1, 1): "down", (1, -1): "up",
          (2, 1): "north", (2, -1): "south"}


def look_ray(obs):
    """(dx, dy, dz) unit direction of the LOOK AXIS -- the ray the engine
    casts for a break, a place and a use."""
    yaw = math.radians(obs["yaw"])
    pitch = math.radians(obs["pitch"])
    cp = math.cos(pitch)
    return -math.sin(yaw) * cp, -math.sin(pitch), math.cos(yaw) * cp


def _solid_and_trust(obs):
    """The record's non-air cells, and how far out absence means air."""
    solid = {(bx, by, bz): bid for bid, bx, by, bz in obs["blocks"] if bid}
    if len(solid) < N_BLOCKS:
        # The scan did not fill, so it carried every non-air cell it saw
        # and the engine's own 16-block radius is the honest bound.
        return solid, 16.0
    px, py, pz = obs["x"], obs["y"], obs["z"]
    far = max(math.dist((bx + 0.5, by + 0.5, bz + 0.5), (px, py, pz))
              for bx, by, bz in solid)
    # One cell of slack, because the record does not say which metric it
    # ranked by and a cell diagonal is the most two sane ones can differ.
    return solid, max(0.0, far - 1.0)


def break_ray(env, reach=None):
    """What the LOOK AXIS -- the engine's break ray -- is on, or None.

    `reach` defaults to as far as the record can vouch for. Pass
    BREAK_REACH for "what would a click break RIGHT NOW"; leave it for
    "what is the crosshair on", which is the question the observation
    asks and which has an honest answer past the 5 blocks a click
    reaches, exactly as their `player_get_targeted_block(8)` answers past
    their 4.5-block survival reach.

    Returns `{"id", "pos", "distance", "side"}` with `pos` the hit cell and
    `distance` the eye-to-face range along the axis. ``side`` is reserved for
    renderers that can identify the struck face.

    None means three different things and deliberately does not
    distinguish them, because a caller that has to act cannot act on any
    of them: nothing solid within `reach`, or the walk left the radius
    the record can vouch for, or the eye is inside a block.
    """
    o = env.obs
    solid, trust = _solid_and_trust(o)
    if reach is None:
        reach = trust
    ex, ey, ez = o["x"], o["y"] + EYE_HEIGHT, o["z"]
    d = look_ray(o)
    eye = (ex, ey, ez)
    cell = [math.floor(ex), math.floor(ey), math.floor(ez)]
    if tuple(cell) in solid:
        return None                       # eye inside a block; no face
    step, t_max, t_delta = [0, 0, 0], [math.inf] * 3, [math.inf] * 3
    for i in range(3):
        if d[i] > 0.0:
            step[i] = 1
            t_max[i] = (cell[i] + 1 - eye[i]) / d[i]
            t_delta[i] = 1.0 / d[i]
        elif d[i] < 0.0:
            step[i] = -1
            t_max[i] = (cell[i] - eye[i]) / d[i]
            t_delta[i] = -1.0 / d[i]
    while True:
        axis = min(range(3), key=lambda i: t_max[i])
        t = t_max[axis]
        if t > reach or t == math.inf:
            return None
        cell[axis] += step[axis]
        t_max[axis] += t_delta[axis]
        key = (cell[0], cell[1], cell[2])
        if math.dist((key[0] + 0.5, key[1] + 0.5, key[2] + 0.5),
                     (o["x"], o["y"], o["z"])) > trust:
            return None                   # past what the record can say
        bid = solid.get(key)
        if bid:
            # RAW id, like `ObsHelpers.crosshair`, because the two are
            # answers to the same question and a caller that switches
            # between them must not also switch value types. The
            # perception channel maps it to a name; the CHECKS
            # (env/task.py, Episode.draw) compare numbers.
            return {"id": int(bid), "pos": list(key), "dist": float(t),
                    "side": _FACES[(axis, step[axis])]}


def breakable_now(env):
    """Return the block a left click can currently break, if any.

    This uses the same ray as :func:`aim_target`, capped at the engine's
    interaction reach. A visible crosshair target can be farther away.
    """
    return break_ray(env, reach=BREAK_REACH)


def aim_target(env):
    """Return the block under the action ray in crosshair-result form.

    The result contains ``id`` and ``dist``, plus ``pos`` and ``side`` when
    available. The look-axis voxel walk is authoritative wherever the bounded
    block sample can certify it. The semantic camera is used only beyond that
    radius, where the current turn cannot interact with the reported block.
    """
    hit = break_ray(env)
    if hit is not None:
        return hit
    ch = env.crosshair()
    _, trust = _solid_and_trust(env.obs)
    if ch["dist"] is not None and ch["dist"] <= trust:
        return {"id": AIR, "dist": None, "pos": None, "side": ""}
    return {"id": int(ch["id"]), "dist": ch["dist"], "pos": None, "side": ""}


def target_block(env):
    """`perception["target_block"]`: what the crosshair is on, or None.

    THE BREAK RAY FIRST, out to BREAK_REACH, because within reach the
    honest answer to "what is the crosshair on" is the block a click
    would break and the camera pixel is 1.576 degrees away from being
    that. Their `player_get_targeted_block(8)` is the client's own
    `Entity.pick`, the very call that fills `minecraft.hitResult` and
    therefore decides the break, so THIS is the port of their mechanism
    and not only of their field names.

    THE CAMERA PIXEL BEYOND IT, unchanged, because past the break reach
    there is no click to disagree with and the camera is the only thing
    that sees out to 48 blocks. Theirs reports past their 4.5-block
    survival reach too, out to the 8 it asks for, so a target named but
    not yet breakable is their behaviour as well as ours.

    None is the empty crosshair, which their raycast answers with when
    the ray leaves the world and which their formatter renders as
    `crosshair_target: none`.
    """
    hit = break_ray(env)
    if hit is not None:
        return {"id": block_name(hit["id"]), "pos": hit["pos"],
                "distance": hit["dist"], "side": hit["side"]}
    ch = aim_target(env)
    if int(ch["id"]) == AIR or ch["dist"] is None:
        return None
    o = env.obs
    dx, dy, dz = crosshair_ray(o)
    t = float(ch["dist"]) + _PAST_FACE
    return {
        "id": block_name(ch["id"]),
        "pos": [math.floor(o["x"] + dx * t),
                math.floor(o["y"] + EYE_HEIGHT + dy * t),
                math.floor(o["z"] + dz * t)],
        "distance": float(ch["dist"]),
        # No face in a depth buffer. Their formatter omits a falsy side,
        # and their own fluid raycast sends "" for the same reason.
        "side": "",
    }


def perception_of(env):
    """Build the structured perception dictionary from one engine record.

    The box is
    dx and dz in [-3, 3] and dy in (-1, 0, 1, 2), iterated dx then dz
    then dy, air excluded from both the counter and the samples, the
    first 24 cells kept as samples, and the counter cut to its 16 most
    common. The ORDER matters and is not decoration: `nearby_samples` is
    the first 24 in iteration order and not the 24 nearest, so a
    different loop nest is a different observation.
    """
    o = env.obs
    x = math.floor(o["x"])
    y = math.floor(o["y"])
    z = math.floor(o["z"])
    # The record's 256 nearest non-air cells, as a lookup. Air is absent
    # from it by construction, so a cell that is not in it is air -- true
    # for the player's own column, which is nearer than anything, and the
    # limit of it is in this module's header.
    solid = {(bx, by, bz): bid for bid, bx, by, bz in o["blocks"] if bid}

    def cell(px, py, pz):
        return block_name(solid.get((px, py, pz), AIR))

    counts = Counter()
    samples = []
    for dx in range(-3, 4):
        for dz in range(-3, 4):
            for dy in (-1, 0, 1, 2):
                bid = solid.get((x + dx, y + dy, z + dz))
                if not bid:
                    continue
                name = block_name(bid)
                counts[name] += 1
                if len(samples) < 24:
                    samples.append({"id": name,
                                    "pos": [x + dx, y + dy, z + dz],
                                    "rel": [dx, dy, dz]})
    return {
        "target_block": target_block(env),
        "standing_on": cell(x, y - 1, z),
        "feet_block": cell(x, y, z),
        "head_block": cell(x, y + 1, z),
        "nearby_blocks": [{"id": bid, "count": n}
                          for bid, n in counts.most_common(16)],
        "nearby_samples": samples,
    }


# ── stable text formatting ───────────────────────────────────────────────

def short_id(value):
    text = str(value or "")
    return text.split("minecraft:", 1)[1] if text.startswith("minecraft:") \
        else text


def one_line(value, limit):
    import re
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    if len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text


def format_perception(perception):
    """Their `_format_perception`, as a list of lines."""
    if perception is None:
        return ["recorder_perception: unavailable"]
    lines = []

    target = perception.get("target_block")
    if isinstance(target, dict) and target:
        block_id = short_id(target.get("id") or target.get("type") or "")
        pos = one_line(target.get("pos") or target.get("position"), 80)
        dist = target.get("distance")
        side = target.get("side")
        dist_text = ""
        try:
            if dist is not None:
                dist_text = f" distance={float(dist):.1f}"
        except (TypeError, ValueError):
            dist_text = f" distance={dist}"
        side_text = f" side={side}" if side else ""
        lines.append(
            f"crosshair_target: {block_id or 'unknown'} pos={pos}"
            f"{dist_text}{side_text}")
    else:
        lines.append("crosshair_target: none")

    for key in ("standing_on", "feet_block", "head_block"):
        value = perception.get(key)
        if value:
            lines.append(f"{key}: {short_id(value)}")

    nearby = perception.get("nearby_blocks") or []
    if isinstance(nearby, dict):
        nearby_items = list(nearby.items())
    elif isinstance(nearby, list):
        nearby_items = [
            (item.get("id"), item.get("count"))
            for item in nearby
            if isinstance(item, dict)
        ]
    else:
        nearby_items = []
    rendered_nearby = []
    for block, count in nearby_items[:12]:
        try:
            n = int(count or 0)
        except (TypeError, ValueError):
            n = 0
        if block and n > 0:
            rendered_nearby.append(f"{short_id(block)} x{n}")
    nearby_text = ", ".join(rendered_nearby)
    lines.append(f"nearby_blocks: {nearby_text or 'none'}")

    samples = perception.get("nearby_samples") or []
    if isinstance(samples, list) and samples:
        rendered = []
        for sample in samples[:10]:
            if not isinstance(sample, dict):
                continue
            rendered.append(
                f"{short_id(sample.get('id'))}@rel="
                f"{one_line(sample.get('rel'), 40)}"
            )
        if rendered:
            lines.append("nearby_samples: " + ", ".join(rendered))
    return lines


def perception_block(env):
    """The whole fenced section, as `<perception>` fences it there."""
    return (["<perception>"] + format_perception(perception_of(env))
            + ["</perception>"])
