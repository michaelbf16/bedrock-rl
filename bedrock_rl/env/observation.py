"""Observation rendering: engine state -> the text the model reads.

Every prompt and every tool response the computer-use agent sees goes through
`render_obs`, so this module is on the critical path of training, rollout, and
reward alike.

The block-id tables and the ASCII glyph map are here because they are
what the semantic camera is rendered through. `render_camera` downsamples
the 36x64 camera to 12x32 characters and marks the crosshair cell.
"""
from bedrock_rl.env.engine import (CAM_H, CAM_W,       # noqa: F401
                                     wrap180)
from bedrock_rl.env.projection import tangents

GLYPHS = {0: ".", 1: "#", 2: "g", 3: "d", 4: "c", 5: "p", 7: "B",
          8: "~", 9: "~", 10: "!", 11: "!", 12: "s", 13: "v",
          14: "G", 15: "I", 16: "C", 17: "L", 18: "%", 31: "'",
          32: "'", 37: "'", 38: "'", 50: "t", 58: "T", 61: "F",
          49: "o", 62: "F", 78: "*", 80: "*", 87: "n",
          90: "P", 161: "%", 162: "L"}
LEGEND = (". sky/air  # stone  g grass  d dirt  c cobble  s sand  v gravel  "
          "* snow\nC coal_ore  I iron_ore  L log  % leaves  ~ water  ! lava  "
          "' plant\nT crafting_table  F furnace  t torch  p planks  ? other"
          "  o obsidian  P portal  n netherrack"
          "\nX crosshair (the block it is on is named above)")

ITEM_NAMES = {0: "nothing", 4: "cobblestone", 5: "planks", 17: "log",
              50: "torch", 58: "crafting_table", 61: "furnace",
              257: "iron_pickaxe", 263: "coal", 265: "iron_ingot",
              259: "flint_and_steel", 264: "diamond", 270: "wooden_pickaxe",
              274: "stone_pickaxe", 278: "diamond_pickaxe", 280: "stick",
              318: "flint"}

ID_NAMES = {0: "air/sky", 1: "stone", 2: "grass_block", 3: "dirt",
            4: "cobblestone", 5: "planks", 8: "water", 9: "water",
            10: "lava", 11: "lava", 12: "sand", 13: "gravel",
            14: "gold_ore", 15: "iron_ore", 16: "coal_ore", 17: "log",
            18: "leaves", 50: "torch", 58: "crafting_table", 61: "furnace",
            49: "obsidian", 62: "furnace", 78: "snow", 80: "snow",
            87: "netherrack", 90: "portal", 161: "leaves",
            162: "log",
            # The deep ores have a NAME here and deliberately no GLYPH
            # above. The name is what the
            # crosshair line reports, so a policy can VERIFY what it is
            # pointing at before it commits. The missing glyph is what
            # keeps the ASCII camera from also reporting which column is
            # which: every one of them draws as "?", so the map shows
            # WHERE the row is and never WHAT is in it. Give these four a
            # glyph and the discrimination the task exists to train is
            # readable out of the text, and the rendered frame stops
            # mattering.
            21: "lapis_ore", 56: "diamond_ore", 73: "redstone_ore",
            74: "redstone_ore", 129: "emerald_ore"}

# The SEMANTIC CAMERA's projection, stated once. CAM_H and CAM_W are
# imported above rather than restated: engine.py unpacks the wire record
# they describe, and this module used to write the same two numbers out
# again in the opposite order, with expert.py importing the pair from
# both files. The 70-degree vertical field of view and the tan(fov/2)
# times aspect form both come from env/projection.py, which is the one
# place the engine's camera is written down.
#
# These are the 64x36 camera's tangents and NOT the rendered frame's.
# The two surfaces share the field of view and differ in aspect, 64/36
# against 428/240, so they differ in TANX by 0.3 percent. expert.py
# inverts THESE to turn a camera pixel into look deltas; adapters/netherite/computer.py
# has its own pair for the frame the model actually clicks on.
TANX, TANY = tangents(CAM_W, CAM_H)


CROSSHAIR_GLYPH = "X"      # not a block id in GLYPHS, so it cannot be one


def render_camera(cam, down_h=3, down_w=2):
    """cam: flat u16 [36*64] -> 12x32 ASCII, centre sample per cell, with
    the crosshair cell marked.

    Every row is exactly CAM_W // down_w characters, so output column k is
    the same camera column on all twelve rows. It used to splice
    "[" + glyph + "]" into a 32-character line without removing anything,
    which made that ONE row 34 characters and shifted every glyph after
    the crosshair two columns against the rows above and below it. A model
    reading columns to judge left from right was misaligned on the row
    that matters most.

    Three characters cannot replace one and keep the width, so the marker
    replaces the glyph rather than wrapping it. Nothing is lost: the block
    under the crosshair is named in full, with its distance, on the
    "crosshair on:" line of every observation.
    """
    lines = []
    ch_r, ch_c = (CAM_H // 2) // down_h, (CAM_W // 2) // down_w
    for r in range(CAM_H // down_h):
        row = []
        for c in range(CAM_W // down_w):
            if r == ch_r and c == ch_c:
                row.append(CROSSHAIR_GLYPH)
                continue
            bid = cam[(r * down_h + down_h // 2) * CAM_W
                      + c * down_w + down_w // 2]
            row.append(GLYPHS.get(int(bid), "?"))
        lines.append("".join(row))
    return "\n".join(lines)


def render_obs(env, max_decisions=None, decisions_used=0, show_camera=True,
               gui_open=False, perception=False):
    """Render one ``MagmaEnv`` observation as model-visible text.

    ``gui_open`` is the episode's inventory-screen latch, maintained from the
    engine event stream because the observation record identifies a world
    container but not the player's inventory panel. ``perception`` appends the
    structured block from :mod:`bedrock_rl.env.perception`; it is task-scoped
    so generation, training, and evaluation share the same prompt contract.
    """
    o = env.obs
    # Within interaction reach, report the action ray so the observation names
    # the block a click would affect. Beyond it, preserve the camera result.
    from bedrock_rl.env.perception import aim_target
    ch = aim_target(env)
    ch_name = ID_NAMES.get(ch["id"], str(ch["id"]))
    inv = [f"{k} x{v}" for k, v in o["inv_counts"].items() if v]
    held_id = o["hotbar_ids"][o["hotbar_sel"]] if 0 <= o["hotbar_sel"] < 9 \
        else 0
    parts = []
    if max_decisions is not None:
        parts.append(f"step budget: {decisions_used}/{max_decisions} used")
    yaw = wrap180(o["yaw"])
    dimension = {0: "overworld", -1: "nether", 1: "end"}.get(
        o.get("dimension"), str(o.get("dimension", "unknown")))
    parts.append(f"dimension: {dimension}")
    parts.append(f"pos x={o['x']:.1f} y={o['y']:.1f} z={o['z']:.1f}  "
                 f"yaw={yaw:.0f}  pitch={o['pitch']:.0f} "
                 "(pitch -90=straight up, 0=horizon, +90=straight down)")
    parts.append("crosshair on: " + ch_name
                 + (f" ({ch['dist']:.1f} blocks away)" if ch["dist"] else ""))
    for kind, label in (("logs", "nearest log"),
                        ("coal", "nearest coal ore")):
        p = env.nearest(kind)
        if p:
            ry, rp, dist = env.rel_angles_to(*p)
            parts.append(f"{label}: turn yaw {ry:+.0f} pitch {rp:+.0f}, "
                         f"{dist:.1f} blocks away")
    parts.append("held: " + ITEM_NAMES.get(held_id,
                                            ID_NAMES.get(held_id,
                                                         str(held_id)))
                 + "   inventory: " + (", ".join(inv) if inv else "empty"))
    if gui_open:
        parts.append("inventory screen: OPEN (the mouse drives the slot "
                     "cursor, not the camera)")
    # every container the engine can put on screen gets a line, because
    # the latch opens for all of them and a screen the text does not name
    # reads as the player's own inventory
    if o["container"] == 1:
        parts.append("crafting table OPEN (3x3 recipes available)")
    elif o["container"] == 2:
        parts.append("furnace OPEN")
    elif o["container"] == 3:
        parts.append("chest OPEN")
    if show_camera:
        parts.append(f"view 12x32 (semantic camera, "
                     f"{CROSSHAIR_GLYPH} = crosshair):")
        parts.append(render_camera(o["cam"]))
        parts.append("legend: " + LEGEND)
    if perception:
        # Imported here rather than at module scope because perception.py
        # imports ID_NAMES from this module; one of the two directions has
        # to be deferred and this is the one that is taken once per turn
        # rather than once per import.
        from bedrock_rl.env.perception import perception_block
        parts.extend(perception_block(env))
    return "\n".join(parts)
