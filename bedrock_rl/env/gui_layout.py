"""Container-screen slot geometry and the clicks that drive it. Where every
slot is and which computer action lands on it, nothing else.

The engine hit-tests slots in FRAMEBUFFER pixels (gm_screen_slot_at) on
a panel gm_screen_layout lays out at gui scale max(1, h/240). Episodes run
at 428x240 and MagmaEnv declares that size to every process it starts,
including processes with rendering disabled, so one table of centres
serves them all. magma/game/test_screen.c asserts the same coordinates
on the C side, so the two cannot drift apart.

Getting this wrong is silent: a driver on another framebuffer clicks the
wrong slots. `Layout` therefore computes centres for any framebuffer size.

Solvers depend on vanilla click semantics. A left click on a stack
picks it whole onto the cursor, a right click on a cell places ONE item,
a left click on the result takes the output, and closing the screen
returns the grid and cursor stacks.

The click builders below are the only place naming an action for a
container screen, so a vocabulary change lands only here.
"""
from bedrock_rl.env.episode import FRAME_H, FRAME_W, SCREEN

GUI_FB_W, GUI_FB_H = FRAME_W, FRAME_H


class Layout:
    """Slot centres AND panel bounds for any framebuffer size, mirroring
    gm_screen_layout.

    The module constants below are this class at the frame size episodes
    run, which is the 428x240 the demos and the C test pin. Anything
    capturing at a different size (the README GIF renders larger so the
    screens are legible) gets its geometry from here, because the virtual
    mouse is in framebuffer pixels and the panel scales with fb_h.

    ONE class, and everything else derived from it. The panel origin used
    to be a literal beside the derived centres, so at a raised frame scale
    the centres moved and the box did not: every compiled click was
    dragged back into a rectangle the panel had left, which lands on
    another slot rather than failing."""

    def __init__(self, fb_w=GUI_FB_W, fb_h=GUI_FB_H):
        self.fb_w, self.fb_h = fb_w, fb_h
        s = max(1, fb_h // 240)
        self.scale = s
        gw = -(-fb_w // s)
        gh = -(-fb_h // s)
        self.px = (gw - 176) // 2 * s
        self.half = 8 * s
        self._s = s
        self._gh = gh

    def _origin_y(self, container):
        ph = 168 if container == 3 else 166
        return (self._gh - ph) // 2 * self._s

    def at(self, gx, gy, container=0):
        """Centre of the slot whose vanilla gui-space origin is (gx,gy)."""
        s = self._s
        return (self.px + gx * s + self.half,
                self._origin_y(container) + gy * s + self.half)

    def hotbar(self, i, container=0):
        return self.at(8 + i * 18, 143 if container == 3 else 142, container)

    def main(self, j, container=0):
        return self.at(8 + (j % 9) * 18,
                       (85 if container == 3 else 84) + (j // 9) * 18,
                       container)

    def grid_2x2(self, i):
        return self.at(98 + (i % 2) * 18, 18 + (i // 2) * 18, 0)

    def result_2x2(self):
        return self.at(154, 28, 0)

    def grid_3x3(self, i):
        return self.at(30 + (i % 3) * 18, 17 + (i // 3) * 18, 1)

    def result_3x3(self):
        return self.at(124, 35, 1)

    def furnace(self, slot):
        """Centre of a furnace slot, by INDEX 0/1/2 or by slot id 46/47/48.

        The module tables are keyed by slot id and this method was keyed
        by index, so the same number meant two things."""
        i = slot - 46 if slot >= 46 else slot
        if i not in (0, 1, 2):
            raise KeyError(f"furnace slot {slot} is 0..2 or 46..48")
        return self.at(56 if i < 2 else 116,
                       17 if i == 0 else 53 if i == 1 else 35, 2)

    def panel_size(self, container=0):
        """(width, height) of the panel in framebuffer pixels."""
        return 176 * self._s, (168 if container == 3 else 166) * self._s

    def panel_box(self, container=0):
        """Inclusive framebuffer bounds of the panel: the region in which
        gm_screen_slot_at answers with a slot rather than GMC_OUTSIDE."""
        w, h = self.panel_size(container)
        y0 = self._origin_y(container)
        return self.px, y0, self.px + w - 1, y0 + h - 1


# The panel this process's frames are laid out on. Derived and not
# written down: the engine lays it out at gui scale max(1, h/240), so at
# a raised NETHERITE_FRAME_SCALE the panel moves and grows with the slot
# centres, and a box that stayed at the 1x rectangle would clamp real
# slots out of reach. At 428x240 these are x 126..301 and y 37..202, the
# numbers verified against the engine by bisection and read from
# gm_screen_layout.
#
# The chest screen is two pixels taller per gui unit and its extra rows
# carry no slot, so the player-screen bounds block nothing on any screen.
_L = Layout(GUI_FB_W, GUI_FB_H)
PANEL_BOX = _L.panel_box()


def clamp_to_panel(px, py):
    """Clamp a framebuffer point to the interactive inventory panel.

    Vanilla interprets an outside click as dropping the carried stack, which is
    outside this action space. Clamping produces a visible no-op at the edge
    that the policy can retry and ensures coordinate-free clicks cannot act on
    an out-of-bounds cursor.
    """
    x0, y0, x1, y1 = PANEL_BOX
    return min(max(int(px), x0), x1), min(max(int(py), y0), y1)

# Slot centres are DERIVED from the framebuffer at the bottom of this module
# rather than written out here, for the same reason the box above is: the
# engine lays the panel out at gui scale max(1, h/240) and a recording that
# renders at 2x or 3x has to click the scaled centres. At 428x240 `Layout`
# reproduces the literal table these lines used to hold, entry for entry,
# which magma/game/test_screen.c also pins.

# Cell order a solver fills.
CELLS_2X2 = (36, 37, 39, 40)
CELLS_3X3 = tuple(range(36, 45))
RESULT_SLOT = 45

CRAFTING_TABLE_BLOCK = 58
CONTAINER_PLAYER = 0
CONTAINER_TABLE = 1
CONTAINER_FURNACE = 2
CONTAINER_CHEST = 3


def slot_xy(slot, screen="player"):
    """Framebuffer pixel centre of a slot id on a given screen."""
    if screen not in SCREENS:
        raise KeyError(f"unknown screen {screen!r}, known screens are "
                       f"{sorted(SCREENS)}")
    for t in SCREENS[screen]:
        if slot in t:
            return t[slot]
    raise KeyError(f"slot {slot} is not on the {screen} screen")


def to_coord(px, py):
    """Framebuffer pixel -> the normalized [0,1000] screen the `computer`
    tool takes, as a list.

    compile_actions maps back with x/1000*(FRAME_W-1), so this is the
    exact inverse. It is the ONE direction of that conversion; there is no
    second copy of it in the action layer."""
    return [int(round(px / (FRAME_W - 1) * SCREEN)),
            int(round(py / (FRAME_H - 1) * SCREEN))]


# ── the clicks ───────────────────────────────────────────────────────────
def open_screen():
    """Open the inventory screen over whatever container is current."""
    return [{"action": "key", "k": "e", "ticks": 1}]


def close_screen():
    """Close it. Must go through the engine's close path or the grid and
    cursor stacks leak."""
    return [{"action": "key", "k": "e", "ticks": 1}]


def click_slot(slot, button="left", screen="player"):
    """One click on one slot: ONE action, because the move and the button
    press compile to a single schedule item in this vocabulary. The turn
    cutter counts actions, so that ratio matters to `max_actions`."""
    name = "left_click" if button == "left" else "right_click"
    return [{"action": name, "coordinate": to_coord(*slot_xy(slot, screen))}]


def select_hotbar(slot):
    """Hold the digit that selects a hotbar slot, so the held item is the
    one a place or a mine will use."""
    return [{"action": "key", "k": str(slot + 1), "ticks": 1}]


def fill_cells(src_slot, cells, screen="player", put_back=True):
    """Pick the ingredient stack up and right-click ONE item into each of
    `cells`, then put the remainder back.

    Putting it back is not tidiness. Taking a craft result needs the
    cursor empty or holding the same item, so a cursor still carrying
    leftovers makes the result click a silent no-op and the craft never
    happens. Pass put_back=False only when the stack was exactly consumed.
    """
    out = list(click_slot(src_slot, "left", screen))
    for c in cells:
        out += click_slot(c, "right", screen)
    if put_back:
        out += click_slot(src_slot, "left", screen)
    return out


def take_result(screen="player"):
    """Left-click the result slot, which is what moves the crafted output
    onto the cursor. Closing the screen is what lands it in inventory."""
    return click_slot(RESULT_SLOT, "left", screen)


def find_stack(env, item_ids, count=1, metas=None):
    """Hotbar slot id holding at least `count` of any of `item_ids`.

    Only the hotbar is searchable. The observation record carries the nine
    hotbar slots and no main-inventory contents, so a stack that overflowed
    into the backpack is invisible and the caller must fail loudly rather
    than click a slot it guessed.
    """
    ids = set(item_ids)
    wanted_meta = None if metas is None else set(metas)
    hotbar_meta = env.obs.get("hotbar_meta")
    for i, (bid, cnt) in enumerate(zip(env.obs["hotbar_ids"],
                                       env.obs["hotbar_counts"])):
        meta_matches = (wanted_meta is None
                        or hotbar_meta is None
                        or hotbar_meta[i] in wanted_meta)
        if bid in ids and cnt >= count and meta_matches:
            return i
    return None


# ── derived slot tables ───────────────────────────────────────
# _L, and so the panel box, is built with the frame constants above.
HOTBAR = {i: _L.hotbar(i) for i in range(9)}
MAIN_INV = {9 + j: _L.main(j) for j in range(27)}
# player inventory screen: 2x2 grid occupies ids 36, 37, 39, 40
GRID_2X2 = dict(zip((36, 37, 39, 40), (_L.grid_2x2(i) for i in range(4))))
RESULT_2X2 = {45: _L.result_2x2()}
# crafting-table screen: the full row-major 3x3, ids 36..44
GRID_3X3 = {36 + i: _L.grid_3x3(i) for i in range(9)}
RESULT_3X3 = {45: _L.result_3x3()}
# furnace screen: input, fuel, output
FURNACE = {46 + i: _L.furnace(46 + i) for i in range(3)}
SCREENS = {"player": (HOTBAR, MAIN_INV, GRID_2X2, RESULT_2X2),
           "table": (HOTBAR, MAIN_INV, GRID_3X3, RESULT_3X3),
           "furnace": (HOTBAR, MAIN_INV, FURNACE)}
