"""Netherite computer-use actions on a normalized [0,1000] screen.

The model sees the rendered frame and emits
generic computer actions on a normalized [0,1000] screen.

There is ONE mouse and ONE keyboard, and what they do depends on what is
on screen, as in the real game. Closed, the mouse is the camera and a
click attacks or uses. Open, it is the slot cursor, a left click takes or
drops a whole stack and a right click places ONE item. `key e` toggles
the screen, `key esc` closes it.

There is deliberately no semantic crafting verb and no verb for opening a
container either. A crafting table is made by clicking planks into the
2x2 grid, and a table's 3x3 grid is reached by right-clicking the placed
block, which is a use of the one mouse and not a new action.

A mouse_move pans the view so the named point becomes the new centre
(coord_to_angles), capped at the screen edge. That mapping is the exact
inverse of the camera ray the engine builds for that point, which means
it depends on the camera's CURRENT PITCH and not on the coordinate
alone: bedrock_rl/env/projection.py derives it and says what the two
independent arctangents that used to stand here were wrong by. GUI
coordinates use that same [0,1000] space on the 428x240 framebuffer the
engine hit-tests, at gui scale 1.

ACTIONS below is the single source of the vocabulary. It renders both the
prompt help and the OpenAI function schema used by every model adapter.
"""
import json
import re
import math

from bedrock_rl.env.actions import ProgramError
from bedrock_rl.env.observation import render_obs
from bedrock_rl.env.projection import (PITCH_LIMIT, angles_to_ndc,
                                         ndc_to_angles, tangents)
from bedrock_rl.env.episode import FRAME_H, FRAME_W, SCREEN
from bedrock_rl.env.gui_layout import clamp_to_panel

# The name the model calls and the tool registry executes.
TOOL_NAME = "computer"
NEXT_COMPUTER_MESSAGE = "Output your next computer action(s) (tool call)."
MAX_HOLD = 600                  # per-action tick cap (budget clamps too)
# A pan is spread over ticks at this rate instead of being applied in one
# or two big jumps. It is what makes the rendered motion read as a hand on
# a mouse rather than a teleport, and it makes the tick cost of a turn
# proportional to its angle: a 3-degree correction still costs one tick, a
# full-screen sweep costs nine.
PAN_DEG_PER_TICK = 6.0
# Movement holds are capped near five blocks. A single action that walks
# twenty blocks is not a plausible computer-use primitive, and reliability
# is flat across the usable range of step sizes, so the cap costs turn
# count and not capability. Mining is a long held attack and is exempt.
MAX_MOVE_HOLD = 30
MOVE_KEYS = ("w", "a", "s", "d", "space", "sneak", "sprint")

KEYMAP = {"w": {"forward": 1}, "s": {"forward": -1},
          "a": {"strafe": -1}, "d": {"strafe": 1},
          "space": {"jump": 1}, "sneak": {"sneak": 1},
          "sprint": {"sprint": 1}}
for _d in range(1, 10):
    KEYMAP[str(_d)] = {"hotbar": _d - 1}

# screen keys: not holds, they change WHICH surface the mouse acts on
GUI_KEYS = ("e", "esc", "escape")


# The RENDERED FRAME's projection, which is the surface the model names
# points on. It is not the semantic camera's: both are a 70-degree
# vertical pinhole, and the aspects are 428/240 here against 64/36 there,
# so the two TANX differ by 0.3 percent. The frame is what the model
# looks at, so the frame's aspect is the one a screen coordinate has to
# be read through.
SCREEN_TANX, SCREEN_TANY = tangents(FRAME_W, FRAME_H)


def coord_to_angles(x, y, pitch):
    """Normalized screen coord [0,1000] -> (dyaw, dpitch) degrees that
    put that point at the centre of the screen.

    y > 500 looks DOWN (pitch +), matching the screen-y-down convention;
    the engine's image plane is y-UP, which is the sign flip below.

    `pitch` is the camera's pitch NOW and is not optional. The engine
    normalizes the whole ray and tilts the up vector under pitch
    (bedrock_rl/env/projection.py), so the same screen point names a
    different turn at every pitch. Treating the two axes as separable
    arctangents, which this did until it was measured against the
    engine, is 11.3 degrees wrong at a corner even at pitch 0.
    """
    return ndc_to_angles(float(x) / 500.0 - 1.0, 1.0 - float(y) / 500.0,
                         pitch, SCREEN_TANX, SCREEN_TANY)


def angles_to_coord(dyaw, dpitch, pitch):
    """(dyaw, dpitch) degrees -> integer screen coord, clamped to the
    screen edge. A clamped move under-turns; callers iterate (see
    moves_for_angles)."""
    nx, ny = angles_to_ndc(max(-PITCH_LIMIT, min(PITCH_LIMIT, float(dyaw))),
                           dpitch, pitch, SCREEN_TANX, SCREEN_TANY)
    x = 500.0 * (1.0 + nx)
    y = 500.0 * (1.0 - ny)
    return (int(round(max(0.0, min(SCREEN, x)))),
            int(round(max(0.0, min(SCREEN, y)))))


def moves_for_angles(dyaw, dpitch, pitch, tol=0.75, max_moves=8):
    """Oracle helper: mouse_move action list achieving a (dyaw, dpitch)
    pan from a camera currently at `pitch`, split into several moves when
    the target lies off-screen.

    The pitch is carried through the loop because each move changes it,
    and the move after it is read through the projection at the pitch it
    will actually be issued from.
    """
    pitch = float(pitch)
    out = []
    while (abs(dyaw) > tol or abs(dpitch) > tol) and len(out) < max_moves:
        x, y = angles_to_coord(dyaw, dpitch, pitch)
        ady, adp = coord_to_angles(x, y, pitch)
        if abs(ady) < 1e-6 and abs(adp) < 1e-6:
            break
        out.append({"action": "mouse_move", "coordinate": [x, y]})
        dyaw -= ady
        dpitch -= adp
        pitch += adp
    return out


def _ticks(a, default=None, hi=MAX_HOLD):
    t = a.get("ticks", default)
    if t is None:
        raise ProgramError(f"missing ticks: {a}")
    try:
        t = int(t)
    # OverflowError is the one that is easy to miss, and it is reachable
    # from the wire. JSON accepts the bare literals `Infinity`,
    # `-Infinity` and `NaN`, so a policy can put a non-finite float in
    # this field; int() of a NaN raises ValueError but int() of an
    # infinity raises OverflowError, which is not a subclass of either of
    # the other two. Uncaught, it leaves this compiler as something that
    # is not a ProgramError, so the episode would not charge the standard
    # malformed-turn penalty. Every way a value can fail to be a tick count
    # costs the same penalty and nothing else.
    except (TypeError, ValueError, OverflowError):
        raise ProgramError(f"bad ticks: {a}")
    if not 1 <= t <= hi:
        raise ProgramError(f"ticks out of range [1,{hi}]: {a}")
    return t


def _coord(a):
    c = a.get("coordinate")
    if not isinstance(c, (list, tuple)) or len(c) != 2:
        raise ProgramError(f"needs coordinate [x,y]: {a}")
    try:
        x, y = float(c[0]), float(c[1])
    # OverflowError for the same reason it is caught in _ticks, arriving
    # from the other direction: JSON integers have no width, and float()
    # of an integer too large to represent raises it. A non-finite float
    # needs no special case here, since it fails the range test below.
    except (TypeError, ValueError, OverflowError):
        raise ProgramError(f"bad coordinate: {a}")
    if not (0 <= x <= SCREEN and 0 <= y <= SCREEN):
        raise ProgramError(f"coordinate outside [0,1000]: {a}")
    return x, y


def _pan(x, y, pitch):
    """Camera pan to center a screen point, as 1-tick schedule items,
    plus the pitch the camera is left at.

    Interpolated at PAN_DEG_PER_TICK so the turn is continuous. The pitch
    returned is the SUM OF THE ROUNDED sub-steps rather than pitch+dp, so
    the compiler's model of the camera is what the engine will actually
    hold after running the schedule and not a third of a degree off it.
    """
    dy, dp = coord_to_angles(x, y, pitch)
    nsub = max(1, math.ceil(max(abs(dy), abs(dp)) / PAN_DEG_PER_TICK))
    step = round(dp / nsub, 3)
    return ([({"dyaw": round(dy / nsub, 3), "dpitch": step}, 1)
             for _ in range(nsub)],
            max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch + step * nsub)))


def _cursor(x, y):
    """Absolute pointer placement, in the framebuffer px the engine
    hit-tests. The engine keeps the pointer where it was left, so a click
    with no coordinate lands wherever the last move put it.

    Clamped into the container panel, which is why a click can no longer
    throw the carried stack on the floor; see
    bedrock_rl/env/gui_layout.py::clamp_to_panel for the whole story.
    """
    px, py = clamp_to_panel(round(x / SCREEN * (FRAME_W - 1)),
                            round(y / SCREEN * (FRAME_H - 1)))
    return {"gui_cursor_x": px, "gui_cursor_y": py}


def compile_actions(actions, pitch, max_actions=12, max_ticks=2000,
                    gui_open=False):
    """JSON actions -> ([(action_key_dict, n_ticks), ...], n_actions), the
    schedule shape Episode.step executes. Raises ProgramError on anything
    malformed (the -0.2 turn-penalty path).

    `pitch` is the camera's pitch in degrees at the START of this call.
    It has no default because a screen coordinate names a different turn
    at every pitch (bedrock_rl/env/projection.py), and a default would
    be a silent claim that the camera is level. A list can hold several
    pans, so the pitch is advanced across the list from the deltas the
    schedule will actually apply; nothing else in the vocabulary moves
    the camera.

    gui_open is the inventory-screen state at the START of this call and
    selects which meaning the mouse actions compile to; `key e` flips it
    mid-list. Episode owns that latch and reads it back off the engine
    after every call (Episode._sync_screen), because a right mouse on a
    placed table or furnace opens a screen the action stream did not ask
    for. It is the state at the start of the call and not a running one,
    so a container opened HERE drives the mouse from the next call
    onward, which is what the schema text tells the model."""
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except (ValueError, TypeError) as e:
            raise ProgramError(f"actions not valid JSON: {e}")
    if isinstance(actions, dict):
        actions = [actions]
    if not isinstance(actions, list) or not actions:
        raise ProgramError("actions must be a JSON action object or a "
                           "non-empty list of them")
    if len(actions) > max_actions:
        raise ProgramError(f"too many actions (max {max_actions})")
    sched = []
    total = 0
    gui = bool(gui_open)
    pitch = float(pitch)
    for a in actions:
        if not isinstance(a, dict):
            raise ProgramError(f"action not an object: {a!r}")
        name = a.get("action")
        if name == "mouse_move":
            x, y = _coord(a)
            if gui:
                items = [(_cursor(x, y), 1)]
            else:
                items, pitch = _pan(x, y, pitch)
            sched += items
            total += len(items)
        elif name in ("left_click", "right_click", "left_click_hold"):
            button = 1 if name.startswith("left") else 2
            hold = _ticks(a) if name == "left_click_hold" else 1
            keys = {}
            sneak = a.get("sneak", False)
            if not isinstance(sneak, bool):
                raise ProgramError(f"click sneak must be true or false: {a}")
            if sneak and gui:
                raise ProgramError(f"click sneak only applies in world: {a}")
            if a.get("coordinate") is not None:
                x, y = _coord(a)
                if gui:
                    keys.update(_cursor(x, y))       # move+click, one tick
                else:
                    pan, pitch = _pan(x, y, pitch)
                    sched += pan
                    total += len(pan)
            if gui:
                keys["gui_click"] = button
                sched.append((keys, 1))
                total += 1
                if hold > 1:
                    # holding a mouse button down over a slot does nothing
                    # beyond the first click; spend the rest as idle ticks
                    # so the tick accounting still matches what was asked
                    sched.append(({}, hold - 1))
                    total += hold - 1
            else:
                keys["attack" if button == 1 else "use"] = 1
                if sneak:
                    keys["sneak"] = 1
                sched.append((keys, hold))
                total += hold
        elif name == "key":
            ks = a.get("k", a.get("key"))
            if isinstance(ks, (str, int)):
                ks = [ks]
            if not isinstance(ks, list) or not ks:
                raise ProgramError(f"key needs k: {a}")
            ks = [str(k).lower() for k in ks]
            if any(k in GUI_KEYS for k in ks):
                if len(ks) != 1:
                    raise ProgramError(f"e/esc cannot be held with other "
                                       f"keys: {a}")
                if ks[0] == "e" and not gui:
                    sched.append(({"gui_open": 1}, 1))
                    gui = True
                elif gui:
                    sched.append(({"gui_open": 0}, 1))
                    gui = False
                else:
                    sched.append(({}, 1))        # esc with nothing open
                total += 1
            else:
                keys = {}
                for k in ks:
                    m = KEYMAP.get(k)
                    if m is None:
                        raise ProgramError(
                            f"unknown key {k!r} (w|a|s|d|space|sneak|"
                            "sprint|e|esc|1-9)")
                    keys.update(m)
                cap = (MAX_MOVE_HOLD if any(k in MOVE_KEYS for k in ks)
                       else MAX_HOLD)
                t = _ticks(a, default=1, hi=cap)
                sched.append((keys, t))
                total += t
        elif name == "wait":
            t = _ticks(a, default=1)
            sched.append(({}, t))
            total += t
        else:
            raise ProgramError(f"unknown action {name!r}")
        if total > max_ticks:
            raise ProgramError("actions exceed tick budget")
    return sched, len(actions)


def cu_payload(args):
    """A `computer` call's arguments -> the action payload as the JSON text
    compile_actions takes.

    This is the one reader of the call shape. Every live harness passes
    through it before ``Episode.step``, so tool adapters do not grow their
    own subtly different argument parsers.

    It never returns None, so nothing downstream drops a call for the
    shape of its payload. A block that names this tool IS a turn; whether
    its contents compile is the action compiler's business and costs the
    malformed-turn penalty.

      arguments as a JSON STRING is parsed, and as a LIST or anything
      else that is not a mapping it is a turn with no payload.
      arguments with no payload returns "", which the compiler rejects
      and charges 0.2 for.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return ""
    if not isinstance(args, dict):
        return ""
    acts = args.get("actions")
    if acts is None and "action" in args:
        acts = args                        # tolerate a bare action object
    if acts is None:
        return ""
    return acts if isinstance(acts, str) else json.dumps(acts)


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def tool_call_attempts(text):
    """Every decoded tool envelope, including malformed attempts.

    Text-only rollout adapters do not have Hermes' structured parser. Dropping
    an invalid or unknown envelope there made malformed calls look identical to
    a voluntary stop, while the verl adapter charged the episode. Return a
    three-tuple for every attempt so all adapters can apply the same accounting.
    """
    text = text or ""
    out = []
    matches = list(TOOL_CALL_RE.finditer(text))
    for match in matches:
        try:
            value = json.loads(match.group(1))
            if not isinstance(value, dict):
                raise TypeError("tool call body must be a JSON object")
            name = value["name"]
            if not isinstance(name, str) or not name:
                raise TypeError("tool call name must be a non-empty string")
            out.append((name, value["arguments"], None))
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            out.append((TOOL_NAME, {}, f"malformed tool envelope: {exc}"))
    if not matches and "<tool_call>" in text:
        out.append((TOOL_NAME, {}, "malformed tool envelope: missing </tool_call>"))
    return out


def tool_calls(text):
    """Parse the decoded-text tool envelope used by local samplers.

    New harnesses exchange structured OpenAI ``tool_calls`` and do not need
    this compatibility parser. Keeping it beside the action language avoids
    coupling execution to a particular reward implementation.
    """
    return [(name, arguments)
            for name, arguments, error in tool_call_attempts(text)
            if error is None]


def cu_action_payloads(text):
    """All decoded ``computer`` payloads in emission order."""
    return [cu_payload(arguments) for name, arguments in tool_calls(text)
            if name == TOOL_NAME]


def cu_tool_call(actions):
    """An action list -> exactly the text a policy has to emit for it.

    The other half of cu_payload. This is the wire format
    reward/__init__.py::TOOL_CALL_RE has to match and cu_payload has to
    read, so writing it anywhere else is writing half of a protocol whose
    other half lives here: the synthetic trainer rows, the SFT
    answers and scripted producers all produce a string that has to
    score, and a stray space would make it score -1.0 for carrying no
    parseable call.
    """
    return ("<tool_call>\n"
            + json.dumps({"name": TOOL_NAME, "arguments": {"actions": actions}})
            + "\n</tool_call>")


def cu_parser(task):
    """Parser hook for Episode: compile `computer` tool-call arguments
    (JSON text or already-decoded structure) into an action schedule.
    Episode passes the live inventory-screen state as gui_open and the
    live camera pitch as pitch, both read off the engine at the moment
    the call arrives."""
    def parse(text, gui_open, pitch):
        return compile_actions(text, pitch, max_actions=task.max_lines,
                               max_ticks=task.max_ticks_per_turn,
                               gui_open=gui_open)
    return parse


# ── prompt text ──────────────────────────────────────────────────────────
# ONE action-vocabulary table drives the prompt and the `computer` schema
# generated by adapters.netherite.chat.tool_schema. compile_actions() above is the
# executable counterpart; keep those surfaces in sync when adding an action.
ACTIONS = [
    ('{"action":"mouse_move","coordinate":[x,y]}',
     "move the mouse to that screen point: with the inventory CLOSED the "
     "camera turns so the point becomes the new center (from a level "
     "view one move pans at most ~51 deg left/right and ~35 deg up/down, "
     "less from a corner and different again when already looking up or "
     "down - repeat to turn farther; y<500 looks UP, y>500 looks DOWN); "
     "with the inventory OPEN the pointer moves there, staying inside "
     "the panel, so a point outside it puts the pointer on the nearest "
     "panel edge"),
    ('{"action":"left_click","coordinate":[x,y]}',
     "left mouse button (coordinate optional: without it the pointer "
     "stays put). Inventory CLOSED: attack/hit once. Inventory OPEN: pick "
     "up the whole stack in the slot under the pointer, or put the "
     "carried stack into it; a click away from the slots does nothing, "
     "and nothing you carry can be thrown away"),
    ('{"action":"left_click_hold","ticks":n}',
     "hold the left button n game ticks (mining takes ~30-160 ticks; keep "
     "the crosshair on the block). In the inventory this is one click"),
    ('{"action":"right_click","coordinate":[x,y]}',
     "right mouse button (coordinate optional). Inventory CLOSED: "
     "use/place, and on a crafting table or furnace standing in the "
     "world this OPENS that block's screen, which is how you reach the "
     "3x3 grid and the furnace. The screen is open from your NEXT call "
     "onward, so end the call after the right click. Inventory OPEN: "
     "place ONE carried item into the slot under the pointer, or pick "
     "up half a stack. Set `sneak:true` on a world right-click to place "
     "against an interactive block without opening it"),
    ('{"action":"key","k":"w","ticks":n}',
     "hold keys n ticks: w=forward s=back a/d=strafe space=jump, "
     "sneak=crouch, sprint=run, digits "
     "1-9 select a hotbar slot; k may be a list to hold several at once "
     '(e.g. ["w","space"] jump-walks and ["w","sneak"] edge-walks; '
     '4 ticks of w cover ~0.65 blocks). '
     f"Movement holds are capped at {MAX_MOVE_HOLD} ticks, about 5 "
     "blocks, so cross a long distance over several calls. "
     'k="e" opens your inventory screen and closes whichever screen is '
     'open, a crafting table or furnace included; k="esc" closes '
     "whichever screen is open"),
    ('{"action":"wait","ticks":n}', "let the game run n ticks"),
]


def cu_help(max_actions, max_ticks=None):
    limits = f"max {max_actions} per call"
    if max_ticks is not None:
        limits += f", max {max_ticks} total game ticks per call"
    lines = [
        "You control the game through a computer-use interface: you see "
        "the rendered screen and act with the mouse and keyboard via the "
        "`computer` tool. Screen coordinates are normalized to [0,1000] "
        "on both axes; [500,500] is the crosshair at screen center.",
        "There is one mouse and one keyboard, and what they do depends on "
        "what is on screen. With the inventory screen open the mouse "
        "drives the pointer over the slots instead of the camera, and "
        "clicks move items. Everything you craft, you craft by clicking "
        "items into the crafting grid.",
        "The 2x2 grid on your own inventory screen is the small one. A "
        "bigger recipe needs a crafting table standing in the world: "
        "select it in the hotbar, right click the ground to place it, "
        "then right click the placed block and its 3x3 screen opens, "
        "with the same mouse and the same clicks. A furnace opens the "
        "same way. The screen is open from your NEXT call onward, so end "
        "the call after the right click. The observation says which "
        "screen is open, `key e` and `key esc` close whichever one is, "
        "and walking more than six blocks from the block closes it too.",
        'Actions (pass one object or a list under "actions"; they run in '
        f"order, {limits}):"]
    lines += [f"  {ex}  {desc}" for ex, desc in ACTIONS]
    return "\n".join(lines)


# The concrete example teaches the envelope and common action shapes. It must
# remain valid at the minimum action and tick budgets because small tasks
# otherwise invite the model to copy a call that their parser rejects.
_CU_EXAMPLE_ACTIONS = (
    {"action": "key", "k": "w", "ticks": 1},
    {"action": "key", "k": "e"},
    {"action": "left_click", "coordinate": [500, 500]},
    {"action": "right_click", "coordinate": [600, 300]},
)
_CU_CLOSED_LOOP = (
    "After every call you receive a fresh screenshot and text observation."
    " Keep calling `computer` until the task is complete.")


def cu_suffix(max_actions=None):
    """The tool-call example appended to the first user message. One copy,
    so it cannot drift out of the action vocabulary compile_actions
    accepts."""
    limit = len(_CU_EXAMPLE_ACTIONS) if max_actions is None else max(
        1, min(int(max_actions), len(_CU_EXAMPLE_ACTIONS)))
    call = {"name": TOOL_NAME,
            "arguments": {"actions": list(_CU_EXAMPLE_ACTIONS[:limit])}}
    example = ("\n\nYou control the game ONLY by calling the `computer` tool. "
               "Emit a tool call in exactly this format:\n<tool_call>\n"
               + json.dumps(call) + "\n</tool_call>\n")
    return example + _CU_CLOSED_LOOP


def validate_cu_example(task):
    """Fail when the canonical few-shot call violates a task's budgets."""
    limit = max(1, min(int(task.max_lines), len(_CU_EXAMPLE_ACTIONS)))
    actions = list(_CU_EXAMPLE_ACTIONS[:limit])
    compile_actions(
        actions, pitch=0, max_actions=task.max_lines,
        max_ticks=task.max_ticks_per_turn, gui_open=False)


def cu_prompt(task, env, gui_open=False):
    goal = str(task.raw.get("goal_cu") or task.goal).strip()
    return ("TASK: " + goal + "\n\n"
            + cu_help(task.max_lines, task.max_ticks_per_turn)
            + "\n\n=== CURRENT OBSERVATION ===\n"
            + render_obs(env, gui_open=gui_open,
                         perception=task.perception)
            + "\n\nCall the `computer` tool now:")


def cu_user_prompt(task, env):
    """The whole first user message of an episode, image marker included.

    Single-sourced because every consumer has to produce the same bytes:
    the RL prompt rows, the synthetic SFT rows, and any evaluation driver.
    The measured failure mode when they drift is that the SFT imprint
    lands on a prompt the policy never sees at rollout time and does not
    transfer at all.
    """
    return "<image>\n" + cu_prompt(task, env) + cu_suffix(task.max_lines)
