"""The event journal: what happened, in order, and the checks that read it.

The observation record is a SNAPSHOT, so it cannot say whether a crafting
table in the inventory was crafted or picked up, it has no notion of
order, and it reports displacement rather than distance travelled.

The journal is the other half. The engine emits a typed record for each
state transition it performs, stamped with the tick and in order, on its
own file descriptor (the C side is magma/game/rl_journal.h). This module
decodes it and turns each event type into a task check.

EVENTS below is the registry and the check name IS the event name. There
is no per-event decoder, check function or registration, because
`register_checks` walks the table. Adding an event type is one row here
and one row in the engine's enum, same code, same slot names.

A misspelled constraint key is an error at task load, not a check that
quietly matches everything.

Ordering is the `sequence` check, which is why the journal is a sequence
and not a counter. Each step must be satisfied by events strictly after
those that satisfied the step before. `all` requires every listed milestone
without imposing order, while `any` holds after one listed event occurs.

Task checks consume these typed events directly.
"""
import os
import struct
from decimal import Decimal, InvalidOperation

# engine.py imports THIS module, so the name tables it owns are imported
# inside _values() rather than here. That call happens once per
# constraint at task load, never on the per-turn path.

# Stream framing. The header is written once when the engine arms the
# journal; the records follow, fixed width, little-endian.
MAGIC = 0x4A4C524E                     # "NRLJ"
# Event rows are additive within this fixed 32-byte framing. ``bucket_target``
# therefore keeps wire version 1; changing the header or record layout would
# require a version bump instead.
VERSION = 1
_HEAD = struct.Struct("<IHH")
_REC = struct.Struct("<HHi6i")
REC_BYTES = _REC.size                  # 32
HEAD_BYTES = _HEAD.size                # 8

# ``player_fluid_state.flags``. These are independent bits: the collision
# volume, eye/near-plane and first-person overlay can cross a boundary on
# different ticks. Public names let artifact validators choose the proof they
# need without copying wire numbers.
FLUID_BODY = 1 << 0
FLUID_VIEWPOINT = 1 << 1
FLUID_OVERLAY = 1 << 2

# ``click_target.face``: adjacent placement cell relative to the hit block.
CLICK_FACES = {
    1: (-1, 0, 0), 2: (1, 0, 0),
    3: (0, -1, 0), 4: (0, 1, 0),
    5: (0, 0, -1), 6: (0, 0, 1),
}


class EventSpec:
    """One row of the event table.

    code     the type id the C enum gives it
    fields   the six payload slot names, None where the slot is unused
    resolve  slot -> how a YAML name becomes ids ("blocks" or "items")
    scale    slot -> multiplier applied before a numeric comparison, so a
             task writes `max_dist: 4.0` in blocks while the wire carries
             quarter-blocks
    flag_field  optional task-facing name for the record header's existing
                16-bit flags word
    """

    __slots__ = ("code", "fields", "resolve", "scale", "flag_field")

    def __init__(self, code, fields, resolve=None, scale=None,
                 flag_field=None):
        self.code = code
        self.fields = tuple(fields)
        self.resolve = resolve or {}
        self.scale = scale or {}
        self.flag_field = flag_field


def _E(code, fields, resolve=None, scale=None, flag_field=None):
    return EventSpec(code, fields, resolve, scale, flag_field)


_BLOCK = "blocks"
_ITEM = "items"

# ── THE EVENT TABLE ─────────────────────────────────────────────────────
# Mirrors the enum in magma/game/rl_journal.h. The slot names here are
# the names a task YAML constrains, so renaming one is a task-facing
# change.
EVENTS = {
    "episode": _E(1, ("index", None, None, None, None, None)),
    "block_broken": _E(
        2, ("block", "x", "y", "z", "tool", "drop"),
        {"block": _BLOCK, "tool": _ITEM, "drop": _ITEM},
        flag_field="meta"),
    "block_placed": _E(
        3, ("block", "x", "y", "z", "item", "slot"),
        {"block": _BLOCK, "item": _ITEM}, flag_field="meta"),
    "item_picked_up": _E(
        4, ("item", "count", "meta", "x", "y", "z"), {"item": _ITEM}),
    "item_dropped": _E(
        5, ("item", "count", "meta", "cause", None, None), {"item": _ITEM}),
    "item_crafted": _E(
        6, ("item", "count", "meta", "width", "screen", None),
        {"item": _ITEM}),
    "item_smelted": _E(
        7, ("item", "count", "meta", "input", None, None),
        {"item": _ITEM, "input": _ITEM}),
    "container_opened": _E(8, ("kind", "x", "y", "z", None, None)),
    "container_closed": _E(9, ("kind", "x", "y", "z", None, None)),
    "slot_changed": _E(
        10, ("slot", "item", "count", "was_item", "was_count", "meta"),
        {"item": _ITEM, "was_item": _ITEM}),
    "cursor_changed": _E(
        11, ("item", "count", "meta", "was_item", "was_count", None),
        {"item": _ITEM, "was_item": _ITEM}),
    "hotbar_selected": _E(
        12, ("slot", "item", None, "was_slot", None, None), {"item": _ITEM}),
    "aimed_at": _E(
        13, ("block", "dist", None, "was_block", None, None),
        {"block": _BLOCK, "was_block": _BLOCK},
        # the wire carries the observation's own u8 depth, dist*4
        {"dist": 0.25}),
    "travelled": _E(
        14, ("blocks", "mm", None, None, None, None), None, {"mm": 0.001}),
    "vitals_changed": _E(
        15, ("health", "food", None, "was_health", "was_food", None)),
    "died": _E(16, (None, None, None, None, None, None)),
    "dimension_changed": _E(
        17, ("dimension", "was_dimension", None, None, None, None)),
    "click_target": _E(
        18, ("block", "x", "y", "z", "was_block", None),
        {"block": _BLOCK, "was_block": _BLOCK}, flag_field="face"),
    "player_fluid_state": _E(
        19, ("block", "x", "y", "z", "was_block", "was_flags"),
        {"block": _BLOCK, "was_block": _BLOCK}, flag_field="flags"),
    "bucket_target": _E(
        20, ("block", "x", "y", "z", "meta", "slot"),
        {"block": _BLOCK}),
    "overflow": _E(63, ("dropped", None, None, None, None, None)),
}

_BY_CODE = {s.code: (name, s) for name, s in EVENTS.items()}

# Container kinds, as the engine numbers them (game/container_live.h).
CONTAINER_KINDS = {"player": 0, "crafting_table": 1, "furnace": 2,
                   "chest": 3}
# item_dropped causes (game/rl_journal.h).
DROP_CAUSES = {"harvest": 0, "toss": 1, "container": 2}
# item_crafted screens.
CRAFT_SCREENS = {"gui": 0, "primitive": 1}
_ENUMS = {"kind": CONTAINER_KINDS, "cause": DROP_CAUSES,
          "screen": CRAFT_SCREENS}


class Event:
    """One decoded record. Slots are reachable by name."""

    __slots__ = ("name", "tick", "v", "spec", "flags")

    def __init__(self, name, spec, tick, v, flags=0):
        self.name = name
        self.spec = spec
        self.tick = tick
        self.v = v
        self.flags = int(flags)

    def __getitem__(self, field):
        if field == self.spec.flag_field:
            return self.flags
        try:
            return self.v[self.spec.fields.index(field)]
        except ValueError:
            raise KeyError(f"{self.name} has no slot {field!r}; its slots "
                           f"are {self.field_names()}")

    def field_names(self):
        names = [field for field in self.spec.fields if field]
        if self.spec.flag_field:
            names.append(self.spec.flag_field)
        return names

    def get(self, field, default=None):
        try:
            return self[field]
        except KeyError:
            return default

    def as_dict(self):
        d = {"event": self.name, "tick": self.tick}
        d.update({f: self.v[i] for i, f in enumerate(self.spec.fields) if f})
        if self.spec.flag_field:
            d[self.spec.flag_field] = self.flags
        return d

    def __repr__(self):
        body = " ".join(f"{f}={self.v[i]}"
                        for i, f in enumerate(self.spec.fields) if f)
        if self.spec.flag_field:
            body += f" {self.spec.flag_field}={self.flags}"
        return f"<t{self.tick} {self.name}{' ' + body if body else ''}>"


class JournalReader:
    """Incremental decoder for one engine process's journal fd.

    The engine writes one tick's records as a single write() before it
    writes that tick's observation, so a caller that has just read an
    observation can drain this and be sure it has everything for that
    tick. Partial records are still buffered rather than assumed away,
    because a write larger than the pipe buffer is split by the kernel.
    """

    def __init__(self):
        self.buf = b""
        self.checked_head = False
        # What this reader could not turn into events. Both used to be
        # thrown away: an unknown code was a `continue` with no counter,
        # and the engine's own RLJ_OVERFLOW record decoded into an event
        # nothing read. Renumbering one row of the enum in
        # magma/game/rl_journal.h therefore made every check over that
        # event return False forever while the task still loaded,
        # validated and ran, which reads exactly like a policy that never
        # does the thing.
        self.unknown = {}               # wire code -> records skipped
        self.dropped = 0                # records the ENGINE never sent
        self.overflows = 0              # RLJ_OVERFLOW records seen
        self._reported = set()

    def feed(self, data):
        """Bytes off the fd -> the Events they complete."""
        self.buf += data
        out = []
        if not self.checked_head:
            if len(self.buf) < HEAD_BYTES:
                return out
            magic, version, rec_bytes = _HEAD.unpack_from(self.buf, 0)
            if magic != MAGIC:
                raise ValueError(
                    f"journal stream magic {magic:#x} is not {MAGIC:#x}; the "
                    "fd is not an event journal")
            if version != VERSION or rec_bytes != REC_BYTES:
                raise ValueError(
                    f"journal is version {version} with {rec_bytes}-byte "
                    f"records, this reader speaks version {VERSION} with "
                    f"{REC_BYTES}-byte records. Rebuild the engine with the "
                    "patch this repo ships, or update bedrock_rl/"
                    "journal.py to match it.")
            self.buf = self.buf[HEAD_BYTES:]
            self.checked_head = True
        n = len(self.buf) // REC_BYTES
        for i in range(n):
            code, flags, tick, *v = _REC.unpack_from(self.buf, i * REC_BYTES)
            row = _BY_CODE.get(code)
            if row is None:
                # An engine this checkout has no row for. The record is
                # well framed and its meaning is not known here, so
                # skipping it is still the only reading available, but it
                # is COUNTED, because the alternative is a check that
                # silently never fires. problems() is what says so.
                self.unknown[code] = self.unknown.get(code, 0) + 1
                continue
            if row[0] == "overflow":
                # The engine buffers at most RLJ_MAX_PER_TICK records per
                # tick and reports what it had to drop rather than losing
                # it quietly. Nothing here read that, so the report went
                # the same way as the records.
                self.overflows += 1
                self.dropped += v[0]
            out.append(Event(row[0], row[1], tick, tuple(v), flags))
        self.buf = self.buf[n * REC_BYTES:]
        return out


    def problems(self):
        """New problems since the last call, as lines for a person.

        New, because a caller drains this every tick and a per-tick
        repeat of the same fact is not a second fact. The counts in each
        line are the totals so far.
        """
        out = []
        for code in sorted(self.unknown):
            if ("code", code) in self._reported:
                continue
            self._reported.add(("code", code))
            out.append(
                f"cannot decode journal record type {code}: EVENTS in "
                f"bedrock_rl/env/journal.py has no row for it, so every "
                f"check over that event answers False for the rest of the "
                f"run, which reads as a policy that never does the thing. "
                f"The two tables are magma/game/rl_journal.h and EVENTS in "
                f"that module and they carry the same numbers or nothing "
                f"works.")
        if self.dropped and "overflow" not in self._reported:
            self._reported.add("overflow")
            out.append(
                f"the engine could not journal everything one tick did and "
                f"started dropping records, {self.dropped}"
                f" of them so far, because one tick produced more than "
                f"RLJ_MAX_PER_TICK. Checks over the dropped events can only "
                f"be wrong in one direction, which is not firing. The "
                f"running totals are on this reader as .dropped and "
                f".overflows.")
        return out


def report_problems(reader, stream=None):
    """Print anything new on `reader.problems()` as a diagnostic.

    A `[journal] ...` line, which is the shape bedrock_rl/reporting.py
    recognises as this repo's own and the verl console filter never
    drops. Returns what it printed.
    """
    import sys
    out = reader.problems()
    for line in out:
        print(f"[journal] {line}", file=stream or sys.stderr, flush=True)
    return out


def drain(fd, reader):
    """Every Event available on a non-blocking journal fd right now."""
    out = []
    while True:
        try:
            chunk = os.read(fd, 1 << 16)
        except BlockingIOError:
            break
        except OSError:
            break
        if not chunk:
            break
        out.extend(reader.feed(chunk))
    return out


# ── the check factory ───────────────────────────────────────────────────
# Keys a journal check spec may carry that are not slot constraints.
_RESERVED = frozenset(("type", "count", "reward", "of"))


class EventMatch:
    """A compiled predicate over one event type.

    Compiled once when the task loads, so a per-turn check is a scan and
    not a re-parse, and so a task with a bad constraint fails at load
    with the list of slots rather than at the first evaluation.
    """

    def __init__(self, name, spec):
        if name not in EVENTS:
            raise KeyError(f"unknown event type {name!r}; the journal "
                           f"emits {sorted(EVENTS)}")
        self.name = name
        self.ev = EVENTS[name]
        self.count = int(spec.get("count", 1))
        if self.count < 1:
            raise ValueError(f"{name}: count must be at least 1")
        self.eq = []                    # (field name, frozenset of values)
        self.lo = []                    # (field name, threshold)
        self.hi = []
        named = [f for f in self.ev.fields if f]
        if self.ev.flag_field:
            named.append(self.ev.flag_field)
        for key, val in spec.items():
            if key in _RESERVED:
                continue
            bound, field = None, key
            if key.startswith("min_"):
                bound, field = "lo", key[4:]
            elif key.startswith("max_"):
                bound, field = "hi", key[4:]
            if field not in named:
                raise KeyError(
                    f"{name} has no slot {field!r} (from key {key!r}); its "
                    f"slots are {named}. A journal check constrains slots "
                    "by name, or bounds them with min_<slot>/max_<slot>.")
            scale = self.ev.scale.get(field, 1.0)
            if bound == "lo":
                self.lo.append((field, float(val) / scale))
            elif bound == "hi":
                self.hi.append((field, float(val) / scale))
            else:
                self.eq.append((field, self._values(field, val)))

    def _values(self, field, val):
        """A YAML constraint value -> the set of wire values it accepts."""
        from bedrock_rl.env.catalog import resolve_blocks, resolve_items
        kind = self.ev.resolve.get(field)
        if kind == _BLOCK:
            return frozenset(resolve_blocks(val))
        if kind == _ITEM:
            return frozenset(resolve_items(val))
        names = _ENUMS.get(field)
        vals = val if isinstance(val, (list, tuple)) else [val]
        out = []
        for v in vals:
            if names is not None and isinstance(v, str):
                if v not in names:
                    raise KeyError(f"{self.name}.{field}: {v!r} is not one "
                                   f"of {sorted(names)}")
                out.append(names[v])
            else:
                scale = self.ev.scale.get(field, 1.0)
                try:
                    wire = Decimal(str(v)) / Decimal(str(scale))
                except (InvalidOperation, ValueError) as exc:
                    raise ValueError(
                        f"{self.name}.{field}: {v!r} is not numeric") from exc
                if (not wire.is_finite()
                        or wire != wire.to_integral_value()):
                    raise ValueError(
                        f"{self.name}.{field}: {v!r} cannot be represented "
                        f"exactly at the journal wire scale {scale:g}")
                out.append(int(wire))
        return frozenset(out)

    def block_ids(self, field="block"):
        """Engine ids of the `block` slot this match constrains, or ().

        The BLOCK A CHECK IS ABOUT, handed out so nobody has to resolve
        the name a second time. `bedrock_rl/env/episode.py` asks a
        task's success check this to find the block a start pose may not
        already be aimed at, and it has to get the ids the check ITSELF
        will compare against: a second call to resolve_blocks on the same
        YAML string is a second opinion, and a name that resolves to two
        ids (`redstone_ore` is 73 and 74) is where two opinions diverge.

        () when the event has no such slot, when the slot resolves to
        ITEMS rather than blocks (`item_crafted.item`), and when this
        match leaves the slot free. `was_block` is deliberately not
        reachable by default: on `aimed_at` it names where the crosshair
        CAME FROM, which is not what the check is about.
        """
        if self.ev.resolve.get(field) != _BLOCK:
            return ()
        for constrained, allowed in self.eq:
            if constrained == field:
                return tuple(sorted(allowed))
        return ()

    def matches(self, e):
        if e.name != self.name:
            return False
        for field, allowed in self.eq:
            if e[field] not in allowed:
                return False
        for field, lo in self.lo:
            if e[field] < lo:
                return False
        for field, hi in self.hi:
            if e[field] > hi:
                return False
        return True

    def scan(self, events, start=0):
        """Index just past the `count`-th match at or after `start`, or
        None. Returning the index rather than a bool is what lets
        `sequence` chain steps without a second mechanism."""
        seen = 0
        for i in range(start, len(events)):
            if self.matches(events[i]):
                seen += 1
                if seen >= self.count:
                    return i + 1
        return None


def compile_check(spec):
    """A task YAML check spec -> the object its evaluation needs, or None
    when the check is not journal backed."""
    t = spec.get("type")
    if t in ("sequence", "all", "any"):
        steps = spec.get("of")
        if not isinstance(steps, list) or not steps:
            raise ValueError(
                f"{t} needs a non-empty `of:` list of journal checks, "
                "each an event type with its constraints")
        compiled = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise TypeError(f"{t}.of[{index}] must be a mapping")
            event = step.get("type")
            if event not in EVENTS:
                raise ValueError(
                    f"{t}.of[{index}] must name one journal event; "
                    f"got {event!r}, expected one of {sorted(EVENTS)}")
            compiled.append(EventMatch(event, step))
        return compiled
    if t in EVENTS:
        return EventMatch(t, spec)
    return None


def register_checks(check_type):
    """Register one check per event type, plus journal composition.

    Called once by bedrock_rl.env.task. There is deliberately no per-event
    function here: the whole point of the table is that adding a row to
    it is the entire Python side of a new event type.
    """
    def journal(check, env, start_obs):
        del start_obs
        return holds(check.type, check.journal, env.journal)

    # keys=None: a journal check's spec keys are the event's own slot
    # names, so EventMatch above is what refuses a misspelled one, and a
    # second list of them here would be a copy that goes stale.
    for name in EVENTS:
        check_type(name, keys=None)(journal)

    @check_type("sequence", keys=None)
    def _sequence(check, env, start_obs):
        """Every step satisfied, each strictly after the one before it."""
        del start_obs
        return holds(check.type, check.journal, env.journal)

    @check_type("all", keys=None)
    def _all(check, env, start_obs):
        """Every listed journal milestone occurred, in any order."""
        del start_obs
        return holds(check.type, check.journal, env.journal)

    @check_type("any", keys=None)
    def _any(check, env, start_obs):
        """At least one listed journal event occurred during the episode."""
        del start_obs
        return holds(check.type, check.journal, env.journal)


def holds(kind, compiled, events, start=0):
    """Evaluate a compiled journal check after an episode-local baseline."""
    start = int(start)
    if kind == "sequence":
        at = start
        for step in compiled:
            at = step.scan(events, at)
            if at is None:
                return False
        return True
    if kind == "all":
        return all(step.scan(events, start) is not None for step in compiled)
    if kind == "any":
        return any(step.scan(events, start) is not None for step in compiled)
    return compiled.scan(events, start) is not None
