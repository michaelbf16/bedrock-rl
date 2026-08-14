"""Live-verified, objective-independent route lighting."""

import math

from .types import PolicyVeto


class WallTorchRouteMixin:
    """Reusable, live-verified wall lighting for scripted routes.

    The mixin deliberately knows nothing about an objective or world seed. A
    policy records the feet cells it actually traversed, and this behavior
    chooses a visible side wall at a prior footprint. It never guesses a
    placement: the public click ray must identify both the support and the
    exact adjacent destination immediately before the click.

    Subclasses may override :meth:`route_light_protected_cells` to keep
    lights out of cells that a later plan will mine or build in, and
    :meth:`route_light_initial_site` to light a just-created workstation as
    soon as torches become available.
    """

    _NON_TORCH_SUPPORTS = frozenset({
        0, 6, 8, 9, 10, 11, 18, 30, 31, 32, 37, 38, 39, 40, 50, 51,
        58, 59, 61, 62, 78, 83, 104, 105, 106, 111, 115,
    })

    def reset_route_lighting(self):
        self.shown_lights = set()
        self.pending_light = None
        self.light_aim_attempts = 0
        self.light_place_attempts = 0
        self.light_retry_position = None
        self.torch_events_seen = 0
        self.recent_route_cells = []
        self.rejected_light_supports = set()

    def route_light_protected_cells(self):
        """Cells reserved by the objective's future plan."""
        return ()

    def route_light_forbidden_support_cells(self):
        """Supports whose later mutation makes a mounted light unsafe.

        A protected destination cannot contain a torch.  A protected solid is
        not automatically an unsafe *support*, however: certified tunnel walls
        are the normal place to mount one.  The default remains conservative
        for simple policies; planners with an immutable/mutable distinction
        can override this independently.
        """
        return self.route_light_protected_cells()

    def route_light_forbidden_destination_cells(self):
        """Air cells in which even a passable torch would obstruct work."""
        return self.route_light_protected_cells()

    def route_light_initial_site(self, env):
        """Optional immediate ``(cell, support, aim_point, kind)`` site."""
        del env
        return None

    def route_light_sneak_supports(self):
        """Interactive supports that require sneak-right-click."""
        return ()

    @staticmethod
    def _route_light_events(env, item):
        from bedrock_rl.env.catalog import resolve_items
        wanted = frozenset(resolve_items([item]))
        return [event for event in env.journal
                if event.name == "block_placed"
                and int(event.get("block", -1)) in wanted]

    @staticmethod
    def _route_light_item_count(env, item):
        from bedrock_rl.env.catalog import resolve_items
        wanted = frozenset(resolve_items([item]))
        return sum(int(count) for found, count in zip(
            env.obs["hotbar_ids"], env.obs["hotbar_counts"])
            if int(found) in wanted)

    @staticmethod
    def _route_light_select(env, item):
        from bedrock_rl.env import gui_layout
        from bedrock_rl.env.catalog import resolve_items
        slot = gui_layout.find_stack(env, resolve_items([item]), 1)
        if slot is None:
            return None
        return ([] if int(env.obs["hotbar_sel"]) == slot
                else gui_layout.select_hotbar(slot))

    @staticmethod
    def _route_light_click(env):
        from bedrock_rl.env.journal import CLICK_FACES
        for event in reversed(env.journal):
            if event.name != "click_target":
                continue
            support = (int(event["x"]), int(event["y"]), int(event["z"]))
            offset = CLICK_FACES.get(int(event.get("face", 0)))
            destination = (None if offset is None else tuple(
                support[index] + offset[index] for index in range(3)))
            return support, destination
        return None, None

    def remember_route_position(self, env):
        """Record one live player footprint for later behind-path lighting."""
        cell = (math.floor(float(env.obs["x"])),
                math.floor(float(env.obs["y"]) + 0.05),
                math.floor(float(env.obs["z"])))
        if not self.recent_route_cells or self.recent_route_cells[-1] != cell:
            self.recent_route_cells.append(cell)
            self.recent_route_cells = self.recent_route_cells[-48:]

    # Private alias retained for policies built against the original mixin API.
    _remember_route_position = remember_route_position

    def choose_route_light_site(self, env):
        """Choose a visible side wall at an already traversed footprint."""
        if not self.torch_events_seen:
            initial = self.route_light_initial_site(env)
            if initial is not None:
                return initial

        feet_y = math.floor(float(env.obs["y"]) + 0.05)
        yaw = math.radians(float(env.obs["yaw"]))
        forward = (-math.sin(yaw), math.cos(yaw))
        hazards = self.observed_hazards(env)
        protected = {tuple(map(int, cell)) for cell in
                     self.route_light_forbidden_destination_cells()}
        forbidden_supports = {tuple(map(int, cell)) for cell in
                              self.route_light_forbidden_support_cells()}
        blocks = {(int(x), int(y), int(z)): int(block)
                  for block, x, y, z in env.obs.get("blocks", ())}
        candidates = []
        for cell in reversed(self.recent_route_cells):
            if cell[1] not in (feet_y, feet_y + 1):
                continue
            dx = cell[0] + 0.5 - float(env.obs["x"])
            dz = cell[2] + 0.5 - float(env.obs["z"])
            distance = math.hypot(dx, dz)
            if distance > 3.5:
                continue
            ahead = dx * forward[0] + dz * forward[1]
            current_cell = distance < 0.65
            previous_level = cell[1] == feet_y + 1
            if current_cell and not previous_level:
                continue
            if not current_cell and ahead > -0.15:
                continue
            if any(abs(hy - feet_y) <= 1
                   and math.hypot(hx + 0.5 - (cell[0] + 0.5),
                                  hz + 0.5 - (cell[2] + 0.5))
                   <= self.behaviors.hazard_clearance_blocks + 0.75
                   for hx, hy, hz in hazards):
                continue
            lateral = abs(dx * forward[1] - dz * forward[0])
            if previous_level and current_cell:
                height_priority = -2
            elif not previous_level and not current_cell:
                height_priority = 0 if lateral >= 0.45 else 1
            else:
                height_priority = 2
            route_score = (height_priority, abs(distance - 1.5), ahead)
            cell_for_torch = (cell[0], cell[1] + 1, cell[2])
            minimum_spacing = max(
                2.0, float(self.behaviors.light_every_blocks) / 2.0)
            too_close = any(
                math.dist(cell_for_torch, shown) < minimum_spacing
                for shown in self.shown_lights)
            if (cell_for_torch in self.shown_lights
                    or cell_for_torch in protected):
                continue
            for ox, oz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                support = (cell_for_torch[0] + ox, cell_for_torch[1],
                           cell_for_torch[2] + oz)
                block = blocks.get(support)
                if (support in self.rejected_light_supports
                        or support in forbidden_supports
                        or (block is not None
                            and block in self._NON_TORCH_SUPPORTS)
                        or (block is not None and self.is_hazard(block))):
                    continue
                along_face = abs(ox * forward[0] + oz * forward[1])
                if along_face > 0.72:
                    continue
                # ``env.rel_angles_to`` accepts block-coordinate arguments
                # and adds the half-block centre offsets itself. Move those
                # arguments just inside the intended support face.
                point = (round(support[0] - ox * 0.49, 6),
                         support[1],
                         round(support[2] - oz * 0.49, 6))
                candidates.append(((1 if too_close else 0,)
                                   + route_score + (along_face,),
                                   cell_for_torch, support, point, "wall"))
            # Open cuts and stair descents may have no side wall within the
            # exact click reach. A standing torch on a certified support is
            # ordinary Minecraft behavior and keeps lighting from becoming a
            # seed filter for otherwise safe surface geometry. Never place it
            # in the player's current body cell.
            floor_cell = cell
            floor_support = (cell[0], cell[1] - 1, cell[2])
            floor_block = blocks.get(floor_support)
            if (not current_cell
                    and floor_cell not in protected
                    and floor_cell not in self.shown_lights
                    and floor_support not in forbidden_supports
                    and floor_support not in self.rejected_light_supports
                    and floor_block is not None
                    and floor_block not in self._NON_TORCH_SUPPORTS
                    and not self.is_hazard(floor_block)):
                point = (float(floor_support[0]),
                         floor_support[1] + 0.49,
                         float(floor_support[2]))
                candidates.append(((1 if too_close else 0, 3)
                                   + route_score,
                                   floor_cell, floor_support, point,
                                   "floor"))
        if not candidates:
            return None
        _score, cell, support, point, kind = min(candidates)
        return cell, support, point, kind

    _light_site = choose_route_light_site

    def light_route(self, env):
        """Place and verify one due wall light behind the player."""
        events = self._route_light_events(env, self.behaviors.light_item)
        if len(events) > self.torch_events_seen:
            event = events[-1]
            placed = (int(event["x"]), int(event["y"]), int(event["z"]))
            meta = int(event.get("meta", -1))
            # The pinned vanilla placement routine may resolve an exact
            # wall-face click as a standing torch when the same destination
            # also has solid floor support.  Both are ordinary, mounted
            # torches.  Preserve the exact destination proof below and reject
            # only unattached/unknown metadata.
            valid_mount = meta in (1, 2, 3, 4, 5)
            if not valid_mount:
                self.pending_light = None
                return PolicyVeto(
                    "route_light_not_wall_mounted",
                    f"route torch at {placed} was not wall/floor-mounted: "
                    f"meta={event.get('meta')}",
                    (placed[0] + 0.5, placed[2] + 0.5), (placed,))
            if (self.pending_light is not None
                    and placed != self.pending_light[0]):
                expected = self.pending_light[0]
                self.pending_light = None
                return PolicyVeto(
                    "route_light_landed_elsewhere",
                    f"route torch landed at {placed}, expected {expected}",
                    (expected[0] + 0.5, expected[2] + 0.5),
                    (placed, expected))
            self.torch_events_seen = len(events)
            self.shown_lights.add(placed)
            self.pending_light = None
            self.light_aim_attempts = 0
            self.light_place_attempts = 0
            self.light_retry_position = None
            self.route_light_placed()
            return [{"action": "wait", "ticks": max(
                1, self.behaviors.light_pause_ticks)}]
        if not self.route_light_due():
            return None
        if self._route_light_item_count(env, self.behaviors.light_item) < 1:
            recover = getattr(self, "recover_hotbar", None)
            if recover is not None:
                replacements = ()
                replacement_hook = getattr(
                    self, "route_light_hotbar_swaps", None)
                if replacement_hook is not None:
                    replacements = replacement_hook(
                        env, self.behaviors.light_item)
                recovered = recover(
                    env, self.behaviors.light_item,
                    replace=replacements)
                if recovered is not None:
                    return recovered
            return None
        if self.pending_light is None:
            self.pending_light = self.choose_route_light_site(env)
            self.light_aim_attempts = 0
            self.light_place_attempts = 0
        if self.pending_light is None:
            return None
        cell, support, point, _kind = self.pending_light
        target, destination = self._route_light_click(env)
        if target != support or destination != cell:
            # Import lazily to avoid a module cycle while allowing this mixin to
            # be composed independently of the public controller.
            from bedrock_rl.sdg.policy.controller import (
                MinecraftController,
            )
            aim = MinecraftController.aim_point(env, point)
            if aim and self.light_aim_attempts < 4:
                self.light_aim_attempts += 1
                return aim
            self.rejected_light_supports.add(support)
            self.pending_light = None
            self.light_aim_attempts = 0
            self.light_place_attempts = 0
            self.light_retry_position = tuple(
                float(env.obs[key]) for key in ("x", "y", "z"))
            return [{"action": "wait", "ticks": 1}]
        selected = self._route_light_select(
            env, self.behaviors.light_item)
        if selected is None:
            return None
        if self.light_place_attempts >= 2:
            self.rejected_light_supports.add(support)
            self.pending_light = None
            self.light_aim_attempts = 0
            self.light_place_attempts = 0
            self.light_retry_position = tuple(
                float(env.obs[key]) for key in ("x", "y", "z"))
            return [{"action": "wait", "ticks": 1}]
        self.light_place_attempts += 1
        action = {"action": "right_click"}
        if support in {tuple(map(int, cell))
                       for cell in self.route_light_sneak_supports()}:
            action["sneak"] = True
        return selected + [action]

    _light_route = light_route
