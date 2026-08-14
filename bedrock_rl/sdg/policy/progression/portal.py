"""Terminal Nether portal operation for a scripted progression policy.

This module owns the live build/ignite/enter state machine. The progression
policy supplies reusable navigation, inventory, aiming, and journal controls.
"""

from __future__ import annotations

import math

from bedrock_rl.sdg.policy import PolicyVeto
from bedrock_rl.env.catalog import resolve_blocks


PORTAL_STAGES = frozenset({
    "return_to_surface", "portal_approach", "portal_clear", "portal_frame",
    "portal_descend", "portal_ignite", "portal_enter", "complete",
})


class NetherPortalExecutor:
    """Advance one explicit ``NetherPortalOperation`` from live state."""

    def __init__(self, policy):
        self.policy = policy

    def actions(self, env):
        policy = self.policy
        plan = policy.portal_plan
        if plan is None:
            policy.stage = "complete"
            return None
        if policy.stage == "return_to_surface":
            clear = policy._clear_return_leg(env)
            if clear is not None:
                return clear
            result = policy._follow_route(env)
            if result is not None:
                return result
            policy._start_route(plan.approach_stands, "portal_approach")

        if policy.stage == "portal_approach":
            result = policy._follow_route(env)
            if result is not None:
                return result
            policy._portal_clear_event_start = len(env.journal)
            policy.stage = "portal_clear"

        if policy.stage == "portal_clear":
            for cell in getattr(plan, "clear_cells", ()):
                if policy.broken(
                        env, cell,
                        since=policy._portal_clear_event_start):
                    policy._portal_clear_attempts.pop(cell, None)
                    policy._portal_clear_control_attempts.pop(cell, None)
                    continue
                attempts = policy._portal_clear_attempts.get(cell, 0)
                if attempts >= 4:
                    return PolicyVeto(
                        "progression_portal_clear_exhausted",
                        f"portal chamber cell {cell} did not break",
                        (cell[0] + 0.5, cell[2] + 0.5), (cell,))
                controls = policy._portal_clear_control_attempts.get(cell, 0)
                if controls >= 16:
                    return PolicyVeto(
                        "progression_portal_clear_control_exhausted",
                        "bounded tool selection, aim, and recenter controls "
                        f"never produced an exact attack on portal chamber "
                        f"cell {cell}",
                        (cell[0] + 0.5, cell[2] + 0.5), (cell,))
                ticks = policy.mining_hold_ticks(
                    env, cell, tool=policy.spec.portal.return_tool,
                    fallback=policy.spec.portal.return_ticks)
                result = policy.mine_exact(
                    env, cell, tool=policy.spec.portal.return_tool,
                    ticks=ticks,
                    stand=(plan.work_stance[0] + 0.5,
                           plan.work_stance[2] + 0.5),
                    since=policy._portal_clear_event_start)
                if isinstance(result, PolicyVeto):
                    return result
                if not result:
                    return PolicyVeto(
                        "progression_portal_clear_unavailable",
                        f"could not clear portal chamber cell {cell}",
                        (cell[0] + 0.5, cell[2] + 0.5), (cell,))
                if any(action.get("action") == "left_click_hold"
                       for action in result):
                    policy._portal_clear_attempts[cell] = attempts + 1
                else:
                    policy._portal_clear_control_attempts[cell] = controls + 1
                return result
            policy._portal_placement_event_start = len(env.journal)
            policy.stage = "portal_frame"

        if policy.stage == "portal_frame":
            while policy._placement_index < len(plan.placements):
                step = plan.placements[policy._placement_index]
                item = (policy.spec.portal.scaffold_item
                        if step.role == "scaffold" else step.material)
                block = item
                # A journal entry proves that a placement happened once, not
                # that the block is still there.  Long progression episodes
                # can legitimately reuse cells cleared by an earlier cast or
                # workstation operation, so only this build's events count.
                present = policy.placed(
                    env, step.cell, block,
                    since=policy._portal_placement_event_start)
                if present:
                    policy._portal_place_control_attempts.pop(
                        policy._placement_index, None)
                    policy._placement_index += 1
                    policy._portal_placement_route_index = None
                    continue
                stance = step.stance or plan.work_stance
                horizontal = math.hypot(
                    float(env.obs["x"]) - (stance[0] + 0.5),
                    float(env.obs["z"]) - (stance[2] + 0.5))
                vertical = float(env.obs["y"]) - stance[1]
                jump_placement = bool(getattr(step, "jump", False))
                airborne_jump_pose = (
                    jump_placement and horizontal <= 0.45
                    and 0.2 < vertical <= 1.35)
                if (horizontal <= 0.45 and 0.2 < vertical <= 1.35
                        and not airborne_jump_pose):
                    return [{"action": "wait", "ticks": 2}]
                if (horizontal > 0.45
                        or (abs(vertical) > 0.2
                            and not airborne_jump_pose)):
                    if (policy._portal_placement_route_index
                            != policy._placement_index):
                        route = policy._live_surface_route(
                            env, stance,
                            f"portal-placement:{policy._placement_index}")
                        policy._start_route(route, "portal_frame")
                        policy._portal_placement_route_index = (
                            policy._placement_index)
                    moved = policy._follow_route(env)
                    if moved is not None:
                        return moved
                    if (math.hypot(
                            float(env.obs["x"]) - (stance[0] + 0.5),
                            float(env.obs["z"]) - (stance[2] + 0.5)) > 0.45
                            or abs(float(env.obs["y"])
                                   - stance[1]) > 0.2):
                        return PolicyVeto(
                            "progression_portal_placement_stance_diverged",
                            f"player did not reach certified placement "
                            f"stance {stance}",
                            (stance[0] + 0.5, stance[2] + 0.5),
                            (stance,))
                controls = policy._portal_place_control_attempts.get(
                    policy._placement_index, 0)
                if controls >= 32:
                    return PolicyVeto(
                        "progression_portal_place_control_exhausted",
                        "bounded movement, inventory, aim, and jump controls "
                        f"did not place portal step {policy._placement_index} "
                        f"at {step.cell}",
                        (step.cell[0] + 0.5, step.cell[2] + 0.5),
                        (step.cell, step.support))
                result = policy.place(
                    env, item, step.cell, step.support, block=block,
                    stand=(stance[0] + 0.5, stance[2] + 0.5),
                    jump_from_y=(stance[1] if jump_placement else None),
                    since=policy._portal_placement_event_start)
                if result is None:
                    recovered = policy.recover_hotbar(
                        env, item,
                        replace=policy._hotbar_swaps(env, wanted=(item,)),
                        minimum=1)
                    if recovered:
                        policy._portal_place_control_attempts[
                            policy._placement_index] = controls + 1
                        return recovered
                    return PolicyVeto(
                        "progression_portal_material_missing",
                        f"no {item!r} is present in the observed inventory "
                        "for portal placement",
                        (step.cell[0] + 0.5, step.cell[2] + 0.5),
                        (step.cell,))
                if not isinstance(result, PolicyVeto):
                    policy._portal_place_control_attempts[
                        policy._placement_index] = controls + 1
                return result
            last = plan.placements[-1].stance or plan.work_stance
            raised_work = (
                plan.work_stance[0], plan.work_stance[1] + 1,
                plan.work_stance[2])
            policy._start_route(
                (last, raised_work, plan.interaction_stance),
                "portal_descend")

        if policy.stage == "portal_descend":
            result = policy._follow_route(env)
            if result is not None:
                return result
            policy.stage = "portal_ignite"

        if policy.stage == "portal_ignite":
            if int(env.obs.get("dimension", 0)) == -1:
                policy.stage = "complete"
                return None
            # Once any interior portal block is observed, stop clicking and
            # enter it. The task's live verifier remains the final authority.
            portal_ids = frozenset(policy._block_ids(
                policy.spec.portal.active_block))
            ray_block, ray_cell = policy.click_target(env)
            if (any(policy.observed_block(env, cell) in portal_ids
                    for cell in plan.open_cells)
                    or (ray_cell in plan.open_cells
                        and ray_block in portal_ids)):
                policy.stage = "portal_enter"
            else:
                fire_ids = frozenset(resolve_blocks(["fire"]))
                burning = tuple(dict.fromkeys(
                    cell for cell in plan.open_cells
                    if (policy.observed_block(env, cell) in fire_ids
                        or (cell == ray_cell and ray_block in fire_ids))))
                if burning:
                    frame = tuple(
                        (cell, policy.observed_block(env, cell))
                        for cell in plan.occupied_cells)
                    return PolicyVeto(
                        "progression_portal_frame_invalid",
                        "flint and steel produced fire but the synchronous "
                        "engine validator did not activate the portal; "
                        f"observed frame={frame}",
                        (plan.interaction_cell[0] + 0.5,
                         plan.interaction_cell[2] + 0.5),
                        burning)
                stance_error = math.hypot(
                    float(env.obs["x"])
                    - (plan.interaction_stance[0] + 0.5),
                    float(env.obs["z"])
                    - (plan.interaction_stance[2] + 0.5))
                if (stance_error > 0.05
                        or abs(float(env.obs["y"])
                               - plan.interaction_stance[1]) > 0.2):
                    if policy._portal_ignite_control_attempts >= 16:
                        return PolicyVeto(
                            "progression_portal_ignite_control_exhausted",
                            "bounded stance and aim controls never reached "
                            "the certified portal ignition ray",
                            (plan.interaction_cell[0] + 0.5,
                             plan.interaction_cell[2] + 0.5),
                            (plan.interaction_stance,))
                    policy._portal_ignite_control_attempts += 1
                    move = policy.face_and_move_to(
                        env, plan.interaction_stance[0] + 0.5,
                        plan.interaction_stance[2] + 0.5,
                        tolerance=0.05, sequential_hop=True,
                        allow_jump=False, movement_cap_blocks=0.3)
                    return move or [{"action": "wait", "ticks": 2}]
                selected = policy.select(env, policy.spec.portal.igniter_item)
                if selected is None:
                    return PolicyVeto(
                        "progression_igniter_missing",
                        f"no {policy.spec.portal.igniter_item!r} is visible "
                        "for portal ignition",
                        (plan.interaction_cell[0] + 0.5,
                         plan.interaction_cell[2] + 0.5))
                if selected:
                    return selected
                target = policy.click_target(env)[1]
                destination = policy.click_destination(env)
                if (target == plan.interaction_support
                        and destination == plan.interaction_cell):
                    if policy._portal_ignite_attempts >= 4:
                        return PolicyVeto(
                            "progression_portal_ignite_exhausted",
                            "portal did not activate after four exact clicks",
                            (plan.interaction_cell[0] + 0.5,
                             plan.interaction_cell[2] + 0.5))
                    policy._portal_ignite_attempts += 1
                    return [{"action": "right_click"},
                            {"action": "wait", "ticks": 4}]
                aim = policy.aim_point(env, tuple(
                    plan.interaction_support[index]
                    + (plan.interaction_cell[index]
                       - plan.interaction_support[index]) * 0.45
                    for index in range(3)))
                if aim:
                    if policy._portal_ignite_control_attempts >= 16:
                        return PolicyVeto(
                            "progression_portal_ignite_control_exhausted",
                            "bounded camera controls never proved the "
                            "portal ignition face",
                            (plan.interaction_cell[0] + 0.5,
                             plan.interaction_cell[2] + 0.5),
                            (() if target is None else (target,)))
                    policy._portal_ignite_control_attempts += 1
                    return aim
                return PolicyVeto(
                    "progression_portal_ignite_ray_unproven",
                    "public ray does not prove the planned ignition cell",
                    (plan.interaction_cell[0] + 0.5,
                     plan.interaction_cell[2] + 0.5),
                    (() if target is None else (target,)))

        if policy.stage == "portal_enter":
            if int(env.obs.get("dimension", 0)) == -1:
                policy.stage = "complete"
                return None
            if (policy._portal_entry_attempts
                    >= policy.spec.portal.max_entry_attempts):
                return PolicyVeto(
                    "progression_portal_entry_exhausted",
                    "engine never verified arrival in the Nether",
                    (plan.entry_stance[0] + 0.5,
                     plan.entry_stance[2] + 0.5))
            policy._portal_entry_attempts += 1
            move = policy.face_and_move_to(
                env, plan.entry_stance[0] + 0.5,
                plan.entry_stance[2] + 0.5,
                tolerance=0.25, sequential_hop=True)
            return move or [{
                "action": "wait",
                "ticks": policy.spec.portal.entry_wait_ticks,
            }]
        return None
