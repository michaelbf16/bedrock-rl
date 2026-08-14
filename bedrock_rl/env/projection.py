"""The engine's pinhole camera and its EXACT inverse.

Both of netherite's cameras build a ray the same way. The semantic camera
is `oc_pixel` in `blaze/core/obs_camera.h`, and the rasterizer that draws
the frame the model looks at is `cr_perspective` in `magma/core/math.c`
fed by `camera_for` in `magma/game/frame_capture.c`. Written out, the
engine takes a normalized point (nx, ny) on the image plane, with nx to
the RIGHT and ny UP, both in [-1, 1], and forms

    d = l + nx*TANX*r + ny*TANY*u          then normalizes d

where l is the look vector at the current (yaw, pitch), r is the
horizontal right vector, u = r x l tilts under pitch, TANY is tan(fov/2)
and TANX is TANY times the image's aspect. Everything in this module is
the inverse of those three lines and nothing else.

The inverse is NOT two independent arctangents. Taking

    dyaw   = atan(nx*TANX)
    dpitch = atan(ny*TANY)

treats the two axes as separable, which they are not: the ray is
normalized as a whole, and u tilts with pitch, so the pitch of the ray
depends on nx and the yaw of the ray depends on ny. The two forms agree
on the two centre axes at pitch 0 and nowhere else. At pitch 0 the naive
form is 11.32 degrees wrong in pitch at a screen corner; at pitch 45 the
bottom corner is 29.11 degrees wrong in yaw and 36.40 wrong in pitch.
Those four numbers are what the separable form costs, measured against
the engine's own raycast; they are here so that a "simplification" back
to the naive form has a price written next to it.

Deriving it takes half a page. Work at yaw 0, which loses nothing because
rotating the yaw rotates the whole basis about the world y axis and
neither the yaw DELTA nor the resulting pitch depends on where that
rotation started. With p the current pitch in radians,

    l = ( 0, -sin p,  cos p)
    r = (-1,  0,      0    )
    u = r x l = ( 0,  cos p,  sin p)

so with A = nx*TANX and B = ny*TANY the unnormalized ray is

    d = (-A,  B cos p - sin p,  cos p + B sin p)

whose norm is sqrt(1 + A^2 + B^2) because r, u and l are orthonormal.
Reading the yaw and pitch back off d with netherite's own convention
(dx = -sin yaw cos pitch, dy = -sin pitch, dz = cos yaw cos pitch) gives
`ndc_to_angles` below. `angles_to_ndc` is the other direction, and it is
just the projection of the target direction onto the same three axes.

Two constants are the ENGINE'S and are restated here: the 70-degree
vertical field of view, and the 89-degree pitch clamp
`magma/game/player_ctl.c` applies after every dyaw/dpitch. Each carries
the C expression it was read from, in the comment on its definition
below, because nothing compares the two copies. A divergence does not
raise: it silently aims every action a fixed number of degrees off, on
every harness using screen coordinates, which would agree with each other
and with nothing in the world.
"""
import math

# Vertical field of view in degrees. The semantic camera states it as the
# comment on OC_TANY (blaze/core/obs_camera.h) and the rasterizer states
# it as `c.fov_deg = 70.0f * fov_mult` (magma/game/frame_capture.c).
# fov_mult is the sprint ease and the underwater 60/70; neither is
# reachable from the model's action vocabulary, which carries no sprint
# key, and an episode that walks into water is outside what any of this
# claims.
FOV_DEG = 70.0

# What the engine clamps pitch to after adding dpitch, magma/game/
# player_ctl.c. A requested turn past it is not a turn the engine will
# make, so the wrapper clamps to the same number instead of emitting a
# delta the engine will silently shorten.
PITCH_LIMIT = 89.0

# Below this the target direction is at or behind the image plane and no
# point on the screen names it. Not a tolerance: it is the sign test on
# d . l, with a hair of slack so a target exactly on the plane takes the
# off-screen branch rather than dividing by a denormal.
_ON_PLANE = 1e-12


def tangents(width, height, fov_deg=FOV_DEG):
    """(TANX, TANY) for an image `width` x `height`.

    TANY is tan(fov/2) and TANX is TANY times the aspect, which is how
    both `cr_perspective` (m[0] = f/aspect, m[5] = f) and OC_TANX state
    it. The aspect is the IMAGE's, so the 428x240 frame the model clicks
    on and the 64x36 semantic camera do not share a TANX.
    """
    tany = math.tan(math.radians(fov_deg) / 2.0)
    return tany * (float(width) / float(height)), tany


def ndc_to_angles(nx, ny, pitch, tanx, tany):
    """Normalized image point -> (dyaw, dpitch) degrees that put that
    point at the centre of the screen.

    nx is +1 at the right edge, ny is +1 at the TOP edge, matching the
    engine. `pitch` is the camera's pitch now, in degrees, and it is not
    optional: the same point names a different ray at every pitch.

    dpitch already carries the engine's own clamp, so the returned delta
    is the one the engine will actually apply.
    """
    a = float(nx) * tanx
    b = float(ny) * tany
    p = math.radians(float(pitch))
    sp, cp = math.sin(p), math.cos(p)
    # dz of the ray in the yaw-0 frame is cos p + B sin p and dx is -A,
    # and netherite reads yaw off a direction as atan2(-dx, dz).
    dyaw = math.degrees(math.atan2(a, cp + b * sp))
    n = math.sqrt(1.0 + a * a + b * b)
    dy = (b * cp - sp) / n
    aimed = -math.degrees(math.asin(max(-1.0, min(1.0, dy))))
    aimed = max(-PITCH_LIMIT, min(PITCH_LIMIT, aimed))
    return dyaw, aimed - float(pitch)


def angles_to_ndc(dyaw, dpitch, pitch, tanx, tany):
    """(dyaw, dpitch) degrees -> the normalized image point that names
    that direction, the exact inverse of `ndc_to_angles`.

    The target direction is projected onto the camera's right, up and
    forward axes; the first two over the third are the image-plane
    coordinates. A target at or behind the image plane has no image
    point, and rather than returning a signed infinity this answers with
    the point on the edge of the [-1,1] square in the target's direction,
    which is what a caller that iterates (`moves_for_angles`) needs.
    """
    p = math.radians(float(pitch))
    y1 = math.radians(float(dyaw))
    p1 = math.radians(max(-PITCH_LIMIT,
                          min(PITCH_LIMIT, float(pitch) + float(dpitch))))
    sp, cp = math.sin(p), math.cos(p)
    sp1, cp1 = math.sin(p1), math.cos(p1)
    right = math.sin(y1) * cp1                         # d . r
    up = math.cos(y1) * cp1 * sp - sp1 * cp            # d . u
    fwd = sp * sp1 + cp * cp1 * math.cos(y1)           # d . l
    if fwd > _ON_PLANE:
        return right / fwd / tanx, up / fwd / tany
    ex, ey = right / tanx, up / tany
    m = max(abs(ex), abs(ey))
    if m == 0.0:
        return 1.0, 0.0        # exactly behind: any edge names it as well
    return ex / m, ey / m
