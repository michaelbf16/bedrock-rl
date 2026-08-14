"""Deterministic visualization of Netherite's semantic camera."""

from __future__ import annotations


# Environment block IDs, not texture samples. Unknown blocks stay neutral so
# adding a block never silently borrows another block's visual identity.
SEMANTIC_COLORS = {
    0: (110, 165, 255), 1: (128, 128, 128), 2: (95, 159, 53),
    3: (134, 96, 67), 4: (100, 100, 100), 7: (40, 40, 40),
    8: (47, 90, 200), 9: (47, 90, 200), 10: (230, 120, 30),
    11: (230, 120, 30), 12: (219, 211, 160), 13: (136, 126, 126),
    14: (143, 140, 125), 15: (135, 130, 126), 16: (105, 105, 105),
    17: (155, 106, 56), 18: (32, 105, 28), 21: (35, 75, 170),
    24: (215, 205, 158), 31: (74, 132, 50), 32: (123, 79, 25),
    37: (255, 220, 60), 38: (215, 60, 60), 49: (28, 18, 38),
    50: (245, 205, 75), 56: (75, 205, 205), 58: (150, 105, 58),
    61: (105, 105, 105), 62: (120, 105, 88), 73: (145, 45, 45),
    74: (175, 55, 55), 78: (240, 248, 255), 79: (145, 180, 235),
    82: (158, 164, 176), 86: (222, 126, 34), 87: (112, 52, 49),
    90: (105, 42, 150), 99: (140, 110, 90), 100: (200, 60, 60),
    106: (25, 90, 25), 129: (45, 185, 105), 161: (32, 105, 28),
    162: (155, 106, 56), 175: (60, 140, 60),
}


def semantic_image(camera, depth, edge, *, width=800, height=None):
    """Colorize one 64x36 semantic observation with nearest-neighbor pixels."""
    from PIL import Image

    from bedrock_rl.env.engine import CAM_H, CAM_W

    if not (len(camera) == len(depth) == len(edge) == CAM_W * CAM_H):
        raise ValueError("semantic camera/depth/edge arrays have wrong size")
    pixels = []
    unknown = (90, 90, 90)
    for block, distance, boundary in zip(camera, depth, edge):
        rgb = SEMANTIC_COLORS.get(int(block), unknown)
        shade = 1.0 if int(block) == 0 else max(
            0.0, 1.0 - float(distance) / 512.0)
        if boundary:
            shade *= 0.55
        pixels.append(tuple(round(channel * shade) for channel in rgb))
    image = Image.new("RGB", (CAM_W, CAM_H))
    image.putdata(pixels)
    width = int(width)
    if width < CAM_W:
        raise ValueError(f"semantic image width must be at least {CAM_W}")
    height = (round(width * CAM_H / CAM_W) if height is None
              else int(height))
    if height < CAM_H:
        raise ValueError(f"semantic image height must be at least {CAM_H}")
    return image.resize((width, height), Image.Resampling.NEAREST)
