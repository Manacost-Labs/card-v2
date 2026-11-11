"""Compose an in-play ("on board") minion: art under an oval frame with empty stat gems."""
from __future__ import annotations

import cv2
import numpy as np

from .register import Placement

# Where the 512 px art sits inside the 300x350 Firestone board frame. Measured by registering a
# reference composite (frame + art) against the art; valid for the normal and the golden frame.
ONBOARD_PLACEMENT = Placement(score=1.0, sx=0.5325, sy=0.5325, ox=12.7, oy=28.7)
ONBOARD_ART_SIZE = 512


def frame_window(alpha: np.ndarray) -> np.ndarray:
    """The art opening: the transparent area enclosed by the frame (uint8 0/1)."""
    height, width = alpha.shape
    solid = (alpha > 127).astype(np.uint8)
    flood = solid.copy()
    cv2.floodFill(flood, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 2)
    hole = (flood == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hole, 8)
    if count < 2:
        return hole
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def compose_onboard(frame_up: np.ndarray, art_hi: np.ndarray, k: float,
                    place: Placement = ONBOARD_PLACEMENT) -> np.ndarray:
    """frame_up: BGRA frame enlarged k times; art_hi: BGR art of any size. Returns BGRA."""
    height, width = frame_up.shape[:2]
    frame = frame_up.astype(np.float32)
    alpha = frame[..., 3] / 255.0
    hole = frame_window(frame_up[..., 3])
    # run the art slightly under the frame's inner edge so no gap shows through
    grow = max(1, int(round(k * 2.5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    under = cv2.GaussianBlur(cv2.dilate(hole, kernel).astype(np.float32), (0, 0), max(k * 0.4, 0.3))
    art_scale = ONBOARD_ART_SIZE / art_hi.shape[0]
    matrix = place.matrix(k, art_scale)
    interp = cv2.INTER_AREA if place.sx * k * art_scale < 1 else cv2.INTER_LANCZOS4
    layer = cv2.warpAffine(art_hi, matrix, (width, height), flags=interp).astype(np.float32)
    a = alpha[..., None]
    out = np.zeros_like(frame)
    out[..., :3] = frame[..., :3] * a + layer * (1 - a)
    out[..., 3] = np.clip(np.maximum(alpha, under) * 255, 0, 255)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
