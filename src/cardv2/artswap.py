"""Replace the upscaled art of a card render with the real, larger art file."""
from __future__ import annotations

import cv2
import numpy as np

from .register import Placement, art_window


def swap_art(render_low: np.ndarray, render_up: np.ndarray, art: np.ndarray, art_hi: np.ndarray,
             place: Placement) -> tuple[np.ndarray, dict]:
    """Composite `art_hi` into `render_up` wherever `render_low` shows the art.

    render_low/render_up: BGRA, render_up being render_low enlarged K times.
    art: BGR art at the size `place` was measured against; art_hi: the same art, larger.
    Colours of the art are fitted to the render so the card keeps its exact look.
    """
    mask, gain, bias = art_window(render_low, art, place)
    info = {"mask_px": int(mask.sum()), "gain": [round(float(v), 3) for v in gain], "bias": [round(float(v), 1) for v in bias]}
    if mask.sum() < 64 or not ((gain > 0.6) & (gain < 1.6)).all():
        return render_up, {**info, "applied": False, "reason": "colour_fit"}
    height, width = render_up.shape[:2]
    k = width / render_low.shape[1]
    art_scale = art.shape[0] / art_hi.shape[0]
    matrix = place.matrix(k, art_scale)
    interp = cv2.INTER_AREA if place.sx * k * art_scale < 1 else cv2.INTER_LANCZOS4
    layer = cv2.warpAffine(art_hi, matrix, (width, height), flags=interp).astype(np.float32) * gain + bias
    soft = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    soft = cv2.resize(soft, (width, height), interpolation=cv2.INTER_CUBIC)
    soft = np.clip(cv2.GaussianBlur(soft, (0, 0), k * 0.9), 0, 1)[..., None]
    out = render_up.astype(np.float32)
    out[..., :3] = out[..., :3] * (1 - soft) + np.clip(layer, 0, 255) * soft
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), {**info, "applied": True}
