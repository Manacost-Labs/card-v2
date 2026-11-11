"""Image loading/saving helpers. Arrays are OpenCV-style: uint8, BGR or BGRA."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .errors import CardError


def load_bgra(path: Path | str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"))
    except (OSError, ValueError) as error:
        raise CardError("bad_image", f"cannot read image {path}: {error}") from error
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def load_bgr(path: Path | str) -> np.ndarray:
    return np.ascontiguousarray(load_bgra(path)[..., :3])


def save_png(path: Path | str, bgra: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgra):
        raise CardError("write_failed", f"cannot write {path}")


def alpha_bbox(bgra: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of non-transparent pixels."""
    ys, xs = np.where(bgra[..., 3] > 0)
    if xs.size == 0:
        return 0, 0, bgra.shape[1], bgra.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_and_fit(up: np.ndarray, low: np.ndarray, width: int, crop: bool = True) -> np.ndarray:
    """Trim `up` to the card's own bounds (taken from the low-res source) and scale to `width`."""
    scale = up.shape[1] / low.shape[1]
    if crop:
        x0, y0, x1, y1 = alpha_bbox(low)
        up = up[int(y0 * scale):int(round(y1 * scale)), int(x0 * scale):int(round(x1 * scale))]
    if width and width != up.shape[1]:
        height = max(1, round(up.shape[0] * width / up.shape[1]))
        interp = cv2.INTER_AREA if width < up.shape[1] else cv2.INTER_LANCZOS4
        up = cv2.resize(up, (width, height), interpolation=interp)
    return up
