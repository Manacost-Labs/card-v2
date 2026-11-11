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


def save_image(path: Path | str, bgra: np.ndarray, quality: int = 90) -> None:
    """Write PNG or WebP, chosen by the file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        save_png(path, bgra)
        return
    if suffix != ".webp":
        raise CardError("bad_format", f"output must end with .png or .webp: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)).save(path, "WEBP", quality=int(quality), method=6)
    except (OSError, ValueError) as error:
        raise CardError("write_failed", f"cannot write {path}: {error}") from error


def fidelity(low: np.ndarray, up: np.ndarray, block: int = 24) -> dict:
    """How far the enlarged picture is from its source when shrunk back to the source size.

    Both pictures are slightly blurred first, so sharpening does not count: what remains is a
    change of content (art painted over the frame, shifted layers, colour casts, wrong art).
    Measured over the source's opaque pixels on a 0-255 scale. `mean_diff` is the whole card,
    `region_diff` the worst block of `block` x `block` source pixels.
    """
    back = cv2.resize(up, (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(back[..., :3].astype(np.float32), (0, 0), 1.5)
    b = cv2.GaussianBlur(low[..., :3].astype(np.float32), (0, 0), 1.5)
    opaque = cv2.erode((low[..., 3] > 250).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not opaque.any():
        return {"mean_diff": 0.0, "region_diff": 0.0}
    diff = np.abs(a - b).max(axis=2)
    diff[~opaque] = 0
    # block means through an integral image; only blocks that are almost fully opaque count
    kernel = (block, block)
    sums = cv2.boxFilter(diff, -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
    cover = cv2.boxFilter(opaque.astype(np.float32), -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT)
    full = cover > 0.9 * block * block
    region = float((sums[full] / cover[full]).max()) if full.any() else 0.0
    return {"mean_diff": round(float(diff[opaque].mean()), 2), "region_diff": round(region, 1)}


def flatten(bgra: np.ndarray, background=(28, 24, 24)) -> np.ndarray:
    """Composite over a solid BGR colour; returns BGR."""
    alpha = bgra[..., 3:4].astype(np.float32) / 255.0
    out = bgra[..., :3].astype(np.float32) * alpha + np.float32(background) * (1 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def compare_sheet(before: np.ndarray, after: np.ndarray, zoom: tuple[float, float, float, float] | None = None,
                  height: int = 0, gap: int = 24, background=(28, 24, 24)) -> np.ndarray:
    """Before/after sheet: the source stretched to the result's size next to the result.

    `before` is trimmed to its own card bounds when its proportions differ from `after`. With
    `zoom` (x0, y0, x1, y1 as fractions of the card) a second row shows that region enlarged.
    """
    ah, aw = after.shape[:2]
    if abs(before.shape[1] / before.shape[0] - aw / ah) > 0.01:
        x0, y0, x1, y1 = alpha_bbox(before)
        before = before[y0:y1, x0:x1]
    stretched = cv2.resize(before, (aw, ah), interpolation=cv2.INTER_LANCZOS4)
    left, right = flatten(stretched, background), flatten(after, background)

    def row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        spacer = np.full((a.shape[0], gap, 3), background, np.uint8)
        return np.hstack([a, spacer, b])

    rows = [row(left, right)]
    if zoom:
        zx0, zy0, zx1, zy1 = zoom
        if not (0 <= zx0 < zx1 <= 1 and 0 <= zy0 < zy1 <= 1):
            raise CardError("bad_zoom", "zoom must be x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        box = (slice(int(zy0 * ah), int(zy1 * ah)), slice(int(zx0 * aw), int(zx1 * aw)))
        crops = [cv2.resize(image[box], (aw, max(1, round(aw * (box[0].stop - box[0].start) / (box[1].stop - box[1].start)))),
                            interpolation=cv2.INTER_LANCZOS4) for image in (left, right)]
        rows.append(np.full((gap, rows[0].shape[1], 3), background, np.uint8))
        rows.append(row(*crops))
    sheet = np.vstack(rows)
    if height and height != sheet.shape[0]:
        width = max(1, round(sheet.shape[1] * height / sheet.shape[0]))
        sheet = cv2.resize(sheet, (width, height), interpolation=cv2.INTER_AREA)
    return np.dstack([sheet, np.full(sheet.shape[:2], 255, np.uint8)])
