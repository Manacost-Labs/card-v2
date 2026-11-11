"""Find the card art inside a card render, and the pixels where the render actually shows it."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class Placement:
    """Art pixel (xa, ya) appears in the render at (sx * xa + ox, sy * ya + oy)."""

    score: float
    sx: float
    sy: float
    ox: float
    oy: float

    def as_dict(self) -> dict:
        return {key: round(float(value), 4) for key, value in asdict(self).items()}

    def matrix(self, k: float = 1.0, art_scale: float = 1.0) -> np.ndarray:
        """Affine art -> render, for a render enlarged k times and an art enlarged 1/art_scale times.

        Enlarged pictures keep pixel centres aligned with their originals, so pixel x of the
        original sits at k * x + (k - 1) / 2; the same holds for the enlarged art.
        """
        shift = (k - 1) / 2
        art_shift = (1 / art_scale - 1) / 2
        ax, ay = self.sx * k * art_scale, self.sy * k * art_scale
        return np.float32([[ax, 0, self.ox * k + shift - ax * art_shift],
                           [0, ay, self.oy * k + shift - ay * art_shift]])


def _centre(art_gray: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, int]:
    size = art_gray.shape[0]
    c0, c1 = int(size * lo), int(size * hi)
    return art_gray[c0:c1, c0:c1], c0


def locate(render_bgr: np.ndarray, art_bgr: np.ndarray, scales=(0.2, 1.0, 161), window=(0.3, 0.7)) -> Placement:
    """Multi-scale template match of the art's centre against the render.

    The centre of the art is used because it is visible on every card type; the search first
    assumes a uniform scale, then allows a slightly different x/y scale.
    """
    render_gray = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY)
    art_gray = cv2.cvtColor(art_bgr, cv2.COLOR_BGR2GRAY)
    template, c0 = _centre(art_gray, *window)

    def match(sx: float, sy: float):
        scaled = _scale_exact(template, sx, sy)
        if scaled.shape[0] >= render_gray.shape[0] or scaled.shape[1] >= render_gray.shape[1] or min(scaled.shape) < 8:
            return -1.0, (0, 0), sx, sy
        _, value, _, loc = cv2.minMaxLoc(cv2.matchTemplate(render_gray, scaled, cv2.TM_CCOEFF_NORMED))
        return float(value), loc, sx, sy

    best = max((match(s, s) for s in np.linspace(*scales)), key=lambda item: item[0])
    for steps in (np.linspace(-0.012, 0.012, 13), np.linspace(-0.012, 0.012, 13), np.linspace(-0.002, 0.002, 9)):
        _, _, sx0, sy0 = best
        best = max([match(sx0 + dx, sy0 + dy) for dx in steps for dy in steps] + [best], key=lambda item: item[0])
    score, loc, sx, sy = best
    dx, dy = _subpixel(render_gray, _scale_exact(template, sx, sy), loc)
    # a scaled picture keeps pixel centres aligned, so its pixel 0 sits at (s - 1) / 2
    return Placement(score, sx, sy, loc[0] + dx + (sx - 1) / 2 - c0 * sx, loc[1] + dy + (sy - 1) / 2 - c0 * sy)


def _scale_exact(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale by exactly (sx, sy). cv2.resize alone rounds the size and so quantises the scale."""
    height, width = image.shape[:2]
    out_w, out_h = int(width * sx), int(height * sy)
    if out_w < 2 or out_h < 2:
        return np.zeros((max(out_h, 1), max(out_w, 1)), image.dtype)
    near_w, near_h = out_w + 2, out_h + 2
    near = cv2.resize(image, (near_w, near_h), interpolation=cv2.INTER_AREA)
    rx, ry = sx * width / near_w, sy * height / near_h
    matrix = np.float32([[rx, 0, (rx - 1) / 2], [0, ry, (ry - 1) / 2]])
    return cv2.warpAffine(near, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _subpixel(render_gray: np.ndarray, scaled: np.ndarray, loc: tuple[int, int]) -> tuple[float, float]:
    """Refine an integer template-match position by fitting a parabola through the response peak."""
    if scaled.shape[0] >= render_gray.shape[0] or scaled.shape[1] >= render_gray.shape[1]:
        return 0.0, 0.0
    response = cv2.matchTemplate(render_gray, scaled, cv2.TM_CCOEFF_NORMED)
    x, y = loc
    if not (0 < x < response.shape[1] - 1 and 0 < y < response.shape[0] - 1):
        return 0.0, 0.0

    def peak(before: float, centre: float, after: float) -> float:
        curve = before - 2 * centre + after
        return float(np.clip(0.5 * (before - after) / curve, -0.5, 0.5)) if curve < -1e-9 else 0.0

    return (peak(response[y, x - 1], response[y, x], response[y, x + 1]),
            peak(response[y - 1, x], response[y, x], response[y + 1, x]))


def colour_fit(art_low: np.ndarray, render: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain and bias that map art colours to render colours inside `mask`."""
    gain, bias = np.ones(3, np.float32), np.zeros(3, np.float32)
    if int(mask.sum()) < 16:
        return gain, bias
    for channel in range(3):
        gain[channel], bias[channel] = np.polyfit(art_low[..., channel][mask], render[..., channel][mask], 1)
    return gain, bias


def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)


def _structure_agrees(render_gray: np.ndarray, art_gray: np.ndarray, floor: float = 16.0, share: float = 0.5) -> np.ndarray:
    """True where the two pictures have the same local drawing, not merely the same colour.

    A frame part can match the art's colour by accident (dark rim next to a dark corner of the
    art), but it brings edges the art does not have. The gradient difference is allowed to grow
    with the edge strength, so real art edges survive small sharpness and alignment differences.
    """
    rx, ry = _gradients(render_gray)
    ax, ay = _gradients(art_gray)
    differ = cv2.GaussianBlur(np.hypot(rx - ax, ry - ay), (0, 0), 0.7)
    strength = cv2.GaussianBlur(np.maximum(np.hypot(rx, ry), np.hypot(ax, ay)), (0, 0), 0.7)
    return differ < floor + share * strength


def _enclosed(mask: np.ndarray) -> np.ndarray:
    """Pixels outside `mask` that cannot be reached from the picture border (holes in it)."""
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    cv2.floodFill(padded, None, (0, 0), 2)
    return padded[1:-1, 1:-1] == 0


def art_window(render_bgra: np.ndarray, art_bgr: np.ndarray, place: Placement, tolerance: float = 34,
               window=(0.3, 0.7)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask of render pixels that show the art, plus the art -> render colour transform.

    Starts from the matched centre, fits colours there, then keeps the connected region around it
    whose colour and drawing both agree with the art; frame parts, name banners and gems drawn
    over the art drop out. Holes left by strong art edges are filled only where the colour agrees.
    Returns (mask bool HxW, gain[3], bias[3]).
    """
    height, width = render_bgra.shape[:2]
    matrix = place.matrix()
    art_low_u8 = cv2.warpAffine(art_bgr, matrix, (width, height), flags=cv2.INTER_AREA)
    art_low = art_low_u8.astype(np.float32)
    inside = cv2.warpAffine(np.full(art_bgr.shape[:2], 255, np.uint8), matrix, (width, height)) > 250
    render = render_bgra[..., :3].astype(np.float32)
    opaque = render_bgra[..., 3] > 250
    render_gray = cv2.cvtColor(np.ascontiguousarray(render_bgra[..., :3]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    size = art_bgr.shape[0]
    c0, c1 = size * window[0], size * window[1]
    seed = np.zeros((height, width), bool)
    seed[max(int(place.oy + c0 * place.sy), 0):int(place.oy + c1 * place.sy),
         max(int(place.ox + c0 * place.sx), 0):int(place.ox + c1 * place.sx)] = True
    mask = seed
    gain, bias = np.ones(3, np.float32), np.zeros(3, np.float32)
    for _ in range(3):
        gain, bias = colour_fit(art_low, render, mask)
        fitted = np.clip(art_low * gain + bias, 0, 255)
        colour_ok = inside & opaque & (np.abs(fitted - render).max(axis=2) < tolerance)
        fitted_gray = cv2.cvtColor(fitted.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        strict = (colour_ok & _structure_agrees(render_gray, fitted_gray)).astype(np.uint8)
        strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(strict, 8)
        if count < 2:
            return np.zeros((height, width), bool), gain, bias
        overlap = [int((labels[seed] == label).sum()) for label in range(1, count)]
        chosen = np.argmax(overlap) if max(overlap) else np.argmax(stats[1:, cv2.CC_STAT_AREA])
        body = labels == (1 + int(chosen))
        body = body | (_enclosed(body) & colour_ok)
        mask = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & colour_ok
    return mask, gain, bias
