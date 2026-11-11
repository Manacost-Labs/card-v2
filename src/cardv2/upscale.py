"""4x neural upscaling (Real-ESRGAN) of RGB and RGBA pictures.

torch and spandrel are imported lazily so the rest of the package (and its tests) work without
them; anything exposing `scale`, `name`, `device`, `rgb()` and `rgba()` can stand in for Upscaler.
"""
from __future__ import annotations

import cv2
import numpy as np

from .errors import CardError
from .sources import MODEL_NAME, model_path


def bleed_edges(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fill transparent pixels with nearby card colours.

    bgr: float32 HxWx3 in [0,1]; alpha: float32 HxWx1 in [0,1]. Without this the model pulls the
    (arbitrary) colour of fully transparent pixels into the card's outline.
    """
    filled, cover = bgr * alpha, alpha.copy()
    for radius in (1, 2, 4, 8, 16):
        spread_colour = cv2.GaussianBlur(filled, (0, 0), radius)
        spread_cover = cv2.GaussianBlur(cover, (0, 0), radius)
        if spread_cover.ndim == 2:
            spread_cover = spread_cover[..., None]
        colour = np.where(spread_cover > 1e-3, spread_colour / np.maximum(spread_cover, 1e-3), 0)
        reach = np.minimum(spread_cover * 4, 1)
        empty = cover < 1e-3
        filled = np.where(empty, colour * reach, filled)
        cover = np.where(empty, reach, cover)
    spread = np.clip(filled / np.maximum(cover, 1e-3), 0, 1)
    return np.where(alpha > 1e-3, bgr, spread).astype(np.float32)


class Upscaler:
    scale = 4
    name = MODEL_NAME

    def __init__(self, device: str = "auto", threads: int | None = None, tile: int = 0):
        try:
            import torch
            from spandrel import ModelLoader
        except ImportError as error:
            raise CardError("missing_dependency", f"torch and spandrel are required for upscaling: {error}") from error
        self._torch = torch
        if threads:
            torch.set_num_threads(threads)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tile = tile
        self._model = ModelLoader().load_from_file(str(model_path())).eval().to(device)

    def _forward(self, tensor):
        torch = self._torch
        _, _, height, width = tensor.shape
        tile, pad, s = self.tile, 16, self.scale
        with torch.no_grad():
            if not tile or (height <= tile and width <= tile):
                return self._model(tensor)
            out = torch.zeros((1, 3, height * s, width * s), dtype=tensor.dtype, device=tensor.device)
            for y in range(0, height, tile):
                for x in range(0, width, tile):
                    y1, x1 = min(y + tile, height), min(x + tile, width)
                    ys, xs = max(y - pad, 0), max(x - pad, 0)
                    ye, xe = min(y1 + pad, height), min(x1 + pad, width)
                    piece = self._model(tensor[..., ys:ye, xs:xe])
                    out[..., y * s:y1 * s, x * s:x1 * s] = piece[..., (y - ys) * s:(y1 - ys) * s, (x - xs) * s:(x1 - xs) * s]
            return out

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """rgb float32 HxWx3 in [0,1] -> 4x upscaled, same layout."""
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        out = self._forward(tensor).squeeze(0).clamp(0, 1).float().cpu().numpy()
        return out.transpose(1, 2, 0)

    def rgb(self, bgr: np.ndarray) -> np.ndarray:
        """uint8 BGR -> uint8 BGR, 4x."""
        out = self._run(bgr[..., ::-1].astype(np.float32) / 255.0)
        return np.ascontiguousarray((out[..., ::-1] * 255 + 0.5).astype(np.uint8))

    def rgba(self, bgra: np.ndarray) -> np.ndarray:
        """uint8 BGRA -> uint8 BGRA, 4x. Colour and alpha go through the model separately."""
        data = bgra.astype(np.float32) / 255.0
        alpha = data[..., 3:4]
        base = bleed_edges(data[..., :3], alpha)
        colour = self._run(np.ascontiguousarray(base[..., ::-1]))[..., ::-1]
        if float(alpha.min()) > 0.999:
            up_alpha = np.ones(colour.shape[:2] + (1,), np.float32)
        else:
            up_alpha = self._run(np.repeat(alpha, 3, axis=2)).mean(axis=2, keepdims=True)
        out = np.concatenate([colour, up_alpha], axis=2)
        return (out * 255 + 0.5).astype(np.uint8)
