"""High-level builds: hand card, on-board minion, animated card. Each returns a JSON-able report."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from . import animate as anim
from . import sources
from .artswap import swap_art
from .errors import CardError
from .imaging import compare_sheet, crop_and_fit, fidelity, load_bgr, load_bgra, save_image, save_png
from .onboard import ONBOARD_ART_SIZE, compose_onboard
from .register import locate

DEFAULT_MIN_ART_SCORE = 0.95
# Fidelity limits on a 0-255 scale (see imaging.fidelity). Eight real builds across every variant
# gave mean 3.4-4.8 and region 5.2-13.3 when faithful. Damage measured during development: art
# painted over the frame, region 21-50; layers shifted by 2 source px, mean 13-19; a wrong art,
# mean 33-48. The margins are narrow and rest on few cards, so exceeding a limit is reported as a
# warning, never as a failure.
DEFAULT_MAX_DIFF = 6.0
DEFAULT_MAX_REGION_DIFF = 18.0
DEFAULT_LOOP_BLEND = 8
ANIMATION_FORMATS = ("webm", "webp", "mp4")


def _check_fidelity(low: np.ndarray, up: np.ndarray, max_diff: float, max_region_diff: float) -> tuple[dict, list[str]]:
    measured = fidelity(low, up)
    faithful = measured["mean_diff"] <= max_diff and measured["region_diff"] <= max_region_diff
    return ({**measured, "max_diff": max_diff, "max_region_diff": max_region_diff, "faithful": faithful},
            [] if faithful else ["fidelity"])


class Builder:
    def __init__(self, upscaler=None, device: str = "auto", threads: int | None = None, tile: int = 0,
                 use_cache: bool = True):
        self._upscaler = upscaler
        self._options = {"device": device, "threads": threads, "tile": tile}
        self.use_cache = use_cache

    @property
    def upscaler(self):
        if self._upscaler is None:
            from .upscale import Upscaler

            self._upscaler = Upscaler(**self._options)
        return self._upscaler

    # ---- upscaling with an on-disk cache keyed by the source file's content ----
    def _upscaled(self, path: Path, alpha: bool) -> tuple[np.ndarray, dict]:
        key = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        cached = sources.cache_dir() / "upscaled" / f"{key}-{self.upscaler.name}-{'rgba' if alpha else 'rgb'}.png"
        started = time.monotonic()
        if self.use_cache and cached.exists():
            image = load_bgra(cached) if alpha else load_bgr(cached)
            hit = True
        else:
            image = self.upscaler.rgba(load_bgra(path)) if alpha else self.upscaler.rgb(load_bgr(path))
            if self.use_cache:
                save_png(cached, image)
            hit = False
        return image, {"model": self.upscaler.name, "scale": self.upscaler.scale, "device": self.upscaler.device,
                       "cached": hit, "seconds": round(time.monotonic() - started, 1)}

    # ---- hand card ----
    def _hand_layers(self, card_id: str, variant: str, mode: str, locale: str, source: str, render: str | None,
                     art: str | None, art_swap: str, min_art_score: float):
        render_ref = render or sources.render_url(card_id, variant, mode, locale, source)
        render_path = sources.resolve(render_ref)
        low = load_bgra(render_path)
        up, upscale_info = self._upscaled(render_path, alpha=True)
        report = {"source": {"render": render_ref}, "upscale": upscale_info}

        want_swap = art_swap == "on" or (art_swap == "auto" and variant == "normal")
        if not want_swap:
            report["art_swap"] = {"applied": False, "reason": "disabled" if art_swap == "off" else "variant"}
            return low, up, report
        art_ref = art or sources.art_url(card_id)
        try:
            art_path = sources.resolve(art_ref)
        except CardError as error:
            if art_swap == "on":
                raise
            report["art_swap"] = {"applied": False, "reason": "art_unavailable", "detail": error.message}
            return low, up, report
        art_image = load_bgr(art_path)
        place = locate(low[..., :3], art_image)
        swap = {"applied": False, "score": round(place.score, 3), "min_score": min_art_score}
        if place.score < min_art_score:
            if art_swap == "on":
                raise CardError("registration_failed", f"art match {place.score:.3f} is below {min_art_score}")
            swap["reason"] = "low_score"
        else:
            art_hi, _ = self._upscaled(art_path, alpha=False)
            up, detail = swap_art(low, up, art_image, art_hi, place)
            swap.update(detail)
        report["source"]["art"] = art_ref
        report["art_swap"] = swap
        return low, up, report

    def hand(self, card_id: str, out: str, variant: str = "normal", mode: str = "constructed", locale: str = "ruRU",
             source: str = "firestone", render: str | None = None, art: str | None = None, width: int = 1000,
             art_swap: str = "auto", min_art_score: float = DEFAULT_MIN_ART_SCORE, crop: bool = True,
             quality: int = 90, max_diff: float = DEFAULT_MAX_DIFF,
             max_region_diff: float = DEFAULT_MAX_REGION_DIFF) -> dict:
        sources.check_card_id(card_id)
        low, up, report = self._hand_layers(card_id, variant, mode, locale, source, render, art, art_swap, min_art_score)
        check, warnings = _check_fidelity(low, up, max_diff, max_region_diff)
        final = crop_and_fit(up, low, width, crop)
        save_image(out, final, quality)
        return {"ok": True, "command": "hand", "card": card_id, "variant": variant, "mode": mode, "locale": locale,
                **report, "fidelity": check, "warnings": warnings,
                "output": {"path": str(out), "width": final.shape[1], "height": final.shape[0]}}

    # ---- on-board minion ----
    def onboard(self, card_id: str, out: str, golden: bool = False, art: str | None = None, frame: str | None = None,
                width: int = 0, quality: int = 90, max_diff: float = DEFAULT_MAX_DIFF,
                max_region_diff: float = DEFAULT_MAX_REGION_DIFF) -> dict:
        sources.check_card_id(card_id)
        kind = "golden" if golden else "normal"
        frame_ref = frame or sources.onboard_frame_url(kind)
        frame_path = sources.resolve(frame_ref)
        art_ref = art or sources.art_url(card_id)
        art_path = sources.resolve(art_ref)
        if load_bgr(art_path).shape[0] != ONBOARD_ART_SIZE:
            raise CardError("bad_image", f"on-board art must be {ONBOARD_ART_SIZE}x{ONBOARD_ART_SIZE}")
        frame_low = load_bgra(frame_path)
        frame_up, upscale_info = self._upscaled(frame_path, alpha=True)
        art_hi, _ = self._upscaled(art_path, alpha=False)
        composed = compose_onboard(frame_up, art_hi, frame_up.shape[1] / frame_low.shape[1])
        check, warnings = _check_fidelity(frame_low, composed, max_diff, max_region_diff)  # frame only; the art is new
        final = crop_and_fit(composed, frame_low, width, crop=False)
        save_image(out, final, quality)
        return {"ok": True, "command": "onboard", "card": card_id, "variant": kind,
                "source": {"frame": frame_ref, "art": art_ref}, "upscale": upscale_info,
                "fidelity": check, "warnings": warnings,
                "output": {"path": str(out), "width": final.shape[1], "height": final.shape[0]}}

    # ---- animated hand card ----
    def animate(self, card_id: str, video: str, out_dir: str, variant: str = "golden", mode: str = "constructed",
                locale: str = "ruRU", source: str = "firestone", render: str | None = None, art: str | None = None,
                width: int = 1000, formats: tuple[str, ...] = ("webm", "mp4"), keep_frames: bool = False,
                loop_blend: int = DEFAULT_LOOP_BLEND, max_diff: float = DEFAULT_MAX_DIFF,
                max_region_diff: float = DEFAULT_MAX_REGION_DIFF) -> dict:
        sources.check_card_id(card_id)
        if not formats or any(name not in ANIMATION_FORMATS for name in formats):
            raise CardError("bad_format", f"animation formats are {ANIMATION_FORMATS}")
        anim.require_ffmpeg()
        video_path = sources.resolve(video, kind="video", max_bytes=sources.VIDEO_MAX_BYTES)
        low, up, report = self._hand_layers(card_id, variant, mode, locale, source, render, None, "off", 1.0)
        check, warnings = _check_fidelity(low, up, max_diff, max_region_diff)
        art_ref = art or sources.art_url(card_id)
        art_image = load_bgr(sources.resolve(art_ref))
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        stem = f"{card_id}-{variant}-animated"
        work = Path(tempfile.mkdtemp(prefix="cardv2-anim-"))
        try:
            frames, fps = anim.decode_frames(video_path, work / "in")
            steps = anim.transfer(frames, art_image, low, up, width)
            info = next(steps)
            frames_dir = out_root / f"{stem}-frames" if keep_frames else work / "out"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True)
            for index, frame in enumerate(steps, start=1):
                save_png(frames_dir / f"f_{index:04d}.png", frame)
            loop = anim.loop_blend(frames_dir, loop_blend)
            info["frames"] = loop.pop("frames")
            outputs = {name: out_root / f"{stem}.{name}" for name in formats}
            anim.encode(frames_dir, fps, (info["width"], info["height"]), outputs)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        report["source"].update({"video": video, "art": art_ref})
        report.pop("art_swap", None)
        return {"ok": True, "command": "animate", "card": card_id, "variant": variant, "locale": locale, **report,
                "fidelity": check, "warnings": warnings,
                "animation": {**info, "fps": fps, "seconds": round(info["frames"] / fps, 2), "loop": loop},
                "output": {**{name: str(path) for name, path in outputs.items()},
                           **({"frames": str(frames_dir)} if keep_frames else {})}}

    # ---- before/after sheet ----
    @staticmethod
    def compare(before: str, after: str, out: str, zoom: tuple[float, float, float, float] | None = None,
                height: int = 0, quality: int = 90) -> dict:
        source = load_bgra(sources.resolve(before))
        result = load_bgra(sources.resolve(after))
        sheet = compare_sheet(source, result, zoom, height)
        save_image(out, sheet, quality)
        return {"ok": True, "command": "compare", "before": before, "after": after,
                "enlargement": round(result.shape[1] / max(1, source.shape[1]), 2),
                "output": {"path": str(out), "width": sheet.shape[1], "height": sheet.shape[0]}}
