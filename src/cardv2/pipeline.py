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
from .imaging import crop_and_fit, load_bgr, load_bgra, save_png
from .onboard import ONBOARD_ART_SIZE, compose_onboard
from .register import locate

DEFAULT_MIN_ART_SCORE = 0.95


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
             art_swap: str = "auto", min_art_score: float = DEFAULT_MIN_ART_SCORE, crop: bool = True) -> dict:
        sources.check_card_id(card_id)
        low, up, report = self._hand_layers(card_id, variant, mode, locale, source, render, art, art_swap, min_art_score)
        final = crop_and_fit(up, low, width, crop)
        save_png(out, final)
        return {"ok": True, "command": "hand", "card": card_id, "variant": variant, "mode": mode, "locale": locale,
                **report, "output": {"path": str(out), "width": final.shape[1], "height": final.shape[0]}}

    # ---- on-board minion ----
    def onboard(self, card_id: str, out: str, golden: bool = False, art: str | None = None, frame: str | None = None,
                width: int = 0) -> dict:
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
        final = crop_and_fit(composed, frame_low, width, crop=False)
        save_png(out, final)
        return {"ok": True, "command": "onboard", "card": card_id, "variant": kind,
                "source": {"frame": frame_ref, "art": art_ref}, "upscale": upscale_info,
                "output": {"path": str(out), "width": final.shape[1], "height": final.shape[0]}}

    # ---- animated hand card ----
    def animate(self, card_id: str, video: str, out_dir: str, variant: str = "golden", mode: str = "constructed",
                locale: str = "ruRU", source: str = "firestone", render: str | None = None, art: str | None = None,
                width: int = 1000, formats: tuple[str, ...] = ("webm", "mp4"), keep_frames: bool = False) -> dict:
        sources.check_card_id(card_id)
        anim.require_ffmpeg()
        video_path = sources.resolve(video, kind="video", max_bytes=sources.VIDEO_MAX_BYTES)
        low, up, report = self._hand_layers(card_id, variant, mode, locale, source, render, None, "off", 1.0)
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
            frames_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(steps, start=1):
                save_png(frames_dir / f"f_{index:04d}.png", frame)
            outputs = {name: out_root / f"{stem}.{name}" for name in formats}
            anim.encode(frames_dir, fps, (info["width"], info["height"]), outputs.get("webm"), outputs.get("mp4"))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        report["source"].update({"video": video, "art": art_ref})
        report.pop("art_swap", None)
        return {"ok": True, "command": "animate", "card": card_id, "variant": variant, "locale": locale, **report,
                "animation": {**info, "fps": fps},
                "output": {**{name: str(path) for name, path in outputs.items()},
                           **({"frames": str(frames_dir)} if keep_frames else {})}}
