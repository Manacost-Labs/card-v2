"""Animate the art of a static HD card with the motion of a small animated render of the same card.

The HD card is untouched outside its art window. Inside, each output frame is the HD art warped
by the optical flow measured in the small video, plus the video's low-frequency brightness change
(glow, flames). Both pictures are registered to the same base art, which gives the mapping
video -> HD card even when the two renders use different layouts or languages.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .errors import CardError
from .imaging import alpha_bbox, load_bgra
from .register import art_window, locate

MIN_SCORE = 0.8


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise CardError("missing_dependency", "ffmpeg is required for animated cards")
    return path


def _run(cmd: list[str]) -> None:
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise CardError("ffmpeg_failed", done.stderr.strip()[-400:] or "ffmpeg failed")


def decode_frames(video: Path, out_dir: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame as BGRA (keeping VP9 alpha) and return them with the frame rate."""
    ffmpeg = require_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    codec = []
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True).stderr
    if "Video: vp9" in probe:
        codec = ["-c:v", "libvpx-vp9"]  # the native vp9 decoder drops the alpha plane
    elif "Video: vp8" in probe:
        codec = ["-c:v", "libvpx"]
    _run([ffmpeg, "-v", "error", "-y", *codec, "-i", str(video), "-pix_fmt", "rgba", str(out_dir / "f_%04d.png")])
    files = sorted(out_dir.glob("f_*.png"))
    if len(files) < 2:
        raise CardError("bad_video", "the video has fewer than two frames")
    fps = 20.0
    for token in probe.split(","):
        token = token.strip()
        if token.endswith(" fps"):
            try:
                fps = float(token.split()[0])
            except ValueError:
                pass
            break
    return [load_bgra(path) for path in files], fps


def transfer(frames: list[np.ndarray], art: np.ndarray, card_low: np.ndarray, card_up: np.ndarray,
             out_width: int, zoom: int = 2):
    """Yield (info dict) first, then one BGRA output frame per input frame."""
    count = len(frames)
    h, w = frames[0].shape[:2]
    rest = np.median(np.stack([frame[..., :3] for frame in frames]), axis=0).astype(np.uint8)
    rest_bgra = np.dstack([rest, frames[0][..., 3]])
    v = locate(rest, art)
    c = locate(card_low[..., :3], art)
    if min(v.score, c.score) < MIN_SCORE:
        raise CardError("registration_failed",
                        f"could not match the art (video {v.score:.2f}, card {c.score:.2f}, need {MIN_SCORE})")
    video_mask, _, _ = art_window(rest_bgra, art, v, tolerance=40)
    card_mask, _, _ = art_window(card_low, art, c, tolerance=40)

    # optical flow, frame -> rest pose, at `zoom` times the video resolution
    size2 = (w * zoom, h * zoom)
    rest2 = cv2.resize(rest, size2, interpolation=cv2.INTER_CUBIC)
    rest2_gray = cv2.cvtColor(rest2, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    frames2, flows = [], np.zeros((count, h * zoom, w * zoom, 2), np.float32)
    for index, frame in enumerate(frames):
        enlarged = cv2.resize(np.ascontiguousarray(frame[..., :3]), size2, interpolation=cv2.INTER_CUBIC)
        frames2.append(enlarged)
        flows[index] = dis.calc(cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY), rest2_gray, None)
    soft_v = cv2.erode(video_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    soft_v = cv2.resize(cv2.GaussianBlur(soft_v, (0, 0), 1.5), size2, interpolation=cv2.INTER_LINEAR)
    for index in range(count):
        flows[index] = cv2.GaussianBlur(flows[index], (0, 0), 2.5) * soft_v[..., None]
    flows = 0.25 * np.roll(flows, 1, axis=0) + 0.5 * flows + 0.25 * np.roll(flows, -1, axis=0)

    # mapping: zoomed-video pixels -> HD card pixels (through the shared art coordinates)
    H, W = card_up.shape[:2]
    k = W / card_low.shape[1]
    ax, ay = c.sx / v.sx, c.sy / v.sy
    m00, m11 = float(k * ax / zoom), float(k * ay / zoom)
    centre = 0.5 / zoom - 0.5
    to_card = np.float32([[m00, 0, k * (ax * (centre - v.ox) + c.ox) + (k - 1) / 2],
                          [0, m11, k * (ay * (centre - v.oy) + c.oy) + (k - 1) / 2]])

    def to_hd(image, interp):
        return cv2.warpAffine(image, to_card, (W, H), flags=interp)

    soft_c = cv2.erode(card_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    soft_c = cv2.GaussianBlur(cv2.resize(soft_c, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), k * 0.9)
    mask = np.clip(np.minimum(to_hd(soft_v, cv2.INTER_LINEAR), soft_c), 0, 1)
    ys, xs = np.where(mask > 0.002)
    if xs.size == 0:
        raise CardError("registration_failed", "the art windows of the video and the card do not overlap")
    pad = 24
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, H)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, W)
    roi_mask = mask[y0:y1, x0:x1, None]
    hy, hx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    gy, gx = np.mgrid[0:h * zoom, 0:w * zoom].astype(np.float32)
    card = card_up.astype(np.float32)
    card_bgr = np.ascontiguousarray(card[..., :3])
    bx0, by0, bx1, by1 = (int(round(value * k)) for value in alpha_bbox(card_low))
    out_width = out_width or (bx1 - bx0)
    out_height = max(2, round((by1 - by0) * out_width / (bx1 - bx0)))

    yield {
        "frames": count,
        "width": out_width,
        "height": out_height,
        "video_match": v.as_dict(),
        "card_match": c.as_dict(),
        "enlargement": round(m00 * zoom, 2),
        "flow_max_px": round(float(np.abs(flows).max()) / zoom, 2),
    }
    for index in range(count):
        flow = flows[index]
        moved_rest = cv2.remap(rest2, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        light = frames2[index].astype(np.float32) - moved_rest.astype(np.float32)
        light = cv2.GaussianBlur(light, (0, 0), 1.6) * soft_v[..., None]
        hd_flow = to_hd(flow, cv2.INTER_LINEAR)[y0:y1, x0:x1]
        hd_light = to_hd(light, cv2.INTER_CUBIC)[y0:y1, x0:x1]
        moved = cv2.remap(card_bgr, (hx + hd_flow[..., 0] * m00).astype(np.float32),
                          (hy + hd_flow[..., 1] * m11).astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        frame = card.copy()
        frame[y0:y1, x0:x1, :3] = frame[y0:y1, x0:x1, :3] * (1 - roi_mask) + np.clip(moved + hd_light, 0, 255) * roi_mask
        cropped = np.clip(frame[by0:by1, bx0:bx1] + 0.5, 0, 255).astype(np.uint8)
        yield cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_AREA)


def encode(frames_dir: Path, fps: float, size: tuple[int, int], webm: Path | None, mp4: Path | None,
           background: str = "18181c") -> None:
    """Encode numbered PNG frames: VP9 WebM keeps transparency, H.264 MP4 is flattened on a colour."""
    ffmpeg = require_ffmpeg()
    pattern = str(frames_dir / "f_%04d.png")
    rate = f"{fps:g}"
    if webm:
        webm.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg, "-v", "error", "-y", "-framerate", rate, "-i", pattern, "-c:v", "libvpx-vp9",
              "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "26", "-row-mt", "1", str(webm)])
    if mp4:
        mp4.parent.mkdir(parents=True, exist_ok=True)
        width, height = size[0] // 2 * 2, size[1] // 2 * 2  # H.264 needs even dimensions
        flatten = f"[1:v]crop={width}:{height}:0:0[card];[0:v][card]overlay=shortest=1,format=yuv420p"
        _run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{background}:s={width}x{height}:r={rate}",
              "-framerate", rate, "-i", pattern, "-filter_complex", flatten,
              "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-movflags", "+faststart", str(mp4)])
