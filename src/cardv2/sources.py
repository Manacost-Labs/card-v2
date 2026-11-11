"""Where source pictures come from, and a small download cache.

Nothing from these sources is stored in the repository: files are fetched on demand into the
cache directory (CARDV2_CACHE, default ~/.cache/cardv2).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .errors import CardError

CARD_ID = re.compile(r"^[A-Za-z0-9_]{2,64}$")
LOCALE = re.compile(r"^[a-z]{2}[A-Z]{2}$")
VARIANTS = ("normal", "golden", "signature", "diamond")
MODES = ("constructed", "bg")
RENDER_SOURCES = ("firestone", "hsjson")

HSJSON_ART = "https://art.hearthstonejson.com/v1/orig/{id}.png"
HSJSON_RENDER = "https://art.hearthstonejson.com/v1/render/latest/{locale}/512x/{id}.png"
HSJSON_BGS = "https://art.hearthstonejson.com/v1/bgs/latest/{locale}/512x/{id}.png"
FIRESTONE_CARD = "https://static.firestoneapp.com/cards/{bgs}{locale}/512/{id}{suffix}.png"
ONBOARD_FRAME = "https://static.zerotoheroes.com/hearthstone/asset/coliseum/images/{name}.png?v=2"
ONBOARD_FRAMES = {"normal": "onboard_minion_frame", "golden": "onboard_minion_frame_premium"}

MODEL_NAME = "RealESRGAN_x4plus"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"

IMAGE_MAX_BYTES = 32 * 1024 * 1024
VIDEO_MAX_BYTES = 128 * 1024 * 1024
MODEL_MAX_BYTES = 256 * 1024 * 1024
USER_AGENT = "cardv2/0.1"


def cache_dir() -> Path:
    root = os.environ.get("CARDV2_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "cardv2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_card_id(card_id: str) -> str:
    if not CARD_ID.match(card_id or ""):
        raise CardError("bad_card_id", f"card id must match {CARD_ID.pattern}: {card_id!r}")
    return card_id


def check_locale(locale: str) -> str:
    if not LOCALE.match(locale or ""):
        raise CardError("bad_locale", f"locale must look like ruRU or enUS: {locale!r}")
    return locale


def render_url(card_id: str, variant: str = "normal", mode: str = "constructed", locale: str = "ruRU",
               source: str = "firestone") -> str:
    """URL of the 512 px hand-card render for this card."""
    check_card_id(card_id)
    check_locale(locale)
    if variant not in VARIANTS:
        raise CardError("bad_variant", f"variant must be one of {VARIANTS}")
    if mode not in MODES:
        raise CardError("bad_mode", f"mode must be one of {MODES}")
    if source == "firestone":
        suffix = "" if variant == "normal" else f"_{variant}"
        return FIRESTONE_CARD.format(bgs="bgs/" if mode == "bg" else "", locale=locale, id=card_id, suffix=suffix)
    if source == "hsjson":
        if variant != "normal":
            raise CardError("unsupported", "HearthstoneJSON has only normal renders; use --source firestone")
        template = HSJSON_BGS if mode == "bg" else HSJSON_RENDER
        return template.format(locale=locale, id=card_id)
    raise CardError("bad_source", f"source must be one of {RENDER_SOURCES}")


def art_url(card_id: str) -> str:
    return HSJSON_ART.format(id=check_card_id(card_id))


def onboard_frame_url(kind: str) -> str:
    if kind not in ONBOARD_FRAMES:
        raise CardError("bad_variant", f"on-board frame must be one of {tuple(ONBOARD_FRAMES)}")
    return ONBOARD_FRAME.format(name=ONBOARD_FRAMES[kind])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like(kind: str, head: bytes) -> bool:
    if kind == "image":
        return head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8") or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")
    if kind == "video":
        return head.startswith(b"\x1a\x45\xdf\xa3") or head[4:8] == b"ftyp"  # webm/mkv, mp4
    return True


def fetch(url: str, kind: str = "image", max_bytes: int = IMAGE_MAX_BYTES, timeout: int = 60) -> Path:
    """Download an https URL into the cache (once) and return the local path."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise CardError("bad_url", f"only https URLs are fetched: {url!r}")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(parsed.path).name)[-80:] or "file"
    target = cache_dir() / kind / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}-{name}"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            head = response.read(16)
            if not _looks_like(kind, head):
                raise CardError("not_found", f"{url} did not return a {kind} (the card or variant may not exist)")
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".part-")
            size = len(head)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(head)
                    while chunk := response.read(1 << 20):
                        size += len(chunk)
                        if size > max_bytes:
                            raise CardError("too_large", f"{url} is larger than {max_bytes} bytes")
                        out.write(chunk)
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    except urllib.error.HTTPError as error:
        raise CardError("not_found", f"{url} returned HTTP {error.code} (the card or variant may not exist)") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise CardError("network", f"could not download {url}: {error}") from error
    return target


def resolve(value: str, kind: str = "image", max_bytes: int = IMAGE_MAX_BYTES) -> Path:
    """A user-supplied input: an existing local file, or an https URL to fetch."""
    if value.startswith("https://"):
        return fetch(value, kind, max_bytes)
    path = Path(value).expanduser()
    if not path.is_file():
        raise CardError("missing_file", f"file not found: {value}")
    return path


def model_path() -> Path:
    """Local path of the upscaler weights, downloaded and checksum-verified on first use."""
    override = os.environ.get("CARDV2_MODEL")
    if override:
        path = Path(override)
        if not path.is_file():
            raise CardError("missing_file", f"CARDV2_MODEL does not exist: {override}")
        return path
    path = cache_dir() / "models" / f"{MODEL_NAME}.pth"
    if not path.exists():
        fetched = fetch(MODEL_URL, kind="models-download", max_bytes=MODEL_MAX_BYTES, timeout=300)
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(fetched, path)
    if sha256_file(path) != MODEL_SHA256:
        path.unlink()
        raise CardError("bad_model", "downloaded model weights failed the checksum and were removed")
    return path
