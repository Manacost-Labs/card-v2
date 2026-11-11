import cv2
import numpy as np
import pytest


class FakeUpscaler:
    """Stands in for the neural upscaler: plain 4x bicubic, no model needed."""

    scale = 4
    name = "fake"
    device = "cpu"

    def rgb(self, bgr):
        return cv2.resize(bgr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    def rgba(self, bgra):
        return cv2.resize(bgra, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDV2_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def fake_upscaler():
    return FakeUpscaler()


@pytest.fixture
def art():
    """A 512x512 picture with structure at several scales, so template matching has something to hold."""
    rng = np.random.default_rng(7)
    image = np.zeros((512, 512, 3), np.float32)
    for sigma, weight in ((40, 120), (12, 70), (4, 40)):
        image += cv2.GaussianBlur(rng.random((512, 512, 3)).astype(np.float32) - 0.5, (0, 0), sigma) * weight * sigma
    image = np.clip(image + 128, 0, 255).astype(np.uint8)
    cv2.circle(image, (200, 220), 60, (40, 60, 220), -1)
    cv2.rectangle(image, (300, 260), (400, 380), (30, 200, 90), -1)
    cv2.line(image, (120, 400), (420, 120), (240, 240, 240), 6)
    return image


PLACE = {"scale": 0.45, "ox": 40.0, "oy": 30.0}


@pytest.fixture
def render(art):
    """A 320x440 card: grey body, the art in an oval window, a banner drawn across the art's lower part."""
    height, width = 440, 320
    card = np.zeros((height, width, 4), np.uint8)
    cv2.rectangle(card, (20, 15), (300, 425), (90, 90, 96, 255), -1)
    matrix = np.float32([[PLACE["scale"], 0, PLACE["ox"]], [0, PLACE["scale"], PLACE["oy"]]])
    placed = cv2.warpAffine(art, matrix, (width, height), flags=cv2.INTER_AREA)
    window = np.zeros((height, width), np.uint8)
    cv2.ellipse(window, (155, 145), (95, 105), 0, 0, 360, 255, -1)
    card[..., :3][window > 0] = placed[window > 0]
    cv2.rectangle(card, (30, 215), (290, 250), (200, 190, 160, 255), -1)
    return card
