import json

import cv2
import numpy as np
import pytest

from cardv2 import cli, sources
from cardv2.animate import transfer
from cardv2.errors import CardError
from cardv2.imaging import load_bgra, save_png
from cardv2.onboard import ONBOARD_PLACEMENT, compose_onboard
from cardv2.pipeline import Builder
from cardv2.register import art_window, locate
from cardv2.upscale import bleed_edges

from .conftest import PLACE


# ---------- sources ----------
def test_render_urls_cover_every_variant_and_mode():
    assert sources.render_url("RLK_706") == "https://static.firestoneapp.com/cards/ruRU/512/RLK_706.png"
    assert sources.render_url("RLK_706", "signature").endswith("/ruRU/512/RLK_706_signature.png")
    assert sources.render_url("BG26_146", mode="bg", locale="enUS").endswith("/cards/bgs/enUS/512/BG26_146.png")
    assert sources.render_url("BG26_146", mode="bg", source="hsjson").endswith("/bgs/latest/ruRU/512x/BG26_146.png")
    assert sources.onboard_frame_url("golden").endswith("onboard_minion_frame_premium.png?v=2")


@pytest.mark.parametrize("card_id", ["../etc/passwd", "RLK 706", "", "a", "RLK_706.png", "x/y"])
def test_card_id_cannot_escape_the_url_path(card_id):
    with pytest.raises(CardError) as caught:
        sources.render_url(card_id)
    assert caught.value.code == "bad_card_id"


def test_unsupported_combinations_are_rejected():
    with pytest.raises(CardError) as caught:
        sources.render_url("RLK_706", "golden", source="hsjson")
    assert caught.value.code == "unsupported"
    with pytest.raises(CardError):
        sources.render_url("RLK_706", locale="russian")


def test_only_https_is_fetched_and_missing_files_are_reported(tmp_path):
    with pytest.raises(CardError) as caught:
        sources.fetch("http://example.com/a.png")
    assert caught.value.code == "bad_url"
    with pytest.raises(CardError) as caught:
        sources.resolve(str(tmp_path / "nope.png"))
    assert caught.value.code == "missing_file"


# ---------- registration ----------
def test_locate_recovers_scale_and_offset(render, art):
    place = locate(render[..., :3], art)
    assert place.score > 0.95
    assert abs(place.sx - PLACE["scale"]) < 0.01 and abs(place.sy - PLACE["scale"]) < 0.01
    assert abs(place.ox - PLACE["ox"]) < 2 and abs(place.oy - PLACE["oy"]) < 2


def test_locate_scores_low_for_a_different_picture(render, art):
    other = cv2.flip(cv2.cvtColor(cv2.cvtColor(art, cv2.COLOR_BGR2HSV), cv2.COLOR_BGR2RGB), 1)
    assert locate(render[..., :3], other).score < 0.8


def test_art_window_excludes_frame_and_banner(render, art):
    mask, gain, bias = art_window(render, art, locate(render[..., :3], art))
    assert mask[120, 155]            # inside the oval
    assert not mask[232, 155]        # under the banner
    assert not mask[400, 155]        # card body
    assert not mask[5, 5]            # transparent corner
    assert np.allclose(gain, 1, atol=0.1) and np.abs(bias).max() < 12


# ---------- upscaler helpers ----------
def test_bleed_edges_fills_transparent_pixels_with_neighbours():
    bgr = np.zeros((20, 20, 3), np.float32)
    alpha = np.zeros((20, 20, 1), np.float32)
    bgr[8:12, 8:12] = (0.2, 0.6, 0.9)
    alpha[8:12, 8:12] = 1
    out = bleed_edges(bgr, alpha)
    assert np.allclose(out[9, 9], (0.2, 0.6, 0.9))             # opaque pixels untouched
    assert np.allclose(out[9, 6], (0.2, 0.6, 0.9), atol=0.05)   # neighbour takes the card colour


# ---------- hand card ----------
def _files(tmp_path, render, art):
    render_path, art_path = tmp_path / "render.png", tmp_path / "art.png"
    save_png(render_path, render)
    save_png(art_path, np.dstack([art, np.full(art.shape[:2], 255, np.uint8)]))
    return str(render_path), str(art_path)


def test_hand_normal_swaps_art_and_crops_to_the_card(tmp_path, render, art, fake_upscaler):
    render_path, art_path = _files(tmp_path, render, art)
    out = tmp_path / "out.png"
    report = Builder(upscaler=fake_upscaler).hand("TEST_001", str(out), render=render_path, art=art_path, width=560)
    assert report["ok"] and report["art_swap"]["applied"] and report["art_swap"]["score"] > 0.95
    image = load_bgra(out)
    assert image.shape[1] == 560 and report["output"]["height"] == image.shape[0]
    assert abs(image.shape[0] / image.shape[1] - 411 / 281) < 0.02  # trimmed to the 281x411 card body
    # the swap must put the right part of the art in the right place: compare with the true art
    plain = Builder(upscaler=fake_upscaler).hand("TEST_001", str(tmp_path / "plain.png"), render=render_path,
                                                 art=art_path, width=560, art_swap="off")
    assert plain["art_swap"] == {"applied": False, "reason": "disabled"}
    scale = 560 / 281
    x0, y0, size = round((PLACE["ox"] - 20) * scale), round((PLACE["oy"] - 15) * scale), round(512 * PLACE["scale"] * scale)
    truth = cv2.resize(art, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    box = (slice(y0 + 150, y0 + 280), slice(x0 + 150, x0 + 300))

    def error(path):
        return float(np.abs(load_bgra(path)[..., :3].astype(np.float32)[box] - truth[150:280, 150:300]).mean())

    assert error(out) < 4.0


def test_hand_special_variants_keep_the_render_art(tmp_path, render, art, fake_upscaler):
    render_path, art_path = _files(tmp_path, render, art)
    report = Builder(upscaler=fake_upscaler).hand("TEST_001", str(tmp_path / "g.png"), variant="golden",
                                                  render=render_path, art=art_path, width=0, crop=False)
    assert report["art_swap"] == {"applied": False, "reason": "variant"}
    assert (report["output"]["width"], report["output"]["height"]) == (1280, 1760)


def test_hand_skips_swap_when_art_does_not_match(tmp_path, render, art, fake_upscaler):
    render_path, _ = _files(tmp_path, render, art)
    wrong = tmp_path / "wrong.png"
    save_png(wrong, cv2.flip(art, 1))
    builder = Builder(upscaler=fake_upscaler)
    report = builder.hand("TEST_001", str(tmp_path / "o.png"), render=render_path, art=str(wrong))
    assert report["art_swap"]["applied"] is False and report["art_swap"]["reason"] == "low_score"
    with pytest.raises(CardError) as caught:
        builder.hand("TEST_001", str(tmp_path / "o.png"), render=render_path, art=str(wrong), art_swap="on")
    assert caught.value.code == "registration_failed"


def test_upscaled_pictures_are_cached_by_content(tmp_path, render, art, fake_upscaler):
    render_path, art_path = _files(tmp_path, render, art)
    builder = Builder(upscaler=fake_upscaler)
    first = builder.hand("TEST_001", str(tmp_path / "a.png"), render=render_path, art=art_path)
    second = builder.hand("TEST_001", str(tmp_path / "b.png"), render=render_path, art=art_path)
    assert first["upscale"]["cached"] is False and second["upscale"]["cached"] is True
    assert np.array_equal(load_bgra(tmp_path / "a.png"), load_bgra(tmp_path / "b.png"))


# ---------- on-board minion ----------
def test_onboard_puts_art_inside_the_frame_only(tmp_path, art, fake_upscaler):
    frame = np.zeros((350, 300, 4), np.uint8)
    cv2.ellipse(frame, (148, 166), (99, 130), 0, 0, 360, (120, 120, 120, 255), 14)
    frame_path, art_path = tmp_path / "frame.png", tmp_path / "art.png"
    save_png(frame_path, frame)
    save_png(art_path, np.dstack([art, np.full(art.shape[:2], 255, np.uint8)]))
    out = tmp_path / "board.png"
    report = Builder(upscaler=fake_upscaler).onboard("TEST_001", str(out), frame=str(frame_path), art=str(art_path))
    image = load_bgra(out)
    assert (report["output"]["width"], report["output"]["height"]) == (1200, 1400) == (image.shape[1], image.shape[0])
    assert image[664, 592, 3] == 255 and image[10, 10, 3] == 0          # art opaque, corner transparent
    ax = round((148 - ONBOARD_PLACEMENT.ox) / ONBOARD_PLACEMENT.sx)
    ay = round((166 - ONBOARD_PLACEMENT.oy) / ONBOARD_PLACEMENT.sy)
    assert np.abs(image[665, 593, :3].astype(int) - art[ay, ax].astype(int)).max() < 40  # the right part of the art


def test_compose_onboard_keeps_frame_pixels(art):
    frame = np.zeros((350, 300, 4), np.uint8)
    cv2.ellipse(frame, (148, 166), (99, 130), 0, 0, 360, (10, 200, 30, 255), 14)
    out = compose_onboard(frame, art, 1.0)
    assert tuple(out[166, 49][:3]) == (10, 200, 30)


# ---------- animation ----------
def test_transfer_moves_only_the_art_window(render, art):
    card_up = cv2.resize(render, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    small = cv2.resize(render, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    frames = []
    for step in range(6):
        frame = small.copy()
        shift = 2.0 * np.sin(step / 6 * 2 * np.pi)
        moved = cv2.warpAffine(small[..., :3], np.float32([[1, 0, shift], [0, 1, 0]]), (small.shape[1], small.shape[0]),
                               borderMode=cv2.BORDER_REPLICATE)
        frame[40:110, 60:130, :3] = moved[40:110, 60:130]   # a patch inside the oval sways sideways
        frames.append(frame)
    steps = transfer(frames, art, render, card_up, out_width=562)
    info = next(steps)
    out = list(steps)
    assert info["frames"] == len(out) == 6 and out[0].shape[1] == 562 == info["width"]
    assert info["video_match"]["score"] > 0.9 and info["card_match"]["score"] > 0.9
    stack = np.stack([frame[..., :3].astype(np.float32) for frame in out])
    motion = stack.std(axis=0).max(axis=2)
    assert motion[150:300, 200:350].mean() > 0.5        # the art moves
    assert motion[620:800, 60:500].max() < 1e-3         # card body below the banner is frozen
    assert np.array_equal(out[0][..., 3], out[3][..., 3])


# ---------- command line ----------
def test_cli_reports_errors_as_json(capsys):
    assert cli.main(["hand", "../x", "--out", "o.png"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["error"]["code"] == "bad_card_id" and payload["command"] == "hand"


def test_cli_hand_and_batch(tmp_path, render, art, fake_upscaler, monkeypatch, capsys):
    render_path, art_path = _files(tmp_path, render, art)
    monkeypatch.setattr(cli, "_builder", lambda args: Builder(upscaler=fake_upscaler))
    out = tmp_path / "cli.png"
    assert cli.main(["hand", "TEST_001", "--out", str(out), "--render", render_path, "--art", art_path, "--width", "400"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and payload["output"]["width"] == 400 and out.exists()

    # batch: an existing output is skipped, a bad id fails without stopping the run
    ids = tmp_path / "ids.txt"
    ids.write_text("# comment\nTEST_001\nbad id\n", encoding="utf-8")
    (tmp_path / "batch").mkdir()
    (tmp_path / "batch" / "TEST_001.png").write_bytes(b"x")
    assert cli.main(["batch", "--ids-file", str(ids), "--out-dir", str(tmp_path / "batch")]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [line["ok"] for line in lines] == [True, False]
    assert lines[0]["skipped"] == "exists" and lines[1]["error"]["code"] == "bad_card_id"


def test_cli_rejects_unknown_variant():
    with pytest.raises(SystemExit) as caught:
        cli.main(["hand", "RLK_706", "--out", "o.png", "--variant", "platinum"])
    assert caught.value.code == 2
