import json
import shutil

import cv2
import numpy as np
import pytest

from cardv2 import cli, sources
from cardv2.animate import decode_frames, encode, loop_blend, transfer
from cardv2.errors import CardError
from cardv2.imaging import compare_sheet, fidelity, load_bgra, save_image, save_png
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
    assert [line["ok"] for line in lines] == [True, False, False]
    assert lines[0]["skipped"] == "exists" and lines[1]["error"]["code"] == "bad_card_id"
    assert lines[2] == {"ok": False, "command": "batch", "built": 0, "skipped": 1, "failed": 1, "warned": 0}


def test_cli_rejects_unknown_variant():
    with pytest.raises(SystemExit) as caught:
        cli.main(["hand", "RLK_706", "--out", "o.png", "--variant", "platinum"])
    assert caught.value.code == 2


# ---------- art window must not leak onto the frame ----------
def test_art_window_stops_at_a_rim_even_when_the_frame_has_the_art_colour(render, art):
    """A frame part with the art's own colours right next to the window must stay frame."""
    place = locate(render[..., :3], art)
    matrix = np.float32([[PLACE["scale"], 0, PLACE["ox"]], [0, PLACE["scale"], PLACE["oy"]]])
    placed = cv2.warpAffine(art, matrix, (render.shape[1], render.shape[0]), flags=cv2.INTER_AREA)
    leaky = render.copy()
    leaky[60:230, 25:62, :3] = placed[60:230, 25:62]                       # same colours as the art, outside the oval
    cv2.ellipse(leaky, (155, 145), (95, 105), 0, 0, 360, (20, 20, 20, 255), 3)  # the rim between them
    cv2.line(leaky, (35, 90), (50, 150), (240, 200, 60, 255), 3)            # a glyph drawn on the frame
    mask, _, _ = art_window(leaky, art, place)
    outside = np.zeros(mask.shape, np.uint8)
    cv2.ellipse(outside, (155, 145), (99, 109), 0, 0, 360, 1, -1)
    assert mask[120, 155] and mask[(outside == 0)].mean() < 0.002
    assert not mask[60:230, 25:55].any()


# ---------- fidelity ----------
def test_fidelity_ignores_sharpening_but_sees_changed_content(render):
    up = cv2.resize(render, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    sharp = cv2.addWeighted(up, 1.6, cv2.GaussianBlur(up, (0, 0), 3), -0.6, 0)
    clean = fidelity(render, sharp)
    assert clean["mean_diff"] < 3 and clean["region_diff"] < 12
    damaged = up.copy()
    damaged[1200:1400, 400:700, :3] = (30, 200, 240)      # something painted over the card body
    assert fidelity(render, damaged)["region_diff"] > 60
    shifted = np.roll(up, 8, axis=1)                        # every layer 2 source pixels to the right
    assert fidelity(render, shifted)["mean_diff"] > 2 * clean["mean_diff"]


def test_hand_reports_a_fidelity_warning_without_failing(tmp_path, render, art, fake_upscaler):
    render_path, art_path = _files(tmp_path, render, art)
    builder = Builder(upscaler=fake_upscaler)
    fine = builder.hand("TEST_001", str(tmp_path / "a.png"), render=render_path, art=art_path)
    assert fine["fidelity"]["faithful"] and fine["warnings"] == []
    strict = builder.hand("TEST_001", str(tmp_path / "b.png"), render=render_path, art=art_path, max_region_diff=0.01)
    assert strict["ok"] and strict["warnings"] == ["fidelity"] and strict["fidelity"]["faithful"] is False


# ---------- output formats and comparison sheets ----------
def test_webp_output_and_unknown_extension(tmp_path, render):
    save_image(tmp_path / "card.webp", render, 80)
    data = (tmp_path / "card.webp").read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert load_bgra(tmp_path / "card.webp").shape == render.shape
    with pytest.raises(CardError) as caught:
        save_image(tmp_path / "card.gif", render)
    assert caught.value.code == "bad_format"


def test_compare_sheet_layout(render):
    after = cv2.resize(render[15:426, 20:301], (562, 822), interpolation=cv2.INTER_CUBIC)
    sheet = compare_sheet(render, after)
    assert sheet.shape == (822, 562 * 2 + 24, 4)
    # the source is trimmed to its card and stretched: both halves show the same picture
    assert np.abs(sheet[400, 281, :3].astype(int) - sheet[400, 562 + 24 + 281, :3].astype(int)).max() < 30
    zoomed = compare_sheet(render, after, zoom=(0.25, 0.1, 0.75, 0.5))
    assert zoomed.shape[1] == sheet.shape[1] and zoomed.shape[0] == 822 + 24 + round(562 * (0.4 * 822) / (0.5 * 562))
    with pytest.raises(CardError):
        compare_sheet(render, after, zoom=(0.5, 0.1, 0.4, 0.5))


def test_cli_compare(tmp_path, render, capsys):
    before, after = tmp_path / "before.png", tmp_path / "after.png"
    save_png(before, render)
    save_png(after, cv2.resize(render[15:426, 20:301], (562, 822), interpolation=cv2.INTER_CUBIC))
    out = tmp_path / "sheet.webp"
    base = ["compare", "--before", str(before), "--after", str(after), "--out", str(out)]
    assert cli.main([*base, "--zoom", "0.2,0.5,0.8,0.9"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and payload["enlargement"] == round(562 / 320, 2) and out.exists()
    assert cli.main([*base, "--zoom", "1,2"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "bad_zoom"


# ---------- animation: loop and encoding ----------
def _write_ramp(directory, count, size=(64, 96)):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        frame = np.full((size[1], size[0], 4), 255, np.uint8)
        frame[..., :3] = index * 10
        save_png(directory / f"f_{index + 1:04d}.png", frame)


def test_loop_blend_removes_the_jump(tmp_path):
    _write_ramp(tmp_path / "frames", 12)
    report = loop_blend(tmp_path / "frames", 4)
    assert report["frames"] == 8 == len(list((tmp_path / "frames").glob("f_*.png")))
    assert report["jump_before"] == 110 and report["jump_after"] < 10 and report["blend_frames"] == 4
    assert load_bgra(tmp_path / "frames" / "f_0001.png")[0, 0, 0] == 64   # 0.2 * frame 1 + 0.8 * frame 9

    _write_ramp(tmp_path / "plain", 12)
    untouched = loop_blend(tmp_path / "plain", 0)
    assert untouched["frames"] == 12 and untouched["jump_after"] == untouched["jump_before"] == 110


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_all_formats_and_decode_back(tmp_path):
    _write_ramp(tmp_path / "frames", 6)
    outputs = {name: tmp_path / f"clip.{name}" for name in ("webm", "webp", "mp4")}
    encode(tmp_path / "frames", 20, (64, 96), outputs)
    assert all(path.stat().st_size > 0 for path in outputs.values())
    assert outputs["webp"].read_bytes()[8:12] == b"WEBP"
    frames, fps = decode_frames(outputs["webm"], tmp_path / "decoded")
    assert len(frames) == 6 and fps == 20 and frames[0].shape == (96, 64, 4)


# ---------- catalogue ----------
def test_card_ids_from_catalogue(tmp_path, monkeypatch):
    catalogue = tmp_path / "cards.json"
    catalogue.write_text(json.dumps([
        {"id": "EX1_298", "dbfId": 1, "collectible": True},
        {"id": "BG_1", "dbfId": 2, "isBattlegroundsPoolMinion": True, "battlegroundsPremiumDbfId": 3},
        {"id": "BG_1_G", "dbfId": 3},
        {"id": "BG_S", "dbfId": 4, "isBattlegroundsPoolSpell": True},
        {"id": "bad id", "dbfId": 5, "isBattlegroundsPoolMinion": True},
    ]), encoding="utf-8")
    monkeypatch.setattr(sources, "fetch", lambda url, **kwargs: catalogue)
    assert sources.card_ids("bg") == ["BG_1,normal,bg", "BG_S,normal,bg"]
    assert sources.card_ids("bg", with_golden=True) == ["BG_1,normal,bg", "BG_1_G,normal,bg", "BG_S,normal,bg"]
    assert sources.card_ids("collectible") == ["BG_1", "BG_1_G", "BG_S", "EX1_298"]
    with pytest.raises(CardError):
        sources.card_ids("everything")


def test_cli_ids_writes_a_batch_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sources, "card_ids", lambda kind, with_golden=False, refresh=False: ["A_1", "B_2"])
    out = tmp_path / "ids.txt"
    assert cli.main(["ids", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "A_1\nB_2\n"
    assert json.loads(capsys.readouterr().out)["count"] == 2
