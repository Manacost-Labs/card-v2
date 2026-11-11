"""Command line: every command prints exactly one JSON object per result on stdout.

Exit codes: 0 success, 1 a build failed (see "error"), 2 bad arguments.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__, sources
from .errors import CardError
from .pipeline import DEFAULT_MIN_ART_SCORE, Builder


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _failure(error: CardError, **context) -> dict:
    return {"ok": False, **context, "error": {"code": error.code, "message": error.message}}


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="where the upscaler runs")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for the upscaler")
    parser.add_argument("--tile", type=int, default=0, help="upscale in tiles of this size to limit memory (0 = whole image)")
    parser.add_argument("--no-cache", action="store_true", help="do not reuse or store upscaled pictures")


def _add_card(parser: argparse.ArgumentParser, default_variant: str) -> None:
    parser.add_argument("--variant", default=default_variant, choices=sources.VARIANTS)
    parser.add_argument("--mode", default="constructed", choices=sources.MODES, help="bg = Battlegrounds render")
    parser.add_argument("--locale", default="ruRU")
    parser.add_argument("--source", default="firestone", choices=sources.RENDER_SOURCES)
    parser.add_argument("--render", help="use this render (file or https URL) instead of --source")
    parser.add_argument("--art", help="use this 512 px art (file or https URL) instead of HearthstoneJSON")
    parser.add_argument("--width", type=int, default=1000, help="output width in px (0 = native 4x)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cardv2", description="Build high-resolution Hearthstone card images.")
    parser.add_argument("--version", action="version", version=f"cardv2 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    hand = sub.add_parser("hand", help="hand card: normal, golden, signature, diamond; constructed or Battlegrounds")
    hand.add_argument("card")
    hand.add_argument("--out", required=True)
    _add_card(hand, "normal")
    hand.add_argument("--art-swap", default="auto", choices=("auto", "on", "off"),
                      help="auto: only for the normal variant and only when the art matches well")
    hand.add_argument("--min-art-score", type=float, default=DEFAULT_MIN_ART_SCORE)
    hand.add_argument("--no-crop", action="store_true", help="keep the source canvas instead of trimming to the card")
    _add_runtime(hand)

    onboard = sub.add_parser("onboard", help="in-play minion (oval frame, empty stat gems)")
    onboard.add_argument("card")
    onboard.add_argument("--out", required=True)
    onboard.add_argument("--golden", action="store_true")
    onboard.add_argument("--art", help="512 px art (file or https URL)")
    onboard.add_argument("--frame", help="300x350 frame (file or https URL) instead of the Firestone frame")
    onboard.add_argument("--width", type=int, default=0, help="output width in px (0 = native 1200)")
    _add_runtime(onboard)

    animate = sub.add_parser("animate", help="animated hand card: motion of a small video on the HD card")
    animate.add_argument("card")
    animate.add_argument("--video", required=True, help="small animated render of the same card (file or https URL)")
    animate.add_argument("--out-dir", required=True)
    _add_card(animate, "golden")
    animate.add_argument("--formats", default="webm,mp4", help="comma list of webm (transparent) and mp4")
    animate.add_argument("--keep-frames", action="store_true")
    _add_runtime(animate)

    batch = sub.add_parser("batch", help="many hand cards with one loaded model; prints one JSON line per card")
    batch.add_argument("--ids-file", required=True, help="text file: one 'CARD_ID[,variant[,mode]]' per line; # comments")
    batch.add_argument("--out-dir", required=True)
    _add_card(batch, "normal")
    batch.add_argument("--art-swap", default="auto", choices=("auto", "on", "off"))
    batch.add_argument("--min-art-score", type=float, default=DEFAULT_MIN_ART_SCORE)
    batch.add_argument("--force", action="store_true", help="rebuild cards whose output file already exists")
    _add_runtime(batch)

    sub.add_parser("doctor", help="report what is installed and reachable")
    return parser


def _builder(args) -> Builder:
    return Builder(device=args.device, threads=args.threads, tile=args.tile, use_cache=not args.no_cache)


def _doctor() -> dict:
    report: dict = {"ok": True, "command": "doctor", "version": __version__, "cache": str(sources.cache_dir())}
    report["ffmpeg"] = shutil.which("ffmpeg")
    try:
        import torch

        report["torch"] = {"version": torch.__version__, "cuda": bool(torch.cuda.is_available())}
    except ImportError:
        report["torch"] = None
    try:
        import spandrel  # noqa: F401

        report["spandrel"] = True
    except ImportError:
        report["spandrel"] = False
    model = sources.cache_dir() / "models" / f"{sources.MODEL_NAME}.pth"
    report["model"] = {"name": sources.MODEL_NAME, "downloaded": model.exists()}
    report["ok"] = bool(report["torch"]) and report["spandrel"]
    report["can_animate"] = report["ok"] and bool(report["ffmpeg"])
    return report


def _batch(args) -> int:
    try:
        lines = Path(args.ids_file).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CardError("missing_file", f"cannot read --ids-file: {error}") from error
    builder = _builder(args)
    out_dir = Path(args.out_dir)
    failures = 0
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        card = parts[0]
        variant = parts[1] if len(parts) > 1 and parts[1] else args.variant
        mode = parts[2] if len(parts) > 2 and parts[2] else args.mode
        target = out_dir / f"{card}{'' if variant == 'normal' else '_' + variant}.png"
        try:
            sources.check_card_id(card)
            if target.exists() and not args.force:
                _emit({"ok": True, "command": "hand", "card": card, "variant": variant, "skipped": "exists",
                       "output": {"path": str(target)}})
                continue
            _emit(builder.hand(card, str(target), variant=variant, mode=mode, locale=args.locale, source=args.source,
                               width=args.width, art_swap=args.art_swap, min_art_score=args.min_art_score))
        except CardError as error:
            failures += 1
            _emit(_failure(error, command="hand", card=card, variant=variant))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = _doctor()
            _emit(report)
            return 0 if report["ok"] else 1
        if args.command == "batch":
            return _batch(args)
        builder = _builder(args)
        if args.command == "hand":
            _emit(builder.hand(args.card, args.out, variant=args.variant, mode=args.mode, locale=args.locale,
                               source=args.source, render=args.render, art=args.art, width=args.width,
                               art_swap=args.art_swap, min_art_score=args.min_art_score, crop=not args.no_crop))
        elif args.command == "onboard":
            _emit(builder.onboard(args.card, args.out, golden=args.golden, art=args.art, frame=args.frame, width=args.width))
        elif args.command == "animate":
            formats = tuple(name for name in (part.strip() for part in args.formats.split(",")) if name)
            if not formats or any(name not in ("webm", "mp4") for name in formats):
                raise CardError("bad_format", "--formats accepts webm and mp4")
            _emit(builder.animate(args.card, args.video, args.out_dir, variant=args.variant, mode=args.mode,
                                  locale=args.locale, source=args.source, render=args.render, art=args.art,
                                  width=args.width, formats=formats, keep_frames=args.keep_frames))
        return 0
    except CardError as error:
        _emit(_failure(error, command=args.command, card=getattr(args, "card", None)))
        return 1


if __name__ == "__main__":
    sys.exit(main())
