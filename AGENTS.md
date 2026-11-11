# Instructions for AI agents

`cardv2` turns existing 512 px Hearthstone card renders into high-resolution images. Call it as a
command-line tool and read its JSON; do not import its modules from other projects.

## Contract

- Every command prints **one JSON object per result on stdout** (`batch`: one per line). Logs and
  usage errors go to stderr.
- Exit code `0` success, `1` a build failed, `2` bad arguments.
- Success: `{"ok": true, "command": ..., "card": ..., "output": {...}, ...}`.
- Failure: `{"ok": false, "command": ..., "card": ..., "error": {"code": ..., "message": ...}}`.
  Branch on `error.code`, never on the message text.

| `error.code` | Meaning | What to do |
|---|---|---|
| `bad_card_id`, `bad_locale`, `bad_variant`, `bad_mode`, `bad_source`, `bad_format`, `bad_url`, `bad_zoom`, `bad_kind` | invalid input | fix the argument |
| `not_found` | the source has no such card or variant | try another `--variant`/`--source`, or pass `--render` |
| `unsupported` | this source lacks that variant | use `--source firestone` |
| `network`, `too_large`, `bad_catalog` | download failed or returned unusable data | retry once, then report |
| `missing_file`, `bad_image`, `bad_video` | an input file is absent or unreadable | check the path |
| `registration_failed` | the art could not be matched | for `hand`: rerun with `--art-swap off`; for `animate`: the video is a different card |
| `missing_dependency` | torch/spandrel or ffmpeg missing | run `cardv2 doctor` |
| `bad_model` | weights failed the checksum | rerun; the file is re-downloaded |
| `ffmpeg_failed`, `write_failed` | encoding or disk problem | report with the message |

## Choosing a command

1. Card as shown in hand → `cardv2 hand CARD_ID --out FILE [--variant golden|signature|diamond] [--mode bg]`.
   The extension of `--out` picks the format: `.png` or `.webp`.
2. Minion as shown on the board → `cardv2 onboard CARD_ID --out FILE [--golden]`. There is no
   signature or diamond board frame.
3. Animated card → `cardv2 animate CARD_ID --video FILE_OR_HTTPS --out-dir DIR`. You must supply a
   small animated render of the **same card and variant**; the tool does not find one.
   `--formats webm,webp,mp4`: webm and webp keep transparency (webp works in `<img>`), mp4 does not.
4. More than a few cards → `cardv2 batch --ids-file FILE --out-dir DIR [--format webp]`. It loads
   the model once and skips outputs that already exist, so an interrupted run can simply be
   restarted. The last line is a summary: `{"command": "batch", "built", "skipped", "failed", "warned"}`.
5. Every card of a kind → `cardv2 ids --kind collectible|bg --out ids.txt`, then `batch`.
6. Showing a result to a person → `cardv2 compare --before SOURCE --after RESULT --out FILE
   [--zoom x0,y0,x1,y1]`. Left is the source stretched to the result's size, right is the result.

Battlegrounds golden minions are separate cards: use their own id (for example `BG26_146_G`) with
`--mode bg`, not `--variant golden`.

## Reading the result

- `output.path`, `output.width`, `output.height`: the file that was written.
- `art_swap.applied`: whether the real art replaced the upscaled art. `false` with reason
  `variant` (golden/signature/diamond), `low_score`, `art_unavailable`, `colour_fit` or `disabled`
  is a normal outcome, not an error. Report it; do not force `--art-swap on` to make it true.
- `art_swap.score`: 0–1 match of the art; values around 0.99 are typical for normal cards.
- `upscale.cached`: `true` means the upscaled picture came from the cache.
- `fidelity`: how far the result is from its source once shrunk back (0-255 scale, blur-tolerant).
  `mean_diff` is the whole card, `region_diff` the worst 24x24 px source region, `faithful` whether
  both are within `max_diff` / `max_region_diff`.
- `warnings`: `["fidelity"]` when the result is not faithful. The file is still written and `ok`
  stays `true`. Treat it as "look at this card before using it": open the image, or build a
  `compare` sheet. Do not raise the limits to silence it.
- `animation`: frame count, size, `video_match`/`card_match` scores, `flow_max_px`, and `loop`
  (`jump_before`/`jump_after` at the loop point next to `step_typical` between frames).

## Rules

- Do not add card images, frames, videos, model weights or build outputs to this repository. They
  belong in the cache (`CARDV2_CACHE`) or in the caller's output directory. The only exception is
  `docs/img`: comparison sheets for the README, made with `cardv2 compare`, WebP, under 400 KB each.
- Do not claim a result looks right from the JSON alone. `ok: true` means the file was written;
  look at the image before saying anything about quality.
- The output is an enlargement of the source render. If the source has the wrong name, text or
  font, so does the output: pick another `--source` or pass `--render`.
- State which card types you actually ran. Weapons, heroes and locations were never tested, and
  `--device cuda` was never run.
- A full card on CPU takes one to three minutes. Use `--threads` to leave cores for other work on
  a shared server, and prefer `batch` over a shell loop.

## Development

`make verify PYTHON=<python>` runs ruff and pytest; GitHub runs the same on every push and pull
request. Tests use a stand-in upscaler and local synthetic pictures, so they need no network, no
model and no torch. Add a test with every change
to registration, compositing or the JSON contract.
