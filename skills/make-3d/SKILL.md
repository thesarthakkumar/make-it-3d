---
name: make-3d
description: Convert any 2D video or movie into stereoscopic 3D - red-cyan anaglyph for paper 3D glasses, side-by-side for VR headsets (Quest, Vision Pro players) and 3D TVs, half-SBS or top-bottom. Use whenever the user asks to make a video/movie/clip "3D", "watch with 3D glasses", "red cyan", "anaglyph", "SBS", "for my VR headset" or "for my 3D TV". Runs locally with Depth Anything V2 depth estimation; no cloud upload.
---

# make-3d

Turns a normal video into 3D by estimating depth for every frame and rendering a separate left-eye
and right-eye view. Everything runs locally (Apple Silicon MPS, NVIDIA CUDA or CPU).

Scripts live next to this file in `scripts/`. Below, `$SKILL_DIR` means this skill's base directory.

## 1. Pick the format (ask if the user didn't say how they'll watch)

| Viewer | `--format` |
|---|---|
| Red-cyan paper glasses, any screen | `anaglyph` (default) |
| VR headset (Quest, Vision Pro via a player app, Pico) | `sbs` |
| 3D TV or 3D projector | `hsbs` (or `tb` if the TV wants over-under) |

Red lens goes on the **left** eye for anaglyph.

## 2. Setup (one time)

```bash
PY=$(bash "$SKILL_DIR/scripts/setup.sh")
```

Needs `ffmpeg` and `python3`. Installs torch + transformers into `~/.cache/make-it-3d/venv`
(about 1 GB, first run only). The depth model (~100 MB) downloads from Hugging Face on first use.

## 3. Always render a short preview first

```bash
"$PY" "$SKILL_DIR/scripts/convert3d.py" "INPUT" --format anaglyph --start 00:10:00 --duration 20 -o "PREVIEW.mp4"
```

Pick a start time with people or objects at different distances (check a few frames with ffmpeg
if unsure; skip opening credits). Pull one frame from the preview and look at it: near objects
should show wider red/cyan fringes than the background, with no tearing around people.

## 4. Full render (run in the background; it takes roughly the video's length on an M-series Mac)

```bash
"$PY" "$SKILL_DIR/scripts/convert3d.py" "INPUT" --format anaglyph -o "OUTPUT.mp4"
```

Output defaults to `<input>_3D_<format>.mp4` next to the input. Audio is copied across
(re-encoded to AAC). Subtitles are not carried over.

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--strength` | 2.0 | Total eye separation, % of width. 1.0 subtle, 3.0 strong. Above 3 gets tiring and shows more artifacts. |
| `--pop` | 0.3 | Share of the depth that comes out of the screen. 0 = window effect (all behind), 0.5+ = lots of pop-out. |
| `--edge-dilation` | 2 | Grows the foreground a few px to hide halos. 0 = sharper edges, 4 = gentlest on the eyes. |
| `--model` | small | `base`/`large` are sharper but CC-BY-NC (non-commercial) and slower. |
| `--max-width` | 1920 | Wider sources (4K) are scaled down to this before processing. |
| `--save-depth PATH` | off | Also writes the depth video (white = near) for debugging. |

If the user says "more 3D", raise `--strength` to 2.5-3 and/or `--pop` to 0.4-0.5. If it hurts
their eyes or ghosts, lower `--strength` to 1.5 and `--pop` to 0.2.

## Honest limits to tell the user

- Depth is estimated, not filmed, so it's a conversion, not native 3D. Faces and people look best;
  thin things (hair, branches, fences) can shimmer.
- Anaglyph mutes colours, especially reds. That's the glasses, not the conversion.
- Only convert videos the user has the right to use; don't help redistribute converted
  copyrighted films.
