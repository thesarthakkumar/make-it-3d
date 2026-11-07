---
name: make-3d
description: Convert any 2D video or movie into stereoscopic 3D - red-cyan anaglyph for paper 3D glasses, side-by-side for VR headsets (Quest, Vision Pro players), half side-by-side 1080p for 3D projectors (Epson, BenQ, Optoma, ViewSonic, DLP-Link) and 3D TVs. Use whenever the user asks to make a video/movie/clip "3D", "watch with 3D glasses", "red cyan", "anaglyph", "SBS", "for my projector", "for my VR headset" or "for my 3D TV". Runs locally on macOS, Windows or Linux with Depth Anything V2 depth estimation; no cloud upload.
---

# make-3d

Turns a normal video into 3D by estimating depth for every frame and rendering a separate left-eye
and right-eye view. Runs locally on macOS (Apple Silicon GPU), Windows and Linux (NVIDIA GPU, or CPU).

Scripts live next to this file in `scripts/`. Below, `$SKILL_DIR` means this skill's base directory.
Claude Code runs commands in bash on every OS (Git Bash on Windows), so the commands below work as-is.

## 1. Pick the preset (ask how they'll watch if they didn't say)

| Watching on | `--preset` | Output |
|---|---|---|
| Red-cyan paper glasses, any screen or ordinary projector | `glasses` (default) | anaglyph, source size |
| 3D projector with active shutter / DLP-Link glasses | `projector` | half side-by-side, 1920x1080, gentler depth for a big screen |
| 3D TV | `tv` | half side-by-side, 1920x1080 |
| VR headset (Quest, Vision Pro player apps, Pico) | `vr` | full side-by-side |

Use `--format tb` with `projector`/`tv` if the device only takes top-bottom.
An ordinary (non-3D) projector can still show the `glasses` version: tell them to watch with red-cyan glasses.

## 2. Setup (one time)

```bash
PY=$(python3 "$SKILL_DIR/scripts/setup.py" || python "$SKILL_DIR/scripts/setup.py" || py -3 "$SKILL_DIR/scripts/setup.py")
```

Needs `ffmpeg` and Python 3.9+ on PATH. If ffmpeg is missing, setup prints the install command
(macOS `brew install ffmpeg`, Windows `winget install Gyan.FFmpeg`, Linux `sudo apt install ffmpeg`).
It installs PyTorch + transformers into a private env (`~/.cache/make-it-3d/venv`, or
`%LOCALAPPDATA%\make-it-3d\venv` on Windows), picking the CUDA build automatically when an NVIDIA
GPU is present. Without a GPU it runs on CPU, which is slow: warn the user before a full movie.

## 3. Always render a short preview first

```bash
"$PY" "$SKILL_DIR/scripts/convert3d.py" "INPUT" --preset projector --start 00:10:00 --duration 20 -o "PREVIEW.mp4"
```

Pick a start time with people or objects at different distances (skip opening credits). Pull a
frame from the preview and look at it before the full run.

## 4. Full render (run in the background; on an Apple M-series or mid-range NVIDIA GPU expect roughly 1-1.5x the video's length at 720p-1080p)

```bash
"$PY" "$SKILL_DIR/scripts/convert3d.py" "INPUT" --preset projector
```

Default output sits next to the input with a name players auto-detect:
`name_3D_anaglyph.mp4`, `name.3D.HSBS.mp4` / `name.3D.HTAB.mp4` (Kodi), `name_3D_LRF_Full_SBS.mp4` (Skybox, DeoVR).
Audio is copied across (as AAC). Subtitles are not carried over.

## 5. Tell the user how to play it

- **3D projector / 3D TV**: play the file fullscreen from a player that outputs 1080p, then set the
  projector's 3D format to **Side by Side** (or Top and Bottom for `tb`) in its 3D menu, and turn
  the glasses on. Kodi and PowerDVD (Windows) can also switch the projector automatically from the
  `.3D.HSBS.` file name. If depth looks inside-out (background floating in front), re-render with
  `--swap-eyes` or use the projector's "3D inverse / L-R swap" setting.
- **Red-cyan**: red lens on the **left** eye, fullscreen, dim room.
- **VR**: copy the file to the headset; the `_LRF_Full_SBS` name makes Skybox / DeoVR pick 3D automatically.

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--strength` | 2.0 (projector 1.5) | Total eye separation, % of width. 1.0 subtle, 3.0 strong. Keep it lower on big screens. |
| `--pop` | 0.3 (projector 0.2) | Share of the depth in front of the screen. 0 = window effect, 0.5+ = lots of pop-out. |
| `--size` | preset | Output frame `WxH`, letterboxed. `source` keeps the input size. |
| `--swap-eyes` | off | Swap left/right. |
| `--edge-dilation` | 2 | Grows the foreground a few px to hide halos. 0 sharper, 4 gentlest. |
| `--model` | small | `base`/`large` are sharper but CC-BY-NC (non-commercial) and slower. |
| `--max-width` | 1920 | Wider sources (4K) are scaled down first. |
| `--device` | auto | `mps`, `cuda` or `cpu`. |
| `--save-depth PATH` | off | Also writes the depth video (white = near). |

"More 3D": raise `--strength` (to 2.5-3, or 2 on a projector) and/or `--pop`.
Eye strain or ghosting: lower `--strength` to 1-1.5 and `--pop` to 0.15.

## Honest limits to tell the user

- Depth is estimated, not filmed, so it's a conversion, not native 3D. People and faces look best;
  hair, branches and fences can shimmer.
- Anaglyph mutes colours, especially reds. That's the glasses.
- Only convert videos the user has the right to use; don't help redistribute converted
  copyrighted films.
