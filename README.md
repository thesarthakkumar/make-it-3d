# make-it-3d

A Claude Code plugin that turns any 2D video or movie into 3D: red-cyan anaglyph for paper 3D glasses, half side-by-side 1080p for 3D projectors and 3D TVs, or full side-by-side for VR headsets. Works on macOS, Windows and Linux.

Just ask Claude: *"make this movie 3D for my projector"*, *"make this 3D for red-cyan glasses"* or *"convert this clip to 3D for my Quest"*.

## Install

In Claude Code:

```
/plugin marketplace add thesarthakkumar/make-it-3d
/plugin install make-it-3d@make-it-3d
```

Requirements: `ffmpeg` and Python 3.9+.

| OS | Install ffmpeg | GPU used |
|---|---|---|
| macOS | `brew install ffmpeg` | Apple Silicon (MPS) |
| Windows 10/11 | `winget install Gyan.FFmpeg` | NVIDIA (CUDA), else CPU |
| Linux | `sudo apt install ffmpeg` | NVIDIA (CUDA), else CPU |

On first use the skill creates its own Python environment (`~/.cache/make-it-3d/venv`, or `%LOCALAPPDATA%\make-it-3d\venv` on Windows), installs PyTorch (the CUDA build automatically if an NVIDIA GPU is found) and transformers, and downloads the depth model (about 100 MB). Everything runs locally. CPU-only works but is slow for full movies.

CI runs the converter on Windows, macOS and Linux on every push.

## Presets

| Watching on | `--preset` | Output |
|---|---|---|
| Red-cyan paper glasses, any screen or ordinary projector | `glasses` (default) | anaglyph (red lens on the left eye) |
| 3D projector (Epson, BenQ, Optoma, ViewSonic, DLP-Link) | `projector` | half side-by-side 1920x1080, gentler depth for big screens |
| 3D TV | `tv` | half side-by-side 1920x1080 |
| VR headset (Quest, Vision Pro player apps, Pico) | `vr` | full side-by-side |

Add `--format tb` for devices that want top-bottom. Output names are auto-detected by players: `name.3D.HSBS.mp4` (Kodi), `name_3D_LRF_Full_SBS.mp4` (Skybox, DeoVR).

### Watching on a 3D projector

1. Convert with `--preset projector`.
2. Play the file fullscreen at 1080p (VLC, Kodi, PowerDVD, a media stick or a laptop over HDMI).
3. In the projector's 3D menu set the format to **Side by Side** and switch the glasses on. Kodi and PowerDVD can switch it automatically from the file name.
4. If the depth looks inside-out, use the projector's L/R swap ("3D inverse") or re-render with `--swap-eyes`.

No 3D projector? Any projector shows the `glasses` (anaglyph) version; watch with red-cyan glasses.

## How it works

1. **Depth**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small estimates depth for every frame (batched, fp16).
2. **Temporal stability**: the depth range is EMA-normalised and frames are blended, with both reset at scene cuts so a new shot never inherits the last one's depth.
3. **Edge dilation**: the foreground grows by a couple of pixels so edges don't leave halos.
4. **Stereo views**: each eye is rendered by z-ordered forward splatting at 2x horizontal supersampling (nearer surfaces win), and disoccluded gaps are filled from the background side.
5. **Packing**: Dubois least-squares anaglyph in linear light, or SBS / half-SBS / TB, letterboxed to the preset's frame size (adds `stereo_mode` metadata when the output is .mkv). Audio is carried across.

Speed on an Apple M5: about 13 fps at 720p, so a 1-hour 720p episode takes about 1.5 hours. A mid-range NVIDIA GPU is in the same range or faster; CPU-only is several times slower.

## Standalone CLI

```bash
PY=$(python3 skills/make-3d/scripts/setup.py)
"$PY" skills/make-3d/scripts/convert3d.py movie.mkv --preset projector --start 00:10:00 --duration 20   # preview
"$PY" skills/make-3d/scripts/convert3d.py movie.mkv --preset projector                                  # full
```

Windows PowerShell:

```powershell
$PY = python skills\make-3d\scripts\setup.py
& $PY skills\make-3d\scripts\convert3d.py movie.mkv --preset projector
```

| Flag | Default | Effect |
|---|---|---|
| `--preset` | glasses | `glasses`, `projector`, `tv`, `vr` |
| `--format` | from preset | `anaglyph`, `sbs`, `hsbs`, `tb` |
| `--strength` | 2.0 (projector 1.5) | Eye separation as % of width. 1 subtle, 3 strong. |
| `--pop` | 0.3 (projector 0.2) | Share of depth in front of the screen. 0 = all behind, 0.5+ = lots of pop-out. |
| `--size` | from preset | Output frame `WxH`, letterboxed; `source` keeps the input size. |
| `--swap-eyes` | off | Swap left and right. |
| `--edge-dilation` | 2 | Foreground growth in px. 0 sharper, 4 gentler on the eyes. |
| `--model` | small | `base` / `large` are sharper but non-commercial (CC-BY-NC-4.0). |
| `--max-width` | 1920 | Wider sources are scaled down first. |
| `--save-depth` | off | Also write the depth video. |

## Limits

- Depth is estimated, so this is a conversion, not native 3D. People and faces convert best; hair, branches and fences can shimmer.
- Anaglyph mutes colours (especially red); that's the glasses.
- Subtitles aren't carried over.
- Only convert videos you have the right to use.

## Credits and related projects

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): the depth model (Small: Apache-2.0).
- [iw3 / nunif](https://github.com/nagadomi/nunif): the most complete open-source 2D-to-3D converter. Its edge dilation, EMA depth normalisation and scene-cut handling inspired this pipeline. Use it if you want more depth models and ML-based warping.
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything): temporally consistent video depth; a possible future backend.
- [StereoCrafter](https://stereocrafter.github.io): diffusion-based stereo inpainting; the high-end of quality, needs a big GPU.

## License

MIT
