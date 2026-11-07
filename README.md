# make-it-3d

A Claude Code plugin that turns any 2D video or movie into 3D: red-cyan anaglyph for paper 3D glasses, side-by-side for VR headsets, or half-SBS / top-bottom for 3D TVs.

Just ask Claude: *"make this movie 3D for red-cyan glasses"* or *"convert this clip to 3D for my Quest"*.

## Install

In Claude Code:

```
/plugin marketplace add thesarthakkumar/make-it-3d
/plugin install make-it-3d@make-it-3d
```

Requirements: `ffmpeg` and `python3`. On first use the skill creates its own Python environment in `~/.cache/make-it-3d/venv` (PyTorch + transformers, about 1 GB) and downloads the depth model (about 100 MB).

Runs fully local on Apple Silicon (MPS), NVIDIA (CUDA) or CPU.

## Formats

| Watching on | Format |
|---|---|
| Red-cyan paper glasses, any screen | `anaglyph` (red lens on the left eye) |
| VR headset (Quest, Vision Pro player apps, Pico) | `sbs` |
| 3D TV / projector | `hsbs` or `tb` |

## How it works

1. **Depth**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) Small estimates depth for every frame (batched, fp16).
2. **Temporal stability**: the depth range is EMA-normalised and frames are blended, with both reset at scene cuts so a new shot never inherits the last one's depth.
3. **Edge dilation**: the foreground grows by a couple of pixels so edges don't leave halos.
4. **Stereo views**: each eye is rendered by z-ordered forward splatting at 2x horizontal supersampling (nearer surfaces win), and disoccluded gaps are filled from the background side.
5. **Packing**: Dubois least-squares anaglyph in linear light, or SBS / half-SBS / TB with `stereo_mode` metadata. Audio is carried across.

Speed on an Apple M5: about 13 fps for 720p anaglyph, so a 1-hour 720p episode takes about 1.5 hours.

## Standalone CLI

```bash
PY=$(bash skills/make-3d/scripts/setup.sh)
"$PY" skills/make-3d/scripts/convert3d.py movie.mkv --format anaglyph --start 00:10:00 --duration 20   # preview
"$PY" skills/make-3d/scripts/convert3d.py movie.mkv --format sbs -o movie_3D_SBS.mp4                    # full
```

| Flag | Default | Effect |
|---|---|---|
| `--strength` | 2.0 | Eye separation as % of width. 1 subtle, 3 strong. |
| `--pop` | 0.3 | Share of depth in front of the screen. 0 = all behind, 0.5+ = lots of pop-out. |
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
