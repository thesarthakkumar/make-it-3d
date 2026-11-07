#!/usr/bin/env python3
"""make-it-3d: convert any 2D video into stereoscopic 3D.

Pipeline (one streaming pass, no temp frames on disk):
  1. decode frames with ffmpeg
  2. depth: Video Depth Anything Small (default, temporally consistent, streamed in 32-frame
     windows) or per-frame Depth Anything V2 (--model fast); or reuse a saved depth video (--depth-in)
  3. temporal stabilisation: EMA-normalised depth range + light frame blending, reset on scene cuts
  4. edge dilation (foreground grows a few px so edges don't ghost)
  5. stereo synthesis per eye: z-ordered forward splatting at 2x horizontal supersampling,
     disocclusion holes filled from the background side
  6. pack as red-cyan anaglyph (Dubois), side-by-side, half-SBS or top-bottom
  7. encode with ffmpeg, original audio copied across

Example:
  python convert3d.py movie.mkv -o movie_3D.mp4 --format anaglyph
  python convert3d.py movie.mkv --format sbs --start 00:15:00 --duration 120
"""
import argparse, json, os, subprocess, sys, time
from fractions import Fraction

import torch
import torch.nn.functional as F

FORMATS = {
    "anaglyph": "red-cyan anaglyph for red/cyan paper glasses (Dubois colour method)",
    "sbs": "full side-by-side, for VR headsets (Quest, Vision Pro players) and 3D TVs",
    "hsbs": "half side-by-side (same width as the source), for most 3D TVs and projectors",
    "tb": "top-bottom (over-under), for some 3D TVs and players",
}
PRESETS = {  # sensible defaults per screen; any explicit flag still wins
    "glasses":   dict(format="anaglyph", strength=2.0, pop=0.3, size=None),
    "vr":        dict(format="sbs", strength=2.0, pop=0.3, size=None),
    "tv":        dict(format="hsbs", strength=2.0, pop=0.3, size=(1920, 1080)),
    # big screens magnify disparity, so keep it gentler and mostly behind the screen
    "projector": dict(format="hsbs", strength=1.5, pop=0.2, size=(1920, 1080)),
}
SUFFIX = {  # names that VR players (Skybox, DeoVR) and Kodi recognise automatically
    "anaglyph": "_3D_anaglyph", "sbs": "_3D_LRF_Full_SBS", "hsbs": ".3D.HSBS", "tb": ".3D.HTAB",
}
MODELS = {  # per-frame models. Small is Apache-2.0; Base/Large are CC-BY-NC-4.0 (non-commercial only)
    "fast": "depth-anything/Depth-Anything-V2-Small-hf",
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def parse_time(t):
    if t is None:
        return None
    if ":" in str(t):
        parts = [float(p) for p in str(t).split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return float(t)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    fr = s.get("avg_frame_rate") or s.get("r_frame_rate")
    if not fr or fr == "0/0":
        fr = s.get("r_frame_rate", "24/1")
    return s["width"], s["height"], fr, float(j["format"].get("duration", 0))


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Depth:
    """Depth Anything V2 wrapper -> relative inverse depth (bigger = nearer)."""

    def __init__(self, model, device, short_side):
        from transformers import AutoModelForDepthEstimation
        self.dev = device
        self.half = device in ("cuda", "mps")
        m = AutoModelForDepthEstimation.from_pretrained(MODELS.get(model, model)).to(device).eval()
        self.m = m.half() if self.half else m
        self.short = short_side
        dt = torch.float16 if self.half else torch.float32
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dt).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dt).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb):  # rgb: B,3,H,W float 0..1 on device
        B, _, H, W = rgb.shape
        s = self.short / min(H, W)
        ih, iw = max(14, round(H * s / 14) * 14), max(14, round(W * s / 14) * 14)
        x = F.interpolate(rgb, (ih, iw), mode="bilinear", align_corners=False, antialias=True)
        x = x.half() if self.half else x
        x = (x - self.mean) / self.std
        d = self.m(pixel_values=x).predicted_depth.unsqueeze(1).float()
        return F.interpolate(d, (H, W), mode="bicubic", align_corners=False)[:, 0]


class CutDetector:
    """Flags the first frame of a new shot (mean change of a 64x36 thumbnail)."""

    def __init__(self, thresh=0.13):
        self.thresh, self.thumb, self.cuts = thresh, None, 0

    def __call__(self, rgb):
        t = F.interpolate(rgb[None], (36, 64), mode="bilinear", antialias=True, align_corners=False)[0]
        cut = self.thumb is not None and (t - self.thumb).abs().mean().item() > self.thresh
        self.thumb = t
        self.cuts += int(cut)
        return cut


class Stabiliser:
    """Normalises depth to 0..1 with an EMA'd percentile range and blends frames,
    resetting both at scene cuts so a new shot never inherits the old one's depth."""

    def __init__(self, decay=0.9, blend=0.35):
        self.decay, self.blend = decay, blend
        self.lo = self.hi = self.prev = None

    def __call__(self, d, cut):
        sample = d[::8, ::8].flatten().float()
        q = torch.quantile(sample, torch.tensor([0.02, 0.98], device=d.device))
        lo, hi = q[0].item(), q[1].item()
        if self.lo is None or cut:
            self.lo, self.hi, self.prev = lo, hi, None
        else:
            self.lo = self.decay * self.lo + (1 - self.decay) * lo
            self.hi = self.decay * self.hi + (1 - self.decay) * hi
        n = ((d - self.lo) / max(self.hi - self.lo, 1e-6)).clamp(0, 1)
        if self.prev is not None:
            n = (1 - self.blend) * n + self.blend * self.prev
        self.prev = n
        return n


def dilate_foreground(depth, px):
    """Grow near regions by px (max filter) so foreground edges carry no background halo."""
    if px <= 0:
        return depth
    k = 2 * px + 1
    return F.max_pool2d(depth[None, None], (k, k), stride=1, padding=px)[0, 0]


class Stereo:
    """Z-ordered forward splatting with background-side hole filling."""

    def __init__(self, H, W, device, ss=2):
        self.H, self.W, self.ss, self.dev = H, W, ss, device
        self.W2 = W * ss
        self.xs = torch.arange(self.W2, device=device).float().view(1, -1).expand(H, -1)
        self.rows = (torch.arange(H, device=device) * self.W2).view(-1, 1).expand(-1, self.W2)
        self.gy = torch.linspace(-1, 1, H, device=device).view(-1, 1).expand(-1, W)
        self.gx = torch.linspace(-1, 1, W, device=device).view(1, -1).expand(H, -1)

    def backward(self, img, disp, sign):
        gx = self.gx - sign * disp * (1.0 / self.W)  # shift by sign*disp/2 px, grid units 2/W
        return F.grid_sample(img[None], torch.stack([gx, self.gy], -1)[None], mode="bilinear",
                             padding_mode="border", align_corners=True)[0]

    def view(self, img, disp, sign, bg_disp):
        """img 3,H,W; disp H,W in source px (positive = in front of screen)."""
        H, W2, ss = self.H, self.W2, self.ss
        img2 = F.interpolate(img[None], (H, W2), mode="bilinear", align_corners=False)[0]
        d2 = F.interpolate(disp[None, None], (H, W2), mode="bilinear", align_corners=False)[0, 0]
        tx = torch.round(self.xs + sign * d2 * ss / 2).long()
        ok = (tx >= 0) & (tx < W2)
        idx = (self.rows + tx.clamp(0, W2 - 1))[ok]
        key = d2[ok]
        z = torch.full((H * W2,), -1e9, device=self.dev)
        z = z.scatter_reduce(0, idx, key, reduce="amax")
        win = key >= z[idx] - 1e-4  # nearest surface wins each target pixel
        out = torch.zeros(3, H * W2, device=self.dev)
        src = img2.reshape(3, -1)[:, ok.flatten()]
        out[:, idx[win]] = src[:, win]
        holes = (z <= -1e8).view(1, H, W2).float()
        out = out.view(3, H, W2)
        # downsample supersampled result and hole mask back to source width
        out = F.avg_pool2d(out[None], (1, ss))[0]
        holes = F.avg_pool2d(holes[None], (1, ss))[0]
        # fill disocclusions from the background (eroded depth) with a backward warp
        fill = self.backward(img, bg_disp, sign)
        cover = (1 - holes).clamp(min=1e-6)
        out = torch.where(holes > 0.99, fill, out / cover * (1 - holes) + fill * holes)
        return out.clamp(0, 1)


DUBOIS_L = [[0.456, 0.500, 0.176], [-0.040, -0.038, -0.016], [-0.015, -0.021, -0.005]]
DUBOIS_R = [[-0.043, -0.088, -0.002], [0.378, 0.734, -0.018], [-0.072, -0.113, 1.226]]


def to_lin(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def fit(img, size):
    """Scale an eye view to fit inside size=(w, h) keeping aspect, padding with black (letterbox)."""
    if not size:
        return img
    cw, ch = size
    _, h, w = img.shape
    s = min(cw / w, ch / h)
    nw, nh = max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)
    img = F.interpolate(img[None], (nh, nw), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)
    px, py = (cw - nw) // 2, (ch - nh) // 2
    return F.pad(img, (px, cw - nw - px, py, ch - nh - py))


def pack(L, R, fmt, ML, MR):
    if fmt == "anaglyph":
        out = torch.einsum("ij,jhw->ihw", ML, to_lin(L)) + torch.einsum("ij,jhw->ihw", MR, to_lin(R))
        return to_srgb(out)
    if fmt == "sbs":
        return torch.cat([L, R], 2)
    if fmt == "hsbs":
        return F.interpolate(torch.cat([L, R], 2)[None], L.shape[1:], mode="bilinear", antialias=True, align_corners=False)[0]
    if fmt == "tb":
        return F.interpolate(torch.cat([L, R], 1)[None], L.shape[1:], mode="bilinear", antialias=True, align_corners=False)[0]
    raise ValueError(fmt)


def read_frames(proc, W, H, ch=3):
    size = W * H * ch
    while True:
        b = proc.stdout.read(size)
        if len(b) < size:
            return
        yield torch.frombuffer(bytearray(b), dtype=torch.uint8).view(H, W, ch) if ch > 1 else \
            torch.frombuffer(bytearray(b), dtype=torch.uint8).view(H, W)


def to_dev(u8, dev):
    return u8.to(dev).permute(2, 0, 1).float() / 255


def frames_fast(frames, a, dev, H, W, cutdet):
    """Per-frame Depth Anything V2, batched. Yields (rgb, depth 0..1)."""
    depth = Depth(a.model, dev, a.depth_res or 294)
    stab = Stabiliser(blend=0.35)
    batch = []

    def run(batch):
        rgb = torch.stack([to_dev(f, dev) for f in batch])
        raw = depth(rgb)
        for i in range(rgb.shape[0]):
            yield rgb[i], stab(raw[i], cutdet(rgb[i]))

    for f in frames:
        batch.append(f)
        if len(batch) == a.batch:
            yield from run(batch)
            batch = []
    if batch:
        yield from run(batch)


def frames_vda(frames, a, dev, H, W, cutdet):
    """Video Depth Anything streamed in 32-frame windows, restarted at scene cuts."""
    from vda_stream import VDAStream
    enc = {"vda": "vits", "vda-base": "vitb", "vda-large": "vitl"}[a.model]
    vda = VDAStream(enc, dev, a.depth_res or 392)
    stab = Stabiliser(blend=0.1)  # the model is already temporally consistent
    queue = []  # (uint8 frame on CPU, is_cut) waiting for their depth

    def emit(depths):
        for dep in depths:
            f, cut = queue.pop(0)
            d = F.interpolate(dep[None, None], (H, W), mode="bicubic", align_corners=False)[0, 0]
            yield to_dev(f, dev), stab(d, cut)

    for f in frames:
        rgb = to_dev(f, dev)
        cut = cutdet(rgb)
        if cut:
            yield from emit(vda.flush())
        queue.append((f, cut))
        yield from emit(vda.push(rgb))
    yield from emit(vda.flush())


def frames_depth_in(frames, path, start, length, dev, H, W):
    """Reuse a depth video written earlier with --save-depth (white = near).
    If it covers the whole source (e.g. a full-movie depth pass) it is seeked to --start like the
    source; if it is shorter, it is assumed to already start at --start."""
    seek = []
    if start:
        dlen = probe(path)[3]
        if dlen >= start + 1:
            seek = ["-ss", str(start)]
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error"] + seek + ["-i", path] + length +
                         ["-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE)
    for f, d in zip(frames, read_frames(p, W, H, 1)):
        yield to_dev(f, dev), d.to(dev).float() / 255
    p.kill()


def main():
    ap = argparse.ArgumentParser(description="Convert a 2D video to 3D.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="output path (default: <input>_3D_<format>.mp4 next to the input)")
    ap.add_argument("--preset", choices=PRESETS, default="glasses",
                    help="glasses (anaglyph), vr (full SBS), tv (half SBS 1080p), projector (half SBS 1080p, gentler depth)")
    ap.add_argument("--format", choices=FORMATS, help="override the preset's packing")
    ap.add_argument("--strength", type=float,
                    help="3D strength: total eye separation as %% of width (1.0 subtle, 2.0 default, 3.0 strong)")
    ap.add_argument("--pop", type=float,
                    help="share of the depth range in front of the screen (0 = all behind, 0.3 default, 0.6 lots of pop-out)")
    ap.add_argument("--size", help="output frame WxH, letterboxed to fit (e.g. 1920x1080); 'source' keeps the input size")
    ap.add_argument("--swap-eyes", action="store_true", help="swap left/right (if the 3D looks inside-out)")
    ap.add_argument("--start", help="start time (seconds or HH:MM:SS)")
    ap.add_argument("--duration", help="length to convert (seconds or HH:MM:SS)")
    ap.add_argument("--model", default="vda",
                    help="vda (Video Depth Anything Small, default: steadiest, Apache-2.0) | fast (per-frame "
                         "Depth Anything V2 Small, ~3x faster, more shimmer) | base | large | vda-large "
                         "(base/large variants are non-commercial)")
    ap.add_argument("--depth-res", type=int, help="depth model input short side (default 392 for vda, 294 for fast)")
    ap.add_argument("--depth-in", help="reuse a depth video saved earlier with --save-depth (skips the model)")
    ap.add_argument("--max-width", type=int, default=1920, help="downscale wider sources to this width")
    ap.add_argument("--edge-dilation", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-depth", help="also write the depth video here (white = near); reuse it with --depth-in")
    a = ap.parse_args()
    pre = PRESETS[a.preset]
    a.format = a.format or pre["format"]
    a.strength = pre["strength"] if a.strength is None else a.strength
    a.pop = pre["pop"] if a.pop is None else a.pop
    if a.size == "source":
        size = None
    elif a.size:
        size = tuple(int(v) for v in a.size.lower().split("x"))
    else:
        size = pre["size"]

    src = os.path.abspath(a.input)
    W0, H0, fps, total = probe(src)
    W, H = W0 - W0 % 2, H0 - H0 % 2  # H.264 4:2:0 needs even sizes
    if W > a.max_width:
        W, H = a.max_width - a.max_width % 2, int(round(H0 * a.max_width / W0 / 2) * 2)
    out = a.output or os.path.splitext(src)[0] + SUFFIX[a.format] + ".mp4"
    EW, EH = size or (W, H)  # per-eye frame size after letterboxing
    start, dur = parse_time(a.start), parse_time(a.duration)
    span = dur if dur else max(total - (start or 0), 0)
    fps_f = float(Fraction(fps))
    est = int(span * fps_f) if span else 0

    dev = pick_device(a.device)
    print(f"[make-it-3d] {os.path.basename(src)} {W0}x{H0} @ {fps_f:.3f} fps -> {a.preset}/{a.format} eye {EW}x{EH}, strength {a.strength}, pop {a.pop}, device={dev}",
          flush=True)

    seek = (["-ss", str(start)] if start else [])
    length = (["-t", str(dur)] if dur else [])
    rd = subprocess.Popen(["ffmpeg", "-loglevel", "error"] + seek + ["-i", src] + length +
                          ["-map", "0:v:0", "-vf", f"scale={W}:{H}:flags=lanczos", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    OW = EW * 2 if a.format == "sbs" else EW
    vcodec = ["-c:v", "libx264", "-crf", str(a.crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
    meta = []
    if a.format in ("sbs", "hsbs", "tb") and out.lower().endswith(".mkv"):
        meta = ["-metadata:s:v:0", "stereo_mode=" + {"sbs": "left_right", "hsbs": "left_right", "tb": "top_bottom"}[a.format]]
    wr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{OW}x{EH}", "-r", fps, "-i", "-"] + seek + ["-i", src] + length +
                          ["-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"] +
                          vcodec + meta + ["-movflags", "+faststart", out], stdin=subprocess.PIPE)
    dw = None
    if a.save_depth:
        dw = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                               "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", "14",
                               "-pix_fmt", "yuv420p", a.save_depth], stdin=subprocess.PIPE)

    cutdet = CutDetector()
    frames = read_frames(rd, W, H)
    if a.depth_in:
        source = frames_depth_in(frames, a.depth_in, start, length, dev, H, W)
        a.model = "depth-in"
    elif a.model.startswith("vda"):
        source = frames_vda(frames, a, dev, H, W, cutdet)
    else:
        source = frames_fast(frames, a, dev, H, W, cutdet)
    print(f"[make-it-3d] depth: {a.model}", flush=True)
    stereo = Stereo(H, W, dev)
    ML = torch.tensor(DUBOIS_L, device=dev)
    MR = torch.tensor(DUBOIS_R, device=dev)
    maxd = a.strength / 100 * W  # total disparity range in px
    bg_k = max(3, int(maxd) | 1)

    n, t0 = 0, time.time()
    with torch.no_grad():
        for img, d in source:
            if dw:
                dw.stdin.write((d * 255 + 0.5).byte().cpu().numpy().tobytes())
            d = dilate_foreground(d, a.edge_dilation)
            disp = maxd * (d - a.pop)
            bg = -F.max_pool2d(-disp[None, None], (1, bg_k), stride=1, padding=(0, bg_k // 2))[0, 0]
            L = stereo.view(img, disp, +1, bg)
            R = stereo.view(img, disp, -1, bg)
            if a.swap_eyes:
                L, R = R, L
            o = pack(fit(L, size), fit(R, size), a.format, ML, MR)
            wr.stdin.write((o * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
            n += 1
            if n % 480 == 0:
                rate = n / (time.time() - t0)
                eta = f", ~{(est - n) / rate / 60:.0f} min left" if est > n else ""
                print(f"[make-it-3d] {n}{'/' + str(est) if est else ''} frames, {rate:.1f} fps{eta}", flush=True)

    if rd.poll() is None:  # depth ran out before the video (short --depth-in): stop the reader
        rd.kill()
    rd.wait()
    wr.stdin.close(); wr.wait()
    if dw:
        dw.stdin.close(); dw.wait()
    el = time.time() - t0
    if est and n < est - 2:
        print(f"[make-it-3d] warning: wrote {n} frames, expected about {est}", flush=True)
    print(f"[make-it-3d] done: {n} frames in {el / 60:.1f} min, {cutdet.cuts} scene cuts -> {out}", flush=True)
    if wr.returncode != 0:
        sys.exit("ffmpeg encode failed")


if __name__ == "__main__":
    main()
