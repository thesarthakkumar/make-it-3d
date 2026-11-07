"""Streaming wrapper around Video Depth Anything (ByteDance, Apache-2.0 for the Small model).

The upstream `infer_video_depth` loads the whole video into memory. This class feeds frames one at a
time and emits temporally consistent depth with a short delay, so a 2-hour film streams through in
constant memory. It reproduces the upstream windowing exactly: 32-frame windows, 10-frame overlap
built from the previous window's keyframes, least-squares scale/shift alignment on two anchor
keyframes, and linear blending across the 8 overlapping frames.

One change for long films: upstream anchors every window to the video's very first frame. Here the
stream restarts at scene cuts, so each shot is anchored to its own first frame and a black intro
frame never steers the depth of the whole movie.

The model code is fetched by setup.py into ~/.cache/make-it-3d/src/Video-Depth-Anything (pinned commit).
"""
import os, sys
from pathlib import Path

import torch
import torch.nn.functional as F

VDA_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
INFER_LEN, OVERLAP, INTERP_LEN = 32, 10, 8
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
STEP = INFER_LEN - OVERLAP
CONFIGS = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}
REPOS = {"vits": "depth-anything/Video-Depth-Anything-Small",
         "vitb": "depth-anything/Video-Depth-Anything-Base",
         "vitl": "depth-anything/Video-Depth-Anything-Large"}


def src_dir():
    base = os.environ.get("MAKE3D_SRC")
    if base:
        return Path(base)
    # setup.py puts the code next to the venv: <root>/venv and <root>/src/Video-Depth-Anything
    return Path(sys.prefix).parent / "src" / "Video-Depth-Anything"


def load_model(encoder, device):
    d = src_dir()
    if not (d / "video_depth_anything").exists():
        raise RuntimeError(f"Video Depth Anything code not found in {d}. Run setup.py --vda first.")
    sys.path.insert(0, str(d))
    import logging
    logging.getLogger("dinov2").setLevel(logging.ERROR)
    from video_depth_anything.dinov2_layers import attention as dino_attn
    from video_depth_anything.video_depth import VideoDepthAnything

    # Upstream falls back to a hand-written q@k^T attention when xformers is missing (always on Mac),
    # which is slow and runs out of memory at 518px. PyTorch's fused SDPA is the same maths.
    def sdpa_forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))

    dino_attn.Attention.forward = sdpa_forward
    dino_attn.MemEffAttention.forward = sdpa_forward
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(REPOS[encoder], f"video_depth_anything_{encoder}.pth")
    m = VideoDepthAnything(**CONFIGS[encoder])
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return m.to(device).eval()


def _fit(pred, target):
    """Least-squares scale and shift so pred*s+b ~= target (upstream compute_scale_and_shift_full)."""
    p, t = pred.flatten().float(), target.flatten().float()
    a00, a01, a11 = (p * p).sum(), p.sum(), torch.tensor(float(p.numel()), device=p.device)
    b0, b1 = (p * t).sum(), t.sum()
    det = a00 * a11 - a01 * a01
    if det.abs().item() < 1e-6:
        return 1.0, 0.0
    return ((a11 * b0 - a01 * b1) / det).item(), ((-a01 * b0 + a00 * b1) / det).item()


class VDAStream:
    def __init__(self, encoder="vits", device="cpu", input_size=518):
        self.dev = device
        self.model = load_model(encoder, device)
        self.input_size = input_size
        self.amp = device in ("cuda", "mps")
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        self._reset()

    def _reset(self):
        self.new = []          # preprocessed inputs waiting for the next window
        self.prev_input = None # previous window's 32 inputs
        self.pending = []      # aligned depth frames not yet emitted
        self.ref = None        # two anchor depths for alignment

    def _prep(self, rgb):
        """rgb 3,H,W float 0..1 -> resized (short side = input_size, multiples of 14) + normalised."""
        _, H, W = rgb.shape
        s = self.input_size / min(H, W)
        if max(H, W) / min(H, W) > 1.78:  # upstream memory guard for very wide frames
            s *= 1.777 / (max(H, W) / min(H, W))
        h, w = max(14, round(H * s / 14) * 14), max(14, round(W * s / 14) * 14)
        x = F.interpolate(rgb[None], (h, w), mode="bicubic", align_corners=False, antialias=True)[0]
        return (x.clamp(0, 1) - self.mean) / self.std

    @torch.no_grad()
    def _run(self, inputs):
        x = torch.stack(inputs)[None]  # 1,T,3,h,w
        with torch.autocast(device_type=self.dev, dtype=torch.float16, enabled=self.amp):
            d = self.model(x)[0]
        return d.float(), x[0]

    def _window(self):
        """Run one window over self.new and return newly finalised depth frames."""
        if self.prev_input is None:
            inputs = self.new[:INFER_LEN]
            d, x = self._run(inputs)
            self.pending = [d[i] for i in range(INFER_LEN)]
            self.ref = [d[KEYFRAMES[0]], d[KEYFRAMES[1]]]
            self.new = self.new[INFER_LEN:]
        else:
            inputs = [self.prev_input[k] for k in KEYFRAMES] + self.new[:STEP]
            d, x = self._run(inputs)
            s, b = _fit(torch.cat([d[0], d[1]]), torch.cat(self.ref))
            post = [(d[i] * s + b).clamp(min=0) for i in range(2, OVERLAP)]
            pre = self.pending[-INTERP_LEN:]
            for i in range(INTERP_LEN):
                w = i / (INTERP_LEN - 1)
                pre[i] = pre[i] * (1 - w) + post[i] * w
            self.pending[-INTERP_LEN:] = pre
            self.pending += [(d[i] * s + b).clamp(min=0) for i in range(OVERLAP, INFER_LEN)]
            self.ref = [self.ref[0], (d[KEYFRAMES[1]] * s + b).clamp(min=0)]
            self.new = self.new[STEP:]
        self.prev_input = x
        out, self.pending = self.pending[:-INTERP_LEN], self.pending[-INTERP_LEN:]
        return out

    def push(self, rgb):
        """Add one frame (3,H,W, 0..1 on device). Returns a list of finished depth maps (model res)."""
        self.new.append(self._prep(rgb))
        need = INFER_LEN if self.prev_input is None else STEP
        return self._window() if len(self.new) >= need else []

    def flush(self):
        """End of stream (or a scene cut): finish the remaining frames and reset."""
        out = []
        n_real = len(self.new)
        if n_real:
            need = INFER_LEN if self.prev_input is None else STEP
            self.new += [self.new[-1]] * (need - n_real)
            done = self._window() + self.pending
            keep = len(done) - (need - n_real)  # drop the padded tail
            out = done[:keep]
        else:
            out = self.pending
        self._reset()
        return out
