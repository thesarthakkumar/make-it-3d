#!/usr/bin/env python3
"""One-time setup for make-it-3d on macOS, Windows or Linux.

Creates a private Python env with PyTorch + transformers and prints the path of its
python executable (the only thing written to stdout). Safe to re-run.

  macOS:   uses the default PyTorch build (Apple Silicon GPU via MPS)
  Windows/Linux with an NVIDIA GPU: installs the CUDA build of PyTorch
  otherwise: CPU build (works, but slow for long videos)

Override the env location with MAKE3D_ENV, the PyTorch wheel index with MAKE3D_TORCH_INDEX.
"""
import io, os, platform, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path

CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
VDA_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"  # keep in sync with vda_stream.py
VDA_ZIP = f"https://github.com/DepthAnything/Video-Depth-Anything/archive/{VDA_COMMIT}.zip"
PKGS = ["transformers", "huggingface_hub", "numpy", "pillow", "einops", "easydict",
        "opencv-python-headless", "tqdm"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def env_dir():
    if os.environ.get("MAKE3D_ENV"):
        return Path(os.environ["MAKE3D_ENV"])
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "make-it-3d" / "venv"
    return Path.home() / ".cache" / "make-it-3d" / "venv"


def env_python(d):
    return d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def vda_dir(env):
    return env.parent / "src" / "Video-Depth-Anything"


def fetch_vda(env):
    """Download the pinned Video Depth Anything model code (Apache-2.0). No git needed."""
    dest = vda_dir(env)
    stamp = dest / ".commit"
    if stamp.exists() and stamp.read_text().strip() == VDA_COMMIT:
        return
    log("Downloading Video Depth Anything code ...")
    data = urllib.request.urlopen(VDA_ZIP, timeout=120).read()
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = z.namelist()[0].split("/")[0]
        z.extractall(dest.parent)
    (dest.parent / root).rename(dest)
    stamp.write_text(VDA_COMMIT)


def has_nvidia():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        return subprocess.run([smi, "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def main():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        hint = {"Windows": "winget install Gyan.FFmpeg  (then open a new terminal)",
                "Darwin": "brew install ffmpeg"}.get(platform.system(), "sudo apt install ffmpeg")
        sys.exit(f"ffmpeg/ffprobe not found on PATH. Install it: {hint}")
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9+ is required.")

    d = env_dir()
    py = env_python(d)
    if not py.exists():
        log(f"Creating Python env in {d} ...")
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(d)], check=True)

    check = "import torch, torchvision, transformers, einops, easydict, cv2"
    ok = subprocess.run([str(py), "-c", check], capture_output=True).returncode == 0
    if not ok:
        # envs made by uv have no pip; bootstrap it
        if subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True).returncode != 0:
            subprocess.run([str(py), "-m", "ensurepip", "--upgrade"], check=True, capture_output=True)
        pip = [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-q"]
        index = os.environ.get("MAKE3D_TORCH_INDEX")
        if not index and platform.system() != "Darwin":
            if has_nvidia():
                index = CUDA_INDEX
                log("NVIDIA GPU found: installing the CUDA build of PyTorch.")
            else:
                index = CPU_INDEX
                log("No NVIDIA GPU found: installing the CPU build of PyTorch (slow for long videos).")
        log("Installing PyTorch + transformers (one time, 1-3 GB) ...")
        subprocess.run(pip + ["--upgrade", "pip"], check=True)
        subprocess.run(pip + ["torch", "torchvision"] + (["--index-url", index] if index else []), check=True)
        subprocess.run(pip + PKGS, check=True)

    fetch_vda(d)

    print(str(py))


if __name__ == "__main__":
    main()
