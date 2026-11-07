#!/usr/bin/env python3
"""One-time setup for make-it-3d on macOS, Windows or Linux.

Creates a private Python env with PyTorch + transformers and prints the path of its
python executable (the only thing written to stdout). Safe to re-run.

  macOS:   uses the default PyTorch build (Apple Silicon GPU via MPS)
  Windows/Linux with an NVIDIA GPU: installs the CUDA build of PyTorch
  otherwise: CPU build (works, but slow for long videos)

Override the env location with MAKE3D_ENV, the PyTorch wheel index with MAKE3D_TORCH_INDEX.
"""
import os, platform, shutil, subprocess, sys
from pathlib import Path

CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


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

    ok = subprocess.run([str(py), "-c", "import torch, transformers"], capture_output=True).returncode == 0
    if not ok:
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
        subprocess.run(pip + ["torch"] + (["--index-url", index] if index else []), check=True)
        subprocess.run(pip + ["transformers", "numpy", "pillow"], check=True)

    print(str(py))


if __name__ == "__main__":
    main()
