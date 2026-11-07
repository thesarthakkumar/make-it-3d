#!/usr/bin/env bash
# One-time setup for make-it-3d: a private Python env with PyTorch + transformers.
# Prints the python path to use. Safe to re-run (no-op once installed).
set -euo pipefail
ENV_DIR="${MAKE3D_ENV:-$HOME/.cache/make-it-3d/venv}"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found. Install it first (macOS: brew install ffmpeg)." >&2; exit 1; }
PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "python3 not found." >&2; exit 1; }

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "Creating Python env in $ENV_DIR ..." >&2
  mkdir -p "$(dirname "$ENV_DIR")"
  if command -v uv >/dev/null; then uv venv -q "$ENV_DIR"; else "$PY" -m venv "$ENV_DIR"; fi
fi
if ! "$ENV_DIR/bin/python" -c "import torch, transformers" 2>/dev/null; then
  echo "Installing torch + transformers (one time, ~1 GB) ..." >&2
  if command -v uv >/dev/null; then
    uv pip install -q --python "$ENV_DIR/bin/python" torch transformers numpy pillow
  else
    "$ENV_DIR/bin/pip" install -q torch transformers numpy pillow
  fi
fi
echo "$ENV_DIR/bin/python"
