#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
FISH_SPEECH_DIR="$ROOT/vendor/fish-speech"
MODEL_MARKER="$ROOT/checkpoints/s2-pro/codec.pth"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
OPEN_BROWSER="${OPEN_BROWSER:-auto}"

log() { printf '\n▸ %s\n' "$*"; }

need_setup() {
  [ ! -d "$VENV" ] || [ ! -x "$VENV/bin/python" ] || [ ! -d "$FISH_SPEECH_DIR" ]
}

run_setup() {
  log "First run — installing dependencies (this may take a few minutes)..."
  bash "$ROOT/setup.sh"
}

download_model() {
  log "Downloading Fish Audio S2 Pro weights (~10 GB)..."
  "$VENV/bin/python" "$ROOT/download_model.py"
}

check_cuda() {
  if ! "$VENV/bin/python" - <<'PY' 2>/dev/null; then
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {name} ({vram:.0f} GB VRAM)")
else:
    raise SystemExit(1)
PY
    log "Warning: CUDA GPU not detected. Inference will be very slow on CPU."
    log "On DigitalOcean GPU droplets, verify NVIDIA drivers with: nvidia-smi"
  fi
}

open_browser() {
  [ "$OPEN_BROWSER" = "0" ] && return
  [ -n "${SSH_CONNECTION:-}" ] && return

  local url="http://127.0.0.1:${PORT}"
  (
    sleep 4
    if command -v open &>/dev/null; then
      open "$url"
    elif command -v xdg-open &>/dev/null; then
      xdg-open "$url" >/dev/null 2>&1 || true
    fi
  ) &
}

print_banner() {
  local network_url
  if [ "$HOST" = "0.0.0.0" ]; then
    local ip
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    network_url="http://${ip:-<server-ip>}:${PORT}"
  else
    network_url="http://${HOST}:${PORT}"
  fi

  echo
  echo "Fish Audio S2 Pro — Web UI"
  echo "  Local:   http://127.0.0.1:${PORT}"
  echo "  Network: ${network_url}"
  echo "  Press Ctrl+C to stop"
  echo
}

if need_setup; then
  run_setup
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ ! -f "$MODEL_MARKER" ]; then
  download_model
fi

check_cuda
open_browser
print_banner

exec python "$ROOT/app.py"
