#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

VENV="$ROOT/.venv"
FISH_SPEECH_DIR="$ROOT/vendor/fish-speech"
MODEL_MARKER="$ROOT/checkpoints/s2-pro/codec.pth"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
OPEN_BROWSER="${OPEN_BROWSER:-auto}"
CACHE_DIR="${CACHE_DIR:-$ROOT/.cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_DIR/torchinductor}"
export FISH_MAX_SEQ_LEN="${FISH_MAX_SEQ_LEN:-4096}"
export FISH_COMPILE="${FISH_COMPILE:-1}"
export FISH_WARMUP="${FISH_WARMUP:-1}"

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

server_ip() {
  local ip
  ip="$(curl -4 -s --max-time 2 ifconfig.me 2>/dev/null || true)"
  if [ -z "$ip" ]; then
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  printf '%s' "${ip:-<server-ip>}"
}

open_firewall_port() {
  if ! command -v ufw &>/dev/null; then
    return
  fi
  if ! ufw status 2>/dev/null | grep -q "Status: active"; then
    return
  fi
  if ufw status 2>/dev/null | grep -q "${PORT}/tcp"; then
    return
  fi
  log "Opening port ${PORT} in ufw..."
  sudo ufw allow "${PORT}/tcp" >/dev/null 2>&1 || true
}

print_access_info() {
  local ip network_url
  ip="$(server_ip)"
  if [ "$HOST" = "0.0.0.0" ]; then
    network_url="http://${ip}:${PORT}"
  else
    network_url="http://${HOST}:${PORT}"
  fi

  echo
  echo "Fish Audio S2 Pro — Web UI"
  echo "  Local:   http://127.0.0.1:${PORT}"
  echo "  Network: ${network_url}"
  if [ -n "${SSH_CONNECTION:-}" ]; then
    echo
    echo "  Remote access (run on your laptop, then open http://127.0.0.1:${PORT}):"
    echo "    ssh -L ${PORT}:127.0.0.1:${PORT} root@${ip}"
  fi
  echo
  echo "  If the network URL does not load, allow TCP ${PORT} in your cloud firewall"
  echo "  (DigitalOcean → Droplet → Networking → Firewalls)."
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
open_firewall_port
open_browser

echo
echo "Fish Audio S2 Pro — loading model (warmup compile may take 1-2 min on first run)..."
print_access_info

exec python "$ROOT/app.py"
