#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

detect_backend() {
  if [ -n "${FISH_BACKEND:-}" ]; then
    printf '%s' "$FISH_BACKEND"
    return
  fi
  if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    printf 'mlx'
  else
    printf 'cuda'
  fi
}

BACKEND="$(detect_backend)"
VENV="$ROOT/.venv"
CUDA_EXTRA="${CUDA_EXTRA:-cu128}"
FISH_SPEECH_DIR="$ROOT/vendor/fish-speech"

resolve_python() {
  local candidate major minor path
  for candidate in python3.13 /opt/homebrew/bin/python3.13 /opt/homebrew/opt/python@3.13/bin/python3.13 python3; do
    if command -v "$candidate" &>/dev/null; then
      path="$(command -v "$candidate")"
    elif [ -x "$candidate" ]; then
      path="$candidate"
    else
      continue
    fi
    IFS=. read -r major minor _ < <("$path" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 13 ]; }; then
      printf '%s' "$path"
      return 0
    fi
  done

  if [ "$BACKEND" = "mlx" ] && command -v brew &>/dev/null; then
    if ! brew list python@3.13 &>/dev/null; then
      echo "Installing Python 3.13 (required for MLX)..." >&2
      brew install python@3.13 >&2
    fi
    printf '%s' "$(brew --prefix python@3.13)/bin/python3.13"
    return 0
  fi

  if command -v python3 &>/dev/null; then
    printf '%s' "$(command -v python3)"
    return 0
  fi

  echo "python3 is required" >&2
  exit 1
}

PYTHON="$(resolve_python)"
echo "Backend: $BACKEND"
echo "Python: $($PYTHON --version)"

if [ "$BACKEND" = "cuda" ] && command -v apt-get &>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    python3-dev \
    git \
    ffmpeg \
    libsox-dev \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0
fi

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel

if [ "$BACKEND" = "mlx" ]; then
  pip install mlx-speech
  pip install -r "$ROOT/requirements.txt"
else
  if [ ! -d "$FISH_SPEECH_DIR" ]; then
    git clone --depth 1 https://github.com/fishaudio/fish-speech.git "$FISH_SPEECH_DIR"
  fi

  pip install torch==2.8.0 torchaudio==2.8.0 \
    --index-url "https://download.pytorch.org/whl/${CUDA_EXTRA}"

  if ! pip install --no-build-isolation pyaudio; then
    cat >&2 <<'EOF'
PyAudio failed to build. On Ubuntu/Debian, run:

  sudo apt-get update
  sudo apt-get install -y build-essential python3-dev portaudio19-dev \
    libportaudio2 libportaudiocpp0 libasound-dev libsox-dev ffmpeg

Then re-run ./setup.sh

With conda: conda install -c conda-forge pyaudio
EOF
    exit 1
  fi

  pip install -e "$FISH_SPEECH_DIR"
  pip install -r "$ROOT/requirements.txt"
fi

python "$ROOT/download_model.py"

echo
echo "Setup complete ($BACKEND backend)."
echo "Run the app: ./run.sh"
