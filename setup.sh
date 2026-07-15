#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
CUDA_EXTRA="${CUDA_EXTRA:-cu128}"
FISH_SPEECH_DIR="$ROOT/vendor/fish-speech"

if command -v apt-get &>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y portaudio19-dev libsox-dev ffmpeg git
fi

if ! command -v python3 &>/dev/null; then
  echo "python3 is required"
  exit 1
fi

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel

if [ ! -d "$FISH_SPEECH_DIR" ]; then
  git clone --depth 1 https://github.com/fishaudio/fish-speech.git "$FISH_SPEECH_DIR"
fi

pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url "https://download.pytorch.org/whl/${CUDA_EXTRA}"
pip install -e "$FISH_SPEECH_DIR"
pip install -r "$ROOT/requirements.txt"

python "$ROOT/download_model.py"

echo
echo "Setup complete."
echo "Run the app: ./run.sh"
