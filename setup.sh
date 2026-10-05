#!/usr/bin/env bash
# Crate setup - installs Essentia and downloads the genre models.
# Run once:  ./setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="${CRATE_MODELS:-$HOME/.crate/models}"
BASE="https://essentia.upf.edu/models"

echo "Crate setup"
echo "-----------"

# --- python -----------------------------------------------------------------
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null; then
  echo "No python3 found. Install it with:  brew install python@3.12"; exit 1
fi
VER=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
MAJ=${VER%%.*}; MIN=${VER##*.}
if [ "$MAJ" -ne 3 ] || [ "$MIN" -lt 9 ]; then
  echo "Python $VER found, but 3.9+ is required."
  echo "Try:  brew install python@3.12 && PYTHON=python3.12 ./setup.sh"; exit 1
fi
echo "Python $VER"

# --- venv -------------------------------------------------------------------
if [ ! -d "$HERE/.venv" ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv "$HERE/.venv"
fi
# shellcheck disable=SC1091
source "$HERE/.venv/bin/activate"
python -m pip install --quiet --upgrade pip

echo "Installing Essentia (this pulls TensorFlow, ~400MB, give it a few minutes)..."
# scikit-learn is used to TRAIN the crate classifier (and to measure it in
# eval_genre.py / tune_genre.py). Predicting never needs it: the trained
# weights are exported into crate_model.json and applied with numpy.
if ! pip install --quiet "essentia-tensorflow" "numpy" "scikit-learn"; then
  echo
  echo "Essentia failed to install on this Python version."
  echo "Wheels exist for Python 3.9-3.14. If you are outside that range:"
  echo "  brew install python@3.12 && rm -rf .venv && PYTHON=python3.12 ./setup.sh"
  exit 1
fi
python -c "import essentia.standard" 2>/dev/null && echo "Essentia OK"

# --- ffmpeg (m4a/wma decoding) ----------------------------------------------
if ! command -v ffmpeg >/dev/null; then
  echo
  echo "Note: ffmpeg is not installed. WAV, MP3, AIFF and FLAC will still work;"
  echo "      M4A and WMA may not. Install with:  brew install ffmpeg"
fi

# --- models -----------------------------------------------------------------
mkdir -p "$MODELS"
fetch () {
  local f="$1" url="$2"
  if [ -s "$MODELS/$f" ]; then echo "  have $f"; return; fi
  echo "  downloading $f"
  if ! curl -fSL --retry 3 --connect-timeout 20 -o "$MODELS/$f.part" "$url"; then
    rm -f "$MODELS/$f.part"; echo "  ! failed: $f"; return 1
  fi
  mv "$MODELS/$f.part" "$MODELS/$f"
}

echo "Fetching models into $MODELS"
OK=1
fetch discogs-effnet-bs64-1.pb \
  "$BASE/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb" || OK=0
fetch genre_discogs400-discogs-effnet-1.pb \
  "$BASE/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb" || OK=0
fetch genre_discogs400-discogs-effnet-1.json \
  "$BASE/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json" || OK=0
fetch voice_instrumental-discogs-effnet-1.pb \
  "$BASE/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb" || true

echo
if [ "$OK" -eq 1 ]; then
  echo "Setup complete."
else
  echo "Setup finished, but some models did not download."
  echo "Tempo, key, cues and acapella detection will still work, but tracks"
  echo "cannot be sorted into crates until the models are there."
  echo "Re-run ./setup.sh to retry the downloads, then Start sorting again."
fi
echo
echo "Now start the app:  ./crate.command   (or double-click crate.command in Finder)"
echo "Command-line analyser instead:"
echo "  source .venv/bin/activate && python analyze.py ~/Music -o ~/Desktop/crate_output"
