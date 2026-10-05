#!/usr/bin/env bash
# Crate — double-click to open the sorter.
# LSUIElement keeps this out of the Dock; it starts the local server (if it
# isn't already up) and opens the browser.

APP_DIR="$HOME/Desktop/crate"
PORT=8420
URL="http://127.0.0.1:$PORT"

notify() { osascript -e "display notification \"$1\" with title \"Crate\"" 2>/dev/null; }

cd "$APP_DIR" 2>/dev/null || {
  osascript -e 'display alert "Crate not found" message "Expected the project at ~/Desktop/crate."'
  exit 1
}

if [ ! -d .venv ]; then
  osascript -e 'display alert "Setup not run" message "Open Terminal in ~/Desktop/crate and run ./setup.sh first."'
  exit 1
fi

# Already running? Just open the tab.
if curl -fs "$URL/api/state" >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi

notify "Starting…"
# Absolute interpreter path on purpose: a GUI-launched .app gets a minimal
# PATH, so "python" after activating the venv is not resolvable here even
# though it is in a Terminal.
PY_BIN="$APP_DIR/.venv/bin/python3"
[ -x "$PY_BIN" ] || PY_BIN="$APP_DIR/.venv/bin/python"
# macOS launched this bundle under Rosetta (x86_64) while the venv's numpy
# and essentia are arm64, so the import failed with an architecture mismatch.
# Pin the interpreter to native arm64 on Apple Silicon.
# NOT `uname -m`: under Rosetta that reports x86_64, so the check would be
# false in exactly the case the fix is for. hw.optional.arm64 reports the
# real hardware regardless of the translation the process is running under.
ARCH_PREFIX=()
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
  ARCH_PREFIX=(arch -arm64)
fi
# The log goes in ~/.crate, which a brand-new user does not have yet; the
# redirect is opened before Python runs, so it must exist first.
mkdir -p "$HOME/.crate"
nohup "${ARCH_PREFIX[@]}" "$PY_BIN" -c "
from pathlib import Path
from crateapp.db import connect
from crateapp.runner import Runner
from crateapp.server import serve
from crateapp.model_file import user_model_path
crate_dir = Path.home()/'.crate'
# Same as crate.command: the model is per user, beside the library, and the
# repo's crate_model.json is never written to (decided before connect()
# creates the database - see crateapp/model_file.py).
model = str(user_model_path(crate_dir, bundled=Path.cwd()/'crate_model.json'))
db = crate_dir/'library.db'
serve(connect(db), model, port=$PORT, runner=Runner(db, model))
" > "$HOME/.crate/server.log" 2>&1 &

# Importing essentia + TensorFlow takes ~20s on a cold start, so a 15s wait
# reported "failed to start" for a server that was simply still loading.
for _ in $(seq 1 240); do
  curl -fs "$URL/api/state" >/dev/null 2>&1 && break
  sleep 0.5
done

if curl -fs "$URL/api/state" >/dev/null 2>&1; then
  open "$URL"
else
  osascript -e 'display alert "Crate failed to start" message "See ~/.crate/server.log"'
fi
