#!/usr/bin/env bash
# Crate — double-click this to start the app.
#
# Idempotent on purpose: clicking twice focuses the running instance rather
# than starting a second server on a busy port.

cd "$(dirname "$0")" || exit 1

PORT=8420
URL="http://127.0.0.1:$PORT"

# macOS has `open`; most Linux desktops have `xdg-open`.
open_url () {
  if command -v open >/dev/null 2>&1; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1
  else echo "Open $1 in your browser."; fi
}

if [ ! -d .venv ]; then
  echo "Setup has not been run yet."
  echo "Run ./setup.sh first, then double-click this again."
  read -r -p "Press return to close."
  exit 1
fi

source .venv/bin/activate

# Already up? Just bring the tab forward.
if curl -fs "$URL/api/state" >/dev/null 2>&1; then
  echo "Crate is already running."
  open_url "$URL"
  exit 0
fi

echo "Starting Crate…"
python - "$PORT" <<'PY' &
import sys
from pathlib import Path
from crateapp.db import connect
from crateapp.runner import Runner
from crateapp.server import serve
from crateapp.model_file import user_model_path

port = int(sys.argv[1])
crate_dir = Path.home() / ".crate"
# The model lives beside the library it was trained from, never in this folder
# (see crateapp/model_file.py). Decide where BEFORE connect() creates the db.
model = user_model_path(crate_dir, bundled=Path.cwd() / "crate_model.json")
db = crate_dir / "library.db"

con = connect(db)
runner = Runner(db, str(model))
serve(con, str(model), port=port, runner=runner)
PY
SERVER_PID=$!

# Wait for it to answer rather than guessing at a sleep duration.
# essentia + TensorFlow take ~20s to import on a cold start.
for _ in $(seq 1 240); do
  if curl -fs "$URL/api/state" >/dev/null 2>&1; then break; fi
  sleep 0.5
done

if ! curl -fs "$URL/api/state" >/dev/null 2>&1; then
  echo "Crate failed to start. Run this to see why:"
  echo "  cd \"$(pwd)\" && source .venv/bin/activate && python -c \\"
  echo "    \"from crateapp.db import connect; from crateapp.server import serve; \\"
  echo "     from pathlib import Path; serve(connect(Path.home()/'.crate/library.db'), str(Path.home()/'.crate/crate_model.json'))\""
  read -r -p "Press return to close."
  exit 1
fi

open_url "$URL"
echo "Crate is running at $URL"
echo "Close this window to stop it."
wait "$SERVER_PID"
