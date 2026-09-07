"""Local HTTP server. All filesystem work happens here, never in the browser -
which is why the UI never needs a file picker.

Binds to 127.0.0.1 only: this is a local tool and must never be reachable
from the network. Stdlib only (http.server / json / sqlite3) - no Flask,
no FastAPI.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from crateapp import crates as crate_ops
from crateapp.classifier import rebuild_centroids
from crateapp.db import LOCK
from crateapp.exporters import export_folders, export_rekordbox
from crateapp.scanner import scan

STATIC = Path(__file__).resolve().parent / "static"


class ApiError(Exception):
    """A clean, expected failure that should become a 4xx JSON error rather
    than an uncaught-exception 500 traceback."""

    def __init__(self, message, code=400):
        super().__init__(message)
        self.message = message
        self.code = code


def make_app(con, model_path, runner=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass                                   # keep the terminal readable

        def _send(self, obj, code=200):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, err):
            self._send({"error": err.message}, err.code)

        def _rows(self, rows):
            return [dict(r) for r in rows]

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if not raw:
                return {}
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                raise ApiError("request body must be valid JSON")
            if not isinstance(payload, dict):
                raise ApiError("request body must be a JSON object")
            return payload

        def _require(self, payload, key):
            if key not in payload:
                raise ApiError(f"missing required field: {key}")
            return payload[key]

        # -- static files ------------------------------------------------

        def _serve_static(self, path):
            name = "index.html" if path == "/" else path[len("/static/"):]
            if not name:
                return self._send({"error": "not found"}, 404)
            f = (STATIC / name).resolve()
            # Refuse anything that escapes the static directory (e.g. a
            # crafted "/static/../server.py").
            if STATIC not in f.parents and f != STATIC:
                return self._send({"error": "not found"}, 404)
            if not f.exists() or not f.is_file():
                return self._send({"error": "not found"}, 404)
            body = f.read_bytes()
            ctype = ("text/html" if f.suffix == ".html"
                     else "text/css" if f.suffix == ".css"
                     else "application/javascript" if f.suffix == ".js"
                     else "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # -- routing -------------------------------------------------------

        def do_GET(self):
            path = urlsplit(self.path).path

            if path == "/" or path.startswith("/static/"):
                return self._serve_static(path)

            try:
                if path == "/api/state":
                    return self._send(self._state())

                if path.startswith("/api/crate/"):
                    name = unquote(path[len("/api/crate/"):])
                    if not name:
                        raise ApiError("crate name cannot be empty")
                    return self._send(self._crate(name))

                if path == "/api/review":
                    with LOCK:
                        return self._send({
                            "uncertain": self._rows(crate_ops.uncertain(con)),
                            "unsorted": self._rows(crate_ops.unsorted(con)),
                        })

                if path == "/api/progress":
                    if runner is None:
                        raise ApiError("no runner configured", code=503)
                    return self._send(runner.status())
            except ApiError as e:
                return self._send_error(e)
            except Exception as e:                 # pragma: no cover - safety net
                return self._send({"error": str(e)}, 500)

            return self._send({"error": "not found"}, 404)

        def do_POST(self):
            try:
                payload = self._read_json()

                if self.path == "/api/correct":
                    return self._send(self._correct(payload))

                if self.path == "/api/scan":
                    folder = self._require(payload, "folder")
                    with LOCK:
                        return self._send(scan(con, folder))

                if self.path == "/api/export":
                    return self._send(self._export(payload))

                if self.path == "/api/start":
                    return self._send(self._start(payload))
            except ApiError as e:
                return self._send_error(e)
            except Exception as e:                 # pragma: no cover - safety net
                return self._send({"error": str(e)}, 500)

            return self._send({"error": "not found"}, 404)

        # -- handlers --------------------------------------------------

        def _state(self):
            with LOCK:
                rows = con.execute(
                    "SELECT c.name, count(a.track_id) n FROM crates c "
                    "LEFT JOIN assignments a ON a.crate_id=c.id "
                    "GROUP BY c.id ORDER BY c.name").fetchall()
                return {
                    "crates": [{"name": r["name"], "count": r["n"]} for r in rows],
                    "uncertain": len(crate_ops.uncertain(con)),
                    "unsorted": len(crate_ops.unsorted(con)),
                }

        def _crate(self, name):
            with LOCK:
                rows = con.execute(
                    "SELECT t.* FROM tracks t JOIN assignments a ON a.track_id=t.id "
                    "JOIN crates c ON c.id=a.crate_id WHERE c.name=? ORDER BY t.id",
                    (name,)).fetchall()
                return {"tracks": self._rows(rows)}

        def _correct(self, payload):
            track_id = self._require(payload, "track_id")
            to_crate = self._require(payload, "to_crate")
            mode = payload.get("mode", "move")
            was_error = payload.get("was_error")
            with LOCK:
                try:
                    crate_ops.correct(con, track_id, to_crate, mode=mode,
                                      was_error=was_error)
                except ValueError as e:
                    # Empty/whitespace crate name (crates.ensure_crate) or a
                    # bad mode value (crates.correct) - both are the caller's
                    # fault, not a server error.
                    raise ApiError(str(e))
                # Corrections are the only thing that changes crate
                # membership, so this is the one place the model needs to
                # learn again. Classifier has no reload(): the cheap, correct
                # move is to rebuild the centroid file and let the next
                # classify() call construct a fresh Classifier(model_path)
                # from it.
                rebuild_centroids(con, model_path)
            return {"ok": True}

        def _export(self, payload):
            dest = self._require(payload, "dest")
            kind = payload.get("kind", "folders")
            with LOCK:
                if kind == "folders":
                    return {"exported": export_folders(con, dest)}
                if kind == "rekordbox":
                    # export_rekordbox does not create its parent directory
                    # (unlike export_folders, which does) - do it here or a
                    # nested destination path raises FileNotFoundError.
                    Path(dest).resolve().parent.mkdir(parents=True, exist_ok=True)
                    return {"exported": export_rekordbox(con, dest)}
            raise ApiError(f"unknown export kind: {kind!r}")

        def _start(self, payload):
            if runner is None:
                raise ApiError("no runner configured", code=503)
            folders = payload.get("folders")
            with LOCK:
                if folders:
                    con.execute(
                        "INSERT OR REPLACE INTO config (key, value) VALUES "
                        "('folders', ?)", (json.dumps(folders),))
                    con.commit()
                else:
                    row = con.execute(
                        "SELECT value FROM config WHERE key='folders'").fetchone()
                    folders = json.loads(row["value"]) if row else []
            if not folders:
                raise ApiError("no folders configured to scan")
            return {"started": runner.start(folders)}

    return Handler


def serve(con, model_path, port=8420, runner=None):
    srv = HTTPServer(("127.0.0.1", port), make_app(con, model_path, runner=runner))
    print(f"Crate running at http://127.0.0.1:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
