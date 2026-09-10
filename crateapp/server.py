"""Local HTTP server. All filesystem work happens here, never in the browser -
which is why the UI never needs a file picker.

Binds to 127.0.0.1 only: this is a local tool and must never be reachable
from the network. Stdlib only (http.server / json / sqlite3) - no Flask,
no FastAPI.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from crateapp import crates as crate_ops
from crateapp.classifier import rebuild_centroids
from crateapp.db import LOCK
from crateapp.exporters import export_folders, export_rekordbox
from crateapp.sets import build_set, dedupe, load_pool, neighbours
from crateapp.scanner import scan

STATIC = Path(__file__).resolve().parent / "static"


class ApiError(Exception):
    """A clean, expected failure that should become a 4xx JSON error rather
    than an uncaught-exception 500 traceback."""

    def __init__(self, message, code=400):
        super().__init__(message)
        self.message = message
        self.code = code


class _Retrainer:
    """Retrain shortly after the DJ stops correcting, not during.

    Fitting the crate model takes about a second on this library. Doing it
    inside the correction request means every correction blocks for a second -
    and blocks other requests too, since it holds the database lock. Someone
    working through a review queue makes corrections in bursts, so the fit is
    debounced: each correction pushes the retrain out, and it lands once the
    burst is over. The model is a moment behind the DJ, which costs nothing;
    a UI that stalls on every click costs them the session.
    """

    def __init__(self, con, model_path, delay=3.0):
        self.con = con
        self.model_path = model_path
        self.delay = delay
        self._timer = None
        self._lock = threading.Lock()

    def schedule(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self):
        try:
            with LOCK:
                rebuild_centroids(self.con, self.model_path)
        except Exception:
            pass          # a failed retrain must never take the server down

    def flush(self):
        """Retrain now, for tests and shutdown."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._run()


def make_app(con, model_path, runner=None):
    retrainer = _Retrainer(con, model_path)
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 so the browser gets keep-alive and well-behaved range
        # requests. Every response below sends an accurate Content-Length,
        # which 1.1 requires.
        protocol_version = "HTTP/1.1"

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
                            "disputed": self._rows(crate_ops.disputed(con)),
                        })

                if path.startswith("/api/preview/"):
                    return self._preview(path[len("/api/preview/"):])

                if path.startswith("/api/audio/"):
                    return self._audio(path[len("/api/audio/"):])

                if path.startswith("/api/track/"):
                    return self._send(self._track(path[len("/api/track/"):]))

                if path == "/api/setpool":
                    q = parse_qs(urlsplit(self.path).query)
                    return self._send(self._setpool(q.get("crate", [None])[0]))

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

                if self.path == "/api/stop":
                    if runner is None:
                        raise ApiError("no runner configured", code=503)
                    # Asks the pipeline to stop after the track in progress;
                    # it does not abandon that track's work half-written.
                    runner.stop()
                    return self._send({"stopping": True})

                if self.path == "/api/set":
                    return self._send(self._build_set(payload))

                if self.path == "/api/next":
                    return self._send(self._next(payload))
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
                    "disputed": len(crate_ops.disputed(con)),
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
            mode = payload.get("mode", "move")
            # A confirm has no destination: the answer is wherever the track
            # already is.
            to_crate = (payload.get("to_crate") if mode == "confirm"
                        else self._require(payload, "to_crate"))
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
            # Corrections are the only thing that changes crate membership, so
            # this is the one place the model needs to learn again - but not
            # on this request's clock. Classifier has no reload(): the next
            # classify() constructs a fresh Classifier(model_path) from
            # whatever the retrain last wrote.
            retrainer.schedule()
            return {"ok": True}

        # -- audio preview ----------------------------------------------
        #
        # A short WAV segment, not the file. Two reasons, both decisive:
        # browsers cannot decode AIFF at all (verified: EncodingError on
        # Chrome) and AIFF is 669 of this library's ~1400 files; and shipping
        # 56MB to audition twenty seconds is absurd when the segment is 1.7MB.
        # Decoding through essentia - already a dependency, and the thing that
        # analysed these files in the first place - makes every format the
        # analyser can read previewable, which is all of them.
        #
        # Addressed by track id, never by path: the browser can name a row the
        # analyser created, not a file to read.

        MAX_PREVIEW_SEC = 45.0

        # Formats a browser decodes itself. Everything else has to be
        # converted before it can be played - AIFF above all, which is most of
        # this library and which Chrome refuses outright.
        NATIVE_AUDIO = {".wav": "audio/wav", ".mp3": "audio/mpeg",
                        ".m4a": "audio/mp4", ".mp4": "audio/mp4",
                        ".flac": "audio/flac", ".ogg": "audio/ogg",
                        ".oga": "audio/ogg", ".aac": "audio/aac"}
        CACHE = Path.home() / ".crate" / "cache"

        def _track_row(self, raw_id):
            try:
                track_id = int(raw_id)
            except ValueError:
                raise ApiError("bad track id")
            with LOCK:
                row = con.execute(
                    "SELECT id, path, filename, bpm, camelot, duration_sec "
                    "FROM tracks WHERE id=?", (track_id,)).fetchone()
            if row is None:
                raise ApiError("no such track", code=404)
            return row

        def _playable(self, row):
            """The file to actually send, converting first if the browser
            cannot read the original.

            AIFF -> WAV is a container change, not a re-encode: both are
            uncompressed PCM, so nothing is lost. The result is cached under
            ~/.crate/cache keyed by the source's mtime, so a track converts
            once and seeks instantly ever after.
            """
            src = Path(row["path"])
            if not src.is_file():
                raise ApiError("file is missing from disk", code=404)

            ctype = self.NATIVE_AUDIO.get(src.suffix.lower())
            if ctype:
                return src, ctype

            self.CACHE.mkdir(parents=True, exist_ok=True)
            dst = self.CACHE / f"{row['id']}-{int(src.stat().st_mtime)}.wav"
            if not dst.exists():
                import wave

                import numpy as np
                import essentia.standard as es
                try:
                    audio, sr, ch, _, _, _ = es.AudioLoader(filename=str(src))()
                except Exception as e:
                    raise ApiError(f"could not decode audio: {e}", code=422)
                pcm = np.clip(np.asarray(audio, dtype="float32"), -1.0, 1.0)
                pcm = (pcm * 32767.0).astype("<i2")
                # Written to a temp name and renamed, so a half-converted file
                # is never served if this dies midway.
                tmp = dst.with_suffix(".part")
                with wave.open(str(tmp), "wb") as w:
                    w.setnchannels(int(ch) or 1)
                    w.setsampwidth(2)
                    w.setframerate(int(sr))
                    w.writeframes(pcm.tobytes())
                tmp.replace(dst)
            return dst, "audio/wav"

        def _audio(self, raw_id):
            """The whole track, seekable - so any song can be played at any
            time, not just auditioned around a mix point."""
            f, ctype = self._playable(self._track_row(raw_id))
            size = f.stat().st_size

            # Range support is not optional: seeking a six-minute track means
            # asking for bytes from the middle, and without 206 the browser
            # refuses to seek and refetches the whole file instead.
            start, end, partial = 0, size - 1, False
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                a, _, b = rng[len("bytes="):].partition("-")
                try:
                    if a:
                        start = int(a)
                        end = int(b) if b else size - 1
                    elif b:                       # bytes=-N -> the last N bytes
                        start = max(0, size - int(b))
                    partial = True
                except ValueError:
                    start, end, partial = 0, size - 1, False
            end = min(end, size - 1)
            if partial and (start > end or start >= size):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            length = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(f, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(262144, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return        # the DJ seeked away; not an error
                    remaining -= len(chunk)

        def _track(self, raw_id):
            """One track in the same shape the runner reports the track it is
            working on, so the panel has a single renderer for both.

            The similarity numbers are recomputed here rather than stored: it
            is the identical normalise-and-dot-product classify() runs, so the
            panel shows the model's actual current opinion, not a stale one
            from whenever the track happened to be analysed.
            """
            row = self._track_row(raw_id)
            track_id = row["id"]
            with LOCK:
                an = con.execute(
                    "SELECT moments, energy, vocal FROM analysis WHERE track_id=?",
                    (track_id,)).fetchone()
                emb = con.execute(
                    "SELECT vector FROM embeddings WHERE track_id=?",
                    (track_id,)).fetchone()
                # Every crate it is in, not one of them. A track filed by hand
                # in one crate and by the model in another is the interesting
                # case, and picking whichever row came back first hid it.
                asgs = self._rows(con.execute(
                    "SELECT c.name, a.band, a.confidence, a.source "
                    "FROM assignments a JOIN crates c ON c.id=a.crate_id "
                    "WHERE a.track_id=? ORDER BY a.source='human' DESC, c.name",
                    (track_id,)).fetchall())
            asg = asgs[0] if asgs else None

            out = {
                "id": track_id,
                "filename": row["filename"],
                "bpm": row["bpm"],
                "camelot": row["camelot"],
                "duration": row["duration_sec"],
                "energy": json.loads(an["energy"]) if an and an["energy"] else [],
                "moments": json.loads(an["moments"]) if an and an["moments"] else [],
                "vocal": json.loads(an["vocal"]) if an and an["vocal"] else None,
                "similarities": [],
                "crate": asg["name"] if asg else None,
                "band": (asg["band"] if asg else None) or "unknown",
                "crates": asgs,
                "margin": None,
                "analysed": bool(an),
            }

            if emb is not None and emb["vector"]:
                import numpy as np
                from crateapp.classifier import Classifier
                try:
                    clf = Classifier(model_path)
                except Exception:
                    clf = None
                if clf is not None and clf.crate_names():
                    v = np.frombuffer(emb["vector"], dtype=np.float32).copy()
                    r = clf.classify(v)
                    out["similarities"] = [
                        {"crate": s["crate"], "similarity": s["p"]}
                        for s in r["scores"]]
                    out["margin"] = r["margin"]
                    out["model_pick"] = r["crate"]
                    out["model_band"] = r["band"]
            return out

        def _preview(self, raw_id):
            import io
            import wave

            import numpy as np

            try:
                track_id = int(raw_id)
            except ValueError:
                raise ApiError("bad track id")

            q = parse_qs(urlsplit(self.path).query)
            def num(key, default):
                try:
                    return float(q.get(key, [default])[0])
                except (TypeError, ValueError):
                    raise ApiError(f"{key} must be a number")
            at = max(0.0, num("at", 0.0))
            length = min(self.MAX_PREVIEW_SEC, max(0.5, num("len", 20.0)))

            with LOCK:
                row = con.execute("SELECT path FROM tracks WHERE id=?",
                                  (track_id,)).fetchone()
            if row is None:
                raise ApiError("no such track", code=404)
            f = Path(row["path"])
            if not f.is_file():
                raise ApiError("file is missing from disk", code=404)

            import essentia.standard as es
            SR = 44100
            try:
                audio = es.MonoLoader(filename=str(f), sampleRate=SR)()
            except Exception as e:
                raise ApiError(f"could not decode audio: {e}", code=422)

            start = min(int(at * SR), len(audio))
            stop = min(start + int(length * SR), len(audio))
            seg = audio[start:stop]
            if not len(seg):
                raise ApiError("that position is past the end of the track")

            pcm = np.clip(np.asarray(seg, dtype="float32"), -1.0, 1.0)
            pcm = (pcm * 32767.0).astype("<i2")

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm.tobytes())
            body = buf.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass          # the DJ moved on; not an error

        def do_HEAD(self):
            return self.do_GET()

        # -- set builder ----------------------------------------------
        #
        # The pool is rebuilt per request rather than cached: 60ms at full
        # library size, against a cache that would have to be invalidated
        # every time the runner finishes a track or the DJ makes a
        # correction. Not worth the staleness bug.

        def _pool(self, crate=None):
            with LOCK:
                return dedupe(load_pool(con, crate=crate))

        # The seed picker shows a name, a tempo and a key. Shipping each
        # track's full moment list too made the listing 393KB for 297 tracks
        # and would be near a megabyte across the whole library, none of it
        # read: mix points come back already computed on /api/set and
        # /api/next.
        LISTING = ("id", "filename", "bpm", "camelot", "duration", "energy",
                   "vocal_ratio", "duplicates")

        def _setpool(self, crate=None):
            return {"tracks": [{k: t.get(k) for k in self.LISTING}
                               for t in self._pool(crate)]}

        def _build_set(self, payload):
            seed = self._require(payload, "seed_id")
            try:
                return build_set(self._pool(payload.get("crate")), seed,
                                 length=int(payload.get("length", 8)),
                                 mode=payload.get("mode", "balanced"),
                                 arc=payload.get("arc", "steady"))
            except ValueError as e:
                raise ApiError(str(e))

        def _next(self, payload):
            track_id = self._require(payload, "track_id")
            try:
                out = neighbours(self._pool(payload.get("crate")), track_id,
                                 mode=payload.get("mode", "balanced"),
                                 arc=payload.get("arc", "steady"),
                                 limit=int(payload.get("limit", 25)))
            except ValueError as e:
                raise ApiError(str(e))
            return {"next": out}

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

    Handler.retrainer = retrainer
    return Handler


def serve(con, model_path, port=8420, runner=None):
    # Threaded, not the plain single-threaded HTTPServer: streaming a 56MB
    # WAV for a transition preview would otherwise block every other request
    # for the length of the download - the UI would freeze the moment the DJ
    # pressed play. The sqlite connection is shared safely (check_same_thread
    # is off and every write goes through db.LOCK).
    srv = ThreadingHTTPServer(("127.0.0.1", port),
                              make_app(con, model_path, runner=runner))
    srv.daemon_threads = True
    print(f"Crate running at http://127.0.0.1:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # A correction made in the last few seconds has a retrain still
        # pending on a timer. Without this it is simply lost, and the model
        # silently stays behind until the next correction happens to fire one.
        try:
            srv.RequestHandlerClass.retrainer.flush()
        except Exception:
            pass
        srv.server_close()
