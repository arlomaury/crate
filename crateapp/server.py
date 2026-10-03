"""Local HTTP server. All filesystem work happens here, never in the browser -
which is why the UI never needs a file picker.

Binds to 127.0.0.1 only: this is a local tool and must never be reachable
from the network. Stdlib only (http.server / json / sqlite3) - no Flask,
no FastAPI.
"""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from crateapp import crates as crate_ops
from crateapp.classifier import rebuild_centroids
from crateapp.db import LOCK
from crateapp.exporters import check_rekordbox_dest, export_folders, export_rekordbox
from crateapp.sets import build_set, dedupe, load_pool, neighbours
from crateapp.scanner import scan

STATIC = Path(__file__).resolve().parent / "static"

# Hostnames a request may legitimately carry in Host / Origin. Anything else
# is another website trying to reach the local API (see Handler._guard).
_LOCAL_NAMES = {"127.0.0.1", "localhost", "::1"}


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
                # Then file anything still filed nowhere with the new model -
                # otherwise a new library, analysed before it had a model,
                # would never be sorted (see runner.file_unsorted).
                from crateapp.runner import file_unsorted
                file_unsorted(self.con, self.model_path)
        except Exception:
            pass          # a failed retrain must never take the server down

    def flush(self):
        """Retrain now, for tests and shutdown."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._run()


_CONVERT_GUARD = threading.Lock()
_CONVERT_LOCKS = {}


def make_app(con, model_path, runner=None):
    retrainer = _Retrainer(con, model_path)
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 so the browser gets keep-alive and well-behaved range
        # requests. Every response below sends an accurate Content-Length,
        # which 1.1 requires.
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass                                   # keep the terminal readable

        def end_headers(self):
            # Defence-in-depth headers on every response: no MIME sniffing,
            # no framing by other sites, and no referrer leaking track paths.
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            super().end_headers()

        def _send(self, obj, code=200):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _guard(self, method):
            """Refuse requests that did not come from Crate's own page.

            Binding to 127.0.0.1 keeps the network out, but not other
            websites open in the same browser: any page can fire a request
            at http://127.0.0.1:8420, and the API can scan folders and write
            exports to disk. Three checks close that off:

            - Host must be a loopback name. Stops DNS rebinding, where an
              attacker's domain is re-pointed at 127.0.0.1 to read responses.
            - Origin, when the browser sends one, must be this server itself
              (loopback, and the same port).
              Stops cross-site form posts and fetches.
            - POSTs must be application/json. A cross-site page can only send
              that after a CORS preflight, which this server never approves.

            Returns True if the request may proceed.
            """
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in _LOCAL_NAMES:
                self._send({"error": "forbidden host"}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin is not None:
                o = urlsplit(origin)
                # The port matters too: a dev server or any other app on
                # another localhost port is a different site.
                if (o.scheme != "http" or (o.hostname or "").lower() not in _LOCAL_NAMES
                        or (o.port or 80) != self.server.server_port):
                    self._send({"error": "forbidden origin"}, 403)
                    return False
            if method == "POST":
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype != "application/json":
                    self._send({"error": "Content-Type must be application/json"}, 415)
                    return False
            return True

        def _send_error(self, err):
            self._send({"error": err.message}, err.code)

        def _rows(self, rows):
            return [dict(r) for r in rows]

        MAX_BODY = 1_000_000          # every request body is a small JSON object

        def _read_json(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError("bad Content-Length")
            if n < 0 or n > self.MAX_BODY:
                raise ApiError("request body too large", code=413)
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

        # Typed field access.  Every value in a request body is checked here,
        # so a wrong type is a clear 400 instead of a crash deep inside
        # SQLite or a filesystem call.
        _SQLITE_MAX = 2 ** 63 - 1

        def _field(self, payload, key, kind, default=None, required=False):
            if key not in payload or payload[key] is None:
                if required:
                    raise ApiError(f"missing required field: {key}")
                return default
            v = payload[key]
            ok = {
                "int": isinstance(v, int) and not isinstance(v, bool) and abs(v) <= self._SQLITE_MAX,
                "str": isinstance(v, str) and 0 < len(v) <= 4096 and "\x00" not in v,
                "bool": isinstance(v, bool),
                "strlist": isinstance(v, list) and all(isinstance(x, str) and 0 < len(x) <= 4096 and "\x00" not in x for x in v),
            }[kind]
            if not ok:
                raise ApiError(f"{key} must be {'an integer' if kind == 'int' else 'text' if kind == 'str' else 'true or false' if kind == 'bool' else 'a list of folder paths'}")
            return v

        def _path_id(self, raw_id):
            try:
                v = int(raw_id)
            except ValueError:
                raise ApiError("bad track id")
            if abs(v) > self._SQLITE_MAX:
                raise ApiError("bad track id")
            return v

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
            if not self._guard("GET"):
                return
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
                            "tag_gaps": crate_ops.tag_gaps(con),
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
            if not self._guard("POST"):
                return
            try:
                payload = self._read_json()

                if self.path == "/api/correct":
                    return self._send(self._correct(payload))

                if self.path == "/api/lock":
                    name = self._field(payload, "crate", "str", required=True)
                    locked = self._field(payload, "locked", "bool", default=True)
                    with LOCK:
                        if not crate_ops.set_locked(con, name, locked):
                            raise ApiError("no such crate", code=404)
                    # A locked crate leaves the model, so the model changes.
                    retrainer.schedule()
                    return self._send({"crate": name, "locked": locked})

                if self.path == "/api/remove":
                    track_id = self._field(payload, "track_id", "int", required=True)
                    with LOCK:
                        name = crate_ops.remove_track(con, track_id)
                    if name is None:
                        raise ApiError("no such track", code=404)
                    # Removing a track removes whatever it was teaching the
                    # model, so the model has to be refitted - debounced, like
                    # any other change to the labels.
                    retrainer.schedule()
                    return self._send({"removed": name})

                if self.path == "/api/scan":
                    folder = str(Path(self._field(payload, "folder", "str", required=True)).expanduser())
                    try:
                        # The lock is taken inside scan, only around the
                        # database work, so the UI keeps answering while a
                        # big folder is walked.
                        return self._send(scan(con, folder, lock=LOCK))
                    except FileNotFoundError as e:
                        raise ApiError(str(e), code=404)

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
                    "SELECT c.name, c.locked, count(a.track_id) n FROM crates c "
                    "LEFT JOIN assignments a ON a.crate_id=c.id "
                    "GROUP BY c.id ORDER BY c.name").fetchall()
                return {
                    "crates": [{"name": r["name"], "count": r["n"],
                                "locked": bool(r["locked"])} for r in rows],
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
            track_id = self._field(payload, "track_id", "int", required=True)
            mode = self._field(payload, "mode", "str", default="move")
            # A confirm has no destination: the answer is wherever the track
            # already is.
            to_crate = self._field(payload, "to_crate", "str", required=(mode != "confirm"))
            was_error = self._field(payload, "was_error", "bool")
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
            track_id = self._path_id(raw_id)
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
            # The browser often asks for the same track twice at once (the
            # player and a range request). One conversion per file: the
            # second request waits for the first instead of racing it.
            with _CONVERT_GUARD:
                lock = _CONVERT_LOCKS.setdefault(str(dst), threading.Lock())
            with lock:
                if not dst.exists():
                    self._convert(src, dst)
            return dst, "audio/wav"

        def _convert(self, src, dst):
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
            tmp = dst.with_name(f"{dst.stem}.{threading.get_ident()}.part")
            try:
                with wave.open(str(tmp), "wb") as w:
                    w.setnchannels(int(ch) or 1)
                    w.setsampwidth(2)
                    w.setframerate(int(sr))
                    w.writeframes(pcm.tobytes())
                tmp.replace(dst)
            finally:
                if tmp.exists():
                    tmp.unlink()

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

            track_id = self._path_id(raw_id)

            q = parse_qs(urlsplit(self.path).query)
            def num(key, default):
                try:
                    v = float(q.get(key, [default])[0])
                except (TypeError, ValueError):
                    raise ApiError(f"{key} must be a number")
                if not math.isfinite(v):
                    raise ApiError(f"{key} must be a finite number")
                return v
            at = min(max(0.0, num("at", 0.0)), 24 * 3600.0)   # no track is a day long
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
            seed = self._field(payload, "seed_id", "int", required=True)
            length = self._field(payload, "length", "int", default=8)
            if not 1 <= length <= 200:
                raise ApiError("length must be between 1 and 200")
            try:
                return build_set(self._pool(self._field(payload, "crate", "str")), seed,
                                 length=length,
                                 mode=self._field(payload, "mode", "str", default="balanced"),
                                 arc=self._field(payload, "arc", "str", default="steady"))
            except ValueError as e:
                raise ApiError(str(e))

        def _next(self, payload):
            track_id = self._field(payload, "track_id", "int", required=True)
            limit = self._field(payload, "limit", "int", default=25)
            if not 1 <= limit <= 500:
                raise ApiError("limit must be between 1 and 500")
            try:
                out = neighbours(self._pool(self._field(payload, "crate", "str")), track_id,
                                 mode=self._field(payload, "mode", "str", default="balanced"),
                                 arc=self._field(payload, "arc", "str", default="steady"),
                                 limit=limit)
            except ValueError as e:
                raise ApiError(str(e))
            return {"next": out}

        def _export(self, payload):
            dest = self._field(payload, "dest", "str", required=True)
            kind = self._field(payload, "kind", "str", default="folders")
            if not Path(dest).expanduser().is_absolute():
                # A relative path would land wherever the server was started
                # from (the app's own folder) and the DJ would never find it.
                raise ApiError("give a full path, e.g. ~/Desktop/Crates or ~/Desktop/rekordbox.xml")
            dest = str(Path(dest).expanduser().resolve())
            try:
                with LOCK:
                    if kind == "folders":
                        skipped = []
                        counts = export_folders(con, dest, skipped)
                        return {"exported": counts, "dest": dest, "skipped": len(skipped),
                                "skippedExamples": skipped[:5]}
                    if kind == "rekordbox":
                        # export_rekordbox does not create its parent directory
                        # (unlike export_folders, which does) - do it here or a
                        # nested destination path raises FileNotFoundError.
                        check_rekordbox_dest(Path(dest))
                        Path(dest).parent.mkdir(parents=True, exist_ok=True)
                        return {"exported": export_rekordbox(con, dest), "dest": dest}
            except ValueError as e:
                raise ApiError(str(e))
            except OSError as e:
                # Bad destination: not writable, a file where a folder is
                # needed, a name too long...  The DJ's mistake, not a crash.
                raise ApiError(f"could not export to {dest}: {e.strerror or e}")
            raise ApiError(f"unknown export kind: {kind!r}")

        def _start(self, payload):
            if runner is None:
                raise ApiError("no runner configured", code=503)
            folders = self._field(payload, "folders", "strlist")
            if folders:
                folders = [str(Path(f).expanduser()) for f in folders]
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
