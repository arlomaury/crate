"""The pipeline runner: connects scanner -> worker -> classifier -> crates.

Nothing else in the app calls classify() or auto_assign() - this is where that
finally happens, in a background daemon thread so the HTTP server stays
responsive while a library-sized run (930 tracks, hours) is in progress.

Threading: server.py serves on a ThreadingHTTPServer (one thread per request,
so streaming a track does not stall the UI), and the whole app shares one
sqlite3 connection opened with check_same_thread=False. This runner adds
another thread that also touches that connection, so every access to it - here
and in server.py - is guarded by the shared crateapp.db.LOCK. The lock is held
per unit of work (one scan, one track's writes), never for the whole run nor
across the 4s of audio analysis, so a run in progress does not freeze the UI.

`self._status` is a separate concern from the DB lock: it is an in-process
dict read by the server thread and written by the runner thread, so it gets
its own small lock. It has nothing to do with sqlite thread-safety.
"""
import json
import threading
from pathlib import Path

import analyze
from crateapp.classifier import Classifier, acapella_verdict
from crateapp.crates import auto_assign
from crateapp.db import LOCK, connect
from crateapp.scanner import pending, scan
from crateapp.worker import analyse_one, analyse_track, load_embedding, store_result


def load_genre_model(models_dir):
    """Module-level so tests can monkeypatch crateapp.runner.load_genre_model
    and never load TensorFlow. Loading the real model takes ~20s."""
    return analyze.GenreModel(models_dir) if models_dir else analyze.GenreModel()


class Runner:
    def __init__(self, db_path, model_path, models_dir=None):
        self.db_path = db_path
        self.model_path = model_path
        self.models_dir = models_dir
        self._thread = None
        self._stop_flag = threading.Event()
        self._status_lock = threading.Lock()
        self._status = {
            "running": False,
            "done": 0,
            "total": 0,
            "current": None,
            "errors": 0,
        }

    # -- public API ------------------------------------------------------

    def start(self, folders):
        """Begin a run over `folders` in a background daemon thread.

        Returns False without doing anything if a run is already in progress
        (idempotent - a second /api/start while sorting does not start a
        second overlapping run).
        """
        with self._status_lock:
            if self._status["running"]:
                return False
            self._status["running"] = True
            self._status["done"] = 0
            self._status["total"] = 0
            self._status["errors"] = 0
            # "current" is deliberately NOT reset here - the panel shows the
            # last decided track at rest, including between runs.
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run, args=(list(folders),), daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Ask a running pipeline to stop after the track in progress."""
        self._stop_flag.set()

    def wait(self):
        """Block until the current run finishes. Used by tests; harmless in
        production (e.g. for a clean shutdown)."""
        if self._thread is not None:
            self._thread.join()

    def status(self):
        with self._status_lock:
            return dict(self._status)

    # -- internals ---------------------------------------------------------

    def _run(self, folders):
        con = connect(self.db_path)
        try:
            with LOCK:
                for folder in folders:
                    scan(con, folder)
                rows = pending(con)

            with self._status_lock:
                self._status["total"] = len(rows)

            gm = load_genre_model(self.models_dir)
            classifier = self._load_classifier()

            for row in rows:
                if self._stop_flag.is_set():
                    break
                try:
                    current = self._process_track(con, gm, classifier, row)
                except Exception:
                    # One bad file must never end a run over 1,000 tracks.
                    current = None
                    with self._status_lock:
                        self._status["errors"] += 1

                with self._status_lock:
                    if current is not None:
                        self._status["current"] = current
                    self._status["done"] += 1
        finally:
            con.close()
            with self._status_lock:
                self._status["running"] = False

    def _load_classifier(self):
        try:
            if Path(self.model_path).exists():
                return Classifier(self.model_path)
        except Exception:
            pass
        return None

    def _process_track(self, con, gm, classifier, row):
        """Analyse one track and, if it produced an embedding and a
        classifier is available, classify and file it. Returns the `current`
        status dict for this track (never None - callers treat an exception
        escaping this method, not a None return, as the failure signal)."""
        # The audio analysis runs OUTSIDE the lock. It takes ~4.3s per track;
        # holding the lock across it meant a 1000-track import serialised the
        # database for over an hour, so every UI click waited seconds while
        # the progress panel kept animating and looked fine. Only the write
        # and the reads that follow need to be serialised.
        result, emb = analyse_track(gm, row["path"])
        ok = not result.get("errors")

        with LOCK:
            store_result(con, row["id"], result, emb)
            trow = con.execute(
                "SELECT * FROM tracks WHERE id=?", (row["id"],)).fetchone()
            arow = con.execute(
                "SELECT energy, vocal FROM analysis WHERE track_id=?",
                (row["id"],)).fetchone()
            vec = load_embedding(con, row["id"]) if ok else None

        energy = json.loads(arow["energy"]) if arow and arow["energy"] else []
        current = {
            "filename": trow["filename"],
            "bpm": trow["bpm"],
            "camelot": trow["camelot"],
            "energy": energy,
            "similarities": [],
            "crate": None,
            "band": "unknown",
            "margin": 0.0,
        }

        # A track with no embedding (analysis failed, or no genre model
        # installed) is recorded as analysed and skipped for classification -
        # never guessed at. Same if no crates exist yet to compare against.
        if vec is None or classifier is None or not classifier.crate_names():
            return current

        # Acapellas are routed out before the genre question is asked - see
        # classifier.acapella_verdict for why the DSP verdict beats the model
        # here.
        vocal = json.loads(arow["vocal"]) if arow and arow["vocal"] else None
        aca = acapella_verdict(vocal, classifier.crate_names())
        if aca:
            result = {"crate": aca, "band": "confident", "similarity": 1.0,
                      "margin": 1.0, "p": 1.0,
                      "scores": [{"crate": aca, "p": 1.0}],
                      "reason": "acapella"}
        else:
            result = classifier.classify(vec)

        # The panel shows the numbers the decision was actually made on, taken
        # straight from classify(). Recomputing a separate score here is how
        # the panel ended up displaying cosine similarities beside a verdict
        # reached some other way.
        current["similarities"] = [{"crate": s["crate"], "similarity": s["p"]}
                                   for s in result["scores"]]
        current["crate"] = result["crate"]
        current["band"] = result["band"]
        current["margin"] = result["margin"]

        with LOCK:
            auto_assign(con, row["id"], result)

        return current
