"""Runs analyze.py over pending tracks and caches everything it produces."""
import json
import tempfile
from datetime import datetime, timezone

import numpy as np

import analyze


def store_result(con, track_id, result, embedding):
    """Persist one analysis. Marks the track analysed even when it failed, so a
    broken file is not retried on every scan."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    err = "; ".join(result.get("errors", [])) or None
    con.execute(
        "UPDATE tracks SET analysed_at=?, bpm=?, camelot=?, duration_sec=?, "
        "standard_genre=?, error=? WHERE id=?",
        (now, result.get("bpm"), result.get("camelot"), result.get("duration_sec"),
         result.get("category"), err, track_id))
    con.execute(
        "INSERT OR REPLACE INTO analysis (track_id, moments, energy, vocal, genre_preds) "
        "VALUES (?,?,?,?,?)",
        (track_id,
         json.dumps(result.get("moments", [])),
         json.dumps(result.get("structure", {}).get("energy_curve", [])),
         json.dumps(result.get("vocal_analysis", {})),
         json.dumps(result.get("genre_predictions", []))))
    if embedding is not None:
        con.execute("INSERT OR REPLACE INTO embeddings (track_id, vector) VALUES (?,?)",
                    (track_id, np.asarray(embedding, dtype=np.float32).tobytes()))
    con.commit()


def load_embedding(con, track_id):
    row = con.execute("SELECT vector FROM embeddings WHERE track_id=?",
                      (track_id,)).fetchone()
    if row is None:
        return None
    # .copy() so callers (e.g. the classifier normalising in place) don't hit
    # a read-only-array error: np.frombuffer over bytes is not writeable.
    return np.frombuffer(row["vector"], dtype=np.float32).copy()


def analyse_one(con, gm, row):
    """Analyse a single track row. Returns True if it produced usable output.

    Reads back the embedding analyze.analyse() already wrote to `emb_dir`
    instead of decoding the audio and running the model a second time -
    that forward pass is the single most expensive step in the pipeline.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = analyze.analyse(row["path"], gm, verbose=False, emb_dir=tmp)
        except Exception as e:
            store_result(con, row["id"], {"errors": [str(e)]}, None)
            return False
        emb = None
        f = analyze.embedding_path(tmp, row["path"])
        if f.exists():
            emb = np.load(f)
    store_result(con, row["id"], result, emb)
    return not result.get("errors")
