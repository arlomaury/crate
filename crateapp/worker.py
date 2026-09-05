"""Runs analyze.py over pending tracks and caches everything it produces."""
import json
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
    return None if row is None else np.frombuffer(row["vector"], dtype=np.float32)


def analyse_one(con, gm, row):
    """Analyse a single track row. Returns True if it produced usable output."""
    try:
        result = analyze.analyse(row["path"], gm, verbose=False)
    except Exception as e:
        store_result(con, row["id"], {"errors": [str(e)]}, None)
        return False
    emb = None
    try:
        a16 = analyze.es.MonoLoader(filename=row["path"], sampleRate=16000,
                                    resampleQuality=4)()
        emb = gm.embed(a16).mean(axis=0) if gm and gm.ok else None
    except Exception:
        pass
    store_result(con, row["id"], result, emb)
    return not result.get("errors")
