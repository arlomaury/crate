import json
import numpy as np
import analyze
from crateapp.db import connect
from crateapp.worker import store_result, load_embedding, analyse_one


def a_track(con, path="/music/a.wav"):
    cur = con.execute("INSERT INTO tracks (path, filename) VALUES (?,?)",
                      (path, "a.wav"))
    return cur.lastrowid


RESULT = {
    "bpm": 128.0, "camelot": "8A", "duration_sec": 210.5,
    "category": "House",
    "moments": [{"type": "drop", "bar": 32, "time": 60.0, "label": "Drop 1"}],
    "structure": {"energy_curve": [0.1, 0.9]},
    "vocal_analysis": {"is_acapella": False},
    "genre_predictions": [{"label": "House", "confidence": 0.4}],
}


def test_store_result_fills_the_track_row(tmp_path):
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    store_result(con, tid, RESULT, np.zeros(1280, dtype=np.float32))
    row = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    assert row["bpm"] == 128.0 and row["camelot"] == "8A"
    assert row["standard_genre"] == "House"
    assert row["analysed_at"] is not None


def test_store_result_keeps_moments_and_curves(tmp_path):
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    store_result(con, tid, RESULT, np.zeros(1280, dtype=np.float32))
    row = con.execute("SELECT * FROM analysis WHERE track_id=?", (tid,)).fetchone()
    assert json.loads(row["moments"])[0]["label"] == "Drop 1"


def test_embedding_round_trips_exactly(tmp_path):
    """Centroids are recomputed from these, so a lossy round trip is a bug."""
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    vec = np.random.rand(1280).astype(np.float32)
    store_result(con, tid, RESULT, vec)
    assert np.array_equal(load_embedding(con, tid), vec)


def test_load_embedding_returns_none_when_absent(tmp_path):
    con = connect(tmp_path / "l.db")
    assert load_embedding(con, a_track(con)) is None


def test_storing_twice_overwrites_rather_than_duplicating(tmp_path):
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    store_result(con, tid, RESULT, np.zeros(1280, dtype=np.float32))
    store_result(con, tid, RESULT, np.ones(1280, dtype=np.float32))
    assert con.execute("SELECT count(*) c FROM embeddings").fetchone()["c"] == 1
    assert load_embedding(con, tid)[0] == 1.0


def test_a_failed_track_records_its_error_and_is_not_retried(tmp_path):
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    store_result(con, tid, {"errors": ["could not decode"]}, None)
    row = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    assert row["error"] == "could not decode"
    assert row["analysed_at"] is not None      # so pending() will not return it


def test_load_embedding_returns_a_writeable_array(tmp_path):
    """The classifier normalises vectors in place; a read-only view (what
    np.frombuffer gives by default) would blow up on that."""
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    store_result(con, tid, RESULT, np.ones(1280, dtype=np.float32))
    vec = load_embedding(con, tid)
    assert vec.flags.writeable
    vec[0] = 42.0  # would raise ValueError if read-only


def test_analyse_one_stores_result_and_embedding_on_success(tmp_path, monkeypatch):
    def fake_analyse(path, gm, verbose=False, emb_dir=None):
        if emb_dir is not None:
            np.save(analyze.embedding_path(emb_dir, path),
                    np.full(1280, 7.0, dtype=np.float32))
        return dict(RESULT)

    monkeypatch.setattr(analyze, "analyse", fake_analyse)
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    row = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()

    ok = analyse_one(con, gm=object(), row=row)

    assert ok is True
    track = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    assert track["bpm"] == 128.0 and track["analysed_at"] is not None
    emb = load_embedding(con, tid)
    assert emb is not None and emb[0] == 7.0


def test_analyse_one_marks_analysed_and_returns_false_when_analyse_raises(
        tmp_path, monkeypatch):
    def fake_analyse(path, gm, verbose=False, emb_dir=None):
        raise RuntimeError("could not decode")

    monkeypatch.setattr(analyze, "analyse", fake_analyse)
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    row = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()

    ok = analyse_one(con, gm=object(), row=row)

    assert ok is False
    track = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    assert track["analysed_at"] is not None    # so pending() won't retry it forever
    assert track["error"] == "could not decode"


def test_analyse_one_returns_false_but_still_records_a_result_level_error(
        tmp_path, monkeypatch):
    def fake_analyse(path, gm, verbose=False, emb_dir=None):
        return {"errors": ["file shorter than 5s, skipped"]}

    monkeypatch.setattr(analyze, "analyse", fake_analyse)
    con = connect(tmp_path / "l.db")
    tid = a_track(con)
    row = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()

    ok = analyse_one(con, gm=object(), row=row)

    assert ok is False
    track = con.execute("SELECT * FROM tracks WHERE id=?", (tid,)).fetchone()
    assert track["analysed_at"] is not None
    assert track["error"] == "file shorter than 5s, skipped"
