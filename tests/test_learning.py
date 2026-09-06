import json
import numpy as np
from crateapp.db import connect
from crateapp.worker import store_result
from crateapp.crates import correct
from crateapp.classifier import Classifier, rebuild_centroids


def seed(con, tid, vec):
    con.execute("INSERT INTO tracks (id, path) VALUES (?,?)", (tid, f"/{tid}.wav"))
    store_result(con, tid, {"bpm": 128.0}, vec)


def test_rebuild_creates_a_centroid_per_crate(tmp_path):
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    for tid in (1, 2):
        seed(con, tid, a)
        correct(con, tid, "Amapiano", mode="move")
    counts = rebuild_centroids(con, tmp_path / "m.json")
    assert counts == {"Amapiano": 2}
    assert "Amapiano" in json.loads((tmp_path / "m.json").read_text())["crates"]


def test_a_new_crate_classifies_its_own_kind_afterwards(tmp_path):
    """The whole point of the learning loop: correcting teaches the model."""
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[3] = 1.0
    for tid in (1, 2, 3):
        seed(con, tid, a)
        correct(con, tid, "Amapiano", mode="move")
    p = tmp_path / "m.json"
    rebuild_centroids(con, p)
    assert Classifier(p).classify(a)["crate"] == "Amapiano"


def test_centroids_ignore_tracks_with_no_embedding(tmp_path):
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    seed(con, 1, a)
    con.execute("INSERT INTO tracks (id, path) VALUES (2, '/2.wav')")  # no embedding
    con.commit()
    for tid in (1, 2):
        correct(con, tid, "House", mode="move")
    assert rebuild_centroids(con, tmp_path / "m.json") == {"House": 1}


def test_rebuild_preserves_unrelated_top_level_keys(tmp_path):
    """crate_model.json also carries a top-level house_submodel key (measured
    real/minimal centroids, 89%/94% precision) plus provenance fields. A
    rebuild must never silently delete those - only the 'crates' key changes."""
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "crates": {"Stale": {"n": 1, "centroid": [0.0] * 8}},
        "house_submodel": {"real": {"centroid": [1.0] * 8}, "minimal": {"centroid": [0.0] * 8}},
        "built": "2026-01-01",
        "source": "rekordbox crates",
    }))
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    seed(con, 1, a)
    correct(con, 1, "Amapiano", mode="move")

    rebuild_centroids(con, p)

    doc = json.loads(p.read_text())
    assert "Stale" not in doc["crates"]
    assert "Amapiano" in doc["crates"]
    assert doc["house_submodel"] == {
        "real": {"centroid": [1.0] * 8}, "minimal": {"centroid": [0.0] * 8}}
    assert doc["built"] == "2026-01-01"
    assert doc["source"] == "rekordbox crates"
