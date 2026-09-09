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
        # Two distinct tracks, not the same vector twice: identical embeddings
        # mean the same recording, and training counts those once.
        seed(con, tid, a + np.random.default_rng(tid).normal(0, .05, 8).astype("float32"))
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


# ------------------------------------------------- what caused the drift

def test_training_ignores_the_models_own_guesses(tmp_path):
    """The bug that ruined DUBSTEP.

    rebuild_centroids used to select every assignment regardless of source, so
    a track the model filed became a track the model learned from. On the real
    library DUBSTEP reached 40 reference tracks of which 2 were the DJ's, and
    filled with music that was not dubstep. Only the DJ's own filing counts.
    """
    from crateapp.classifier import training_set
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    for tid in (1, 2, 3):
        seed(con, tid, a)
    correct(con, 1, "DUBSTEP", mode="move")           # the DJ's call
    con.execute("INSERT OR IGNORE INTO crates (name) VALUES ('DUBSTEP')")
    cid = con.execute("SELECT id FROM crates WHERE name='DUBSTEP'").fetchone()["id"]
    for tid in (2, 3):                                 # the model's guesses
        con.execute("INSERT INTO assignments (track_id, crate_id, source, band) "
                    "VALUES (?,?,'auto','confident')", (tid, cid))
    con.commit()

    ids, labels = training_set(con)
    assert ids == [1], "auto assignments must never become training data"
    assert labels == ["DUBSTEP"]


def test_a_track_filed_in_two_crates_trains_nothing(tmp_path):
    """It is not a clean label for a single-label model."""
    from crateapp.classifier import training_set
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    seed(con, 1, a)
    correct(con, 1, "house", mode="move")
    correct(con, 1, "tech", mode="add", was_error=False)
    assert training_set(con)[0] == []


def test_training_reports_crates_too_thin_to_learn(tmp_path):
    """A crate with two examples cannot define a decision boundary. Say so
    rather than pretending to have learned it."""
    from crateapp.classifier import train
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    for tid in (1, 2):
        seed(con, tid, a)
        correct(con, tid, "Amapiano", mode="move")
    out = train(con, tmp_path / "m.json", min_per_class=5)
    assert "Amapiano" in out["too_thin"]
    assert out["trained"] is False


def test_a_thin_crate_still_gets_a_centroid(tmp_path):
    """It cannot be predicted, but it must still count for 'is this like
    anything I own' - otherwise a new crate makes its own tracks look alien."""
    from crateapp.classifier import train
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    seed(con, 1, a)
    correct(con, 1, "Amapiano", mode="move")
    train(con, tmp_path / "m.json", min_per_class=5)
    doc = json.loads((tmp_path / "m.json").read_text())
    assert "Amapiano" in doc["crates"]


def test_a_trained_model_predicts_from_the_learned_layer(tmp_path):
    from crateapp.classifier import train
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    tid = 0
    for vec, crate in ((a, "house"), (b, "DUBSTEP")):
        for _ in range(6):
            tid += 1
            noisy = vec + np.random.default_rng(tid).normal(0, 0.05, 8).astype("float32")
            seed(con, tid, noisy)
            correct(con, tid, crate, mode="move")
    p = tmp_path / "m.json"
    out = train(con, p, min_per_class=5)
    assert out["trained"] is True
    m = Classifier(p)
    assert m.trained()
    assert m.classify(a)["crate"] == "house"
    assert m.classify(b)["crate"] == "DUBSTEP"


def test_music_unlike_anything_owned_is_still_flagged_as_new(tmp_path):
    """A softmax always sums to 1, so the trained layer will happily pick a
    crate for music unlike anything in the library. The centroid distance is
    what catches that, and it must survive training."""
    from crateapp.classifier import train
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    tid = 0
    for vec, crate in ((a, "house"), (b, "DUBSTEP")):
        for _ in range(6):
            tid += 1
            noisy = vec + np.random.default_rng(tid).normal(0, 0.05, 8).astype("float32")
            seed(con, tid, noisy)
            correct(con, tid, crate, mode="move")
    p = tmp_path / "m.json"
    train(con, p, min_per_class=5)
    alien = np.zeros(8, dtype=np.float32); alien[7] = 1.0
    out = Classifier(p).classify(alien)
    assert out["band"] == "unknown" and out["crate"] is None


def test_two_crates_do_not_all_collapse_into_one(tmp_path):
    """With exactly two classes scikit-learn fits binary logistic regression
    and returns a single coefficient row, not one per class. Taking a softmax
    over that one logit files every track into the alphabetically-first crate.
    A two-crate library is the normal starting point, so this must hold."""
    from crateapp.classifier import train
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    tid = 0
    for vec, crate in ((a, "zzz_last"), (b, "aaa_first")):
        for _ in range(6):
            tid += 1
            noisy = vec + np.random.default_rng(tid).normal(0, 0.05, 8).astype("float32")
            seed(con, tid, noisy)
            correct(con, tid, crate, mode="move")
    p = tmp_path / "m.json"
    train(con, p, min_per_class=5)
    m = Classifier(p)
    assert m.classify(a)["crate"] == "zzz_last"
    assert m.classify(b)["crate"] == "aaa_first"


def test_duplicate_copies_do_not_reweight_training(tmp_path):
    """A DJ library holds the same recording many times. Counting each copy
    would let one crate's duplicates pull the decision boundary toward it -
    on the real library that turned tech's 286 examples into 539."""
    from crateapp.classifier import train
    con = connect(tmp_path / "l.db")
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    tid = 0
    for _ in range(5):                    # five distinct 'house' tracks
        tid += 1
        seed(con, tid, a + np.random.default_rng(tid).normal(0, .05, 8).astype("float32"))
        correct(con, tid, "house", mode="move")
    for _ in range(5):                    # five distinct 'tech' tracks
        tid += 1
        seed(con, tid, b + np.random.default_rng(tid).normal(0, .05, 8).astype("float32"))
        correct(con, tid, "tech", mode="move")
    for _ in range(20):                   # twenty identical copies of one tech track
        tid += 1
        seed(con, tid, b)
        correct(con, tid, "tech", mode="move")

    out = train(con, tmp_path / "m.json", min_per_class=3)
    # 5 house + 5 tech + 1 surviving representative of the 20 copies
    assert out["counts"]["tech"] == 6
    assert out["counts"]["house"] == 5
