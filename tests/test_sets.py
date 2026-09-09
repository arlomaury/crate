import numpy as np
import pytest

from crateapp.db import connect
from crateapp.sets import (build_set, dedupe, energy_level, load_pool,
                           neighbours)


@pytest.fixture
def con(tmp_path):
    return connect(tmp_path / "t.db")


def _add(con, name, bpm, camelot, vec, vocal=0.3, energy=None, moments=None,
         crate="TEST"):
    """Insert a fully analysed track the way the runner would."""
    import json
    cur = con.execute(
        "INSERT INTO tracks (path, filename, analysed_at, bpm, camelot, "
        "duration_sec) VALUES (?,?,datetime('now'),?,?,300.0)",
        (f"/music/{name}.wav", f"{name}.wav", bpm, camelot))
    tid = cur.lastrowid
    con.execute(
        "INSERT INTO analysis (track_id, moments, energy, vocal) VALUES (?,?,?,?)",
        (tid, json.dumps(moments or []), json.dumps(energy or [0.5] * 8),
         json.dumps({"vocal_ratio": vocal})))
    con.execute("INSERT INTO embeddings (track_id, vector) VALUES (?,?)",
                (tid, np.asarray(vec, dtype="float32").tobytes()))
    con.execute("INSERT OR IGNORE INTO crates (name) VALUES (?)", (crate,))
    cid = con.execute("SELECT id FROM crates WHERE name=?", (crate,)).fetchone()["id"]
    con.execute("INSERT INTO assignments (track_id, crate_id, source, band) "
                "VALUES (?,?,'auto','confident')", (tid, cid))
    con.commit()
    return tid


# ------------------------------------------------------------------- pool

def test_pool_carries_everything_the_scorer_needs(con):
    _add(con, "a", 128.0, "8A", [1, 0, 0, 0])
    pool = load_pool(con)
    assert len(pool) == 1
    t = pool[0]
    for field in ("id", "filename", "bpm", "camelot", "vec", "vocal_ratio",
                  "energy", "moments", "duration"):
        assert field in t, f"pool is missing {field}"
    assert t["vec"] is not None


def test_pool_skips_unanalysed_tracks(con):
    """A track with no tempo or key cannot be placed in a set."""
    _add(con, "good", 128.0, "8A", [1, 0, 0, 0])
    con.execute("INSERT INTO tracks (path, filename) VALUES ('/m/x.wav','x.wav')")
    con.commit()
    assert [t["filename"] for t in load_pool(con)] == ["good.wav"]


def test_pool_can_be_limited_to_one_crate(con):
    _add(con, "a", 128.0, "8A", [1, 0, 0, 0], crate="UKG")
    _add(con, "b", 128.0, "8A", [1, 0, 0, 0], crate="DUBSTEP")
    assert [t["filename"] for t in load_pool(con, crate="UKG")] == ["a.wav"]


def test_energy_level_summarises_a_curve():
    assert energy_level([0.2, 0.2, 0.2]) == pytest.approx(0.2)
    assert energy_level([]) is None


# ------------------------------------------------------------- neighbours

def test_neighbours_rank_the_best_mix_first(con):
    a = _add(con, "seed", 128.0, "8A", [1, 0, 0, 0])
    _add(con, "clash", 128.0, "3B", [0, 1, 0, 0])
    _add(con, "match", 128.0, "8A", [1, 0, 0, 0])
    ranked = neighbours(load_pool(con), a)
    assert ranked[0]["track"]["filename"] == "match.wav"
    assert ranked[0]["score"] > ranked[-1]["score"]


def test_neighbours_never_return_the_seed_itself(con):
    a = _add(con, "seed", 128.0, "8A", [1, 0, 0, 0])
    _add(con, "other", 128.0, "8A", [1, 0, 0, 0])
    assert all(n["track"]["id"] != a for n in neighbours(load_pool(con), a))


def test_every_neighbour_carries_mix_points(con):
    a = _add(con, "seed", 128.0, "8A", [1, 0, 0, 0],
             moments=[{"type": "breakdown", "time": 200.0}])
    _add(con, "next", 128.0, "8A", [1, 0, 0, 0],
         moments=[{"type": "drop", "time": 48.0}])
    n = neighbours(load_pool(con), a)[0]
    assert n["mix"]["out_at"] == 200.0
    assert n["mix"]["in_at"] is not None


# ---------------------------------------------------------------- the set

def test_build_set_returns_the_requested_length(con):
    ids = [_add(con, f"t{i}", 128.0, "8A", [1, 0, 0, 0]) for i in range(6)]
    s = build_set(load_pool(con), ids[0], length=4)
    assert len(s["tracks"]) == 4


def test_build_set_never_repeats_a_track(con):
    ids = [_add(con, f"t{i}", 128.0, "8A", [1, 0, 0, 0]) for i in range(5)]
    s = build_set(load_pool(con), ids[0], length=5)
    seen = [t["id"] for t in s["tracks"]]
    assert len(set(seen)) == len(seen)


def test_build_set_stops_short_rather_than_padding(con):
    """Three tracks cannot make a ten-track set. Returning a short set is
    honest; repeating tracks to hit the number is not."""
    ids = [_add(con, f"t{i}", 128.0, "8A", [1, 0, 0, 0]) for i in range(3)]
    s = build_set(load_pool(con), ids[0], length=10)
    assert len(s["tracks"]) == 3


def test_build_set_starts_with_the_seed(con):
    ids = [_add(con, f"t{i}", 128.0, "8A", [1, 0, 0, 0]) for i in range(4)]
    s = build_set(load_pool(con), ids[1], length=3)
    assert s["tracks"][0]["id"] == ids[1]


def test_the_first_track_has_no_transition_into_it(con):
    ids = [_add(con, f"t{i}", 128.0, "8A", [1, 0, 0, 0]) for i in range(3)]
    s = build_set(load_pool(con), ids[0], length=3)
    assert s["tracks"][0]["from_previous"] is None
    assert s["tracks"][1]["from_previous"] is not None


def test_every_transition_is_explained(con):
    ids = [_add(con, f"t{i}", 128.0, "8A", [1, 0, 0, 0]) for i in range(3)]
    s = build_set(load_pool(con), ids[0], length=3)
    for t in s["tracks"][1:]:
        assert t["from_previous"]["reasons"]


def test_a_build_arc_climbs(con):
    """Asked to build, the set should not open on its biggest track."""
    low = _add(con, "low", 128.0, "8A", [1, 0, 0, 0], energy=[0.2] * 8)
    _add(con, "mid", 128.0, "8A", [1, 0, 0, 0], energy=[0.5] * 8)
    _add(con, "high", 128.0, "8A", [1, 0, 0, 0], energy=[0.9] * 8)
    s = build_set(load_pool(con), low, length=3, arc="build")
    levels = [t["energy"] for t in s["tracks"]]
    assert levels[-1] > levels[0]


def test_an_unknown_seed_is_an_error(con):
    _add(con, "a", 128.0, "8A", [1, 0, 0, 0])
    with pytest.raises(ValueError):
        build_set(load_pool(con), 9999, length=3)


# --------------------------------------------------------------- duplicates

def _dupe_pool():
    """Two copies of one record, plus a track that merely sounds similar.

    Measured on the real library: byte-identical copies score cosine 1.0000,
    the same song from a different rip scores 0.9958, and genuinely distinct
    tracks top out at 0.9785.

    Distinct records differ along different axes, so `other` deviates on its
    own dimension rather than further along the same one.
    """
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    near = np.array([1.0, 0.09, 0.0, 0.0], dtype="float32")     # cos ~0.996
    other = np.array([1.0, 0.0, 0.21, 0.0], dtype="float32")    # cos ~0.978
    mk = lambda i, vec, path, dur: {
        "id": i, "filename": path.split("/")[-1], "path": path, "vec": vec,
        "duration": dur, "bpm": 128.0, "camelot": "8A", "energy": 0.5,
        "vocal_ratio": 0.3, "moments": []}
    return [mk(1, v, "/Contents/artist/album/Song.aiff", 300.0),
            mk(2, near, "/AllSongs/Song.aiff", 300.4),
            mk(3, other, "/AllSongs/Different.aiff", 300.0)]


def test_dedupe_collapses_the_same_record():
    kept = dedupe(_dupe_pool())
    assert len(kept) == 2


def test_dedupe_keeps_tracks_that_merely_sound_alike():
    """0.978 is two different records in the same style. Merging them would
    quietly delete tracks from the DJ's library."""
    kept = dedupe(_dupe_pool())
    assert "Different.aiff" in [t["filename"] for t in kept]


def test_dedupe_prefers_the_shallower_copy():
    """Of two copies, keep the one in the flat working folder rather than the
    one buried in an artist/album tree."""
    kept = dedupe(_dupe_pool())
    song = [t for t in kept if t["filename"] == "Song.aiff"][0]
    assert song["path"] == "/AllSongs/Song.aiff"


def test_dedupe_records_what_it_merged():
    kept = dedupe(_dupe_pool())
    song = [t for t in kept if t["filename"] == "Song.aiff"][0]
    assert song["duplicates"] == 1


def test_dedupe_leaves_a_clean_pool_alone():
    pool = _dupe_pool()[1:]
    assert len(dedupe(pool)) == 2


def test_a_set_never_offers_a_track_its_own_duplicate(con):
    """Ranking a track's own copy as the best next track is the single most
    useless answer the tool could give."""
    a = _add(con, "x", 128.0, "8A", [1, 0, 0, 0])
    _add(con, "x-copy", 128.0, "8A", [1, 0, 0, 0])
    _add(con, "real", 129.0, "8A", [1, 0.3, 0, 0])
    ranked = neighbours(dedupe(load_pool(con)), a)
    assert [n["track"]["filename"] for n in ranked] == ["real.wav"]


def test_the_set_reports_its_own_weakest_link(con):
    """A DJ needs to know which transition to practise."""
    ids = [_add(con, f"t{i}", 128.0, "8A", [1, 0, 0, 0]) for i in range(2)]
    ids.append(_add(con, "odd", 128.0, "3B", [0, 1, 0, 0]))
    s = build_set(load_pool(con), ids[0], length=3)
    assert "weakest" in s and s["weakest"]["position"] >= 1
