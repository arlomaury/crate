import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import pytest
from http.server import HTTPServer
from crateapp.db import connect
from crateapp.crates import correct
from crateapp.server import make_app


@pytest.fixture
def base(tmp_path):
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, filename, analysed_at, bpm) "
                "VALUES (1, '/a.wav', 'a.wav', 'now', 128.0)")
    con.commit()
    correct(con, 1, "House", mode="move")
    srv = HTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_state_reports_crates_and_counts(base):
    state = get(base + "/api/state")
    assert state["crates"][0]["name"] == "House"
    assert state["crates"][0]["count"] == 1


def test_crate_lists_its_tracks(base):
    assert get(base + "/api/crate/House")["tracks"][0]["filename"] == "a.wav"


def test_review_endpoint_exists(base):
    body = get(base + "/api/review")
    assert "uncertain" in body and "unsorted" in body


def test_correct_moves_a_track(base):
    post(base + "/api/correct",
         {"track_id": 1, "to_crate": "Dubstep", "mode": "move"})
    assert get(base + "/api/crate/Dubstep")["tracks"][0]["id"] == 1
    assert get(base + "/api/crate/House")["tracks"] == []


def test_correct_can_also_add(base):
    post(base + "/api/correct",
         {"track_id": 1, "to_crate": "Party", "mode": "add", "was_error": False})
    assert len(get(base + "/api/crate/House")["tracks"]) == 1
    assert len(get(base + "/api/crate/Party")["tracks"]) == 1


def test_the_ui_is_served_at_root(base):
    with urllib.request.urlopen(base + "/") as r:
        assert b"<title>" in r.read()


def test_crate_name_with_slash_round_trips(base):
    """Crate names may legitimately contain '/' (see crateapp.crates.ensure_crate).
    The API must accept a percent-encoded name (encodeURIComponent style,
    where '/' becomes %2F) and resolve it back to the literal crate name."""
    post(base + "/api/correct",
         {"track_id": 1, "to_crate": "Drum & Bass / Jungle", "mode": "move"})
    body = get(base + "/api/crate/" + urllib.parse.quote("Drum & Bass / Jungle", safe=""))
    assert body["tracks"][0]["id"] == 1


def test_correct_rejects_a_blank_crate_name_cleanly(base):
    req = urllib.request.Request(
        base + "/api/correct",
        data=json.dumps({"track_id": 1, "to_crate": "   ", "mode": "move"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req)
        assert False, "expected an HTTPError"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        body = json.loads(e.read())
        assert "error" in body


# ------------------------------------------------------------ set builder

@pytest.fixture
def sets_base(tmp_path):
    """A server with enough analysed, embedded tracks to build a real set.

    The embeddings are deliberately spread apart: anything closer than cosine
    0.99 is treated as the same recording and collapsed by dedupe(), so
    near-identical fixture vectors would silently leave a one-track pool.
    """
    import numpy as np
    con = connect(tmp_path / "s.db")
    for i, (bpm, cam, vec) in enumerate([
            (128.0, "8A", [1, 0, 0, 0]), (128.5, "8A", [1, 0.3, 0, 0]),
            (129.0, "9A", [1, 0, 0.3, 0]), (175.0, "3B", [0, 1, 0, 0])], start=1):
        con.execute(
            "INSERT INTO tracks (id, path, filename, analysed_at, bpm, camelot,"
            " duration_sec) VALUES (?,?,?,'now',?,?,300.0)",
            (i, f"/m/t{i}.wav", f"t{i}.wav", bpm, cam))
        con.execute("INSERT INTO analysis (track_id, moments, energy, vocal) "
                    "VALUES (?,?,?,?)",
                    (i, json.dumps([{"type": "drop", "bar": 16, "time": 40.0}]),
                     json.dumps([0.5] * 8), json.dumps({"vocal_ratio": 0.3})))
        con.execute("INSERT INTO embeddings (track_id, vector) VALUES (?,?)",
                    (i, np.asarray(vec, dtype="float32").tobytes()))
    con.commit()
    srv = HTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_set_pool_lists_seedable_tracks(sets_base):
    pool = get(sets_base + "/api/setpool")["tracks"]
    assert len(pool) == 4
    assert {"id", "filename", "bpm", "camelot"} <= set(pool[0])


def test_set_pool_never_ships_embeddings(sets_base):
    """A 1280-float vector per track has no business crossing into JSON."""
    assert "vec" not in get(sets_base + "/api/setpool")["tracks"][0]


def test_build_a_set_over_http(sets_base):
    s = post(sets_base + "/api/set", {"seed_id": 1, "length": 3})
    assert [t["id"] for t in s["tracks"]][0] == 1
    assert len(s["tracks"]) == 3
    assert s["tracks"][1]["from_previous"]["reasons"]


def test_the_set_endpoint_honours_the_arc(sets_base):
    s = post(sets_base + "/api/set", {"seed_id": 1, "length": 3, "arc": "build"})
    assert s["arc"] == "build"


def test_next_track_suggestions_over_http(sets_base):
    n = post(sets_base + "/api/next", {"track_id": 1, "limit": 2})["next"]
    assert len(n) == 2
    assert n[0]["score"] >= n[1]["score"]
    assert n[0]["mix"]["out_at"] is not None


def test_an_unmixable_track_ranks_last(sets_base):
    """The 175 BPM track cannot be beatmatched against the 128s."""
    n = post(sets_base + "/api/next", {"track_id": 1})["next"]
    assert n[-1]["track"]["filename"] == "t4.wav"


def test_a_bad_seed_is_a_clean_error_not_a_traceback(sets_base):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(sets_base + "/api/set", {"seed_id": 9999})
    assert e.value.code == 400


def test_the_set_endpoint_requires_a_seed(sets_base):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(sets_base + "/api/set", {})
    assert e.value.code == 400
