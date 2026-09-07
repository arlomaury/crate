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
