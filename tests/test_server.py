import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import pytest
from http.server import ThreadingHTTPServer
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
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
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
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
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


# ----------------------------------------------------------- audio preview

@pytest.fixture
def audio_base(tmp_path):
    """A real, decodable WAV on disk, plus a row whose file is gone."""
    import wave

    import numpy as np
    f = tmp_path / "song.wav"
    sr = 44100
    t = np.arange(sr * 5) / sr                      # 5 seconds of a 440Hz tone
    pcm = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    con = connect(tmp_path / "a.db")
    con.execute("INSERT INTO tracks (id, path, filename) VALUES (1, ?, 'song.wav')",
                (str(f),))
    con.execute("INSERT INTO tracks (id, path, filename) VALUES "
                "(2, '/nope/gone.wav', 'gone.wav')")
    con.commit()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _wav_seconds(body):
    import io
    import wave
    with wave.open(io.BytesIO(body), "rb") as w:
        return w.getnframes() / w.getframerate()


def test_preview_returns_a_playable_wav(audio_base):
    """WAV specifically: browsers cannot decode AIFF, which is most of this
    library, so the preview must not hand back the source format."""
    with urllib.request.urlopen(audio_base + "/api/preview/1?at=0&len=2") as r:
        assert r.headers["Content-Type"] == "audio/wav"
        assert _wav_seconds(r.read()) == pytest.approx(2.0, abs=0.05)


def test_preview_starts_where_it_was_asked_to(audio_base):
    with urllib.request.urlopen(audio_base + "/api/preview/1?at=4&len=5") as r:
        # Only one second of track remains after 4s; it returns what exists.
        assert _wav_seconds(r.read()) == pytest.approx(1.0, abs=0.05)


def test_preview_length_is_capped(audio_base):
    """An unbounded len is a request to buffer the whole library into RAM."""
    with urllib.request.urlopen(audio_base + "/api/preview/1?at=0&len=99999") as r:
        assert _wav_seconds(r.read()) <= 45.0


def test_a_position_past_the_end_is_a_clean_error(audio_base):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(audio_base + "/api/preview/1?at=600&len=5")
    assert e.value.code == 400


def test_preview_is_addressed_by_id_never_by_path(audio_base):
    """The browser can name a row the analyser created, not a file to read."""
    for bad in ["/etc/passwd", "../../../../etc/passwd", "abc"]:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(
                audio_base + "/api/preview/" + urllib.parse.quote(bad, safe=""))
        assert e.value.code in (400, 404)


def test_a_track_whose_file_vanished_is_404_not_500(audio_base):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(audio_base + "/api/preview/2")
    assert e.value.code == 404


def test_a_bad_number_is_rejected_cleanly(audio_base):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(audio_base + "/api/preview/1?at=banana")
    assert e.value.code == 400


def test_a_long_request_does_not_block_other_requests(audio_base):
    """Decoding and previewing must not freeze the UI. On the single-threaded
    HTTPServer every poll queued behind the work."""
    import time
    threads, slow = [], []

    def hit():
        t0 = time.time()
        urllib.request.urlopen(audio_base + "/api/preview/1?at=0&len=45").read()
        slow.append(time.time() - t0)

    for _ in range(3):
        th = threading.Thread(target=hit); th.start(); threads.append(th)
    t0 = time.time()
    get(audio_base + "/api/state")
    assert time.time() - t0 < 2.0
    for th in threads:
        th.join()


# --------------------------------------------------------- full playback

def test_audio_serves_the_whole_track_seekably(audio_base):
    """Any song, any time - so it must be a full, seekable response."""
    with urllib.request.urlopen(audio_base + "/api/audio/1") as r:
        assert r.status == 200
        assert r.headers["Accept-Ranges"] == "bytes"
        assert r.headers["Content-Type"] == "audio/wav"
        assert len(r.read()) == int(r.headers["Content-Length"])


def test_audio_honours_a_range_request(audio_base):
    """Without 206 the browser refuses to seek and refetches the whole file."""
    req = urllib.request.Request(audio_base + "/api/audio/1",
                                 headers={"Range": "bytes=100-199"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 206
        assert r.headers["Content-Length"] == "100"
        assert "/" in r.headers["Content-Range"]
        assert len(r.read()) == 100


def test_a_suffix_range_returns_the_tail(audio_base):
    req = urllib.request.Request(audio_base + "/api/audio/1",
                                 headers={"Range": "bytes=-50"})
    with urllib.request.urlopen(req) as r:
        assert len(r.read()) == 50


def test_a_range_past_the_end_is_416(audio_base):
    req = urllib.request.Request(audio_base + "/api/audio/1",
                                 headers={"Range": "bytes=99999999-"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 416


def test_audio_is_addressed_by_id_never_by_path(audio_base):
    for bad in ["/etc/passwd", "../../../../etc/passwd", "abc"]:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(
                audio_base + "/api/audio/" + urllib.parse.quote(bad, safe=""))
        assert e.value.code in (400, 404)


def test_audio_404s_when_the_file_vanished(audio_base):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(audio_base + "/api/audio/2")
    assert e.value.code == 404


# ------------------------------------------------------------ track detail

def test_track_detail_carries_what_the_panel_draws(audio_base):
    t = get(audio_base + "/api/track/1")
    for k in ("id", "filename", "bpm", "camelot", "duration", "energy",
              "moments", "similarities", "crate", "band", "analysed"):
        assert k in t, f"track detail is missing {k}"


def test_track_detail_says_when_a_track_was_never_analysed(audio_base):
    """The panel must not imply an empty curve is a flat track."""
    t = get(audio_base + "/api/track/1")
    assert t["analysed"] is False
    assert t["energy"] == [] and t["moments"] == []


def test_track_detail_404s_for_an_unknown_track(audio_base):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(audio_base + "/api/track/9999")
    assert e.value.code == 404


def test_track_detail_matches_the_shape_the_runner_reports(sets_base):
    """The panel has one renderer for the live run and for a clicked track,
    so the two payloads must agree on field names."""
    t = get(sets_base + "/api/track/1")
    assert set(("filename", "bpm", "camelot", "energy", "similarities",
                "crate", "band")) <= set(t)
    assert t["analysed"] is True
    assert t["moments"] and t["energy"]


def test_track_detail_reports_every_crate_it_is_in(audio_base):
    """A track filed by hand in one crate and by the model in another is the
    interesting case. Returning one arbitrary row hid it - and pinned the
    model's margin to whichever crate happened to come back first."""
    t = get(audio_base + "/api/track/1")
    assert "crates" in t and isinstance(t["crates"], list)


def test_crates_puts_the_djs_own_call_first(tmp_path):
    """When the DJ and the model disagree, the DJ's call leads."""
    con = connect(tmp_path / "c.db")
    con.execute("INSERT INTO tracks (id, path, filename, analysed_at) "
                "VALUES (1, '/x.wav', 'x.wav', 'now')")
    for name, source in (("house", "auto"), ("tech", "human")):
        con.execute("INSERT OR IGNORE INTO crates (name) VALUES (?)", (name,))
        cid = con.execute("SELECT id FROM crates WHERE name=?",
                          (name,)).fetchone()["id"]
        con.execute("INSERT INTO assignments (track_id, crate_id, source, band) "
                    "VALUES (1, ?, ?, 'confident')", (cid, source))
    con.commit()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t = get(f"http://127.0.0.1:{srv.server_port}/api/track/1")
        assert [c["name"] for c in t["crates"]][0] == "tech"
        assert len(t["crates"]) == 2
    finally:
        srv.shutdown()


# ------------------------------------------------------ retraining is async

def test_a_correction_does_not_wait_for_the_model_to_retrain(base):
    """Fitting the model takes about a second and holds the database lock.
    Doing it inside the request made every correction stall the whole UI,
    which is unusable when working through a review queue."""
    import time
    t0 = time.time()
    post(base + "/api/correct",
         {"track_id": 1, "to_crate": "Dubstep", "mode": "move"})
    assert time.time() - t0 < 0.5


def test_the_correction_itself_is_applied_immediately(base):
    """Deferring the retrain must not defer the filing."""
    post(base + "/api/correct",
         {"track_id": 1, "to_crate": "Dubstep", "mode": "move"})
    assert get(base + "/api/crate/Dubstep")["tracks"][0]["id"] == 1


def test_a_burst_of_corrections_retrains_once_at_the_end(tmp_path):
    """Each correction pushes the retrain out, so a burst costs one fit."""
    from crateapp.server import _Retrainer
    calls = []
    r = _Retrainer(None, tmp_path / "m.json", delay=0.15)
    r._run = lambda: calls.append(1)
    for _ in range(5):
        r.schedule()
    assert calls == [], "must not fit while corrections are still arriving"
    import time
    time.sleep(0.4)
    assert calls == [1], "exactly one fit after the burst settles"


def test_a_failed_retrain_does_not_take_the_server_down(tmp_path):
    from crateapp.server import _Retrainer
    r = _Retrainer(None, tmp_path / "nope" / "m.json", delay=0.01)
    r.flush()          # would raise if the failure escaped


# ---------------------------------------------------------- confirming

def _uncertain_base(tmp_path):
    from crateapp.crates import auto_assign
    con = connect(tmp_path / "u.db")
    con.execute("INSERT INTO tracks (id, path, filename, analysed_at, bpm) "
                "VALUES (1,'/a.wav','a.wav','now',128.0)")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "uncertain",
                         "similarity": 0.61})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return con, srv, f"http://127.0.0.1:{srv.server_port}"


def test_confirm_over_http_needs_no_destination(tmp_path):
    """The answer is wherever the track already is, so requiring to_crate
    would make the one-click case a two-step one."""
    con, srv, base = _uncertain_base(tmp_path)
    try:
        assert len(get(base + "/api/review")["uncertain"]) == 1
        post(base + "/api/correct", {"track_id": 1, "mode": "confirm"})
        assert get(base + "/api/review")["uncertain"] == []
        assert get(base + "/api/crate/house")["tracks"][0]["id"] == 1
    finally:
        srv.shutdown()


def test_a_move_still_requires_a_destination(tmp_path):
    con, srv, base = _uncertain_base(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(base + "/api/correct", {"track_id": 1, "mode": "move"})
        assert e.value.code == 400
    finally:
        srv.shutdown()


def test_confirming_an_unfiled_track_is_a_clean_error(tmp_path):
    con = connect(tmp_path / "n.db")
    con.execute("INSERT INTO tracks (id, path, filename, analysed_at) "
                "VALUES (1,'/a.wav','a.wav','now')")
    con.commit()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(f"http://127.0.0.1:{srv.server_port}/api/correct",
                 {"track_id": 1, "mode": "confirm"})
        assert e.value.code == 400
    finally:
        srv.shutdown()


def test_stop_asks_the_runner_to_stop(tmp_path):
    """A library run is hours long. Without this the only way to call one off
    was killing the server."""
    class FakeRunner:
        def __init__(self): self.stopped = False
        def stop(self): self.stopped = True
        def status(self): return {"running": True}
    con = connect(tmp_path / "s.db")
    r = FakeRunner()
    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              make_app(con, tmp_path / "m.json", runner=r))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        assert post(base + "/api/stop", {})["stopping"] is True
        assert r.stopped
    finally:
        srv.shutdown()


def test_stop_without_a_runner_is_a_clean_503(tmp_path):
    con = connect(tmp_path / "s.db")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(f"http://127.0.0.1:{srv.server_port}/api/stop", {})
        assert e.value.code == 503
    finally:
        srv.shutdown()


def test_a_pending_retrain_is_flushed_on_shutdown(tmp_path):
    """A correction made seconds before quitting leaves a retrain on a timer.
    Without a flush it is simply dropped and the model silently stays behind
    until some later correction happens to fire one."""
    from crateapp.server import _Retrainer
    calls = []
    r = _Retrainer(None, tmp_path / "m.json", delay=999)
    r._run = lambda: calls.append(1)
    r.schedule()
    assert calls == []
    r.flush()
    assert calls == [1]


def test_flush_is_reachable_from_the_server_object(tmp_path):
    """serve() reaches the retrainer through the handler class on shutdown;
    if that attribute ever moves, the flush silently stops happening."""
    con = connect(tmp_path / "s.db")
    Handler = make_app(con, tmp_path / "m.json")
    assert hasattr(Handler, "retrainer")
    assert callable(Handler.retrainer.flush)


def test_remove_over_http(tmp_path):
    con = connect(tmp_path / "r.db")
    con.execute("INSERT INTO tracks (id, path, filename, analysed_at) "
                "VALUES (1,'/a.wav','a.wav','now')")
    con.commit()
    from crateapp.crates import correct as _c
    _c(con, 1, "house", mode="move")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        assert post(base + "/api/remove", {"track_id": 1})["removed"] == "a.wav"
        assert get(base + "/api/crate/house")["tracks"] == []
        with pytest.raises(urllib.error.HTTPError) as e:
            post(base + "/api/remove", {"track_id": 1})
        assert e.value.code == 404
    finally:
        srv.shutdown()


def test_folders_added_before_a_reload_stay_in_every_run(tmp_path):
    """The page only knows folders added since it opened. Starting a run used
    to REPLACE the saved list with that, so a folder added before a reload
    silently dropped out of all future runs."""
    con = connect(tmp_path / "l.db")
    seen = {}

    class FakeRunner:
        def start(self, folders):
            seen["folders"] = list(folders); return True
        def status(self): return {"running": False}

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json", runner=FakeRunner()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    a = tmp_path / "A"; a.mkdir(); b = tmp_path / "B"; b.mkdir()
    try:
        post(base + "/api/scan", {"folder": str(a)})      # added, then the page reloads
        post(base + "/api/start", {"folders": [str(b)]})  # the new page only knows B
        assert seen["folders"] == [str(a), str(b)]
        post(base + "/api/start", {})                     # a later run with no page state
        assert seen["folders"] == [str(a), str(b)]
        post(base + "/api/start", {"folders": [str(tmp_path / "typo")]})
        assert seen["folders"] == [str(a), str(b)], "a path that is not a folder is not saved"
    finally:
        srv.shutdown()


def test_folder_export_does_not_hold_the_database_lock_while_copying(tmp_path, monkeypatch):
    from crateapp import exporters
    from crateapp.db import LOCK
    music = tmp_path / "music"; music.mkdir()
    f = music / "a.wav"; f.write_bytes(b"x")
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, filename) VALUES (1, ?, 'a.wav')", (str(f),))
    con.commit()
    correct(con, 1, "House", mode="move")
    free = []

    def fake_copy(src, tmp):
        got = []
        t = threading.Thread(target=lambda: got.append(LOCK.acquire(timeout=1) and (LOCK.release() or True)))
        t.start(); t.join()
        free.append(bool(got and got[0]))
        import shutil; shutil.copyfile(src, tmp)

    monkeypatch.setattr(exporters, "_copy", fake_copy)
    assert exporters.export_folders(con, tmp_path / "out", lock=LOCK) == {"House": 1}
    assert free == [True]
