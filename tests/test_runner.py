import json
import numpy as np
import pytest
from crateapp.db import connect
from crateapp.runner import Runner


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A library of two tracks and a two-crate model, with analysis stubbed."""
    music = tmp_path / "music"
    music.mkdir()
    for n in ("a.wav", "b.wav"):
        (music / n).write_bytes(b"RIFF0000WAVE")

    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    model = tmp_path / "m.json"
    model.write_text(json.dumps({"crates": {
        "House":   {"n": 9, "centroid": a.tolist()},
        "Dubstep": {"n": 9, "centroid": b.tolist()}}}))

    import crateapp.runner as runner_mod

    def fake_analyse_track(gm, path):
        """Stands in for the real analysis: returns (result, embedding) and
        touches no database, matching analyse_track's contract."""
        from pathlib import Path as _P
        vec = a if _P(path).name == "a.wav" else b
        return ({"bpm": 128.0, "camelot": "8A", "duration_sec": 100,
                 "category": "House",
                 "structure": {"energy_curve": [0.1, 0.9, 0.5]},
                 "moments": [], "vocal_analysis": {}, "genre_predictions": []},
                vec)

    monkeypatch.setattr(runner_mod, "analyse_track", fake_analyse_track)
    monkeypatch.setattr(runner_mod, "load_genre_model", lambda d: None)
    return tmp_path, music, model


def test_a_run_analyses_and_files_every_track(setup):
    tmp, music, model = setup
    r = Runner(tmp / "l.db", model)
    r.start([str(music)])
    r.wait()
    con = connect(tmp / "l.db")
    assert con.execute("SELECT count(*) c FROM tracks").fetchone()["c"] == 2
    assert con.execute(
        "SELECT count(*) c FROM tracks WHERE analysed_at IS NOT NULL"
    ).fetchone()["c"] == 2
    assert con.execute("SELECT count(*) c FROM assignments").fetchone()["c"] == 2


def test_tracks_land_in_the_crate_they_match(setup):
    tmp, music, model = setup
    r = Runner(tmp / "l.db", model); r.start([str(music)]); r.wait()
    con = connect(tmp / "l.db")
    rows = {row["filename"]: row["name"] for row in con.execute(
        "SELECT t.filename, c.name FROM assignments a "
        "JOIN tracks t ON t.id=a.track_id JOIN crates c ON c.id=a.crate_id")}
    assert rows == {"a.wav": "House", "b.wav": "Dubstep"}


def test_status_reports_progress_and_finishes(setup):
    tmp, music, model = setup
    r = Runner(tmp / "l.db", model); r.start([str(music)]); r.wait()
    s = r.status()
    assert s["running"] is False
    assert s["done"] == 2 and s["total"] == 2


def test_status_holds_the_last_decided_track_at_rest(setup):
    """The panel is always visible; at rest it shows the last thing decided."""
    tmp, music, model = setup
    r = Runner(tmp / "l.db", model); r.start([str(music)]); r.wait()
    cur = r.status()["current"]
    assert cur is not None
    assert cur["filename"] in ("a.wav", "b.wav")
    assert cur["band"] in ("confident", "uncertain", "unknown")


def test_status_exposes_real_per_crate_similarities(setup):
    """Every number the panel shows must be one the classifier actually used."""
    tmp, music, model = setup
    r = Runner(tmp / "l.db", model); r.start([str(music)]); r.wait()
    sims = r.status()["current"]["similarities"]
    assert {s["crate"] for s in sims} == {"House", "Dubstep"}
    assert sims == sorted(sims, key=lambda s: -s["similarity"])
    assert all(-1.0 <= s["similarity"] <= 1.0 for s in sims)


def test_status_before_any_run_is_empty_but_valid(tmp_path):
    model = tmp_path / "m.json"
    model.write_text(json.dumps({"crates": {}}))
    s = Runner(tmp_path / "l.db", model).status()
    assert s["running"] is False and s["current"] is None and s["done"] == 0


def test_a_second_run_skips_already_analysed_tracks(setup):
    tmp, music, model = setup
    r = Runner(tmp / "l.db", model); r.start([str(music)]); r.wait()
    r2 = Runner(tmp / "l.db", model); r2.start([str(music)]); r2.wait()
    assert r2.status()["total"] == 0


def test_unplugged_folder_does_not_stop_the_run(tmp_path):
    from crateapp.db import connect
    from crateapp.runner import Runner
    lib = tmp_path / "lib"; lib.mkdir()
    db = tmp_path / "l.db"
    connect(db).close()
    r = Runner(db, tmp_path / "m.json")
    r.start([str(tmp_path / "unplugged"), str(lib)])
    r.wait()
    st = r.status()
    assert st["unreachable"] == [str(tmp_path / "unplugged")]
    assert st["running"] is False


def test_tracks_analysed_before_the_models_existed_are_sorted_once_they_do(setup, monkeypatch):
    """Without the genre models a run gives tempo and key but no embedding.
    Those tracks used to stay unsorted for good (analysis runs once); the
    run now says the models are missing, and picks the tracks up again once
    they are installed."""
    import types
    import crateapp.runner as runner_mod
    tmp, music, model = setup
    real = runner_mod.analyse_track
    monkeypatch.setattr(runner_mod, "load_genre_model", lambda d: types.SimpleNamespace(ok=False))
    monkeypatch.setattr(runner_mod, "analyse_track", lambda gm, p: (real(gm, p)[0], None))
    r = Runner(tmp / "l.db", model); r.start([str(music)]); r.wait()
    assert r.status()["noModel"] is True
    con = connect(tmp / "l.db")
    assert con.execute("SELECT count(*) FROM assignments").fetchone()[0] == 0

    monkeypatch.setattr(runner_mod, "load_genre_model", lambda d: types.SimpleNamespace(ok=True))
    monkeypatch.setattr(runner_mod, "analyse_track", real)
    r2 = Runner(tmp / "l.db", model); r2.start([str(music)]); r2.wait()
    assert r2.status()["total"] == 2 and r2.status()["noModel"] is False
    assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 2
    # And a third run has nothing left to do.
    r3 = Runner(tmp / "l.db", model); r3.start([str(music)]); r3.wait()
    assert r3.status()["total"] == 0
