"""A brand-new library: analysed before any model exists, then taught a few
crates. The tracks already analysed must get filed once the model is trained,
or a new user's library never sorts (runner.file_unsorted)."""
import json

import numpy as np

from crateapp.classifier import rebuild_centroids
from crateapp.crates import auto_assign, correct, unsorted
from crateapp.db import connect
from crateapp.exporters import check_rekordbox_dest
from crateapp.runner import file_unsorted

import pytest


def _library(tmp_path):
    con = connect(tmp_path / "l.db")
    rng = np.random.default_rng(0)
    centres = {"A": np.eye(16)[0] * 3, "B": np.eye(16)[1] * 3}
    for i in range(1, 15):
        side = "A" if i % 2 else "B"
        vec = (centres[side] + rng.normal(0, 0.3, 16)).astype("float32")
        con.execute("INSERT INTO tracks (id, path, filename, analysed_at, bpm) VALUES (?,?,?,'now',128)",
                    (i, f"/m/{side}{i}.wav", f"{side}{i}.wav"))
        con.execute("INSERT INTO analysis (track_id, moments, energy, vocal) VALUES (?,?,?,?)",
                    (i, "[]", "[]", json.dumps({"vocal_ratio": 0.3})))
        con.execute("INSERT INTO embeddings (track_id, vector) VALUES (?,?)", (i, vec.tobytes()))
    con.commit()
    return con


def test_tracks_analysed_before_a_model_get_filed_after_teaching(tmp_path):
    con = _library(tmp_path)
    model = tmp_path / "model.json"
    assert file_unsorted(con, model) == 0                 # no model yet: nothing guessed
    assert len(unsorted(con)) == 14
    for i in range(1, 11):                                 # the DJ files 5 of each
        correct(con, i, "House" if i % 2 else "DNB", mode="move")
    rebuild_centroids(con, model)
    filed = file_unsorted(con, model)
    assert filed == 4 and unsorted(con) == []
    got = dict(con.execute("SELECT a.track_id, c.name FROM assignments a JOIN crates c "
                           "ON c.id=a.crate_id WHERE a.track_id > 10").fetchall())
    assert got == {11: "House", 12: "DNB", 13: "House", 14: "DNB"}
    # The DJ's own filing is untouched.
    assert {r[0] for r in con.execute("SELECT source FROM assignments WHERE track_id <= 10")} == {"human"}


def test_reanalysis_keeps_a_reviewed_auto_placement(tmp_path):
    con = _library(tmp_path)
    auto_assign(con, 1, {"crate": "house", "band": "uncertain", "similarity": 0.7})
    correct(con, 1, "party", mode="add", was_error=False)
    before = sorted(tuple(r) for r in con.execute("SELECT crate_id, source, band FROM assignments WHERE track_id=1"))
    auto_assign(con, 1, {"crate": "tech", "band": "confident", "similarity": 0.9})
    after = sorted(tuple(r) for r in con.execute("SELECT crate_id, source, band FROM assignments WHERE track_id=1"))
    assert after == before


def test_moving_onto_an_also_add_crate_is_still_a_correction(tmp_path):
    con = _library(tmp_path)
    auto_assign(con, 1, {"crate": "house", "band": "uncertain", "similarity": 0.7})
    correct(con, 1, "tech", mode="add", was_error=False)
    correct(con, 1, "tech", mode="move")
    assert con.execute("SELECT was_error FROM corrections ORDER BY rowid DESC").fetchone()[0] == 1


def test_a_rekordbox_collection_is_never_overwritten(tmp_path):
    rek = tmp_path / "rek.xml"
    rek.write_text('<?xml version="1.0"?>\n<DJ_PLAYLISTS Version="1.0.0">\n'
                   '  <PRODUCT Name="rekordbox" Version="7.0"/>\n</DJ_PLAYLISTS>')
    with pytest.raises(ValueError):
        check_rekordbox_dest(rek)
    ours = tmp_path / "crate.xml"
    ours.write_text('<?xml version="1.0"?>\n<DJ_PLAYLISTS Version="1.0.0">\n'
                    '  <PRODUCT Name="Crate" Version="1.0" Company="local"/>\n</DJ_PLAYLISTS>')
    check_rekordbox_dest(ours)                              # our own export: fine to replace
