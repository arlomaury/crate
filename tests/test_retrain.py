"""retrain.py rebuilds the labels from the DJ's Rekordbox playlists."""
import json

import numpy as np

from crateapp.crates import correct, set_locked
from crateapp.db import connect


def _xml(tmp_path, entries):
    tracks = "".join(f'<TRACK TrackID="{i}" Location="file://localhost{p}"/>'
                     for i, (p, _) in enumerate(entries, 1))
    by = {}
    for i, (_, pl) in enumerate(entries, 1):
        by.setdefault(pl, []).append(i)
    nodes = "".join(f'<NODE Name="{pl}" Type="1">' + "".join(f'<TRACK Key="{i}"/>' for i in ids) + "</NODE>"
                    for pl, ids in by.items())
    x = tmp_path / "rek.xml"
    x.write_text(f'<DJ_PLAYLISTS><COLLECTION>{tracks}</COLLECTION>'
                 f'<PLAYLISTS><NODE Name="ROOT" Type="0">{nodes}</NODE></PLAYLISTS></DJ_PLAYLISTS>')
    return x


def test_a_reseed_keeps_hand_picked_locked_crates(tmp_path):
    """Locked crates are not genre playlists, so Rekordbox cannot recreate
    them. Wiping every assignment emptied the DJ's curated crates."""
    from retrain import reseed
    con = connect(tmp_path / "l.db")
    for i, name in enumerate(["a.wav", "b.wav", "fav.wav"], 1):
        con.execute("INSERT INTO tracks (id, path, filename) VALUES (?,?,?)",
                    (i, f"/music/{name}", name))
    con.commit()
    correct(con, 3, "the song", mode="move")
    set_locked(con, "the song")
    correct(con, 1, "rap", mode="move")                  # a stale correction: replaced
    reseed(con, _xml(tmp_path, [("/music/a.wav", "house"), ("/music/b.wav", "tech")]))
    rows = {(r["track_id"], r["name"]) for r in con.execute(
        "SELECT a.track_id, c.name FROM assignments a JOIN crates c ON c.id=a.crate_id")}
    assert rows == {(1, "house"), (2, "tech"), (3, "the song")}


def test_reclassify_files_the_way_a_sorting_run_does(tmp_path):
    """An acapella the audio model disagrees with is flagged for a look, the
    same as in a sorting run (the old hand-copied chain never flagged it)."""
    from crateapp.classifier import train
    from retrain import reclassify
    con = connect(tmp_path / "l.db")
    rng = np.random.default_rng(0)
    tid = 0
    for c, crate in ((0, "house"), (1, "vocals")):
        for _ in range(6):
            tid += 1
            v = np.zeros(8, "float32"); v[c] = 1; v += rng.normal(0, .05, 8).astype("float32")
            con.execute("INSERT INTO tracks (id, path, filename, analysed_at) VALUES (?,?,?, 'x')",
                        (tid, f"/m/{tid}.wav", f"{tid}.wav"))
            con.execute("INSERT INTO embeddings VALUES (?,?)", (tid, v.tobytes()))
            con.execute("INSERT INTO analysis (track_id, vocal) VALUES (?, ?)", (tid, json.dumps({"is_acapella": False})))
            con.commit()
            correct(con, tid, crate, mode="move")
    # An isolated vocal that SOUNDS like house to the model.
    tid += 1
    v = np.zeros(8, "float32"); v[0] = 1
    con.execute("INSERT INTO tracks (id, path, filename, analysed_at) VALUES (?,?,?, 'x')",
                (tid, "/m/aca.wav", "aca.wav"))
    con.execute("INSERT INTO embeddings VALUES (?,?)", (tid, v.tobytes()))
    con.execute("INSERT INTO analysis (track_id, vocal) VALUES (?, ?)", (tid, json.dumps({"is_acapella": True})))
    con.commit()
    m = tmp_path / "m.json"
    train(con, m, min_per_class=5)
    bands = reclassify(con, m)
    row = con.execute("SELECT c.name, a.disputed FROM assignments a JOIN crates c ON c.id=a.crate_id "
                      "WHERE a.track_id=?", (tid,)).fetchone()
    assert row["name"] == "vocals" and row["disputed"] == 1
    assert bands["acapella -> vocals"] == 1 and bands["kept (yours)"] == 12
