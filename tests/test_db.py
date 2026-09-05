import sqlite3
from crateapp.db import connect


def test_connect_creates_all_tables(tmp_path):
    con = connect(tmp_path / "library.db")
    names = {r["name"] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tracks", "embeddings", "analysis", "crates",
            "assignments", "corrections", "config"} <= names


def test_connect_is_idempotent(tmp_path):
    p = tmp_path / "library.db"
    connect(p).close()
    con = connect(p)          # must not raise on second open
    assert con.execute("SELECT count(*) c FROM tracks").fetchone()["c"] == 0


def test_rows_are_addressable_by_name(tmp_path):
    con = connect(tmp_path / "library.db")
    con.execute("INSERT INTO crates (name) VALUES ('House')")
    assert con.execute("SELECT name FROM crates").fetchone()["name"] == "House"


def test_a_track_can_be_in_two_crates(tmp_path):
    """Multi-crate membership is required by 'also add' - see spec section 5."""
    con = connect(tmp_path / "library.db")
    con.execute("INSERT INTO tracks (id, path) VALUES (1, '/a.wav')")
    con.execute("INSERT INTO crates (id, name) VALUES (1, 'House'), (2, 'Party')")
    con.execute("INSERT INTO assignments (track_id, crate_id, source) "
                "VALUES (1, 1, 'auto'), (1, 2, 'human')")
    assert con.execute(
        "SELECT count(*) c FROM assignments WHERE track_id=1").fetchone()["c"] == 2


def test_the_same_track_cannot_join_one_crate_twice(tmp_path):
    con = connect(tmp_path / "library.db")
    con.execute("INSERT INTO tracks (id, path) VALUES (1, '/a.wav')")
    con.execute("INSERT INTO crates (id, name) VALUES (1, 'House')")
    con.execute("INSERT INTO assignments (track_id, crate_id, source) VALUES (1,1,'auto')")
    try:
        con.execute("INSERT INTO assignments (track_id, crate_id, source) VALUES (1,1,'human')")
        raise AssertionError("expected a uniqueness violation")
    except sqlite3.IntegrityError:
        pass
