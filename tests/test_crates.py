import pytest
from crateapp.db import connect
from crateapp.crates import (ensure_crate, auto_assign, correct, uncertain, unsorted)


@pytest.fixture
def con(tmp_path):
    c = connect(tmp_path / "l.db")
    c.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1, '/a.wav', 'now')")
    c.commit()
    return c


def test_ensure_crate_is_idempotent(con):
    assert ensure_crate(con, "House") == ensure_crate(con, "House")


def test_a_confident_result_is_assigned(con):
    auto_assign(con, 1, {"crate": "House", "band": "confident", "similarity": 0.9})
    row = con.execute("SELECT * FROM assignments WHERE track_id=1").fetchone()
    assert row["source"] == "auto" and row["band"] == "confident"


def test_an_uncertain_result_is_still_assigned(con):
    auto_assign(con, 1, {"crate": "House", "band": "uncertain", "similarity": 0.7})
    assert con.execute("SELECT count(*) c FROM assignments").fetchone()["c"] == 1
    assert [r["id"] for r in uncertain(con)] == [1]


def test_an_unknown_result_is_assigned_nowhere(con):
    auto_assign(con, 1, {"crate": None, "band": "unknown", "similarity": 0.2})
    assert con.execute("SELECT count(*) c FROM assignments").fetchone()["c"] == 0
    assert [r["id"] for r in unsorted(con)] == [1]


def test_a_move_replaces_membership(con):
    auto_assign(con, 1, {"crate": "House", "band": "uncertain", "similarity": 0.7})
    correct(con, 1, "Dubstep", mode="move")
    crates = [r["name"] for r in con.execute(
        "SELECT c.name FROM assignments a JOIN crates c ON c.id=a.crate_id "
        "WHERE a.track_id=1")]
    assert crates == ["Dubstep"]


def test_a_move_is_always_recorded_as_an_error(con):
    auto_assign(con, 1, {"crate": "House", "band": "uncertain", "similarity": 0.7})
    correct(con, 1, "Dubstep", mode="move")
    assert con.execute("SELECT was_error FROM corrections").fetchone()["was_error"] == 1


def test_an_also_add_keeps_both_crates(con):
    auto_assign(con, 1, {"crate": "House", "band": "confident", "similarity": 0.9})
    correct(con, 1, "Party", mode="add", was_error=False)
    crates = sorted(r["name"] for r in con.execute(
        "SELECT c.name FROM assignments a JOIN crates c ON c.id=a.crate_id "
        "WHERE a.track_id=1"))
    assert crates == ["House", "Party"]


def test_an_also_add_records_the_djs_own_verdict(con):
    """Counting every also-add as a success would inflate measured accuracy."""
    auto_assign(con, 1, {"crate": "House", "band": "confident", "similarity": 0.9})
    correct(con, 1, "Party", mode="add", was_error=True)
    assert con.execute("SELECT was_error FROM corrections").fetchone()["was_error"] == 1


def test_a_corrected_track_leaves_the_uncertain_list(con):
    auto_assign(con, 1, {"crate": "House", "band": "uncertain", "similarity": 0.7})
    correct(con, 1, "Dubstep", mode="move")
    assert uncertain(con) == []


def test_correcting_into_a_new_crate_creates_it(con):
    auto_assign(con, 1, {"crate": None, "band": "unknown", "similarity": 0.2})
    correct(con, 1, "Amapiano", mode="move")
    assert con.execute("SELECT count(*) c FROM crates WHERE name='Amapiano'"
                       ).fetchone()["c"] == 1


def test_move_cannot_be_talked_out_of_being_an_error(con):
    """The DB has no CHECK constraint for rule 3 - the application code must
    hold the line even when a caller explicitly passes was_error=False."""
    auto_assign(con, 1, {"crate": "House", "band": "confident", "similarity": 0.9})
    correct(con, 1, "Dubstep", mode="move", was_error=False)
    assert con.execute("SELECT was_error FROM corrections").fetchone()["was_error"] == 1


def test_unsorted_excludes_tracks_that_are_not_analysed_yet(con):
    """A freshly scanned track has not been judged yet - it is not 'unsorted',
    it is simply pending. Otherwise a scan buries the real unknowns."""
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (2, '/b.wav', NULL)")
    con.commit()
    assert 2 not in [r["id"] for r in unsorted(con)]


def test_a_move_does_not_destroy_a_deliberate_also_add(con):
    """A move only corrects the model's own (auto) placement. A separate,
    deliberate human also-add (e.g. to a personal 'Party' crate) is the DJ's
    own statement and must survive an unrelated correction."""
    auto_assign(con, 1, {"crate": "House", "band": "confident", "similarity": 0.9})
    correct(con, 1, "Party", mode="add", was_error=False)
    correct(con, 1, "Tech House", mode="move")
    crates = sorted(r["name"] for r in con.execute(
        "SELECT c.name FROM assignments a JOIN crates c ON c.id=a.crate_id "
        "WHERE a.track_id=1"))
    assert crates == ["Party", "Tech House"]


def test_an_also_add_clears_the_track_from_uncertain(con):
    """Acting on a track - even by also-adding it elsewhere - resolves it.
    It must not keep reappearing in the review queue, but it must stay filed
    in the crate it was auto-assigned to."""
    auto_assign(con, 1, {"crate": "House", "band": "uncertain", "similarity": 0.7})
    correct(con, 1, "Party", mode="add", was_error=False)
    assert uncertain(con) == []
    crates = sorted(r["name"] for r in con.execute(
        "SELECT c.name FROM assignments a JOIN crates c ON c.id=a.crate_id "
        "WHERE a.track_id=1"))
    assert crates == ["House", "Party"]


def test_reclassifying_a_track_replaces_the_prior_auto_row(con):
    """Calling auto_assign twice for the same track (e.g. after a centroid
    rebuild) must not accumulate stale auto rows - the new verdict replaces
    the old one."""
    auto_assign(con, 1, {"crate": "House", "band": "confident", "similarity": 0.9})
    auto_assign(con, 1, {"crate": "Dubstep", "band": "confident", "similarity": 0.85})
    rows = con.execute(
        "SELECT c.name FROM assignments a JOIN crates c ON c.id=a.crate_id "
        "WHERE a.track_id=1 AND a.source='auto'").fetchall()
    assert [r["name"] for r in rows] == ["Dubstep"]
