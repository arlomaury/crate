import pytest
from crateapp.db import connect
from crateapp.crates import (ensure_crate, auto_assign, correct, disputed,
                             remove_track, uncertain, unsorted)


@pytest.fixture
def con(tmp_path):
    c = connect(tmp_path / "l.db")
    c.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1, '/a.wav', 'now')")
    c.commit()
    return c


def test_ensure_crate_is_idempotent(con):
    assert ensure_crate(con, "House") == ensure_crate(con, "House")


def test_ensure_crate_rejects_blank_names(con):
    with pytest.raises(ValueError):
        ensure_crate(con, "   ")


def test_ensure_crate_keeps_path_hostile_names_verbatim(con):
    """Sanitisation belongs at the exporter, not the database - a crate is
    free to be called anything meaningful, including something that would
    not be a safe folder name."""
    cid = ensure_crate(con, "Drum & Bass / Jungle")
    row = con.execute("SELECT name FROM crates WHERE id=?", (cid,)).fetchone()
    assert row["name"] == "Drum & Bass / Jungle"


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


# ------------------------------------------------------------ confirming

def test_confirm_takes_a_track_out_of_review(tmp_path):
    """The classifier was right. Saying so must clear the flag."""
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "uncertain", "similarity": 0.6})
    assert len(uncertain(con)) == 1
    correct(con, 1, mode="confirm")
    assert uncertain(con) == []


def test_confirm_leaves_the_track_where_it_is(tmp_path):
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "uncertain", "similarity": 0.6})
    correct(con, 1, mode="confirm")
    rows = con.execute(
        "SELECT c.name FROM assignments a JOIN crates c ON c.id=a.crate_id "
        "WHERE a.track_id=1").fetchall()
    assert [r["name"] for r in rows] == ["house"]


def test_a_confirmed_track_becomes_training_data(tmp_path):
    """The point of confirming: with only a few hundred labels to learn from,
    agreeing with the model is worth as much as correcting it."""
    from crateapp.classifier import training_set
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "uncertain", "similarity": 0.6})
    assert training_set(con)[0] == [], "an auto guess is not training data"
    correct(con, 1, mode="confirm")
    assert training_set(con) == ([1], ["house"])


def test_confirm_is_not_recorded_as_an_error(tmp_path):
    """Counting agreement as failure would make the model look worse the more
    the DJ agreed with it."""
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "uncertain", "similarity": 0.6})
    correct(con, 1, mode="confirm")
    assert con.execute("SELECT was_error FROM corrections").fetchone()["was_error"] == 0


def test_confirming_a_track_the_model_gave_up_on_is_an_error(tmp_path):
    """An 'unknown' track was filed nowhere, so there is no placement to
    agree with. Silently doing nothing would look like it worked."""
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": None, "band": "unknown"})
    with pytest.raises(ValueError):
        correct(con, 1, mode="confirm")


def test_confirm_survives_a_later_move(tmp_path):
    """Changing your mind after confirming must replace, not duplicate."""
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "uncertain", "similarity": 0.6})
    correct(con, 1, mode="confirm")
    correct(con, 1, "tech", mode="move")
    rows = con.execute(
        "SELECT c.name FROM assignments a JOIN crates c ON c.id=a.crate_id "
        "WHERE a.track_id=1").fetchall()
    assert [r["name"] for r in rows] == ["tech"]


# -------------------------------------------------- confident but doubted

def test_disputed_surfaces_confident_tracks_the_signals_disagree_on(tmp_path):
    """A confident mistake never reaches uncertain(), which is exactly why it
    is dangerous. Measured on the real library: 25% of these are misfiled,
    against 2.9% of the confident tracks where both signals agree."""
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "confident",
                         "similarity": 0.8, "disputed": True})
    assert [r["id"] for r in disputed(con)] == [1]
    assert uncertain(con) == [], "it is confident; it must not be in that queue"


def test_an_agreed_confident_track_is_not_disputed(tmp_path):
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "confident",
                         "similarity": 0.8, "disputed": False})
    assert disputed(con) == []


def test_confirming_clears_a_dispute(tmp_path):
    """Agreeing with it resolves the doubt - the row must leave the list."""
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "confident",
                         "similarity": 0.8, "disputed": True})
    correct(con, 1, mode="confirm")
    assert disputed(con) == []


def test_disputed_carries_the_crate_it_was_filed_into(tmp_path):
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, analysed_at) VALUES (1,'/a.wav','now')")
    con.commit()
    auto_assign(con, 1, {"crate": "house", "band": "confident",
                         "similarity": 0.8, "disputed": True})
    assert disputed(con)[0]["crate"] == "house"


# --------------------------------------------------------------- removing

def test_removing_a_track_takes_it_out_of_every_crate(tmp_path):
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, filename, analysed_at) "
                "VALUES (1,'/a.wav','a.wav','now')")
    con.commit()
    correct(con, 1, "house", mode="move")
    assert remove_track(con, 1) == "a.wav"
    assert con.execute("SELECT count(*) FROM assignments").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM tracks").fetchone()[0] == 0


def test_removing_a_track_stops_it_training_the_model(tmp_path):
    """Its rows have to go with it, or a track the DJ deleted keeps teaching
    the classifier from beyond the grave."""
    from crateapp.classifier import training_set
    from crateapp.worker import store_result
    import numpy as np
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, filename) VALUES (1,'/a.wav','a.wav')")
    store_result(con, 1, {"bpm": 128.0}, np.ones(8, dtype=np.float32))
    correct(con, 1, "house", mode="move")
    assert training_set(con)[0] == [1]
    remove_track(con, 1)
    assert training_set(con)[0] == []
    assert con.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM analysis").fetchone()[0] == 0


def test_removing_an_unknown_track_says_so(tmp_path):
    con = connect(tmp_path / "l.db")
    assert remove_track(con, 999) is None


def test_removing_never_touches_the_file_on_disk(tmp_path):
    """The standing invariant: source audio is never modified, moved or
    deleted. Removing is about the DJ's record of a track, not the track."""
    f = tmp_path / "real.wav"
    f.write_bytes(b"RIFFsomething")
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, filename) VALUES (1,?,'real.wav')",
                (str(f),))
    con.commit()
    remove_track(con, 1)
    assert f.exists() and f.read_bytes() == b"RIFFsomething"


# ----------------------------------------------------- the artist shortcut

def test_normalise_artist_collapses_credits():
    from crateapp.crates import normalise_artist
    n = normalise_artist
    assert n("Sammy Virji") == n("Sammy Virji, Issey Cross") == n("sammy  virji")
    assert n("MPH & Dread MC") == n("MPH")
    assert n("Four Tet feat. Nelly Furtado") == n("Four Tet")
    assert n("") is None and n(None) is None


def test_artist_crate_uses_only_the_djs_own_filing(tmp_path):
    """Measured leave-one-out, same-artist-same-crate is right 95.6% of the
    time - and 97.3% on house vs tech, which no amount of audio analysis
    separates above ~74%. But only the DJ's filing counts; the model's own
    guesses about an artist say nothing."""
    from crateapp.crates import artist_crate
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, artist) VALUES (1,'/a.wav','mph')")
    con.execute("INSERT INTO tracks (id, path, artist) VALUES (2,'/b.wav','mph')")
    con.commit()
    auto_assign(con, 2, {"crate": "house", "band": "confident", "similarity": .9})
    assert artist_crate(con, "MPH") == (None, 0), "an auto guess is not evidence"
    correct(con, 1, "UKG", mode="move")
    assert artist_crate(con, "MPH, Dread MC") == ("UKG", 1)


def test_artist_crate_declines_when_the_dj_was_inconsistent(tmp_path):
    """Drake is filed in both house and rap - originals against edits. Those
    artists are genuinely ambiguous, so the rule must decline, not pick."""
    from crateapp.crates import artist_crate
    con = connect(tmp_path / "l.db")
    for i, crate in ((1, "house"), (2, "rap")):
        con.execute("INSERT INTO tracks (id, path, artist) VALUES (?,?,'drake')",
                    (i, f"/{i}.wav"))
        con.commit()
        correct(con, i, crate, mode="move")
    assert artist_crate(con, "Drake") == (None, 0)


def test_artist_crate_is_silent_without_a_tag(tmp_path):
    from crateapp.crates import artist_crate
    con = connect(tmp_path / "l.db")
    assert artist_crate(con, None) == (None, 0)
    assert artist_crate(con, "Nobody At All") == (None, 0)


def test_credited_remixer_strips_genre_qualifiers():
    """"(Boucho UKG Edit)" is by Boucho; UKG is a genre, not part of a name.
    Without this, their UKG edit and their plain remix are different people."""
    from crateapp.crates import credited_remixer as r
    assert r("Somebody (Boucho UKG Edit).aiff") == r("Thing (Boucho Remix).wav")
    assert r("Blame (Gorgon City Remix).aiff") == "gorgoncity"
    assert r("no credit here.aiff") is None


def test_a_remix_is_filed_by_its_remixer_not_the_original_artist(tmp_path):
    """A remix is the remixer's record - their drums, their bassline, their
    genre. The artist tag names the original act, which is one of the ways
    filing by artist goes wrong."""
    from crateapp.crates import who_made_it
    con = connect(tmp_path / "l.db")
    # The DJ has filed a Sammy Virji track under UKG by hand...
    con.execute("INSERT INTO tracks (id, path, filename, artist) "
                "VALUES (1,'/a.wav','Sammy Virji - Thing.wav','sammyvirji')")
    con.commit()
    correct(con, 1, "UKG", mode="move")
    # ...and separately filed Drake under rap.
    con.execute("INSERT INTO tracks (id, path, filename, artist) "
                "VALUES (2,'/b.wav','Drake - Song.wav','drake')")
    con.commit()
    correct(con, 2, "rap", mode="move")
    # A Drake track remixed by Sammy Virji belongs with the remixer.
    con.execute("INSERT INTO tracks (id, path, filename, artist, remixer) "
                "VALUES (3,'/c.wav','Drake - Song (Sammy Virji Remix).wav',"
                "'drake','sammyvirji')")
    con.commit()
    who, crate, n = who_made_it(con, 3)
    assert crate == "UKG", "the remixer decides, not the original artist"
    assert who == "sammyvirji"


def test_who_made_it_falls_back_to_the_artist(tmp_path):
    from crateapp.crates import who_made_it
    con = connect(tmp_path / "l.db")
    con.execute("INSERT INTO tracks (id, path, filename, artist) "
                "VALUES (1,'/a.wav','MPH - One.wav','mph')")
    con.commit()
    correct(con, 1, "UKG", mode="move")
    con.execute("INSERT INTO tracks (id, path, filename, artist) "
                "VALUES (2,'/b.wav','MPH - Two.wav','mph')")
    con.commit()
    assert who_made_it(con, 2)[1] == "UKG"


def test_who_made_it_is_silent_on_an_unknown_track(tmp_path):
    from crateapp.crates import who_made_it
    con = connect(tmp_path / "l.db")
    assert who_made_it(con, 999) == (None, None, 0)


# ------------------------------------------------------- the genre tag rule

def _tagged(con, tid, tag, crate=None):
    con.execute("INSERT INTO tracks (id, path, filename, genre_tag) "
                "VALUES (?,?,?,?)", (tid, f"/{tid}.wav", f"{tid}.wav", tag))
    con.commit()
    if crate:
        correct(con, tid, crate, mode="move")


def test_the_tag_mapping_is_learned_not_hardcoded(tmp_path):
    """Nothing in the code knows that "Tech House" means tech. It is read off
    the DJ's own filing, so it follows their vocabulary and not Beatport's."""
    from crateapp.crates import tag_crate
    con = connect(tmp_path / "l.db")
    for i in (1, 2, 3):
        _tagged(con, i, "Tech House", "my weird crate")
    assert tag_crate(con, "tech house")[0] == "my weird crate"
    assert tag_crate(con, "TECH HOUSE")[0] == "my weird crate", "case-insensitive"


def test_a_tag_seen_once_decides_nothing(tmp_path):
    """One example is a coincidence, not a mapping."""
    from crateapp.crates import tag_crate
    con = connect(tmp_path / "l.db")
    _tagged(con, 1, "Tech House", "tech")
    assert tag_crate(con, "Tech House") == (None, 0, 0.0)


def test_a_tag_that_points_both_ways_decides_nothing(tmp_path):
    """"Hip-Hop/Rap" covers rap, pop crossovers and acapellas here. A tag has
    to point mostly one way before it is worth following."""
    from crateapp.crates import tag_crate
    con = connect(tmp_path / "l.db")
    for i, c in ((1, "rap"), (2, "pop"), (3, "vocals"), (4, "house")):
        _tagged(con, i, "Mixed Bag", c)
    assert tag_crate(con, "Mixed Bag") == (None, 0, 0.0)


def test_the_tag_rule_ignores_the_models_own_guesses(tmp_path):
    """Same rule as everywhere else: only the DJ's filing is evidence."""
    from crateapp.crates import tag_crate
    con = connect(tmp_path / "l.db")
    for i in (1, 2, 3):
        _tagged(con, i, "Tech House")
        auto_assign(con, i, {"crate": "tech", "band": "confident",
                             "similarity": 0.9})
    assert tag_crate(con, "Tech House") == (None, 0, 0.0)


def test_an_untagged_track_falls_through(tmp_path):
    from crateapp.crates import tag_crate
    con = connect(tmp_path / "l.db")
    assert tag_crate(con, None) == (None, 0, 0.0)
    assert tag_crate(con, "   ") == (None, 0, 0.0)


def test_tag_purity_ignores_acapellas(tmp_path):
    """The acapella rule runs first, so isolated vocals never reach the tag
    rule. Counting them drags a tag's purity down for a decision it is not
    being asked to make - on the real library "Hip-Hop/Rap" reads 46% pure
    with them and 76% without, which is the difference between unusable and
    useful."""
    import json
    from crateapp.crates import tag_crate
    con = connect(tmp_path / "l.db")
    for i, (crate, aca) in enumerate(
            [("rap", 0), ("rap", 0), ("rap", 0),
             ("vocals", 1), ("vocals", 1), ("vocals", 1), ("vocals", 1)], start=1):
        con.execute("INSERT INTO tracks (id, path, filename, genre_tag) "
                    "VALUES (?,?,?,'Hip-Hop/Rap')", (i, f"/{i}.wav", f"{i}.wav"))
        con.execute("INSERT INTO analysis (track_id, vocal) VALUES (?,?)",
                    (i, json.dumps({"is_acapella": bool(aca)})))
        con.commit()
        correct(con, i, crate, mode="move")
    crate, n, purity = tag_crate(con, "Hip-Hop/Rap")
    assert crate == "rap", "the acapellas must not outvote the rap tracks"
    assert n == 3 and purity == 1.0


def test_tag_gaps_ranks_by_how_many_tracks_it_would_settle(tmp_path):
    """Teaching a tag is worth more than correcting a track: one decision
    settles every other track carrying it. So the richest tag leads."""
    from crateapp.crates import tag_gaps
    con = connect(tmp_path / "l.db")
    tid = 0
    for tag, n in (("big tag", 5), ("small tag", 2)):
        for _ in range(n):
            tid += 1
            con.execute("INSERT INTO tracks (id, path, filename, genre_tag) "
                        "VALUES (?,?,?,?)", (tid, f"/{tid}.wav", f"{tid}.wav", tag))
            con.commit()
            auto_assign(con, tid, {"crate": "house", "band": "confident",
                                   "similarity": 0.8})
    gaps = tag_gaps(con)
    assert [g["tag"] for g in gaps] == ["big tag", "small tag"]
    assert gaps[0]["would_unlock"] == 5
    assert gaps[0]["track_id"] is not None, "must name a track to act on"


def test_a_tag_that_already_maps_is_not_a_gap(tmp_path):
    """Once it decides something it is no longer worth the DJ's attention."""
    from crateapp.crates import tag_gaps
    con = connect(tmp_path / "l.db")
    for i in (1, 2):
        con.execute("INSERT INTO tracks (id, path, filename, genre_tag) "
                    "VALUES (?,?,?,'known')", (i, f"/{i}.wav", f"{i}.wav"))
        con.commit()
        correct(con, i, "tech", mode="move")
    con.execute("INSERT INTO tracks (id, path, filename, genre_tag) "
                "VALUES (9,'/9.wav','9.wav','known')")
    con.commit()
    auto_assign(con, 9, {"crate": "tech", "band": "confident", "similarity": .9})
    assert [g["tag"] for g in tag_gaps(con)] == []
