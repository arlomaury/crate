import pytest

from crateapp.bootstrap import (add_folder, get_folders, import_rekordbox_xml,
                                remove_folder)
from crateapp.db import connect

XML = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <COLLECTION Entries="3">
    <TRACK TrackID="7" Name="A" Genre="House" Location="file://localhost{a}"/>
    <TRACK TrackID="8" Name="B" Genre="Dubstep" Location="file://localhost{b}"/>
    <TRACK TrackID="9" Name="Gone" Genre="House" Location="file://localhost/nope/x.wav"/>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="2">
      <NODE Name="house" Type="1" Entries="2">
        <TRACK Key="7"/><TRACK Key="9"/>
      </NODE>
      <NODE Name="dubstep" Type="1" Entries="1"><TRACK Key="8"/></NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>"""


@pytest.fixture
def xml_and_db(tmp_path):
    a = tmp_path / "a.wav"; a.write_bytes(b"RIFF")
    b = tmp_path / "b.wav"; b.write_bytes(b"RIFF")
    x = tmp_path / "r.xml"
    x.write_text(XML.format(a=a, b=b))
    return x, connect(tmp_path / "l.db")


def test_import_creates_a_crate_per_playlist(xml_and_db):
    x, con = xml_and_db
    counts, skipped = import_rekordbox_xml(con, x, container_share=1.0)
    assert counts == {"house": 1, "dubstep": 1}
    assert skipped == {}
    names = {r["name"] for r in con.execute("SELECT name FROM crates")}
    assert names == {"house", "dubstep"}


def test_import_skips_tracks_whose_audio_is_gone(xml_and_db):
    """A crate of dead paths teaches nothing and shows phantom entries."""
    x, con = xml_and_db
    import_rekordbox_xml(con, x, container_share=1.0)
    assert con.execute("SELECT count(*) c FROM tracks").fetchone()["c"] == 2
    assert import_rekordbox_xml(con, x, container_share=1.0)[0]["house"] == 1


def test_imported_membership_counts_as_human_not_a_guess(xml_and_db):
    """It is the DJ's own filing, so it must not be overwritten as if automatic."""
    x, con = xml_and_db
    import_rekordbox_xml(con, x, container_share=1.0)
    sources = {r["source"] for r in con.execute("SELECT source FROM assignments")}
    assert sources == {"human"}


def test_importing_twice_does_not_duplicate(xml_and_db):
    x, con = xml_and_db
    import_rekordbox_xml(con, x, container_share=1.0)
    import_rekordbox_xml(con, x, container_share=1.0)
    assert con.execute("SELECT count(*) c FROM assignments").fetchone()["c"] == 2
    assert con.execute("SELECT count(*) c FROM crates").fetchone()["c"] == 2


def test_an_xml_with_no_playlists_is_not_a_crash(tmp_path):
    x = tmp_path / "empty.xml"
    x.write_text('<?xml version="1.0"?><DJ_PLAYLISTS Version="1.0.0">'
                 '<COLLECTION Entries="0"></COLLECTION></DJ_PLAYLISTS>')
    assert import_rekordbox_xml(connect(tmp_path / "l.db"), x) == ({}, {})


def test_folders_are_remembered(tmp_path):
    con = connect(tmp_path / "l.db")
    add_folder(con, "/Users/me/Music")
    assert get_folders(con) == ["/Users/me/Music"]


def test_adding_a_folder_twice_keeps_one(tmp_path):
    con = connect(tmp_path / "l.db")
    add_folder(con, "/Users/me/Music")
    add_folder(con, "/Users/me/Music")
    assert get_folders(con) == ["/Users/me/Music"]


def test_folders_survive_a_reopen(tmp_path):
    """Configured once, never asked for again - including across restarts."""
    p = tmp_path / "l.db"
    add_folder(connect(p), "/Users/me/Music")
    assert get_folders(connect(p)) == ["/Users/me/Music"]


def test_a_folder_can_be_removed(tmp_path):
    con = connect(tmp_path / "l.db")
    add_folder(con, "/a"); add_folder(con, "/b")
    assert remove_folder(con, "/a") == ["/b"]


def test_get_folders_on_a_fresh_database_is_empty(tmp_path):
    assert get_folders(connect(tmp_path / "l.db")) == []


def test_named_playlists_are_excluded_from_becoming_crates(tmp_path):
    """AllSongs and Contents are containers, not taste. There is no reliable
    way to detect that automatically (see bootstrap.py), so the caller names
    them and this asserts they are honoured and reported, never silently dropped."""
    a = tmp_path / "a.wav"; a.write_bytes(b"RIFF")
    b = tmp_path / "b.wav"; b.write_bytes(b"RIFF")
    x = tmp_path / "r.xml"
    x.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <COLLECTION Entries="2">
    <TRACK TrackID="1" Location="file://localhost{a}"/>
    <TRACK TrackID="2" Location="file://localhost{b}"/>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="2">
      <NODE Name="AllSongs" Type="1" Entries="2">
        <TRACK Key="1"/><TRACK Key="2"/>
      </NODE>
      <NODE Name="afro" Type="1" Entries="1"><TRACK Key="1"/></NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>""".format(a=a, b=b))

    con = connect(tmp_path / "l.db")
    counts, skipped = import_rekordbox_xml(con, x, exclude=["AllSongs"])
    assert counts == {"afro": 1}
    assert skipped == {"AllSongs": 2}
    names = {r["name"] for r in con.execute("SELECT name FROM crates")}
    assert names == {"afro"}


def test_folders_added_at_the_same_time_are_all_kept(tmp_path):
    # Two requests remembering folders at once must not each write back a
    # list that lacks the other's folder.
    import threading
    con = connect(tmp_path / "l.db")
    paths = [f"/Volumes/Drive{i}" for i in range(40)]
    threads = [threading.Thread(target=add_folder, args=(con, p)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(get_folders(con)) == sorted(paths)
