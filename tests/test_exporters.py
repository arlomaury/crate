import json
import xml.etree.ElementTree as ET
import pytest
from crateapp.db import connect
from crateapp.crates import correct
from crateapp.exporters import export_folders, export_rekordbox


@pytest.fixture
def con(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    p = music / "song.wav"; p.write_bytes(b"RIFF0000WAVEDATA")
    c = connect(tmp_path / "l.db")
    c.execute("INSERT INTO tracks (id, path, filename, bpm, camelot, duration_sec) "
              "VALUES (1, ?, 'song.wav', 128.0, '8A', 200)", (str(p),))
    c.execute("INSERT INTO analysis (track_id, moments) VALUES (1, ?)",
              (json.dumps([{"label": "Drop 1", "time": 60.0, "type": "drop"}]),))
    c.commit()
    correct(c, 1, "House", mode="move")
    return c


def test_export_folders_copies_into_named_crates(con, tmp_path):
    dest = tmp_path / "out"
    assert export_folders(con, dest) == {"House": 1}
    assert (dest / "House" / "song.wav").exists()


def test_exported_copies_are_byte_identical(con, tmp_path):
    dest = tmp_path / "out"
    export_folders(con, dest)
    src = con.execute("SELECT path FROM tracks WHERE id=1").fetchone()["path"]
    assert (dest / "House" / "song.wav").read_bytes() == open(src, "rb").read()


def test_export_never_touches_the_original(con, tmp_path):
    src = con.execute("SELECT path FROM tracks WHERE id=1").fetchone()["path"]
    before = open(src, "rb").read()
    export_folders(con, tmp_path / "out")
    assert open(src, "rb").read() == before


def test_rekordbox_export_lists_the_tracks(con, tmp_path):
    out = tmp_path / "rekordbox.xml"
    assert export_rekordbox(con, out) == 1
    root = ET.parse(out).getroot()
    assert root.find("COLLECTION").get("Entries") == "1"
    assert root.find("COLLECTION/TRACK").get("Genre") == "House"


def test_rekordbox_export_contains_a_real_playlist(con, tmp_path):
    """An empty playlist tree makes the import look empty in Rekordbox."""
    out = tmp_path / "rekordbox.xml"
    export_rekordbox(con, out)
    root = ET.parse(out).getroot()
    node = root.find("PLAYLISTS/NODE/NODE")
    assert node is not None and node.get("Name") == "House"
    assert len(node.findall("TRACK")) == 1


def test_rekordbox_export_writes_coloured_hot_cues(con, tmp_path):
    out = tmp_path / "rekordbox.xml"
    export_rekordbox(con, out)
    marks = ET.parse(out).getroot().findall("COLLECTION/TRACK/POSITION_MARK")
    hot = [m for m in marks if m.get("Num") != "-1"]
    assert hot and hot[0].get("Red") is not None
