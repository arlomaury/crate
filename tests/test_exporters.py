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


# --------------------------------------------------------- fix round 1


def test_export_folders_confines_hostile_crate_names_to_dest(con, tmp_path):
    """A crate named '/etc/passwd' or '../escape' must not let Path's '/'
    operator (which discards the left side for an absolute right side) or a
    literal '..' write outside the destination directory."""
    music = tmp_path / "music"
    evil_names = ["/etc/passwd", "../escape", "Drum & Bass / Jungle"]
    for i, name in enumerate(evil_names, start=2):
        p = music / f"song{i}.wav"
        p.write_bytes(b"DATA" + str(i).encode())
        con.execute(
            "INSERT INTO tracks (id, path, filename, bpm, camelot, duration_sec) "
            "VALUES (?, ?, ?, 128.0, '8A', 200)", (i, str(p), f"song{i}.wav"))
        con.commit()
        correct(con, i, name, mode="move")

    dest = tmp_path / "out"
    counts = export_folders(con, dest)

    assert sum(counts.values()) == 4  # song.wav + the three evil crates
    dest_resolved = dest.resolve()
    found = list(dest_resolved.rglob("*"))
    assert found, "nothing was exported"
    for path in found:
        # every single file/dir written must resolve to somewhere inside dest
        assert dest_resolved == path.resolve() or dest_resolved in path.resolve().parents


def test_export_folders_disambiguates_basename_collisions(con, tmp_path):
    """Two different tracks sharing a filename must not silently collapse
    into one exported file with an inflated count."""
    music = tmp_path / "music"
    p2 = music / "dup.wav"; p2.write_bytes(b"AAAA")
    sub = music / "sub"; sub.mkdir()
    p3 = sub / "dup.wav"; p3.write_bytes(b"BBBB")
    con.execute("INSERT INTO tracks (id, path, filename, bpm, camelot, duration_sec) "
                "VALUES (2, ?, 'dup.wav', 128.0, '8A', 200)", (str(p2),))
    con.execute("INSERT INTO tracks (id, path, filename, bpm, camelot, duration_sec) "
                "VALUES (3, ?, 'dup.wav', 128.0, '8A', 200)", (str(p3),))
    con.commit()
    correct(con, 2, "House", mode="move")
    correct(con, 3, "House", mode="move")

    dest = tmp_path / "out"
    counts = export_folders(con, dest)

    assert counts["House"] == 3  # song.wav (id 1) + two distinct dup.wav
    files = list((dest / "House").iterdir())
    assert len(files) == 3
    contents = {f.read_bytes() for f in files}
    assert b"AAAA" in contents and b"BBBB" in contents


def test_export_folders_cleans_up_a_failed_copy_so_a_retry_finishes_it(
        con, tmp_path, monkeypatch):
    """A copy that fails partway must leave no truncated file behind, so a
    resumed run cannot mistake a partial copy for a finished one."""
    import crateapp.exporters as exporters

    dest = tmp_path / "out"
    real_copy = exporters._copy
    calls = {"n": 0}

    def flaky_copy(src, tmp):
        calls["n"] += 1
        if calls["n"] == 1:
            tmp.write_bytes(b"TRUNCATED")
            raise RuntimeError("simulated copy failure")
        real_copy(src, tmp)

    monkeypatch.setattr(exporters, "_copy", flaky_copy)

    with pytest.raises(RuntimeError):
        export_folders(con, dest)

    target = dest / "House" / "song.wav"
    assert not target.exists()
    assert not any((dest / "House").glob("*.part-*"))

    counts = export_folders(con, dest)
    assert counts == {"House": 1}
    src = con.execute("SELECT path FROM tracks WHERE id=1").fetchone()["path"]
    assert target.read_bytes() == open(src, "rb").read()


def test_rekordbox_export_escapes_quotes_and_ampersands(con, tmp_path):
    wav = tmp_path / "weird.wav"; wav.write_bytes(b"DATA")
    con.execute("INSERT INTO tracks (id, path, filename, bpm, camelot, duration_sec) "
                "VALUES (9, ?, 'weird.wav', 128.0, '8A', 200)", (str(wav),))
    con.execute("INSERT INTO analysis (track_id, moments) VALUES (9, ?)",
                (json.dumps([{"label": 'Say "Hi" & Bye', "time": 1.0,
                              "type": "drop"}]),))
    con.commit()
    correct(con, 9, 'Rock & "Roll"', mode="move")

    out = tmp_path / "rekordbox.xml"
    export_rekordbox(con, out)

    # ElementTree raises ParseError outright if the quote wasn't escaped.
    root = ET.parse(out).getroot()
    genres = {t.get("Genre") for t in root.findall("COLLECTION/TRACK")}
    assert 'Rock & "Roll"' in genres
    node = root.find("PLAYLISTS/NODE/NODE[@Name='Rock & \"Roll\"']")
    assert node is not None
    marks = root.findall("COLLECTION/TRACK/POSITION_MARK")
    assert any(m.get("Name") == 'Say "Hi" & Bye' for m in marks)
