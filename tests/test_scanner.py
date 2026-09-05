from crateapp.db import connect
from crateapp.scanner import scan, pending


def make_audio(folder, name, content=b"RIFF0000WAVE"):
    p = folder / name
    p.write_bytes(content)
    return p


def test_scan_registers_audio_files(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    make_audio(music, "a.wav"); make_audio(music, "b.mp3")
    con = connect(tmp_path / "l.db")
    assert scan(con, music)["added"] == 2
    assert con.execute("SELECT count(*) c FROM tracks").fetchone()["c"] == 2


def test_scan_ignores_non_audio(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    make_audio(music, "a.wav")
    (music / "notes.txt").write_text("hello")
    (music / "cover.jpg").write_bytes(b"\xff\xd8")
    con = connect(tmp_path / "l.db")
    assert scan(con, music)["added"] == 1


def test_rescanning_adds_nothing(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    make_audio(music, "a.wav")
    con = connect(tmp_path / "l.db")
    scan(con, music)
    second = scan(con, music)
    assert second["added"] == 0 and second["unchanged"] == 1


def test_scan_finds_files_in_subfolders(tmp_path):
    music = tmp_path / "music"; (music / "sub").mkdir(parents=True)
    make_audio(music / "sub", "deep.wav")
    con = connect(tmp_path / "l.db")
    assert scan(con, music)["added"] == 1


def test_an_edited_file_is_marked_for_reanalysis(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    p = make_audio(music, "a.wav")
    con = connect(tmp_path / "l.db")
    scan(con, music)
    con.execute("UPDATE tracks SET analysed_at='2026-01-01'")
    con.commit()
    p.write_bytes(b"RIFF0000WAVELONGER")     # size changes
    assert scan(con, music)["changed"] == 1
    assert len(pending(con)) == 1


def test_a_vanished_file_is_flagged_not_deleted(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    p = make_audio(music, "a.wav")
    con = connect(tmp_path / "l.db")
    scan(con, music)
    p.unlink()
    assert scan(con, music)["missing"] == 1
    assert con.execute("SELECT count(*) c FROM tracks").fetchone()["c"] == 1


def test_pending_lists_only_unanalysed(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    make_audio(music, "a.wav"); make_audio(music, "b.wav")
    con = connect(tmp_path / "l.db")
    scan(con, music)
    assert len(pending(con)) == 2
    con.execute("UPDATE tracks SET analysed_at='now' WHERE filename='a.wav'")
    con.commit()
    assert [r["filename"] for r in pending(con)] == ["b.wav"]
