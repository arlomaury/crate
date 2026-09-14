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


def test_scanning_one_folder_does_not_flag_a_sibling_with_a_shared_prefix(tmp_path):
    music = tmp_path / "Music"; music.mkdir()
    other = tmp_path / "MusicOld"; other.mkdir()
    (music / "a.wav").write_bytes(b"RIFF0000WAVE")
    (other / "b.wav").write_bytes(b"RIFF0000WAVE")
    con = connect(tmp_path / "l.db")
    scan(con, music)
    scan(con, other)
    scan(con, music)          # rescanning Music must not touch MusicOld's track
    row = con.execute("SELECT missing FROM tracks WHERE filename='b.wav'").fetchone()
    assert row["missing"] == 0


def test_pending_lists_only_unanalysed(tmp_path):
    music = tmp_path / "music"; music.mkdir()
    make_audio(music, "a.wav"); make_audio(music, "b.wav")
    con = connect(tmp_path / "l.db")
    scan(con, music)
    assert len(pending(con)) == 2
    con.execute("UPDATE tracks SET analysed_at='now' WHERE filename='a.wav'")
    con.commit()
    assert [r["filename"] for r in pending(con)] == ["b.wav"]


def test_read_tags_takes_genre_from_the_right_field(tmp_path):
    """MetadataReader returns (title, artist, album, comment, genre, track,
    date, ...). Genre is index 4; index 5 is the track NUMBER. Reading 5 by
    mistake produces a rule that confidently maps the tag "1" to a crate,
    which is exactly what happened once."""
    from crateapp.scanner import read_tags
    import wave
    f = tmp_path / "x.wav"
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    artist, genre = read_tags(f)
    # An untagged file must yield nothing rather than a stray number.
    assert genre is None or not genre.strip().isdigit()


def test_read_tags_survives_an_unreadable_file(tmp_path):
    from crateapp.scanner import read_tags
    f = tmp_path / "broken.wav"
    f.write_bytes(b"not audio at all")
    assert read_tags(f) == (None, None)
