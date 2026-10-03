"""Find audio on disk and register it. Reads only - never writes to the library."""
from pathlib import Path

from analyze import AUDIO_EXT


from crateapp.crates import credited_remixer  # noqa: E402  (small, no cycle)


def read_tags(path):
    """`(artist, genre)` from the file's own tags, or `(None, None)`.

    Costs about a millisecond a file - the whole 1,600-track library reads in
    under a second - because only the tag header is parsed, never the audio.
    Worth far more than that: the genre written in by whoever sold the track
    predicts the DJ's filing 90.6% of the time, and 98.2% on house vs tech
    house, which no amount of audio analysis managed above ~76%.

    MetadataReader returns (title, artist, album, comment, genre, track, date,
    pool). The genre is index 4 - index 5 is the track NUMBER, and reading
    that one instead produces a rule that confidently maps "1" to a crate.
    """
    from crateapp.crates import normalise_artist
    try:
        import essentia.standard as es
        md = es.MetadataReader(filename=str(path), failOnError=False)()
        genre = (md[4] or "").strip().lower() or None
        return normalise_artist(md[1]), genre
    except Exception:
        return None, None


def scan(con, folder):
    """Register new audio under `folder`; flag changed and vanished files."""
    # Resolved, not just expanded: stored paths are always Path(p).resolve(),
    # so comparing against an unresolved caller path (e.g. a /var symlink that
    # resolves to /private/var on macOS) would make every existing track look
    # like it fell outside `folder` and get wrongly flagged missing.
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        # Typo, or an external drive that is not plugged in.  Scanning it would
        # flag every track stored under it as missing, so refuse instead.
        raise FileNotFoundError(f"folder not found: {folder}")
    stats = {"added": 0, "unchanged": 0, "changed": 0, "missing": 0}
    seen = set()

    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT or p.name.startswith("._"):
            continue
        try:
            path = str(p.resolve())
            st = p.stat()
        except OSError:
            continue        # broken shortcut or unreadable file: skip it, keep scanning
        seen.add(path)
        row = con.execute("SELECT id, size, mtime FROM tracks WHERE path=?",
                          (path,)).fetchone()
        if row is None:
            artist, genre = read_tags(path)
            con.execute(
                "INSERT INTO tracks (path, filename, size, mtime, artist, "
                "remixer, genre_tag) VALUES (?,?,?,?,?,?,?)",
                (path, p.name, st.st_size, st.st_mtime, artist,
                 credited_remixer(p.name), genre))
            stats["added"] += 1
        elif row["size"] != st.st_size or row["mtime"] != st.st_mtime:
            # Content changed, so any cached analysis is stale - and so are
            # the tags, which is why they are re-read here and not only on
            # first sight.
            artist, genre = read_tags(path)
            con.execute("UPDATE tracks SET size=?, mtime=?, analysed_at=NULL, "
                        "missing=0, artist=?, remixer=?, genre_tag=? "
                        "WHERE id=?",
                        (st.st_size, st.st_mtime, artist,
                         credited_remixer(p.name), genre, row["id"]))
            stats["changed"] += 1
        else:
            con.execute("UPDATE tracks SET missing=0 WHERE id=?", (row["id"],))
            stats["unchanged"] += 1

    # A file that has gone is flagged, never deleted - its crate membership and
    # corrections are still worth keeping if the drive comes back.
    for row in con.execute("SELECT id, path FROM tracks WHERE missing=0"):
        if row["path"] not in seen and Path(row["path"]).is_relative_to(folder):
            con.execute("UPDATE tracks SET missing=1 WHERE id=?", (row["id"],))
            stats["missing"] += 1

    con.commit()
    return stats


def pending(con):
    """Tracks still needing analysis, oldest first."""
    return con.execute(
        "SELECT * FROM tracks WHERE analysed_at IS NULL AND missing=0 ORDER BY id"
    ).fetchall()
