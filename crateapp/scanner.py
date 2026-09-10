"""Find audio on disk and register it. Reads only - never writes to the library."""
from pathlib import Path

from analyze import AUDIO_EXT


def read_artist(path):
    """The artist tag, normalised for comparison, or None.

    Costs about a millisecond a file - the whole 1,600-track library reads in
    under a second - because it only parses the tag header, never the audio.
    Worth it: which artist a track is by turns out to predict the DJ's filing
    far better than the audio does (see crates.artist_crate).
    """
    from crateapp.crates import normalise_artist
    try:
        import essentia.standard as es
        md = es.MetadataReader(filename=str(path), failOnError=False)()
        return normalise_artist(md[1])
    except Exception:
        return None


def scan(con, folder):
    """Register new audio under `folder`; flag changed and vanished files."""
    # Resolved, not just expanded: stored paths are always Path(p).resolve(),
    # so comparing against an unresolved caller path (e.g. a /var symlink that
    # resolves to /private/var on macOS) would make every existing track look
    # like it fell outside `folder` and get wrongly flagged missing.
    folder = Path(folder).expanduser().resolve()
    stats = {"added": 0, "unchanged": 0, "changed": 0, "missing": 0}
    seen = set()

    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT or p.name.startswith("._"):
            continue
        path = str(p.resolve())
        seen.add(path)
        st = p.stat()
        row = con.execute("SELECT id, size, mtime FROM tracks WHERE path=?",
                          (path,)).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO tracks (path, filename, size, mtime, artist) "
                "VALUES (?,?,?,?,?)",
                (path, p.name, st.st_size, st.st_mtime, read_artist(path)))
            stats["added"] += 1
        elif row["size"] != st.st_size or row["mtime"] != st.st_mtime:
            # Content changed, so any cached analysis is stale.
            con.execute("UPDATE tracks SET size=?, mtime=?, analysed_at=NULL, "
                        "missing=0, artist=? WHERE id=?",
                        (st.st_size, st.st_mtime, read_artist(path), row["id"]))
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
