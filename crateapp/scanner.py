"""Find audio on disk and register it. Reads only - never writes to the library."""
from pathlib import Path

from analyze import AUDIO_EXT


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
                "INSERT INTO tracks (path, filename, size, mtime) VALUES (?,?,?,?)",
                (path, p.name, st.st_size, st.st_mtime))
            stats["added"] += 1
        elif row["size"] != st.st_size or row["mtime"] != st.st_mtime:
            # Content changed, so any cached analysis is stale.
            con.execute("UPDATE tracks SET size=?, mtime=?, analysed_at=NULL, "
                        "missing=0 WHERE id=?", (st.st_size, st.st_mtime, row["id"]))
            stats["changed"] += 1
        else:
            con.execute("UPDATE tracks SET missing=0 WHERE id=?", (row["id"],))
            stats["unchanged"] += 1

    # A file that has gone is flagged, never deleted - its crate membership and
    # corrections are still worth keeping if the drive comes back.
    for row in con.execute("SELECT id, path FROM tracks WHERE missing=0"):
        if row["path"] not in seen and row["path"].startswith(str(folder)):
            con.execute("UPDATE tracks SET missing=1 WHERE id=?", (row["id"],))
            stats["missing"] += 1

    con.commit()
    return stats


def pending(con):
    """Tracks still needing analysis, oldest first."""
    return con.execute(
        "SELECT * FROM tracks WHERE analysed_at IS NULL AND missing=0 ORDER BY id"
    ).fetchall()
