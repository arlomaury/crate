"""Find audio on disk and register it. Reads only - never writes to the library."""
from contextlib import nullcontext
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


def scan(con, folder, lock=None):
    """Register new audio under `folder`; flag changed and vanished files.

    `lock` (the shared database lock) is held only while reading and writing
    the database, not while walking the folder and reading tags. Walking a
    large external drive takes seconds, and holding the lock for all of it
    froze every other request - the UI could not even load a crate while a
    scan ran. Without a lock everything runs straight through.
    """
    lock = lock if lock is not None else nullcontext()
    # Resolved, not just expanded: stored paths are always Path(p).resolve(),
    # so comparing against an unresolved caller path (e.g. a /var symlink that
    # resolves to /private/var on macOS) would make every existing track look
    # like it fell outside `folder` and get wrongly flagged missing.
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        # Typo, or an external drive that is not plugged in.  Scanning it would
        # flag every track stored under it as missing, so refuse instead.
        raise FileNotFoundError(f"folder not found: {folder}")

    # 1. What the library already knows (a quick read under the lock).
    with lock:
        known = {r["path"]: (r["size"], r["mtime"])
                 for r in con.execute("SELECT path, size, mtime FROM tracks")}

    # 2. The slow part, with no lock held: walk the folder and read the tags
    #    of anything new or changed.
    found = []          # (path, filename, size, mtime, tags or None)
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT or p.name.startswith("._"):
            continue
        try:
            path = str(p.resolve())
            st = p.stat()
        except OSError:
            continue        # broken shortcut or unreadable file: skip it, keep scanning
        old = known.get(path)
        fresh = old is None or old != (st.st_size, st.st_mtime)
        found.append((path, p.name, st.st_size, st.st_mtime, read_tags(path) if fresh else None))

    # 3. Write it all under the lock. Each row is looked up again here, since
    #    the library may have changed while the folder was being walked.
    stats = {"added": 0, "unchanged": 0, "changed": 0, "missing": 0}
    seen = set()
    with lock:
        for path, name, size, mtime, tags in found:
            seen.add(path)
            row = con.execute("SELECT id, size, mtime FROM tracks WHERE path=?",
                              (path,)).fetchone()
            if row is not None and row["size"] == size and row["mtime"] == mtime:
                con.execute("UPDATE tracks SET missing=0 WHERE id=?", (row["id"],))
                stats["unchanged"] += 1
                continue
            artist, genre = tags if tags is not None else read_tags(path)
            if row is None:
                con.execute(
                    "INSERT INTO tracks (path, filename, size, mtime, artist, "
                    "remixer, genre_tag) VALUES (?,?,?,?,?,?,?)",
                    (path, name, size, mtime, artist, credited_remixer(name), genre))
                stats["added"] += 1
            else:
                # Content changed, so any cached analysis is stale - and so
                # are the tags, which is why they are re-read and not only on
                # first sight.
                con.execute("UPDATE tracks SET size=?, mtime=?, analysed_at=NULL, "
                            "missing=0, artist=?, remixer=?, genre_tag=? "
                            "WHERE id=?",
                            (size, mtime, artist, credited_remixer(name), genre, row["id"]))
                stats["changed"] += 1

        # A file that has gone is flagged, never deleted - its crate membership
        # and corrections are still worth keeping if the drive comes back.
        gone = [r["id"] for r in con.execute("SELECT id, path FROM tracks WHERE missing=0")
                if r["path"] not in seen and Path(r["path"]).is_relative_to(folder)]
        for track_id in gone:
            con.execute("UPDATE tracks SET missing=1 WHERE id=?", (track_id,))
        stats["missing"] = len(gone)
        con.commit()
    return stats


def pending(con, include_unembedded=False):
    """Tracks still needing analysis, oldest first.

    With `include_unembedded`, also tracks that were analysed cleanly but
    have no embedding - analysed before the genre models were installed.
    Without this they kept their tempo and key but could never be sorted,
    since a track is analysed once."""
    sql = "SELECT * FROM tracks WHERE missing=0 AND (analysed_at IS NULL"
    if include_unembedded:
        sql += (" OR (error IS NULL AND id NOT IN (SELECT track_id FROM embeddings))")
    return con.execute(sql + ") ORDER BY id").fetchall()
