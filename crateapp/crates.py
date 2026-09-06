"""Crate membership: filing classifier results, and applying the DJ's
corrections. Crates are virtual - membership lives in SQLite rows. This
module never touches the filesystem: no file is ever moved, copied, or
created."""


def ensure_crate(con, name):
    """Get-or-create a crate by name. Idempotent."""
    con.execute("INSERT OR IGNORE INTO crates (name) VALUES (?)", (name,))
    con.commit()
    return con.execute("SELECT id FROM crates WHERE name=?", (name,)).fetchone()["id"]


def auto_assign(con, track_id, result):
    """File a classifier result (the dict returned by Classifier.classify).

    Confident and uncertain results are both real memberships - an uncertain
    track still belongs to its crate and browses/exports with it; it is
    merely also surfaced by uncertain() for review. An unknown result
    (crate is None) is filed nowhere: it is left for unsorted() rather than
    forced into a crate the DJ never played.
    """
    if result.get("crate") is None:
        return
    cid = ensure_crate(con, result["crate"])
    con.execute(
        "INSERT OR REPLACE INTO assignments "
        "(track_id, crate_id, source, confidence, band) VALUES (?,?,'auto',?,?)",
        (track_id, cid, result.get("similarity"), result.get("band")))
    con.commit()


def correct(con, track_id, to_crate, mode="move", was_error=None):
    """Apply a human decision about a track's crate membership.

    mode='move' - the track leaves its current (auto) crate for `to_crate`.
        The model's original placement was therefore wrong: this is ALWAYS
        recorded as an error (was_error=1), regardless of what the caller
        passes for `was_error` - there is no CHECK constraint enforcing this
        in the schema, so the guarantee lives here.
    mode='add'  - the track additionally joins `to_crate`, keeping its
        existing membership(s). Whether this counts as an error is the DJ's
        own call: "you were right, and it belongs here too" (not an error)
        or "you were wrong, but I'll leave it where it is too" (an error).
        The caller's `was_error` is honoured as-is for adds.
    """
    if mode not in ("move", "add"):
        raise ValueError("mode must be 'move' or 'add'")
    to_id = ensure_crate(con, to_crate)

    from_id = None
    if mode == "move":
        row = con.execute(
            "SELECT crate_id FROM assignments WHERE track_id=? AND source='auto'",
            (track_id,)).fetchone()
        from_id = row["crate_id"] if row else None
        con.execute("DELETE FROM assignments WHERE track_id=?", (track_id,))
        was_error = True  # a move is always a correction of a wrong auto placement

    con.execute(
        "INSERT OR REPLACE INTO assignments "
        "(track_id, crate_id, source, confidence, band) "
        "VALUES (?,?,'human',1.0,'confident')", (track_id, to_id))
    con.execute(
        "INSERT INTO corrections (track_id, from_crate, to_crate, was_error) "
        "VALUES (?,?,?,?)", (track_id, from_id, to_id, 1 if was_error else 0))
    con.commit()


def uncertain(con):
    """Tracks filed provisionally by the classifier and still awaiting a
    human verdict. They are real, browsable/exportable members of their
    crate - this is a view over that provisional state, not a holding pen."""
    return con.execute(
        "SELECT t.* FROM tracks t JOIN assignments a ON a.track_id=t.id "
        "WHERE a.band='uncertain' AND a.source='auto' ORDER BY t.id").fetchall()


def unsorted(con):
    """Tracks that matched no crate at all - the classifier judged 'unknown'
    (or the track otherwise has no membership) and it was filed nowhere
    rather than forced into an existing crate."""
    return con.execute(
        "SELECT t.* FROM tracks t LEFT JOIN assignments a ON a.track_id=t.id "
        "WHERE a.track_id IS NULL AND t.missing=0 ORDER BY t.id").fetchall()
