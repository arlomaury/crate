"""Crate membership: filing classifier results, and applying the DJ's
corrections. Crates are virtual - membership lives in SQLite rows. This
module never touches the filesystem: no file is ever moved, copied, or
created."""


def ensure_crate(con, name):
    """Get-or-create a crate by name. Idempotent.

    Only genuinely meaningless names (empty or whitespace-only) are
    rejected here. A crate is free to be called anything else, including
    something like "Drum & Bass / Jungle" that would not be a safe folder
    name - that sanitisation belongs at the point a crate name becomes a
    filesystem path (crateapp.exporters), not here in the database layer.
    """
    if name is None or not name.strip():
        raise ValueError("crate name cannot be empty")
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

    Re-classifying a track (e.g. after a model rebuild) must not leave stale
    auto rows behind: any previous source='auto' assignment for this track is
    cleared first, so a track re-filed into a different crate ends up with
    exactly one auto row, not two.
    """
    con.execute("DELETE FROM assignments WHERE track_id=? AND source='auto'",
               (track_id,))
    if result.get("crate") is None:
        con.commit()
        return
    cid = ensure_crate(con, result["crate"])
    con.execute(
        "INSERT OR REPLACE INTO assignments "
        "(track_id, crate_id, source, confidence, band, disputed) "
        "VALUES (?,?,'auto',?,?,?)",
        (track_id, cid, result.get("similarity"), result.get("band"),
         1 if result.get("disputed") else 0))
    con.commit()


def correct(con, track_id, to_crate=None, mode="move", was_error=None):
    """Apply a human decision about a track's crate membership.

    mode='move' - the track leaves its current primary crate (its 'auto'
        assignment, or a previous move's 'human' assignment - see below) for
        `to_crate`. The model's original placement was therefore wrong: this
        is ALWAYS recorded as an error (was_error=1), regardless of what the
        caller passes for `was_error` - there is no CHECK constraint
        enforcing this in the schema, so the guarantee lives here.

        A move clears the track's previous *primary* placement, which is
        either its source='auto' row, or a source='human' row left by an
        earlier move (band='confident' - the same band a move has always
        written). A second move must replace the first, not pile up next to
        it, or a track corrected twice ends up filed in both crates at once.
        Deliberate also-adds (mode='add', band='added') are the DJ's own,
        separate statements and are never destroyed as collateral damage of
        an unrelated correction - a move to fix the genre must not silently
        undo a separate, correct also-add.
    mode='add'  - the track additionally joins `to_crate`, keeping its
        existing membership(s), recorded with band='added' so a later move
        knows not to remove it. Whether this counts as an error is the DJ's
        own call: "you were right, and it belongs here too" (not an error)
        or "you were wrong, but I'll leave it where it is too" (an error).
        The caller's `was_error` is honoured as-is for adds.

        Acting on a track is acting on it: an also-add resolves any pending
        review the same way a move does, so it clears the track's auto row
        out of uncertain() too. It does this by marking the auto row's band
        'confirmed' rather than deleting it - the track must stay filed in
        the crate it was auto-assigned to, only the "still needs a look"
        flag is cleared.
    mode='confirm' - the classifier already put it in the right crate. No
        membership changes; `to_crate` is ignored and may be omitted, because
        the answer is wherever the track already is.

        This does more than dismiss the row. The provisional 'auto' placement
        is promoted to a source='human' one, which means a confirmed track
        becomes training data - and with only a few hundred labelled
        recordings to learn from, a confirmation is worth as much to the model
        as a correction. Recorded with was_error=0: the model was right, and
        counting confirmations as failures would make its accuracy look worse
        the more the DJ agreed with it.

        Only tracks the classifier actually filed can be confirmed. A track it
        gave up on (unknown, filed nowhere) has no placement to agree with, so
        confirming one is a caller error, not a silent no-op.
    """
    if mode not in ("move", "add", "confirm"):
        raise ValueError("mode must be 'move', 'add' or 'confirm'")

    if mode == "confirm":
        row = con.execute(
            "SELECT crate_id FROM assignments "
            "WHERE track_id=? AND source='auto'", (track_id,)).fetchone()
        if row is None:
            raise ValueError(
                "nothing to confirm: this track was not filed automatically")
        to_id = row["crate_id"]
        con.execute("DELETE FROM assignments WHERE track_id=? AND source='auto'",
                    (track_id,))
        con.execute(
            "INSERT OR REPLACE INTO assignments "
            "(track_id, crate_id, source, confidence, band) "
            "VALUES (?,?,'human',1.0,'confident')", (track_id, to_id))
        con.execute(
            "INSERT INTO corrections (track_id, from_crate, to_crate, was_error) "
            "VALUES (?,?,?,0)", (track_id, to_id, to_id))
        con.commit()
        return

    to_id = ensure_crate(con, to_crate)

    from_id = None
    if mode == "move":
        primary = ("source='auto' OR (source='human' AND band='confident')")
        row = con.execute(
            f"SELECT crate_id FROM assignments WHERE track_id=? AND ({primary})",
            (track_id,)).fetchone()
        from_id = row["crate_id"] if row else None
        con.execute(
            f"DELETE FROM assignments WHERE track_id=? AND ({primary})",
            (track_id,))
        was_error = True  # a move is always a correction of a wrong auto placement
        band = "confident"
    else:
        con.execute(
            "UPDATE assignments SET band='confirmed' "
            "WHERE track_id=? AND source='auto' AND band='uncertain'",
            (track_id,))
        band = "added"

    con.execute(
        "INSERT OR REPLACE INTO assignments "
        "(track_id, crate_id, source, confidence, band) "
        "VALUES (?,?,'human',1.0,?)", (track_id, to_id, band))
    con.execute(
        "INSERT INTO corrections (track_id, from_crate, to_crate, was_error) "
        "VALUES (?,?,?,?)", (track_id, from_id, to_id, 1 if was_error else 0))
    con.commit()


def uncertain(con):
    """Tracks filed provisionally by the classifier and still awaiting a
    human verdict. They are real, browsable/exportable members of their
    crate - this is a view over that provisional state, not a holding pen."""
    # Carries the crate it was filed into, so the UI can default its picker to
    # where the track actually is. Defaulting to the alphabetically-first crate
    # made "Move" silently misfile anything the DJ did not manually re-pick.
    return con.execute(
        "SELECT t.*, c.name AS crate FROM tracks t "
        "JOIN assignments a ON a.track_id=t.id "
        "JOIN crates c ON c.id=a.crate_id "
        "WHERE a.band='uncertain' AND a.source='auto' ORDER BY t.id").fetchall()


def disputed(con):
    """Confidently filed, but the two signals wanted different crates.

    These never surface in uncertain() - the model is sure - which is exactly
    the problem: a confident mistake is invisible. Measured on this library,
    25% of these are genuinely misfiled against 2.9% of the confident tracks
    where both signals agree, so a short list of them is the cheapest place
    left to find real errors.
    """
    return con.execute(
        "SELECT t.*, c.name AS crate FROM tracks t "
        "JOIN assignments a ON a.track_id=t.id "
        "JOIN crates c ON c.id=a.crate_id "
        "WHERE a.source='auto' AND a.band='confident' AND a.disputed=1 "
        "ORDER BY t.id").fetchall()


def unsorted(con):
    """Analysed tracks that matched no crate at all - the classifier reached
    a verdict of 'unknown' and it was filed nowhere rather than forced into
    an existing crate. A track that simply hasn't been analysed yet is not
    unsorted, it is pending - `analysed_at IS NOT NULL` keeps a fresh scan
    from burying genuine unknowns under everything still waiting its turn."""
    return con.execute(
        "SELECT t.* FROM tracks t LEFT JOIN assignments a ON a.track_id=t.id "
        "WHERE a.track_id IS NULL AND t.analysed_at IS NOT NULL AND t.missing=0 "
        "ORDER BY t.id").fetchall()
