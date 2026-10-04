"""Crate membership: filing classifier results, and applying the DJ's
corrections. Crates are virtual - membership lives in SQLite rows. This
module never touches the filesystem: no file is ever moved, copied, or
created."""
import re


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


def set_locked(con, name, locked=True):
    """Curate a crate by hand only.

    A locked crate is invisible to every automatic route: nothing is filed
    into it, and nothing inside it is evidence for any rule. That second half
    matters more than it looks - without it, a crate holding one favourite
    track teaches "this artist belongs here" and the rules duly start filing
    that artist's whole catalogue into it.

    The DJ can still move or add tracks by hand; locking constrains the tool,
    not them.
    """
    cur = con.execute("UPDATE crates SET locked=? WHERE name=?",
                      (1 if locked else 0, name))
    con.commit()
    return cur.rowcount > 0


def is_locked(con, name):
    row = con.execute("SELECT locked FROM crates WHERE name=?", (name,)).fetchone()
    return bool(row and row["locked"])


def locked_names(con):
    return [r["name"] for r in
            con.execute("SELECT name FROM crates WHERE locked=1 ORDER BY name")]


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
    if con.execute("SELECT 1 FROM assignments WHERE track_id=? AND source='human'",
                   (track_id,)).fetchone():
        # The DJ has acted on this track. Re-analysis (its tags were edited,
        # say) must never replace, add to or remove any of its filing - not
        # their own rows, which are the training data, and not an automatic
        # placement they reviewed and kept with an also-add.
        return
    con.execute("DELETE FROM assignments WHERE track_id=? AND source='auto'",
               (track_id,))
    crate = result.get("crate")
    if crate is None or is_locked(con, crate):
        # A locked crate is the DJ's alone. Refused here as well as upstream,
        # so no future caller can route around it.
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
        is recorded as an error (was_error=1), regardless of what the caller
        passes for `was_error` - there is no CHECK constraint enforcing this
        in the schema, so the guarantee lives here. The one exception is a
        move to a crate the track is already in: that is agreement ("this
        crate, and only this one"), recorded with was_error=0.

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
    # Checked before anything is written: a stale screen (the track removed in
    # another tab) must not leave behind an empty crate created for it.
    if con.execute("SELECT 1 FROM tracks WHERE id=?", (track_id,)).fetchone() is None:
        raise ValueError("no such track - it may have been removed; refresh the page")

    if mode == "confirm":
        row = con.execute(
            "SELECT crate_id FROM assignments "
            "WHERE track_id=? AND source='auto'", (track_id,)).fetchone()
        if row is None and to_crate:
            # Confirming a track that is already the DJ's own filing (they
            # pressed Move on a crate it is in). Nothing changes; it is
            # recorded so the agreement is not lost.
            row = con.execute(
                "SELECT a.crate_id FROM assignments a JOIN crates c "
                "ON c.id=a.crate_id WHERE a.track_id=? AND c.name=?",
                (track_id, to_crate)).fetchone()
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

    # Moving a track to the crate it is already in is agreement, not a
    # correction. The picker defaults to where the track already is, so this
    # is what happens when the DJ works the queue with Move instead of
    # Correct - and recording it as the model being wrong both understates
    # the measured accuracy and teaches the learning loop the opposite of
    # what they meant. 35 of the first 152 corrections were this.
    # Only the primary placement counts as "already there": a move onto a
    # crate the track was merely also-added to still corrects the model.
    already = con.execute(
        "SELECT 1 FROM assignments WHERE track_id=? AND crate_id=? AND "
        "(source='auto' OR (source='human' AND band='confident'))",
        (track_id, to_id)).fetchone() is not None

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
        # A move is a correction of a wrong placement - unless the track was
        # already in that crate, which is the DJ saying "this one, and only
        # this one". It still becomes their filing (and leaves every other
        # primary crate), but it is agreement, not an error.
        was_error = not already
        band = "confident"
    else:
        # Reviewed: the automatic placement stays, and is no longer flagged.
        con.execute(
            "UPDATE assignments SET band='confirmed', disputed=0 "
            "WHERE track_id=? AND source='auto' AND (band='uncertain' OR disputed=1)",
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


def normalise_artist(name):
    """A comparable key for an artist tag.

    Collapses the credit to its FIRST named artist and strips punctuation, so
    "Sammy Virji, Issey Cross", "Sammy Virji & Friends" and "sammy virji" all
    meet. Features and remixer credits vary far too much to match on the whole
    string.
    """
    import re
    a = (name or "").strip().lower()
    if not a:
        return None
    a = re.split(r"\s*(?:,|&|feat\.|ft\.|featuring|vs\.?|\bx\b|with )\s*", a)[0]
    a = re.sub(r"[^a-z0-9]+", "", a)
    return a or None


# "(Gorgon City Remix)", "(Boucho UKG Edit)", "(WESH Bootleg)". The artist
# TAG on these names the original act, but a remix is the remixer's record -
# it is their drums, their bassline, their genre. Keyed on the remixer this
# rule was right 10 times out of 10 on the labelled set; keyed on the original
# artist, remixes are one of the ways it goes wrong.
REMIX_CREDIT = re.compile(
    r"\(([^)]{2,40}?)\s+(?:remix|edit|bootleg|flip|rework|vip|refix|dub mix)\)",
    re.I)


# Words that trail a remixer's name rather than belong to it: "(Boucho UKG
# Edit)" is by Boucho. Stripped so that their UKG edit and their plain remix
# resolve to the same person. Kept to a short known list - guessing more
# aggressively would start merging genuinely different remixers.
CREDIT_NOISE = {"ukg", "vip", "extended", "club", "radio", "dub", "original",
                "instrumental", "official", "bootleg", "jungle", "dnb",
                "house", "techno", "garage", "bass", "hard", "slowed"}


def credited_remixer(filename):
    """The remixer named in a filename, normalised, or None."""
    m = REMIX_CREDIT.search(filename or "")
    if not m:
        return None
    words = [w for w in m.group(1).split()
             if w.strip(".,&").lower() not in CREDIT_NOISE]
    return normalise_artist(" ".join(words)) if words else None


# A tag has to be seen a few times before its mapping means anything, and it
# has to point mostly one way. "Hip-Hop/Rap" covers the DJ's rap tracks, their
# pop crossovers and their acapellas, so it lands at 56% and earns its keep
# only because the acapella rule runs first and takes the vocals out.
TAG_MIN_SEEN = 2
TAG_MIN_PURITY = 0.5


def tag_crate(con, genre_tag):
    """The crate the DJ files this genre tag into, learned from their filing.

    The tag is whatever the shop or the ripper wrote into the file - "Tech
    House", "UK Garage / Bassline", "Hip-Hop/Rap". Nothing is hardcoded: the
    mapping from tag to crate is read off the DJ's own human-filed tracks, so
    it follows their vocabulary rather than anyone else's taxonomy.

    Measured held-out on this library: 90.6% accurate over 95% of tracks, and
    **98.2% on house vs tech house** - a distinction three separate audio
    signal families could not push past ~76%. That is because the DJ files
    that pair by the tag in the first place, which makes it a labelling
    convention the audio was never going to recover.

    Returns (crate, n_seen, purity) or (None, 0, 0.0).
    """
    tag = (genre_tag or "").strip().lower()
    if not tag:
        return None, 0, 0.0
    # Acapellas are excluded, because the acapella rule runs before this one
    # and they never reach it. Counting them drags a tag's purity down for a
    # decision it is not being asked to make: "Hip-Hop/Rap" covers 39 rap
    # tracks, 12 pop and 32 isolated vocals here, which reads as 46% pure and
    # unusable - but 76% once the vocals are taken out by the rule that
    # actually handles them.
    rows = con.execute(
        "SELECT c.name, count(*) n FROM assignments a "
        "JOIN crates c ON c.id = a.crate_id "
        "JOIN tracks t ON t.id = a.track_id "
        "LEFT JOIN analysis an ON an.track_id = t.id "
        "WHERE a.source = 'human' AND lower(t.genre_tag) = ? "
        "  AND coalesce(json_extract(an.vocal, '$.is_acapella'), 0) = 0 "
        "  AND coalesce(c.locked, 0) = 0 "
        "GROUP BY c.name ORDER BY n DESC", (tag,)).fetchall()
    if not rows:
        return None, 0, 0.0
    total = sum(r["n"] for r in rows)
    best = rows[0]
    purity = best["n"] / total
    if total < TAG_MIN_SEEN or purity < TAG_MIN_PURITY:
        return None, 0, 0.0
    return best["name"], total, round(purity, 3)


def tag_gaps(con, limit=8):
    """Tags that would file a lot of tracks, if the DJ filed a couple first.

    The genre tag decides more tracks than everything else combined, but only
    once it has been taught: two tracks, pointing the same way. Until then
    every track carrying that tag falls through to the audio model, which is
    ~15 points worse.

    So this finds the tags doing nothing and ranks them by how many tracks
    each would settle. On this library "minimal / deep tech" sits on 34
    unfiled tracks and has never been filed once - two minutes of the DJ's
    attention against 34 tracks sorted properly.

    Returns [{tag, would_unlock, filed_so_far, track_id, filename}], richest
    first, with one representative track to act on.
    """
    rows = con.execute(
        "SELECT t.genre_tag tag, count(*) n FROM tracks t "
        "JOIN assignments a ON a.track_id = t.id AND a.source = 'auto' "
        "WHERE t.genre_tag IS NOT NULL AND trim(t.genre_tag) != '' "
        "GROUP BY lower(t.genre_tag) ORDER BY n DESC").fetchall()

    out = []
    for r in rows:
        if tag_crate(con, r["tag"])[0]:
            continue                      # already teaching us something
        # Counted exactly as tag_crate counts its evidence (no acapellas, no
        # locked crates), so "one more decides it" is true when the UI says it.
        filed = con.execute(
            "SELECT count(*) n FROM assignments a "
            "JOIN tracks t ON t.id = a.track_id "
            "JOIN crates c ON c.id = a.crate_id "
            "LEFT JOIN analysis an ON an.track_id = t.id "
            "WHERE a.source='human' AND lower(t.genre_tag) = ? "
            "  AND coalesce(json_extract(an.vocal, '$.is_acapella'), 0) = 0 "
            "  AND coalesce(c.locked, 0) = 0", (r["tag"].lower(),)).fetchone()["n"]
        pick = con.execute(
            "SELECT t.id, t.filename FROM tracks t "
            "JOIN assignments a ON a.track_id = t.id AND a.source = 'auto' "
            "WHERE lower(t.genre_tag) = ? ORDER BY t.id LIMIT 1",
            (r["tag"].lower(),)).fetchone()
        if pick is None:
            continue
        out.append({"tag": r["tag"], "would_unlock": r["n"],
                    "filed_so_far": filed, "track_id": pick["id"],
                    "filename": pick["filename"]})
        if len(out) >= limit:
            break
    return out


# Two shapes appear in this library: "Artist - Title (Extended Mix)" and
# "Title-Artist". Neither is metadata, and reading a name out of one is
# guesswork compared to a tag - which is why a name recovered this way has to
# clear a higher bar before it decides anything.
#
# This is NOT the genre-from-keywords dead end. No word in the filename is
# read as a genre. It recovers WHO made the track and then asks the DJ's own
# filing about that person - the same question the artist rule already
# answers, for the 345 auto-filed tracks whose artist tag is simply missing.
NAME_MIN_FILED = 2


def artist_from_filename(filename):
    """The artist a filename names, normalised, or None."""
    stem = re.sub(r"\.(aiff?|wav|mp3|m4a|flac|aif)$", "", filename or "",
                  flags=re.I)
    stem = re.sub(r"\s*\([^)]*\)\s*$", "", stem)      # trailing (Extended Mix)
    stem = re.sub(r"^\d{1,2}[\s.\-]+", "", stem)        # leading track number
    if " - " in stem:
        return normalise_artist(stem.split(" - ")[0])
    m = re.match(r"^[^-]{3,}-(.+)$", stem)              # Title-Artist
    return normalise_artist(m.group(1)) if m else None


def who_made_it(con, track_id):
    """`(key, crate, n)` - the person whose filing decides this track.

    The remixer first where there is one, then the credited artist. Both are
    only consulted where the DJ's own filing for that person is unanimous.
    """
    row = con.execute("SELECT filename, artist FROM tracks WHERE id=?",
                      (track_id,)).fetchone()
    if row is None:
        return None, None, 0
    rx = credited_remixer(row["filename"])
    if rx:
        crate, n = _filed_under(con, rx)
        if crate:
            return rx, crate, n
    crate, n = artist_crate(con, row["artist"])
    if crate:
        return row["artist"], crate, n

    # No usable artist tag. The filename usually still names them.
    guessed = artist_from_filename(row["filename"])
    if guessed:
        crate, n = _filed_under(con, guessed)
        if crate and n >= NAME_MIN_FILED:
            return guessed, crate, n
    return None, None, 0


def _filed_under(con, key):
    """The crate the DJ unanimously filed this normalised name into."""
    if not key:
        return None, 0
    # Acapellas are excluded here for the same reason as in tag_crate, and it
    # matters more than it looks: "vocals" is a FORMAT, not a genre. The DJ
    # files their own acapella stems there, so counting those taught the rule
    # "Lil Yachty -> vocals" and it duly filed his full tracks alongside the
    # stems. Who made a track says nothing about whether it is an isolated
    # vocal - the acapella rule answers that, and it runs first.
    rows = con.execute(
        "SELECT c.name, count(*) n FROM assignments a "
        "JOIN crates c ON c.id = a.crate_id "
        "JOIN tracks t ON t.id = a.track_id "
        "LEFT JOIN analysis an ON an.track_id = t.id "
        "WHERE a.source = 'human' AND (t.artist = ? OR t.remixer = ?) "
        "  AND coalesce(json_extract(an.vocal, '$.is_acapella'), 0) = 0 "
        "  AND coalesce(c.locked, 0) = 0 "
        "GROUP BY c.name", (key, key)).fetchall()
    if len(rows) != 1:
        return None, 0
    return rows[0]["name"], rows[0]["n"]


def artist_crate(con, artist):
    """The crate the DJ has filed this artist into, if they have been consistent.

    This is the strongest signal available and the tool ignored it for a long
    time. Measured leave-one-out across 659 labelled tracks whose artist has
    another labelled track: predicting "same artist, same crate" is right
    **95.6%** of the time - and **97.3%** on house vs tech house, the pair no
    amount of audio analysis could separate above ~74%.

    Only the DJ's own filing counts, and only when it is unanimous: of 276
    artists with more than one labelled track, 9 are split across crates
    (Drake in house and rap, Rihanna across house, pop and vocals - originals
    against edits). Those are genuinely ambiguous, so the rule declines rather
    than picking a side.

    Returns (crate, n_tracks) or (None, 0).
    """
    key = normalise_artist(artist)
    if not key:
        return None, 0
    # Acapellas are excluded here for the same reason as in tag_crate, and it
    # matters more than it looks: "vocals" is a FORMAT, not a genre. The DJ
    # files their own acapella stems there, so counting those taught the rule
    # "Lil Yachty -> vocals" and it duly filed his full tracks alongside the
    # stems. Who made a track says nothing about whether it is an isolated
    # vocal - the acapella rule answers that, and it runs first.
    rows = con.execute(
        "SELECT c.name, count(*) n FROM assignments a "
        "JOIN crates c ON c.id = a.crate_id "
        "JOIN tracks t ON t.id = a.track_id "
        "LEFT JOIN analysis an ON an.track_id = t.id "
        "WHERE a.source = 'human' AND t.artist = ? "
        "  AND coalesce(json_extract(an.vocal, '$.is_acapella'), 0) = 0 "
        "  AND coalesce(c.locked, 0) = 0 "
        "GROUP BY c.name", (key,)).fetchall()
    if len(rows) != 1:
        return None, 0                    # unfiled, or filed inconsistently
    return rows[0]["name"], rows[0]["n"]


def remove_track(con, track_id):
    """Drop a track from the catalogue entirely.

    **The file on disk is never touched.** This module does not do filesystem
    work at all, and the project's standing invariant is that source audio is
    never modified, moved or deleted - so this removes the DJ's *record* of a
    track, not the track. A sample, a sound effect, an interview clip: things
    that are in the folder but are not music they will ever play.

    Every dependent row goes with it (analysis, embedding, crate memberships,
    correction history) through ON DELETE CASCADE, so a removed track cannot
    keep training the model from beyond the grave.

    Returns the filename that was removed, or None if there was no such track.
    """
    row = con.execute("SELECT filename FROM tracks WHERE id=?",
                      (track_id,)).fetchone()
    if row is None:
        return None
    con.execute("DELETE FROM tracks WHERE id=?", (track_id,))
    con.commit()
    return row["filename"]


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
