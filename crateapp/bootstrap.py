"""First-run setup.

Two jobs: remember which folders hold the music, and seed crates from an
existing Rekordbox export so the app is useful the moment it opens rather
than after weeks of corrections.

The seeding matters more than it looks. A DJ who has already sorted their
library in Rekordbox has, in effect, already labelled their training data;
importing it means the crate model starts out knowing their taste instead of
starting from nothing.
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

from crateapp.crates import ensure_crate
from crateapp.db import LOCK


def add_folder(con, path):
    """Remember a music folder. Stored once; never asked for again."""
    # Read and write under one hold of the (re-entrant) lock: two requests
    # adding folders at once would otherwise each write back a list missing
    # the other's folder.
    with LOCK:
        folders = get_folders(con)
        p = str(Path(path).expanduser())
        if p not in folders:
            folders.append(p)
            con.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES ('folders', ?)",
                (json.dumps(folders),))
            con.commit()
    return folders


def remove_folder(con, path):
    with LOCK:
        folders = [f for f in get_folders(con) if f != str(Path(path).expanduser())]
        con.execute("INSERT OR REPLACE INTO config (key, value) VALUES ('folders', ?)",
                    (json.dumps(folders),))
        con.commit()
    return folders


def get_folders(con):
    with LOCK:
        row = con.execute("SELECT value FROM config WHERE key='folders'").fetchone()
    if not row or not row["value"]:
        return []
    try:
        return json.loads(row["value"])
    except (ValueError, TypeError):
        return []


# Some playlists are containers or set-lists rather than genre judgements -
# "AllSongs" and "Contents" each hold the whole library, and "MAIN"/"PARTY" are
# gig selections spanning every genre. A centroid built from those describes
# nothing in particular and then competes with the real crates on every
# classification.
#
# There is no reliable way to detect them automatically. Share-of-collection
# fails (AllSongs is 1166 of a 3134-track collection, only 43%) and so does
# superset detection, because the same song in two playlists can carry two
# different TrackIDs when the files live in different folders. So the caller
# names them; this is the DJ's judgement, not a heuristic.
CONTAINER_SHARE = 1.0


def import_rekordbox_xml(con, xml_path, container_share=CONTAINER_SHARE,
                         exclude=()):
    """Seed crates from the playlists in a Rekordbox XML export.

    Only tracks whose audio still exists on disk are imported - a crate full of
    dead paths teaches the model nothing and would show the DJ phantom entries.
    Playlist membership becomes a *human* assignment, because that is exactly
    what it is: the DJ's own filing, not a guess.

    Returns (imported, skipped) where `skipped` names the container playlists
    that were deliberately not turned into crates, so the caller can say so
    rather than silently dropping them.
    """
    root = ET.parse(str(xml_path)).getroot()

    collection = root.find("COLLECTION")
    if collection is None:
        return {}, {}

    # TrackID -> path, for the tracks that still exist
    loc = {}
    for tr in collection.findall("TRACK"):
        raw = tr.get("Location", "")
        p = unquote(raw).replace("file://localhost", "")
        if p and os.path.exists(p):
            loc[tr.get("TrackID")] = p

    counts, skipped = {}, {}
    limit = max(1, int(len(loc) * container_share))
    exclude = set(exclude)

    def walk(node):
        for n in node.findall("NODE"):
            if n.get("Type") == "1":            # a playlist, not a folder
                paths = [loc[k.get("Key")] for k in n.findall("TRACK")
                         if k.get("Key") in loc]
                if paths:
                    if n.get("Name") in exclude or len(paths) > limit:
                        skipped[n.get("Name")] = len(paths)
                    else:
                        _seed_crate(con, n.get("Name"), paths)
                        counts[n.get("Name")] = len(paths)
            walk(n)

    playlists = root.find("PLAYLISTS")
    if playlists is not None:
        walk(playlists)
    return counts, skipped


def _seed_crate(con, name, paths):
    cid = ensure_crate(con, name)
    with LOCK:
        for p in paths:
            con.execute(
                "INSERT OR IGNORE INTO tracks (path, filename) VALUES (?,?)",
                (str(Path(p).resolve()), Path(p).name))
            row = con.execute("SELECT id FROM tracks WHERE path=?",
                              (str(Path(p).resolve()),)).fetchone()
            con.execute(
                "INSERT OR REPLACE INTO assignments "
                "(track_id, crate_id, source, confidence, band) "
                "VALUES (?,?,'human',1.0,'confident')", (row["id"], cid))
        con.commit()
