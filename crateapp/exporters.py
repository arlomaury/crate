"""Opt-in exports. Neither one modifies the source library."""
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from xml.sax.saxutils import escape
from urllib.parse import quote

from analyze import CUE_COLOURS, DEFAULT_CUE_COLOUR, tempo_element


# Characters XML 1.0 forbids outright (control characters other than tab,
# newline and carriage return, plus lone surrogates and U+FFFE/FFFF).
_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _esc(s):
    """Escape a string for use inside a double-quoted XML attribute.

    xml.sax.saxutils.escape() by itself only escapes &, < and > - not the
    double quote - so a title containing a literal '"' produces malformed
    XML that Rekordbox rejects outright. The extra entities map fixes that
    everywhere this is used.  Characters XML cannot carry at all (stray
    control bytes in ripped filenames) are dropped, since one of them makes
    the whole file unreadable.
    """
    return escape(_XML_ILLEGAL.sub("", s or ""), {'"': "&quot;", "\n": "&#10;", "\r": "&#13;", "\t": "&#9;"})


def _members(con):
    """crate name -> list of track rows."""
    rows = con.execute(
        "SELECT c.name, t.* FROM assignments a "
        "JOIN crates c ON c.id=a.crate_id JOIN tracks t ON t.id=a.track_id "
        "WHERE t.missing=0 ORDER BY c.name, t.id").fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["name"], []).append(r)
    return out


def _safe_folder_name(name, fallback):
    """Map a crate name to a single safe filesystem path segment.

    Crate names are free text - a DJ might legitimately call one
    "Drum & Bass / Jungle" - and are never sanitised in the database or in
    the UI (see crateapp.crates.ensure_crate). Only here, at the point a
    name becomes a folder name, are path separators and traversal
    sequences neutralised: Python's Path "/" operator silently discards
    the left side when the right side is absolute, so an unsanitised crate
    named "/etc/evil" would write straight to /etc/evil, and "../../x"
    would traverse out of the destination.
    """
    safe = (name or "").strip()
    safe = safe.replace("/", "-").replace("\\", "-")
    while ".." in safe:
        safe = safe.replace("..", "-")
    safe = safe.strip().strip(".").strip()
    return safe or fallback


def _fs_key(name):
    """A filename as macOS's default (APFS, case-insensitive) disk sees it:
    case and Unicode normalisation do not distinguish two names."""
    return unicodedata.normalize("NFC", name).casefold()


def export_folders(con, dest, skipped=None):
    """Real folders of real files. Uses APFS copy-on-write clones where possible,
    so a 49GB library costs almost no extra disk; falls back to a plain copy.

    Never writes, moves or transcodes a source file - only ever reads it via
    `cp`. A copy is always made under a temporary name in the destination
    crate folder and renamed into place only once it has fully landed, so a
    copy that fails partway can never be mistaken for a finished one on a
    later, resumed run.
    """
    dest = Path(dest).expanduser().resolve()
    crate_ids = {r["name"]: r["id"]
                 for r in con.execute("SELECT id, name FROM crates").fetchall()}
    counts = {}
    if skipped is None:
        skipped = []
    members = _members(con)
    # Two crates can map to one folder: "House" and "house" are the same
    # folder on a Mac's disk, and "Drum/Bass" and "Drum-Bass" sanitise to the
    # same name. Their files would silently merge, so a colliding crate gets
    # its id appended instead.
    folder_counts = {}
    for crate in members:
        k = _fs_key(_safe_folder_name(crate, f"crate-{crate_ids.get(crate, 'unknown')}"))
        folder_counts[k] = folder_counts.get(k, 0) + 1
    for crate, rows in members.items():
        safe = _safe_folder_name(crate, f"crate-{crate_ids.get(crate, 'unknown')}")
        if folder_counts[_fs_key(safe)] > 1:
            safe = f"{safe} ({crate_ids.get(crate, 'x')})"
        folder = dest / safe
        resolved = folder.resolve()
        # Belt and braces: even after sanitising, refuse to write anywhere
        # that isn't actually inside dest.
        if resolved != dest and dest not in resolved.parents:
            raise ValueError(
                f"refusing to export crate {crate!r}: {resolved} escapes {dest}")
        folder.mkdir(parents=True, exist_ok=True)

        # Two different tracks can share a filename (e.g. two rips both
        # called "song.wav"). Silently skipping the second as "already
        # exported" would understate the crate and drop a track with no
        # error - exactly the wrong failure mode for this project. So any
        # name that collides within this crate is disambiguated by the
        # track's own id, which is stable across runs.
        # Compared the way the Mac's disk compares names: "Song.wav" and
        # "song.wav", or "Café" typed two different Unicode ways, are ONE file
        # there. Counted as different, the second copy found the first already
        # in place, was reported exported, and was never copied.
        name_counts = {}
        for r in rows:
            n = _fs_key(Path(r["path"]).name)
            name_counts[n] = name_counts.get(n, 0) + 1

        n = 0
        for r in rows:
            src = Path(r["path"])
            if not src.is_file():
                # Deleted or moved since the last scan.  Skip it and say so,
                # rather than abandoning the export halfway through.
                skipped.append(str(src))
                continue
            if name_counts[_fs_key(src.name)] > 1:
                target = folder / f"{src.stem}_{r['id']}{src.suffix}"
            else:
                target = folder / src.name
            if target.exists():
                n += 1
                continue
            tmp = target.with_name(f"{target.name}.part-{os.getpid()}-{r['id']}")
            try:
                _copy(src, tmp)
                tmp.rename(target)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
            n += 1
        counts[crate] = n
    return counts


def _copy(src, tmp):
    """Copy src to tmp, preferring an APFS copy-on-write clone."""
    try:
        subprocess.run(["cp", "-c", str(src), str(tmp)], check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(["cp", str(src), str(tmp)], check=True, capture_output=True)


def export_rekordbox(con, dest):
    """rekordbox.xml with a playlist per crate, plus coloured hot cues."""
    members = _members(con)
    tracks, ids = [], {}
    for rows in members.values():
        for r in rows:
            if r["id"] not in ids:
                ids[r["id"]] = len(ids) + 1
                tracks.append(r)

    L = ['<?xml version="1.0" encoding="UTF-8"?>', '<DJ_PLAYLISTS Version="1.0.0">',
         '  <PRODUCT Name="Crate" Version="1.0" Company="local"/>',
         f'  <COLLECTION Entries="{len(tracks)}">']
    genre_of = {r["id"]: c for c, rows in members.items() for r in rows}

    for r in tracks:
        row = con.execute("SELECT moments FROM analysis WHERE track_id=?",
                          (r["id"],)).fetchone()
        moments = json.loads(row["moments"]) if row and row["moments"] else []
        L.append(
            f'    <TRACK TrackID="{ids[r["id"]]}" Name="{_esc(Path(r["path"]).stem)}" '
            f'Kind="Audio File" Location="file://localhost{quote(r["path"])}" '
            f'AverageBpm="{r["bpm"] or 0}" Tonality="{_esc(r["camelot"] or "")}" '
            f'Genre="{_esc(genre_of.get(r["id"], ""))}" '
            f'TotalTime="{int(r["duration_sec"] or 0)}">')
        tempo = tempo_element(moments, r["bpm"])
        if tempo:
            L.append(tempo)
        for i, m in enumerate(moments):
            lab, start = _esc(m["label"]), f'{m["time"]:.3f}'
            L.append(f'      <POSITION_MARK Name="{lab}" Type="0" '
                     f'Start="{start}" Num="-1"/>')
            if i < 8:
                cr, cg, cb = CUE_COLOURS.get(m.get("type", ""), DEFAULT_CUE_COLOUR)
                L.append(f'      <POSITION_MARK Name="{lab}" Type="0" '
                         f'Start="{start}" Num="{i}" '
                         f'Red="{cr}" Green="{cg}" Blue="{cb}"/>')
        L.append('    </TRACK>')

    L += ['  </COLLECTION>', '  <PLAYLISTS>',
          f'    <NODE Type="0" Name="ROOT" Count="{len(members)}">']
    for crate, rows in members.items():
        L.append(f'      <NODE Name="{_esc(crate)}" Type="1" KeyType="0" '
                 f'Entries="{len(rows)}">')
        for r in rows:
            L.append(f'        <TRACK Key="{ids[r["id"]]}"/>')
        L.append('      </NODE>')
    L += ['    </NODE>', '  </PLAYLISTS>', '</DJ_PLAYLISTS>']

    out = Path(dest).expanduser()
    check_rekordbox_dest(out)
    tmp = out.with_name(f".{out.name}.part")
    tmp.write_text("\n".join(L), encoding="utf-8")
    tmp.replace(out)
    return len(tracks)


def check_rekordbox_dest(path):
    """Refuse any destination that is not an .xml file, or that already
    holds something other than a previous Rekordbox export. A mistyped or
    pasted path must never be able to overwrite a track or anything else."""
    path = Path(path)
    if path.suffix.lower() != ".xml":
        raise ValueError("the Rekordbox export must be an .xml file, e.g. ~/Desktop/rekordbox.xml")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"{path} is a folder, not a file")
        with open(path, "rb") as f:
            head = f.read(4096)
        # Only a file Crate itself wrote. A collection exported from
        # Rekordbox has the same root tag, but it is the DJ's own library -
        # and the ground truth retrain.py learns from - so it is never
        # replaced.
        if b"<DJ_PLAYLISTS" not in head or b'<PRODUCT Name="Crate"' not in head:
            raise ValueError(f"{path} already exists and is not an export from Crate - "
                             "choose another name so it is not overwritten")
