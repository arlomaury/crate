"""Opt-in exports. Neither one modifies the source library."""
import json
import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape
from urllib.parse import quote

from analyze import CUE_COLOURS, DEFAULT_CUE_COLOUR


def _esc(s):
    """Escape a string for use inside a double-quoted XML attribute.

    xml.sax.saxutils.escape() by itself only escapes &, < and > - not the
    double quote - so a title containing a literal '"' produces malformed
    XML that Rekordbox rejects outright. The extra entities map fixes that
    everywhere this is used.
    """
    return escape(s or "", {'"': "&quot;"})


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


def export_folders(con, dest):
    """Real folders of real files. Uses APFS copy-on-write clones where possible,
    so a 49GB library costs almost no extra disk; falls back to a plain copy.

    Never writes, moves or transcodes a source file - only ever reads it via
    `cp`. A copy is always made under a temporary name in the destination
    crate folder and renamed into place only once it has fully landed, so a
    copy that fails partway can never be mistaken for a finished one on a
    later, resumed run.
    """
    dest = Path(dest).resolve()
    crate_ids = {r["name"]: r["id"]
                 for r in con.execute("SELECT id, name FROM crates").fetchall()}
    counts = {}
    for crate, rows in _members(con).items():
        safe = _safe_folder_name(crate, f"crate-{crate_ids.get(crate, 'unknown')}")
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
        name_counts = {}
        for r in rows:
            n = Path(r["path"]).name
            name_counts[n] = name_counts.get(n, 0) + 1

        n = 0
        for r in rows:
            src = Path(r["path"])
            if name_counts[src.name] > 1:
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
        L.append(f'      <TEMPO Inizio="0.000" Bpm="{r["bpm"] or 0}" '
                 f'Metro="4/4" Battito="1"/>')
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

    Path(dest).write_text("\n".join(L), encoding="utf-8")
    return len(tracks)
