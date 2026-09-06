"""Opt-in exports. Neither one modifies the source library."""
import json
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape
from urllib.parse import quote

from analyze import CUE_COLOURS, DEFAULT_CUE_COLOUR


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


def export_folders(con, dest):
    """Real folders of real files. Uses APFS copy-on-write clones where possible,
    so a 49GB library costs almost no extra disk; falls back to a plain copy."""
    dest = Path(dest)
    counts = {}
    for crate, rows in _members(con).items():
        folder = dest / crate
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in rows:
            target = folder / Path(r["path"]).name
            if target.exists():
                n += 1
                continue
            try:
                subprocess.run(["cp", "-c", r["path"], str(target)], check=True,
                               capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                subprocess.run(["cp", r["path"], str(target)], check=True)
            n += 1
        counts[crate] = n
    return counts


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
            f'    <TRACK TrackID="{ids[r["id"]]}" Name="{escape(Path(r["path"]).stem)}" '
            f'Kind="Audio File" Location="file://localhost{quote(r["path"])}" '
            f'AverageBpm="{r["bpm"] or 0}" Tonality="{escape(r["camelot"] or "")}" '
            f'Genre="{escape(genre_of.get(r["id"], ""))}" '
            f'TotalTime="{int(r["duration_sec"] or 0)}">')
        L.append(f'      <TEMPO Inizio="0.000" Bpm="{r["bpm"] or 0}" '
                 f'Metro="4/4" Battito="1"/>')
        for i, m in enumerate(moments):
            lab, start = escape(m["label"]), f'{m["time"]:.3f}'
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
        L.append(f'      <NODE Name="{escape(crate)}" Type="1" KeyType="0" '
                 f'Entries="{len(rows)}">')
        for r in rows:
            L.append(f'        <TRACK Key="{ids[r["id"]]}"/>')
        L.append('      </NODE>')
    L += ['    </NODE>', '  </PLAYLISTS>', '</DJ_PLAYLISTS>']

    Path(dest).write_text("\n".join(L), encoding="utf-8")
    return len(tracks)
