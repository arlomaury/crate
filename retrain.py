#!/usr/bin/env python3
"""Reset the ground truth from the DJ's own Rekordbox filing, then retrain.

Run this when the model has drifted, or after re-exporting rek.xml.

What it does, and why each step is needed:

1. Delete every automatic assignment. They are the model's own guesses. The
   old training code learned from them, which is how DUBSTEP ended up with 40
   reference tracks of which 2 were the DJ's.
2. Re-import the genre playlists from rek.xml as human assignments. Those
   playlists are the DJ's actual filing - the only labels worth learning from.
   Set lists (MAIN, PARTY, REMIX) and containers (AllSongs, Contents) are not
   genres and are skipped.
3. Train the layer and the centroids on what is left.
4. Re-classify every analysed track with the retrained model.

Usage:
    python retrain.py [--xml ~/Documents/rek.xml] [--dry-run]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from crateapp.classifier import Classifier, acapella_verdict, train
from crateapp.crates import auto_assign, tag_crate, who_made_it
from crateapp.db import LOCK, connect
from crateapp.worker import load_embedding
from eval_genre import GENRE_PLAYLISTS, xml_labels


def reseed(con, xml_path, dry_run=False):
    """Replace every label in the database with the DJ's Rekordbox filing."""
    labels = xml_labels(xml_path)

    rows = con.execute("SELECT id, path FROM tracks").fetchall()
    by_path = {Path(r["path"]): r["id"] for r in rows}
    by_name = {}
    for r in rows:
        by_name.setdefault(Path(r["path"]).name, r["id"])

    # One Rekordbox entry can correspond to several catalogued copies of the
    # same file; label every copy, or the duplicates train nothing and get
    # auto-filed by a model that never saw them.
    wanted, unmatched = [], 0
    for p, crate in labels.items():
        tid = by_path.get(p)
        if tid is None:
            tid = by_name.get(p.name)
        if tid is None:
            unmatched += 1
            continue
        wanted.append((tid, crate))
        for r in rows:
            if Path(r["path"]).name == p.name and r["id"] != tid:
                wanted.append((r["id"], crate))

    counts = Counter(c for _, c in wanted)
    print(f"Rekordbox genre playlists: {len(labels)} cleanly labelled tracks")
    print(f"Matched in the library:    {len(wanted)} rows "
          f"(including duplicate copies of the same recording)")
    if unmatched:
        print(f"Not in the library:        {unmatched}")
    for c, n in counts.most_common():
        print(f"    {c:<10} {n:>5}")

    if dry_run:
        return counts

    with LOCK:
        auto = con.execute(
            "SELECT count(*) FROM assignments WHERE source='auto'").fetchone()[0]
        human = con.execute(
            "SELECT count(*) FROM assignments WHERE source='human'").fetchone()[0]
        print(f"\nClearing {auto} automatic and {human} existing human "
              f"assignments, then re-seeding from Rekordbox.")
        con.execute("DELETE FROM assignments")
        for crate in sorted(set(counts)):
            con.execute("INSERT OR IGNORE INTO crates (name) VALUES (?)", (crate,))
        cid = {r["name"]: r["id"]
               for r in con.execute("SELECT id, name FROM crates").fetchall()}
        for tid, crate in wanted:
            con.execute(
                "INSERT OR REPLACE INTO assignments "
                "(track_id, crate_id, source, confidence, band) "
                "VALUES (?,?,'human',1.0,'confident')", (tid, cid[crate]))
        con.commit()
    return counts


def reclassify(con, model_path):
    """Re-file every analysed track with the retrained model."""
    clf = Classifier(model_path)
    if not clf.crate_names():
        print("No crates to classify against.", file=sys.stderr)
        return {}

    import json as _json
    with LOCK:
        ids = [r["id"] for r in con.execute(
            "SELECT t.id FROM tracks t JOIN embeddings e ON e.track_id=t.id "
            "WHERE t.analysed_at IS NOT NULL AND t.missing=0").fetchall()]
        vocals = {r["track_id"]: _json.loads(r["vocal"])
                  for r in con.execute(
                      "SELECT track_id, vocal FROM analysis "
                      "WHERE vocal IS NOT NULL").fetchall()}
        tags = {r["id"]: r["genre_tag"] for r in con.execute(
            "SELECT id, genre_tag FROM tracks "
            "WHERE genre_tag IS NOT NULL").fetchall()}

        # Human labels are the ground truth and must survive re-filing.
        human = {r["track_id"] for r in con.execute(
            "SELECT track_id FROM assignments WHERE source='human'").fetchall()}

    bands = Counter()
    for tid in ids:
        if tid in human:
            bands["kept (yours)"] += 1
            continue
        vec = load_embedding(con, tid)
        if vec is None:
            continue
        # Same order as runner._classify: acapella, tag, person, audio.
        aca = acapella_verdict(vocals.get(tid), clf.crate_names())
        chosen, why = None, None
        if not aca:
            by_tag, _n, _p = tag_crate(con, tags.get(tid))
            if by_tag:
                chosen, why = by_tag, "by genre tag"
            else:
                _who, by_person, _n2 = who_made_it(con, tid)
                if by_person:
                    chosen, why = by_person, "by artist (yours)"
        if chosen:
            model = clf.classify(vec)
            auto_assign(con, tid, {
                "crate": chosen, "band": "confident", "p": 1.0,
                "similarity": model.get("similarity", 1.0), "margin": 1.0,
                "scores": model.get("scores", []),
                "disputed": bool(model.get("crate")
                                 and model["crate"] != chosen)})
            bands[why] += 1
            continue
        if aca:
            result = {"crate": aca, "band": "confident", "similarity": 1.0,
                      "margin": 1.0, "p": 1.0,
                      "scores": [{"crate": aca, "p": 1.0}]}
            bands["acapella -> vocals"] += 1
            with LOCK:
                auto_assign(con, tid, result)
            continue
        result = clf.classify(vec)
        bands[result["band"]] += 1
        with LOCK:
            auto_assign(con, tid, result)
    return bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(Path.home() / "Documents" / "rek.xml"))
    ap.add_argument("--db", default=str(Path.home() / ".crate" / "library.db"))
    ap.add_argument("--model", default="crate_model.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    con = connect(a.db)
    reseed(con, a.xml, dry_run=a.dry_run)
    if a.dry_run:
        return

    print("\nTraining…")
    out = train(con, a.model)
    if not out["trained"]:
        print(f"  not trained: {out.get('reason')}", file=sys.stderr)
        return
    print(f"  learned {len(out['classes'])} crates from {out['n']} of the "
          f"DJ's own tracks: {', '.join(out['classes'])}")
    if out["too_thin"]:
        print(f"  too thin to learn (kept as crates, not predicted): "
              f"{', '.join(out['too_thin'])}")

    print("\nRe-filing the library…")
    bands = reclassify(con, a.model)
    for b, n in bands.most_common():
        print(f"    {b:<14} {n:>5}")


if __name__ == "__main__":
    main()
