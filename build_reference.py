#!/usr/bin/env python3
"""
Build/refresh genre_reference.json - the reference centroids analyze.py uses
to catch genres the Discogs-400 taxonomy has no label for (see GenreModel and
GenreModel.taxonomy_gap_signal() in analyze.py).

Why this exists: genre_discogs400 has no "Afro House" class, so a real Afro
House track always gets filed under some neighbouring house label with no
signal anything is wrong. The fix is to compare a track's raw embedding
(discarded by the classification head, but still meaningful) against
centroids built from the DJ's own hand-sorted playlist folders - the model
can't be retrained, but a nearest-centroid check on top of it can flag what
it's structurally unable to say.

Run this again whenever a reference playlist folder changes size or a new
gap genre gets added to GAP_GENRES - the bigger and cleaner these folders
are, the better the centroid and the more reliable the signal. It rebuilds
genre_reference.json in place; analyze.py picks up the new file on its next
run automatically.

Usage:
    python build_reference.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import essentia.standard as es

from analyze import GenreModel, MODEL_DIR, AUDIO_EXT

HERE = Path(__file__).resolve().parent
OUT = HERE / "genre_reference.json"

# slug -> (folder path, display name shown in needs_review reasons,
#          is this a genre gap - i.e. does the Discogs-400 taxonomy have no
#          matching label for it at all? Only gap genres get the
#          needs_review override; the rest just strengthen the reference
#          space so nearest-centroid comparisons have real competition.)
FOLDERS = {
    "RAP":        (Path.home() / "Desktop" / "RAP",        "Rap",        False),
    "TECH_HOUSE": (Path.home() / "Desktop" / "TECH HOUSE",  "Tech House", False),
    "AFROHOUSE":  (Path.home() / "Desktop" / "AFROHOUSE",   "Afro House", True),
    "UKG":        (Path.home() / "Desktop" / "UKG",         "UK Garage",  False),
    "DUBSTEP":    (Path.home() / "Desktop" / "DUBSTEP",     "Dubstep",    False),
    # Add new hand-sorted genre folders here as they show up. Mark a folder
    # `is_gap=True` only if you've checked GenreModel.labels and confirmed
    # the Discogs-400 taxonomy really has no matching class (e.g. "Riddim"
    # doesn't either, as of the 2026-09 label set) - RAP/TECH_HOUSE/UKG/
    # DUBSTEP all correctly resolve to real Discogs labels on their own and
    # don't need this treatment.
}


def main():
    gm = GenreModel(MODEL_DIR)
    if not gm.ok:
        print("Genre model not installed - run setup.sh first.", file=sys.stderr)
        sys.exit(1)

    ref = {"embedding_model": "discogs-effnet-bs64-1", "folders": {}}

    for slug, (folder, display, is_gap) in FOLDERS.items():
        if not folder.is_dir():
            print(f"  ! skipping {slug}: {folder} not found")
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
        vecs, cats = [], Counter()
        for p in files:
            try:
                a16 = es.MonoLoader(filename=str(p), sampleRate=16000,
                                    resampleQuality=4)()
                preds, _voice, emb = gm.predict(a16)
                vecs.append(emb.mean(axis=0))
                cats[preds[0]["label"]] += 1
            except Exception as e:
                print(f"  ! {p.name}: {e}")
        if not vecs:
            print(f"  ! skipping {slug}: no usable audio")
            continue
        centroid = np.mean(vecs, axis=0)
        ref["folders"][slug] = {
            "display_name": display,
            "is_gap": is_gap,
            "n": len(vecs),
            "centroid": centroid.tolist(),
            # Labels the genre model itself gave this folder's own tracks -
            # the data-driven "family" a gap genre's embedding match has to
            # also land in before it's trusted, replacing any hand-guessed
            # list of "plausible" neighbouring labels.
            "compatible_categories": sorted(cats.keys()),
        }
        print(f"  {slug:12s} n={len(vecs):3d}  is_gap={is_gap}  "
              f"compatible_categories={sorted(cats.keys())}")

    OUT.write_text(json.dumps(ref))
    print(f"\nWrote {OUT} ({OUT.stat().st_size:,} bytes, "
          f"{len(ref['folders'])} reference folders, "
          f"{sum(1 for f in ref['folders'].values() if f['is_gap'])} gap genres)")


if __name__ == "__main__":
    main()
