#!/usr/bin/env python3
"""Measure how well we can actually predict this DJ's genre calls.

Why this exists
---------------
The shipped classifier is nearest-centroid on Discogs-EffNet embeddings: one
mean vector per crate, cosine similarity, argmax. That is the simplest thing
that could work, and it was never compared against anything.

It also silently retrained on its own output. `rebuild_centroids` selected
every assignment regardless of `source`, so the model's guesses became its
own training data - DUBSTEP ended up with 40 reference tracks of which 2 were
the DJ's, and drifted accordingly.

So: rebuild the ground truth from the DJ's own Rekordbox playlists, and
compare methods under stratified cross-validation. Report per-genre, because
"which genres does it confuse" is the useful question, not "what is the
accuracy".

Ground truth
------------
The DJ's Rekordbox playlists ARE the labels - their own filing, not tags
scraped from a shop. Container playlists (AllSongs, Contents) and non-genre
playlists (MAIN, PARTY, REMIX) are excluded: they are set lists, not genres.

Usage
-----
    python eval_genre.py [--xml ~/Documents/rek.xml] [--folds 5]
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from crateapp.db import connect

# The DJ's playlists, mapped onto the crate vocabulary the app uses. Two
# playlists can mean the same crate ('tech' and 'TECH HOUSE'), which is why
# this is a mapping and not just a name list.
GENRE_PLAYLISTS = {
    "tech": "tech", "TECH HOUSE": "tech",
    "house": "house",
    "pop": "pop",
    "vocals": "vocals",
    "afro": "afro", "AFROHOUSE": "afro",
    "rap": "rap", "RAP": "rap",
    "UKG": "UKG",
    "DUBSTEP": "DUBSTEP",
}
# Set lists and containers, not genres. Filing by these would teach the model
# that "tracks I play at parties" is a sound.
NOT_GENRES = {"MAIN", "PARTY", "REMIX", "AllSongs", "Contents",
              "CUE Analysis Playlist", "garage", "ROOT"}


def xml_labels(xml_path):
    """{absolute track path: crate} from the DJ's own playlists."""
    root = ET.parse(xml_path).getroot()

    by_id = {}
    for t in root.iter("TRACK"):
        tid, loc = t.get("TrackID"), t.get("Location")
        if tid and loc:
            by_id[tid] = Path(unquote(urlparse(loc).path))

    labels, multi = {}, Counter()
    for node in root.find("PLAYLISTS").iter("NODE"):
        name = node.get("Name")
        if name in NOT_GENRES or name not in GENRE_PLAYLISTS:
            continue
        crate = GENRE_PLAYLISTS[name]
        for entry in node.findall("TRACK"):
            p = by_id.get(entry.get("Key"))
            if p is None:
                continue
            if p in labels and labels[p] != crate:
                multi[(labels[p], crate)] += 1
                # A track the DJ filed under two genres is not a clean label.
                labels[p] = None
            elif p not in labels:
                labels[p] = crate
    if multi:
        print("Tracks filed under two genres (dropped, they are not clean "
              "single labels):", file=sys.stderr)
        for (a, b), n in multi.most_common():
            print(f"    {a} + {b}: {n}", file=sys.stderr)
    return {p: c for p, c in labels.items() if c}


def load_features(con, labels):
    """Embeddings + tempo + vocal ratio for every labelled track we analysed."""
    rows = con.execute(
        "SELECT t.id, t.path, t.filename, t.bpm, an.vocal, e.vector "
        "FROM tracks t JOIN embeddings e ON e.track_id = t.id "
        "LEFT JOIN analysis an ON an.track_id = t.id").fetchall()

    # Match on path, then fall back to filename: the DJ's library holds the
    # same recording in several folders, and the Rekordbox entry may point at
    # a copy we catalogued under a different path.
    by_path = {Path(r["path"]): r for r in rows}
    by_name = {}
    for r in rows:
        by_name.setdefault(Path(r["path"]).name, r)

    X, y, ids, names, missed = [], [], [], [], 0
    for p, crate in sorted(labels.items()):
        r = by_path.get(p) or by_name.get(p.name)
        if r is None:
            missed += 1
            continue
        vec = np.frombuffer(r["vector"], dtype=np.float32).copy()
        voc = 0.0
        if r["vocal"]:
            voc = float(json.loads(r["vocal"]).get("vocal_ratio") or 0.0)
        X.append(vec)
        y.append(crate)
        ids.append(r["id"])
        names.append(r["filename"])
    if missed:
        print(f"({missed} labelled tracks are not analysed in the library; "
              f"skipped)", file=sys.stderr)
    return np.stack(X), np.array(y), ids, names


def tempo_features(con, ids):
    """Tempo, as something a linear model can use.

    Raw BPM is a bad feature: dubstep sits at 140 AND at 70 for the same
    music, so the relationship to genre is not monotonic. Half/double folded
    into one octave fixes that, and sin/cos of the position within the octave
    keeps 138 and 142 close together instead of splitting them at the wrap.
    """
    bpm = {}
    for r in con.execute("SELECT id, bpm FROM tracks").fetchall():
        bpm[r["id"]] = r["bpm"]
    out = []
    for i in ids:
        b = bpm.get(i) or 0.0
        if b <= 0:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        folded = b
        while folded >= 180:
            folded /= 2
        while 0 < folded < 90:
            folded *= 2
        # 90..180 mapped onto a circle, plus the raw value scaled
        frac = (folded - 90) / 90.0
        out.append([folded / 180.0,
                    np.sin(2 * np.pi * frac),
                    np.cos(2 * np.pi * frac),
                    1.0])
    return np.asarray(out, dtype=np.float32)


def discogs400(X):
    """Full 400-label Discogs distribution, recomputed from the embeddings.

    The head is a small graph on top of the embedding, so this costs no audio
    decoding at all - the expensive part is already cached in the database.
    Only the top 5 labels were ever stored, which is not enough to learn from.
    """
    import analyze
    gm = analyze.GenreModel()
    if not gm.ok:
        print("(genre head unavailable; skipping Discogs400 features)",
              file=sys.stderr)
        return None
    out = []
    for i in range(0, len(X), 256):
        out.append(np.asarray(gm.head(X[i:i + 256])))
    return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------- the methods

def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


class NearestCentroid:
    """What ships today: one mean unit-vector per crate, cosine, argmax."""
    name = "nearest centroid (current)"

    def fit(self, X, y):
        Xu = unit(X)
        self.classes = np.unique(y)
        self.cent = unit(np.stack([Xu[y == c].mean(axis=0) for c in self.classes]))
        return self

    def predict(self, X):
        return self.classes[np.argmax(unit(X) @ self.cent.T, axis=1)]


def evaluate(name, make, X, y, folds, seed=0):
    """Stratified k-fold, returning predictions for every sample."""
    from sklearn.model_selection import StratifiedKFold
    pred = np.empty_like(y)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    acc = float((pred == y).mean())
    return {"name": name, "acc": acc, "pred": pred}


def report(res, y, labels):
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n{'='*74}\n{res['name']}  —  overall {res['acc']*100:.1f}%\n{'='*74}")
    print(classification_report(y, res["pred"], labels=labels,
                                zero_division=0, digits=3))
    cm = confusion_matrix(y, res["pred"], labels=labels)
    w = max(len(l) for l in labels) + 1
    print("confusion (rows = truth, cols = predicted)")
    print(" " * w + "".join(f"{l[:6]:>7}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:<{w}}" + "".join(f"{v:>7}" for v in row))


def calibrate(X, y, make, folds, seed=0):
    """Where to put the "don't guess, flag it" line.

    The shipped threshold (0.60 cosine) was reasoned, not measured, which the
    handover notes admit. The right value is whatever separates the
    predictions that turn out right from the ones that turn out wrong, so:
    collect out-of-fold probabilities and read the trade-off off the data.

    Also measured here: how far a track sits from every crate it could belong
    to. A softmax always sums to 1, so a discriminative model reports high
    confidence even for music unlike anything the DJ owns - it has to pick
    something. The distance to the nearest centroid is what actually answers
    "is this like anything I know", and it is the signal that keeps the app
    able to spot a genuinely new sound.
    """
    from sklearn.model_selection import StratifiedKFold
    probs = np.zeros(len(y))
    margins = np.zeros(len(y))
    correct = np.zeros(len(y), dtype=bool)
    cos = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = make().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        pred = m.classes_[np.argmax(p, axis=1)]
        probs[te] = p.max(axis=1)
        top2 = np.sort(p, axis=1)[:, -2:]
        margins[te] = top2[:, 1] - top2[:, 0]
        correct[te] = pred == y[te]
        nc = NearestCentroid().fit(X[tr], y[tr])
        cos[te] = (unit(X[te]) @ nc.cent.T).max(axis=1)

    print(f"\n{'='*74}\nWhere to draw the 'flag it instead of guessing' line\n{'='*74}")
    print(f"{'threshold':>10} {'filed':>7} {'of all':>7} {'right':>8}  "
          f"{'wrong, filed anyway':>20}")
    for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        take = probs >= t
        if not take.any():
            continue
        prec = correct[take].mean()
        print(f"{t:>10.2f} {take.sum():>7} {take.mean()*100:>6.0f}% "
              f"{prec*100:>7.1f}%  {int((~correct[take]).sum()):>20}")
    print(f"\nconfidence when right:  median {np.median(probs[correct]):.3f}")
    print(f"confidence when wrong:  median {np.median(probs[~correct]):.3f}")

    # A saturated softmax is a poor uncertainty signal - 0.91 median even when
    # wrong. How close the runner-up came is a different question, and often a
    # sharper one: a near-tie is a coin flip whatever the top number says.
    print(f"\n{'margin':>10} {'filed':>7} {'of all':>7} {'right':>8}  "
          f"{'wrong, filed anyway':>20}")
    for t in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95):
        take = margins >= t
        if not take.any():
            continue
        print(f"{t:>10.2f} {take.sum():>7} {take.mean()*100:>6.0f}% "
              f"{correct[take].mean()*100:>7.1f}%  "
              f"{int((~correct[take]).sum()):>20}")
    print(f"\nmargin when right:      median {np.median(margins[correct]):.3f}")
    print(f"margin when wrong:      median {np.median(margins[~correct]):.3f}")
    lo, hi = np.percentile(cos, [1, 5])
    print(f"\ndistance to nearest crate (cosine) across known-genre tracks:")
    print(f"  1st percentile {lo:.3f}   5th percentile {hi:.3f}   "
          f"min {cos.min():.3f}")
    print(f"  -> a track below about {lo:.2f} is unlike anything filed, "
          f"which is the 'new sound' case")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(Path.home() / "Documents" / "rek.xml"))
    ap.add_argument("--db", default=str(Path.home() / ".crate" / "library.db"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--from-db", action="store_true",
                    help="use every label in the library, including the DJ's "
                         "in-app confirmations, instead of just rek.xml")
    a = ap.parse_args()

    con = connect(a.db)
    if a.from_db:
        from crateapp.classifier import training_set
        ids, labs = training_set(con)
        paths = {r["id"]: Path(r["path"])
                 for r in con.execute("SELECT id, path FROM tracks").fetchall()}
        labels = {paths[i]: l for i, l in zip(ids, labs) if i in paths}
        print(f"\nGround truth from the library: {len(labels)} labelled tracks")
        print("  NOTE: this includes tracks confirmed in the app, which are by "
              "definition\n  cases the model already got right. Accuracy "
              "measured on them reads high\n  for that reason - compare "
              "against the rek.xml-only number, not instead of it.")
    else:
        labels = xml_labels(a.xml)
        print(f"\nGround truth from {a.xml}: "
              f"{len(labels)} cleanly labelled tracks")
    X, y, ids, names = load_features(con, labels)

    # Deduplicate BEFORE cross-validation. This library holds the same
    # recording several times over, and an identical copy sitting in the
    # training fold while its twin is scored in the test fold is leakage: the
    # model is being asked to recognise a track it has already memorised.
    # Without this the headline number is flattering and wrong.
    _, first = np.unique(np.round(X, 5), axis=0, return_index=True)
    keep = np.sort(first)
    if len(keep) < len(X):
        print(f"Deduplicated {len(X)} labelled rows to {len(keep)} distinct "
              f"recordings (identical copies would leak across folds)")
    X, y = X[keep], y[keep]
    ids = [ids[i] for i in keep]
    names = [names[i] for i in keep]

    counts = Counter(y)
    print(f"Matched to analysed tracks: {len(y)}")
    for c, n in counts.most_common():
        print(f"    {c:<10} {n:>4}")

    # A class with fewer members than folds cannot be cross-validated.
    keep = np.array([counts[c] >= a.folds for c in y])
    if not keep.all():
        dropped = sorted({c for c in y[~keep]})
        print(f"\nDropped (fewer than {a.folds} examples): {dropped}")
        X, y = X[keep], y[keep]
        ids = [i for i, k in zip(ids, keep) if k]
    order = sorted(set(y))

    tempo = tempo_features(con, ids)
    d400 = discogs400(X)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, Normalizer
    from sklearn.svm import LinearSVC

    def lr():
        return make_pipeline(Normalizer(), StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0))

    def svm():
        return make_pipeline(Normalizer(), StandardScaler(),
                             LinearSVC(C=0.1, max_iter=5000))

    runs = [
        ("nearest centroid (current)", NearestCentroid, X),
        ("logistic regression, embedding", lr, X),
        ("linear SVM, embedding", svm, X),
        ("logistic regression, embedding + tempo", lr, np.hstack([X, tempo])),
    ]
    if d400 is not None:
        runs += [
            ("logistic regression, Discogs400 only", lr, d400),
            ("logistic regression, embedding + Discogs400", lr,
             np.hstack([X, d400])),
            ("logistic regression, everything", lr,
             np.hstack([X, d400, tempo])),
        ]

    results = []
    for name, make, feats in runs:
        r = evaluate(name, make, feats, y, a.folds)
        results.append(r)
        print(f"  {name:<48} {r['acc']*100:6.1f}%")

    calibrate(X, y, lr, a.folds)

    results.sort(key=lambda r: -r["acc"])
    print(f"\n\nBest: {results[0]['name']} at {results[0]['acc']*100:.1f}% "
          f"(baseline {[r for r in results if 'current' in r['name']][0]['acc']*100:.1f}%)")
    report(results[0], y, order)
    base = [r for r in results if "current" in r["name"]][0]
    if base is not results[0]:
        report(base, y, order)


if __name__ == "__main__":
    main()
