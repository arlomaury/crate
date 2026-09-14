#!/usr/bin/env python3
"""Is averaging the whole track throwing away the answer?

analyze.py computes a Discogs-EffNet embedding per frame and then keeps only
`emb.mean(axis=0)` - one vector for a six-minute record. That average includes
the intro, the breakdown and the outro, which is not how a DJ decides what
something is. They listen to the groove.

This tests whether a different reduction of the same frames carries more:

    mean            what ships
    mean + std      adds how much the track VARIES, not just its average
    drop            only the frames inside the main drop section, located
                    from the structural moments already stored per track
    drop + mean     both
    percentiles     25th/50th/75th, a shape-preserving summary
    max             the most extreme frame

Everything is measured on the same tracks, same folds, so the only thing
changing is the reduction.

Usage:
    python pooling_experiment.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import analyze
from crateapp.classifier import training_set
from crateapp.db import connect

CACHE = Path.home() / ".crate" / "pooling.npz"
VARIANTS = ["mean", "meanstd", "drop", "dropmean", "pct", "max"]


def pool(emb, dur, moments):
    """Every reduction of one track's frames, as a dict of vectors."""
    emb = np.asarray(emb, dtype="float32")
    n = len(emb)
    out = {}
    out["mean"] = emb.mean(axis=0)
    out["meanstd"] = np.concatenate([emb.mean(axis=0), emb.std(axis=0)])
    out["pct"] = np.concatenate([np.percentile(emb, p, axis=0)
                                 for p in (25, 50, 75)]).astype("float32")
    out["max"] = emb.max(axis=0)

    # The frames inside the main drop. Structural moments are already stored
    # per track, so this costs nothing to locate.
    seg = None
    if dur and n > 4 and moments:
        drops = [m for m in moments if m.get("type") == "drop"
                 and m.get("time") is not None]
        if drops:
            start = drops[0]["time"]
            nxt = [m["time"] for m in moments
                   if m.get("time") is not None and m["time"] > start + 5]
            end = min(nxt) if nxt else min(start + 60.0, dur)
            a = int(max(0, min(n - 1, round(start / dur * n))))
            b = int(max(a + 1, min(n, round(end / dur * n))))
            if b - a >= 2:
                seg = emb[a:b]
    if seg is None:
        seg = emb
    out["drop"] = seg.mean(axis=0)
    out["dropmean"] = np.concatenate([out["mean"], out["drop"]])
    return out


def extract(rows):
    cached = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        for i, k in enumerate(z["ids"]):
            cached[int(k)] = {v: z[v][i] for v in VARIANTS}
    todo = [r for r in rows if r[0] not in cached]
    if not todo:
        return cached

    gm = analyze.GenreModel()
    if not gm.ok:
        print("genre model unavailable", file=sys.stderr)
        sys.exit(1)
    import essentia.standard as es

    print(f"embedding {len(todo)} tracks…", flush=True)
    t0 = time.time()
    for n, (tid, path, dur, moments) in enumerate(todo, 1):
        try:
            audio = es.MonoLoader(filename=str(path), sampleRate=16000)()
            if len(audio) < 16000 * 5:
                continue
            emb = gm.embed(audio)
            if len(emb) < 2:
                continue
            cached[tid] = pool(emb, dur, moments)
        except Exception:
            continue
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el/n:.1f}s each, "
                  f"{(len(todo)-n)*el/n/60:.0f} min left", flush=True)
    keys = sorted(cached)
    np.savez(CACHE, ids=np.array(keys),
             **{v: np.stack([cached[i][v] for i in keys]) for v in VARIANTS})
    return cached


def cv(X, y, folds=5, seed=0, C=0.001):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    if min(Counter(y).values()) < folds:
        return None
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, y):
        m = make_pipeline(Normalizer(), StandardScaler(),
                          LogisticRegression(max_iter=5000, C=C))
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean())


def repeated(X, y, n=3):
    got = [cv(X, y, seed=s) for s in range(n)]
    got = [g for g in got if g is not None]
    return (float(np.mean(got)), float(np.std(got))) if got else (None, None)


def main():
    con = connect(Path.home() / ".crate" / "library.db")
    ids, labs = training_set(con)
    label = dict(zip(ids, labs))
    rows = []
    for r in con.execute(
            "SELECT t.id, t.path, t.duration_sec, an.moments FROM tracks t "
            "LEFT JOIN analysis an ON an.track_id=t.id WHERE t.missing=0"):
        if r["id"] in label:
            rows.append((r["id"], r["path"], r["duration_sec"],
                         json.loads(r["moments"]) if r["moments"] else []))
    print(f"{len(rows)} labelled tracks")

    cached = extract(rows)
    tids = [t for t, *_ in rows if t in cached]
    y = np.array([label[t] for t in tids])

    # Deduplicate on the shipped representation, so identical recordings
    # cannot sit in the training and test folds at once.
    base = np.stack([cached[t]["mean"] for t in tids])
    _, first = np.unique(np.round(base, 5), axis=0, return_index=True)
    keep = np.sort(first)
    tids = [tids[i] for i in keep]
    y = y[keep]
    print(f"{len(y)} distinct recordings\n")

    HT = {"house", "tech"}
    hm = np.isin(y, list(HT))
    print(f"{'reduction':<14}{'dims':>6}{'all crates':>14}{'house vs tech':>16}")
    for v in VARIANTS:
        X = np.stack([cached[t][v] for t in tids])
        a, sa = repeated(X, y)
        b, sb = repeated(X[hm], y[hm])
        tag = "  <- ships" if v == "mean" else ""
        print(f"{v:<14}{X.shape[1]:>6}{a*100:>12.1f}% {b*100:>14.1f}%{tag}")


if __name__ == "__main__":
    main()
