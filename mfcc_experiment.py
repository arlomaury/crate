#!/usr/bin/env python3
"""Do MFCCs add anything the Discogs-EffNet embedding does not already have?

MFCCs are the classic genre-classification feature: a compact description of
the spectral envelope, which is to say of timbre. They were the standard
baseline for two decades before learned embeddings.

The reason to doubt they help here is that the embedding is a deep model
trained on millions of tracks for exactly this purpose, and it is 1280
dimensions against MFCC's few dozen. The reason to test anyway is that they
are not the same thing: MFCCs describe the recording - its brightness, its
production, its mastering - in a way a genre-trained network has no particular
reason to preserve, since it was optimised to ignore differences that do not
change the genre label.

So: MFCC statistics alone, and concatenated with the embedding, against the
embedding by itself. Same tracks, same folds.

Usage:
    python mfcc_experiment.py
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from crateapp.classifier import training_set
from crateapp.db import connect
from eval_genre import load_features

CACHE = Path.home() / ".crate" / "mfcc.npz"


def mfcc_features(path, sr=44100, n_coef=20):
    """Mean, std and delta-statistics of the MFCCs over the whole track.

    A track is a sequence of frames; a classifier needs one vector. Mean and
    standard deviation give the average timbre and how much it moves, and the
    mean absolute delta gives how fast it moves - a steady loop and a track
    that keeps changing have different values even at the same average.
    """
    import essentia.standard as es
    try:
        audio = es.MonoLoader(filename=str(path), sampleRate=sr)()
        if len(audio) < sr * 5:
            return None
        w = es.Windowing(type="hann")
        spec = es.Spectrum()
        mfcc = es.MFCC(numberCoefficients=n_coef, inputSize=1025)
        frames = []
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024,
                                       startFromZero=True):
            _, coeffs = mfcc(spec(w(frame)))
            frames.append(coeffs)
        if len(frames) < 8:
            return None
        M = np.asarray(frames, dtype="float32")
        d = np.abs(np.diff(M, axis=0))
        return np.concatenate([M.mean(axis=0), M.std(axis=0),
                               d.mean(axis=0)]).astype("float32")
    except Exception:
        return None


def extract(rows):
    cached = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cached = {int(i): v for i, v in zip(z["ids"], z["feats"])}
    todo = [r for r in rows if r[0] not in cached]
    if todo:
        print(f"extracting MFCCs for {len(todo)} tracks…", flush=True)
        t0 = time.time()
        for n, (tid, path) in enumerate(todo, 1):
            f = mfcc_features(path)
            if f is not None:
                cached[tid] = f
            if n % 50 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el/n:.1f}s each, "
                      f"{(len(todo)-n)*el/n/60:.0f} min left", flush=True)
        keys = sorted(cached)
        np.savez(CACHE, ids=np.array(keys),
                 feats=np.stack([cached[i] for i in keys]))
    return cached


def cv(mk, X, y, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    if min(Counter(y).values()) < folds:
        return None
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, y):
        pred[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean())


def repeated(mk, X, y, n=3):
    got = [cv(mk, X, y, seed=s) for s in range(n)]
    got = [g for g in got if g is not None]
    return (float(np.mean(got)), float(np.std(got))) if got else (None, None)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    con = connect(Path.home() / ".crate" / "library.db")
    ids, labs = training_set(con)
    meta = {r["id"]: r["path"] for r in
            con.execute("SELECT id, path FROM tracks WHERE missing=0")}
    X, y, tids, names = load_features(
        con, {Path(meta[i]): l for i, l in zip(ids, labs) if i in meta})
    _, first = np.unique(np.round(X, 5), axis=0, return_index=True)
    keep = np.sort(first)
    X, y, tids = X[keep], y[keep], [tids[i] for i in keep]
    counts = Counter(y)
    ok = np.array([counts[c] >= 5 for c in y])
    X, y, tids = X[ok], y[ok], [t for t, o in zip(tids, ok) if o]

    cached = extract([(t, meta[t]) for t in tids])
    have = np.array([t in cached for t in tids])
    X, y = X[have], y[have]
    M = np.stack([cached[t] for t, h in zip(tids, have) if h])
    print(f"\n{len(y)} recordings; MFCC vector is {M.shape[1]} numbers "
          f"against the embedding's {X.shape[1]}\n")

    def big():
        return make_pipeline(Normalizer(), StandardScaler(),
                             LogisticRegression(max_iter=5000, C=0.003))

    def small(C=1.0):
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=C))

    HT = {"house", "tech"}
    hm = np.isin(y, list(HT))
    print(f"{'features':<34}{'all crates':>13}{'house vs tech':>16}")
    rows = [("embedding (ships)", lambda: big(), X)]
    for C in (0.01, 0.1, 1.0):
        rows.append((f"MFCC only (C={C})", (lambda c: lambda: small(c))(C), M))
    rows.append(("embedding + MFCC", lambda: big(), np.hstack([X, M])))
    for name, mk, feats in rows:
        a, _ = repeated(mk, feats, y)
        b, _ = repeated(mk, feats[hm], y[hm])
        print(f"{name:<34}{a*100:>11.1f}% {b*100:>14.1f}%")


if __name__ == "__main__":
    main()
