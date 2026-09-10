#!/usr/bin/env python3
"""Does GROOVE separate house from tech house where timbre cannot?

Why this experiment exists
--------------------------
69% of the classifier's remaining errors are house-vs-tech-house, and the
Discogs-EffNet embedding has plateaued: going from 364 to 520 labels bought
0.4 accuracy points. More labels will not fix it.

The published work on EDM subgenre classification (Hsu et al., "Deep Learning
Based EDM Subgenre Classification using Mel-Spectrogram and Tempogram
Features", arXiv:2110.08862) reports that the most important features in
tree-based EDM subgenre classifiers are all tempo/rhythm related, and that
adding tempogram representations to an auto-tagging model helps. That is a
concrete, cheap thing to test here - and it is orthogonal to the embedding,
which is trained for timbre and production style.

Plain BPM was already tested and did nothing (+0.2%, inside the noise). This
is different: a beat-synchronous profile of WHERE energy sits inside the bar.
Tech house and house share tempo and timbre but not groove - the offbeat bass,
the hat placement, the swing.

What it measures
----------------
Three feature sets, on the binary house-vs-tech problem in isolation (where
the error actually is) and on the full 8-crate problem:

    embedding only     - what ships
    groove only        - can rhythm alone tell them apart?
    embedding + groove - does it add anything the embedding lacks?

Usage:
    python groove_experiment.py            # extracts, caches, then reports
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from crateapp.classifier import training_set
from crateapp.db import connect
from eval_genre import load_features

CACHE = Path.home() / ".crate" / "groove.npz"


def groove_features(path, sr=44100):
    """A beat-synchronous picture of where energy sits inside the bar.

    BeatsLoudness gives, per beat, an overall loudness and its split across 5
    frequency bands. Folding those onto the beat's position within a 4-beat
    bar turns them into a groove signature: which beats carry the kick, where
    the hats sit, whether the bass lands on the offbeat.

    Returns a fixed-length vector, or None if the file cannot be read.
    """
    import essentia.standard as es
    try:
        audio = es.MonoLoader(filename=str(path), sampleRate=sr)()
        if len(audio) < sr * 5:
            return None
        bpm, beats, conf, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
        if len(beats) < 8:
            return None
        loud, band = es.BeatsLoudness(beats=beats, sampleRate=sr)(audio)
        loud = np.asarray(loud, dtype="float32")
        band = np.asarray(band, dtype="float32")          # (n_beats, 5)
        if band.ndim != 2 or len(band) < 8:
            return None
        n = min(len(loud), len(band))
        loud, band = loud[:n], band[:n]

        # Normalise away absolute level - this is about the SHAPE of the
        # groove, not how loud the master is.
        loud = loud / (np.mean(loud) + 1e-9)
        band = band / (np.mean(band, axis=0, keepdims=True) + 1e-9)

        pos = np.arange(n) % 4
        feats = []
        for p in range(4):
            m = pos == p
            if not m.any():
                feats.extend([0.0] * 12)
                continue
            feats.append(float(loud[m].mean()))
            feats.append(float(loud[m].std()))
            feats.extend(band[m].mean(axis=0).tolist())    # 5
            feats.extend(band[m].std(axis=0).tolist())     # 5
        # Beat-to-beat variation: a steady four-to-the-floor differs from a
        # shuffling groove even at the same tempo.
        feats.append(float(np.mean(np.abs(np.diff(loud)))))
        feats.append(float(conf))
        folded = float(bpm)
        while folded >= 180:
            folded /= 2
        while 0 < folded < 90:
            folded *= 2
        feats.append(folded / 180.0)
        feats.append(float(es.Danceability()(audio)[0]))
        return np.asarray(feats, dtype="float32")
    except Exception:
        return None


def extract(con, ids, paths):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cached = {int(k): v for k, v in zip(z["ids"], z["feats"])}
    else:
        cached = {}
    todo = [i for i in ids if i not in cached]
    if todo:
        print(f"extracting groove features for {len(todo)} tracks "
              f"(~2s each, {len(todo)*2/60:.0f} min)…", flush=True)
        t0 = time.time()
        for n, tid in enumerate(todo, 1):
            f = groove_features(paths[tid])
            if f is not None:
                cached[tid] = f
            if n % 25 == 0:
                done = time.time() - t0
                print(f"  {n}/{len(todo)}  {done/n:.1f}s each, "
                      f"{(len(todo)-n)*done/n/60:.0f} min left", flush=True)
        k = sorted(cached)
        np.savez(CACHE, ids=np.array(k),
                 feats=np.stack([cached[i] for i in k]))
    return cached


def cv(make, X, y, folds=5):
    from sklearn.model_selection import StratifiedKFold
    if min(Counter(y).values()) < folds:
        return None
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=0).split(X, y):
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean())


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    con = connect(Path.home() / ".crate" / "library.db")
    ids_all, labs = training_set(con)
    paths = {r["id"]: Path(r["path"])
             for r in con.execute("SELECT id, path FROM tracks").fetchall()}
    labels = {paths[i]: l for i, l in zip(ids_all, labs) if i in paths}

    X, y, tids, names = load_features(con, labels)
    _, first = np.unique(np.round(X, 5), axis=0, return_index=True)
    keep = np.sort(first)
    X, y, tids = X[keep], y[keep], [tids[i] for i in keep]
    print(f"{len(X)} distinct labelled recordings")

    cached = extract(con, tids, paths)
    have = np.array([t in cached for t in tids])
    X, y = X[have], y[have]
    G = np.stack([cached[t] for t, h in zip(tids, have) if h])
    print(f"groove features for {len(G)} of them, {G.shape[1]} dims\n")

    def lr(C=0.001):
        return lambda: make_pipeline(Normalizer(), StandardScaler(),
                                     LogisticRegression(max_iter=5000, C=C))

    def lr_plain(C=1.0):
        # Groove is 46 dims, not 1280 - it does not need the same clamp.
        return lambda: make_pipeline(StandardScaler(),
                                     LogisticRegression(max_iter=5000, C=C))

    XG = np.hstack([X, G])

    print("=" * 62)
    print("THE HARD PAIR: house vs tech house, on its own")
    print("=" * 62)
    m = np.isin(y, ["house", "tech"])
    yb, Xb, Gb, XGb = y[m], X[m], G[m], XG[m]
    print(f"  {len(yb)} tracks  ({Counter(yb)})")
    print(f"  always-guess-the-bigger-class:  "
          f"{max(Counter(yb).values())/len(yb)*100:5.1f}%")
    print(f"  embedding only                  {cv(lr(), Xb, yb)*100:5.1f}%")
    for C in (0.01, 0.1, 1.0):
        print(f"  groove only        (C={C:<5})    {cv(lr_plain(C), Gb, yb)*100:5.1f}%")
    print(f"  embedding + groove              {cv(lr(), XGb, yb)*100:5.1f}%")

    print("\n" + "=" * 62)
    print("ALL EIGHT CRATES")
    print("=" * 62)
    print(f"  embedding only                  {cv(lr(), X, y)*100:5.1f}%")
    print(f"  groove only                     {cv(lr_plain(0.1), G, y)*100:5.1f}%")
    print(f"  embedding + groove              {cv(lr(), XG, y)*100:5.1f}%")


if __name__ == "__main__":
    main()
