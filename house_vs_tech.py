#!/usr/bin/env python3
"""The one distinction that is left: house vs tech house.

Why only this pair
------------------
69% of the classifier's remaining errors are between these two crates, and
they hold 283 of the 520 labelled recordings. Merging them takes the same
model from 76.9% to 90.5%, so everything else is close to solved. Fixing this
pair is the whole remaining problem.

Three candidate signals, each with a reason to believe:

  embedding    what ships. Trained for timbre and production style, and it
               has plateaued - 364 to 520 labels bought 0.4 points.

  Discogs      the taxonomy has explicit House / Tech House / Techno labels,
               and it uses them well: when it says "Tech House" it is right
               47 times out of 53. A single House-minus-TechHouse axis
               separates the pair at 77.7% on its own. Throwing all 400
               labels at the 8-class problem did nothing, because 400 dims
               cannot compete with 1280 under heavy regularisation - but a
               handful of relevant ones, on this pair alone, is a different
               proposition.

  groove       the published EDM subgenre work (arXiv:2110.08862) finds the
               most important features for this task are all rhythm-related.
               Tech house and house share tempo and timbre but not feel.

Usage:
    python house_vs_tech.py         # needs groove_experiment.py run first
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import analyze
from crateapp.classifier import training_set
from crateapp.db import connect
from eval_genre import discogs400, load_features

CACHE = Path.home() / ".crate" / "groove.npz"

# The Discogs labels that speak to this distinction. Chosen because the
# taxonomy actually assigns them to these tracks, not because they sound
# relevant - see the counts printed by the exploration in the session log.
RELEVANT = [
    "Electronic---House", "Electronic---Tech House", "Electronic---Techno",
    "Electronic---Deep House", "Electronic---Minimal Techno",
    "Electronic---Progressive House", "Electronic---Tribal House",
    "Electronic---Electro House", "Electronic---Acid House",
    "Electronic---Bassline", "Electronic---Tech Trance",
    "Electronic---Hard Techno", "Electronic---Garage House",
    "Electronic---Deep Techno", "Electronic---Dub Techno",
    "Electronic---Micro House", "Electronic---Funky House",
    "Electronic---Disco", "Electronic---Nu-Disco", "Electronic---Breakbeat",
]


def cv(make, X, y, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    if min(Counter(y).values()) < folds:
        return None
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, y):
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean())


def repeated(make, X, y, n=5):
    """Averaged over several shuffles: on 283 samples a single split moves
    by a couple of points on nothing but the seed."""
    got = [cv(make, X, y, seed=s) for s in range(n)]
    got = [g for g in got if g is not None]
    return (float(np.mean(got)), float(np.std(got))) if got else (None, None)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    con = connect(Path.home() / ".crate" / "library.db")
    ids, labs = training_set(con)
    paths = {r["id"]: Path(r["path"])
             for r in con.execute("SELECT id, path FROM tracks").fetchall()}
    X, y, tids, names = load_features(
        con, {paths[i]: l for i, l in zip(ids, labs) if i in paths})
    _, first = np.unique(np.round(X, 5), axis=0, return_index=True)
    keep = np.sort(first)
    X, y, tids = X[keep], y[keep], [tids[i] for i in keep]

    D = discogs400(X)
    idx = {l: i for i, l in enumerate(analyze.GenreModel().labels)}
    cols = [idx[l] for l in RELEVANT if l in idx]
    Dsub = D[:, cols]
    print(f"{len(X)} labelled recordings; {len(cols)} relevant Discogs labels")

    G = None
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        have = {int(i): v for i, v in zip(z["ids"], z["feats"])}
        mask = np.array([t in have for t in tids])
        if mask.sum() > 100:
            G = np.stack([have[t] for t, m in zip(tids, mask) if m])
            X, y, Dsub, D = X[mask], y[mask], Dsub[mask], D[mask]
            tids = [t for t, m in zip(tids, mask) if m]
            print(f"groove features for {len(G)} of them")
    if G is None:
        print("(no groove cache yet - run groove_experiment.py first)",
              file=sys.stderr)

    m = np.isin(y, ["house", "tech"])
    yb = y[m]
    print(f"\n{'='*64}\nhouse vs tech house — {len(yb)} tracks "
          f"({dict(Counter(yb))})\n{'='*64}")
    base = max(Counter(yb).values()) / len(yb)
    print(f"  {'always guess the bigger crate':<40} {base*100:5.1f}%")

    def big(C=0.001):
        return lambda: make_pipeline(Normalizer(), StandardScaler(),
                                     LogisticRegression(max_iter=5000, C=C))

    def small(C=1.0):
        return lambda: make_pipeline(StandardScaler(),
                                     LogisticRegression(max_iter=5000, C=C))

    runs = [("embedding (what ships)", big(), X[m])]
    for C in (0.1, 1.0, 10.0):
        runs.append((f"Discogs labels only (C={C})", small(C), Dsub[m]))
    runs.append(("embedding + Discogs labels", big(), np.hstack([X, Dsub])[m]))
    if G is not None:
        runs.append(("groove only", small(1.0), G[m]))
        runs.append(("embedding + groove", big(), np.hstack([X, G])[m]))
        runs.append(("Discogs + groove", small(1.0), np.hstack([Dsub, G])[m]))
        runs.append(("everything", big(), np.hstack([X, Dsub, G])[m]))

    best = (0, None)
    for name, mk, feats in runs:
        mean, sd = repeated(mk, feats, yb)
        if mean is None:
            continue
        print(f"  {name:<40} {mean*100:5.1f}%  (+/-{sd*100:.1f})")
        if mean > best[0]:
            best = (mean, name)
    print(f"\n  best: {best[1]} at {best[0]*100:.1f}%")
    print(f"  the shipped embedding manages "
          f"{repeated(big(), X[m], yb)[0]*100:.1f}%")


if __name__ == "__main__":
    main()
