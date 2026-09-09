#!/usr/bin/env python3
"""Find the best settings for the crate classifier, and the ceiling above it.

Two questions this answers.

**How much is recoverable by tuning.** 452 distinct labelled recordings against
a 1280-dimensional embedding is a badly over-parameterised problem, so the
regularisation strength matters far more than the choice of model. Sweep it,
and sweep a PCA squeeze in front of it.

**How much is not.** house and tech house account for 69% of all errors (81 of
117). Those two crates may not be acoustically separable at all - the DJ's own
filing follows Beatport's tagging, which is a labelling convention rather than
a property of the sound. Scoring the same models with those two merged says
how much of the remaining error is that one distinction, and therefore how
much of it any amount of tuning could ever fix.

Usage:
    python tune_genre.py
"""
import numpy as np

from crateapp.db import connect
from eval_genre import (NearestCentroid, load_features, unit, xml_labels)
from pathlib import Path


def cv_accuracy(make, X, y, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=seed).split(X, y):
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean()), pred


def main():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    from sklearn.svm import SVC

    labels = xml_labels(Path.home() / "Documents" / "rek.xml")
    con = connect(Path.home() / ".crate" / "library.db")
    X, y, ids, names = load_features(con, labels)
    _, first = np.unique(np.round(X, 5), axis=0, return_index=True)
    keep = np.sort(first)
    X, y = X[keep], y[keep]
    print(f"{len(X)} distinct labelled recordings, {X.shape[1]} dimensions\n")

    def lr(C=1.0, weight=None, pca=None):
        steps = [Normalizer(), StandardScaler()]
        if pca:
            steps.append(PCA(n_components=pca, random_state=0))
        steps.append(LogisticRegression(max_iter=5000, C=C,
                                        class_weight=weight))
        return lambda: make_pipeline(*steps)

    print("regularisation sweep (plain logistic regression)")
    best = (0, None, None)
    for C in (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        acc, _ = cv_accuracy(lr(C), X, y)
        print(f"    C={C:<7} {acc*100:5.1f}%")
        if acc > best[0]:
            best = (acc, f"logistic regression C={C}", lr(C))

    print("\nPCA squeeze before the classifier")
    for n in (32, 64, 128, 256):
        acc, _ = cv_accuracy(lr(0.3, pca=n), X, y)
        print(f"    {n:>3} dims  {acc*100:5.1f}%")
        if acc > best[0]:
            best = (acc, f"PCA({n}) + logistic regression C=0.3", lr(0.3, pca=n))

    print("\nbalanced class weights (helps the small crates: afro, UKG, rap)")
    for C in (0.03, 0.1, 0.3):
        acc, _ = cv_accuracy(lr(C, weight="balanced"), X, y)
        print(f"    C={C:<7} {acc*100:5.1f}%")
        if acc > best[0]:
            best = (acc, f"logistic regression C={C}, balanced", lr(C, "balanced"))

    print("\nRBF support vector machine")
    for C in (1.0, 10.0):
        mk = (lambda c: lambda: make_pipeline(
            Normalizer(), StandardScaler(), SVC(C=c, gamma="scale")))(C)
        acc, _ = cv_accuracy(mk, X, y)
        print(f"    C={C:<7} {acc*100:5.1f}%")
        if acc > best[0]:
            best = (acc, f"RBF SVM C={C}", mk)

    base, _ = cv_accuracy(NearestCentroid, X, y)
    print(f"\n    nearest centroid (what shipped)   {base*100:5.1f}%")
    print(f"\nBest: {best[1]} at {best[0]*100:.1f}%")

    # ---- the ceiling ----------------------------------------------------
    print("\n" + "=" * 70)
    print("How much of the error is house-vs-tech alone")
    print("=" * 70)
    merged = np.array(["house/tech" if v in ("house", "tech") else v for v in y])
    acc_m, _ = cv_accuracy(best[2], X, merged)
    base_m, _ = cv_accuracy(NearestCentroid, X, merged)
    print(f"    with house and tech treated as one crate:")
    print(f"        best model      {acc_m*100:5.1f}%   (was {best[0]*100:.1f}%)")
    print(f"        nearest centroid{base_m*100:5.1f}%   (was {base*100:.1f}%)")
    print(f"\n    So {(acc_m-best[0])*100:.1f} points of the error is that one "
          f"distinction.\n    Everything else together accounts for "
          f"{(1-acc_m)*100:.1f}%.")

    # ---- per-crate detail for the winner --------------------------------
    from sklearn.metrics import classification_report
    _, pred = cv_accuracy(best[2], X, y)
    print("\n" + "=" * 70)
    print(f"{best[1]} — per crate")
    print("=" * 70)
    print(classification_report(y, pred, zero_division=0, digits=3))


if __name__ == "__main__":
    main()
