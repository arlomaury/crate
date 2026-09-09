"""Deciding which of the DJ's crates a track belongs in.

Two signals, because they answer two different questions.

**Which crate** is a discriminative question, answered by multinomial logistic
regression over the Discogs-EffNet embedding. It replaced nearest-centroid,
which assumed each genre is a spherical blob around its mean.

Measured by stratified 5-fold cross-validation on the DJ's own Rekordbox
filing - 452 DISTINCT recordings, after collapsing the duplicate copies this
library is full of:

    nearest centroid (what shipped)        72.1%
    logistic regression, tuned             76.1%

An earlier run of this comparison reported 90.7%. That number was wrong. It
was measured before deduplication, so identical copies of the same recording
sat in the training fold and the test fold at once and the model was scoring
tracks it had memorised. Deduplicating first is not optional; see eval_genre.py.

Four points of accuracy is a modest gain, and the per-crate picture matters
more than the total. Precision on the crates that were visibly wrong:

    DUBSTEP   0.800 -> 1.000        (a fifth of that crate was not dubstep)
    UK Garage 0.385 -> 0.778        (worse than a coin flip before)
    vocals    1.000 -> 1.000
    rap       0.757 -> 0.944

**house vs tech house is 69% of all remaining error** - 81 of 117 mistakes,
between two crates holding 280 of the 452 labelled tracks. Merging them takes
the same model to 90.5%, which says the rest of the taxonomy is close to
solved and this one distinction is nearly all that is left. It may not be
solvable from audio at all: the DJ's filing here follows Beatport's tagging,
which is a labelling convention rather than a property of the sound.

Tried and rejected, so nobody repeats them: the full 400-label Discogs genre
distribution as features (71.7% alone, no gain concatenated - that head is
itself a linear layer on this embedding, so it carries strictly less
information), tempo features (+0.2%, inside the noise), PCA to 32-256 dims
(worse at every size), an RBF SVM (73.5%), and class_weight="balanced" (77.0%
overall, but it buys recall on small crates by giving up precision on them,
which is the wrong trade here - see `train`).

**Is this like anything the DJ owns at all** is a different question, and
logistic regression cannot answer it: a softmax sums to one, so it reports a
crate even for music unlike anything in the library - it has to pick
something. Distance to the nearest crate centroid does answer it, so the
centroids are kept for exactly that. This is what lets the app say "this is a
new sound" instead of forcing every track into an existing crate.

Every threshold below is measured. eval_genre.py and tune_genre.py print the
tables they came from.
"""
import json
from pathlib import Path

import numpy as np

# Below this cosine to the nearest crate, a track is unlike anything filed.
# Measured: across the labelled set the 1st percentile of cosine-to-nearest-
# crate is 0.574, so this sits just under it - a track below it is more
# unusual than 99% of the music the DJ has already sorted.
UNKNOWN_FLOOR = 0.56

# File it, or send it to review? Measured trade-off at the regularisation this
# model actually uses:
#
#     threshold   filed   precision on what was filed
#         0.50     79%        80.6%
#         0.60     60%        84.9%
#         0.70     40%        92.2%
#         0.80     19%        96.5%
#
# 0.70 is chosen because the complaint that prompted all of this was about
# precision - wrong tracks turning up inside a crate - and because a track
# left in the review queue costs a few seconds of the DJ's attention while a
# wrongly filed one has to be found first. It does mean the queue is long.
# Lower it to 0.60 to file more and review less.
#
# Note this cannot be carried over from a differently regularised model:
# at C=1.0 the same model saturates near 1.0 and 0.70 would file everything.
CONFIDENT_PROB = 0.70

# Only used when there is no trained layer yet (a brand new library). The old
# measured value: crate members sit at 0.73+ cosine to their centroid, so a
# clear winner needs to beat the runner-up by this much.
FALLBACK_MARGIN = 0.02

# Measured and then dropped: thresholding on the gap to the runner-up gives
# the same trade-off curve as thresholding on the top probability (0.90 margin
# and 0.95 probability both land on 96.0% precision), and it cannot bind
# anyway - if the top class is above 0.95 the gap is necessarily above 0.90.
# One threshold, not two.


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a / (np.linalg.norm(a) + 1e-9)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


class Classifier:
    """Inference only, and deliberately numpy-only.

    The weights are trained by `train` below (which needs scikit-learn) and
    stored in the model file, so predicting a crate never needs anything but
    numpy - one matrix multiply against a 1280-wide layer.
    """

    def __init__(self, model_path):
        data = json.loads(Path(model_path).read_text())
        self.names = sorted(data.get("crates", {}))
        mat = np.array([data["crates"][n]["centroid"] for n in self.names],
                       dtype=np.float32) if self.names else np.zeros((0, 0),
                                                                     np.float32)
        self.centroids = _unit(mat) if len(mat) else mat

        lin = data.get("linear")
        self.linear = None
        if lin:
            self.linear = {
                "classes": list(lin["classes"]),
                "w": np.asarray(lin["w"], dtype=np.float32),
                "b": np.asarray(lin["b"], dtype=np.float32),
                "mean": np.asarray(lin["mean"], dtype=np.float32),
                "scale": np.asarray(lin["scale"], dtype=np.float32),
            }

    def crate_names(self):
        return list(self.names)

    def trained(self):
        return self.linear is not None

    def _probabilities(self, v_unit):
        """Softmax over the trained layer. `v_unit` is already L2-normalised,
        matching the Normalizer -> StandardScaler -> LogisticRegression order
        the weights were fitted under."""
        m = self.linear
        z = (v_unit - m["mean"]) / m["scale"]
        logits = m["w"] @ z + m["b"]
        logits -= logits.max()
        e = np.exp(logits)
        return e / (e.sum() + 1e-12)

    def classify(self, vec):
        """`{crate, band, p, margin, similarity, scores}`.

        `band` is the honest part:
          unknown    - unlike anything filed; a candidate for a new crate
          uncertain  - filed, but the DJ should confirm it
          confident  - filed
        """
        v = _unit(vec)
        sims = self.centroids @ v if len(self.centroids) else np.zeros(0)
        best_cos = float(sims.max()) if len(sims) else 0.0

        if self.linear is None:
            # No trained layer yet (a fresh library, or a model file from
            # before training existed). Fall back to nearest centroid rather
            # than refuse to classify at all.
            if not len(sims):
                return {"crate": None, "band": "unknown", "p": 0.0,
                        "margin": 0.0, "runner_up": None,
                        "similarity": 0.0, "scores": []}
            order = np.argsort(sims)[::-1]
            top = self.names[int(order[0])]
            runner_up = self.names[int(order[1])] if len(order) > 1 else None
            second = float(sims[int(order[1])]) if len(order) > 1 else -1.0
            margin = best_cos - second
            scores = [{"crate": n, "p": float(sc)}
                      for n, sc in sorted(zip(self.names, sims),
                                          key=lambda t: -t[1])]
            if best_cos < UNKNOWN_FLOOR:
                band, crate = "unknown", None
            elif margin < FALLBACK_MARGIN:
                band, crate = "uncertain", top
            else:
                band, crate = "confident", top
            return {"crate": crate, "band": band, "p": round(best_cos, 4),
                    "margin": round(margin, 4), "runner_up": runner_up,
                    "similarity": round(best_cos, 4), "scores": scores}

        probs = self._probabilities(v)
        classes = self.linear["classes"]
        order = np.argsort(probs)[::-1]
        top_i = int(order[0])
        p_top = float(probs[top_i])
        p_second = float(probs[int(order[1])]) if len(order) > 1 else 0.0
        margin = p_top - p_second
        scores = [{"crate": classes[i], "p": float(probs[i])} for i in order]

        # Order matters: "unlike anything I own" outranks "which of these".
        if best_cos < UNKNOWN_FLOOR:
            band, crate = "unknown", None
        elif p_top < CONFIDENT_PROB:
            band, crate = "uncertain", classes[top_i]
        else:
            band, crate = "confident", classes[top_i]

        return {"crate": crate, "band": band, "p": round(p_top, 4),
                "margin": round(margin, 4),
                "runner_up": classes[int(order[1])] if len(order) > 1 else None,
                "similarity": round(best_cos, 4), "scores": scores}


def training_set(con, human_only=True):
    """`(ids, names, labels)` for the tracks the model may learn from.

    `human_only` is the whole point. The previous version of this trained on
    every assignment regardless of source, on the reasoning that "a
    confidently auto-filed track is still evidence". It is not: it is the
    model's own opinion, and feeding it back makes errors self-reinforcing.
    On this library it drove DUBSTEP to 40 reference tracks of which 2 were
    the DJ's, and the crate filled up with music that was not dubstep.

    A track the DJ has filed by hand, or confirmed, is evidence. Nothing else
    is.
    """
    sql = ("SELECT c.name, a.track_id FROM assignments a "
           "JOIN crates c ON c.id = a.crate_id")
    if human_only:
        sql += " WHERE a.source = 'human'"
    rows = con.execute(sql).fetchall()

    # A track filed under two crates is not a clean label for a single-label
    # model, so it trains nothing.
    seen = {}
    for r in rows:
        seen.setdefault(r["track_id"], set()).add(r["name"])
    ids, labels = [], []
    for tid, crates in seen.items():
        if len(crates) == 1:
            ids.append(tid)
            labels.append(next(iter(crates)))
    return ids, labels


def train(con, model_path, human_only=True, min_per_class=5):
    """Fit the layer and the centroids from the DJ's own filing, and save both.

    Returns a summary dict: what it trained on, and what it left out.
    """
    from crateapp.worker import load_embedding

    ids, labels = training_set(con, human_only=human_only)

    X, y = [], []
    for tid, lab in zip(ids, labels):
        v = load_embedding(con, tid)
        if v is not None:
            X.append(v)
            y.append(lab)
    if not X:
        # Same shape as every other return: a caller reading ["counts"] must
        # not have to know which branch produced the summary.
        return {"trained": False, "counts": {}, "too_thin": [], "n": 0,
                "human_only": human_only,
                "reason": "no labelled tracks with embeddings"}

    X = _unit(np.stack(X))
    y = np.array(y)

    # A DJ library holds the same recording several times over (a working
    # folder, an artist/album tree, "track (1).aiff"). Those copies are
    # byte-identical, so their embeddings are identical, and keeping them all
    # would silently reweight the class priors - on this library tech would
    # count 539 examples instead of its real 286 and pull the boundaries
    # toward itself. Every copy still gets filed; only training dedupes.
    _, first = np.unique(np.round(X, 5), axis=0, return_index=True)
    keep = np.sort(first)
    X, y = X[keep], y[keep]

    # A crate with almost no examples cannot be learned, and including it
    # mostly steals tracks from crates that can. Keep its centroid so the
    # crate still exists in the app - just do not let the layer predict it.
    counts = {c: int((y == c).sum()) for c in sorted(set(y))}
    usable = np.array([counts[c] >= min_per_class for c in y])
    thin = sorted(c for c, n in counts.items() if n < min_per_class)

    doc = {}
    path = Path(model_path)
    if path.exists():
        # Preserve provenance and anything else living at the top level -
        # overwriting the whole document would silently drop it.
        doc = json.loads(path.read_text())

    # Centroids for every crate, including the thin ones: they are the
    # "is this like anything I own" signal, and a mean needs far fewer
    # examples than a decision boundary.
    crates = {}
    for c in counts:
        crates[c] = {"n": counts[c],
                     "centroid": _unit(X[y == c].mean(axis=0)).tolist()}
    doc["crates"] = crates

    summary = {"trained": False, "counts": counts, "too_thin": thin,
               "n": int(usable.sum()), "human_only": human_only}

    if len(set(y[usable])) >= 2:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        Xu, yu = X[usable], y[usable]
        scaler = StandardScaler().fit(Xu)
        # C=0.001 is heavy regularisation, and it is not a guess: with ~450
        # distinct labelled recordings against 1280 dimensions this problem is
        # badly over-parameterised, and the sweep in tune_genre.py shows
        # accuracy climbing all the way down from C=3 (73.5%) to C=0.001
        # (76.1%) before collapsing below it.
        #
        # class_weight="balanced" scores higher still (77.0%) but is NOT used:
        # it buys recall on the small crates by giving up precision on them
        # (afro 1.000 -> 0.619, DUBSTEP 1.000 -> 0.958). Wrong tracks showing
        # up in a crate is the failure the DJ actually reported; a track the
        # model is unsure of goes to the review queue instead, which is the
        # cheaper mistake.
        lr = LogisticRegression(max_iter=5000, C=0.001).fit(
            scaler.transform(Xu), yu)

        w, b = lr.coef_, lr.intercept_
        # With exactly two crates scikit-learn fits ordinary binary logistic
        # regression and returns ONE row of coefficients, not one per class.
        # Taking a softmax over that single logit makes every track come out
        # as the first class. Expand it to the two-row form the softmax below
        # expects: the decision function is +z for the positive class,
        # -z for the other.
        if w.shape[0] == 1:
            w = np.vstack([-w[0], w[0]])
            b = np.array([-b[0], b[0]])

        doc["linear"] = {
            "classes": [str(c) for c in lr.classes_],
            "w": w.astype(np.float32).tolist(),
            "b": b.astype(np.float32).tolist(),
            "mean": scaler.mean_.astype(np.float32).tolist(),
            "scale": scaler.scale_.astype(np.float32).tolist(),
        }
        summary["trained"] = True
        summary["classes"] = list(map(str, lr.classes_))
    else:
        doc.pop("linear", None)
        summary["reason"] = "fewer than two crates have enough examples"

    path.write_text(json.dumps(doc))
    return summary


def rebuild_centroids(con, model_path):
    """Kept as the name the app calls after a correction; retrains everything.

    The DJ's correction is exactly the signal worth learning from, so a
    correction retrains the layer, not just the means. Returns the per-crate
    counts it trained on, which is what callers of the old name expect.
    """
    return train(con, model_path)["counts"]
