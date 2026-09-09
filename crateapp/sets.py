"""Building a set out of the library.

`mixing.py` answers "do these two records mix". This answers "what do I play
next, and then what after that" - which is the same question asked repeatedly,
because a set is played in order and each transition only has to work against
the one immediately before it.

So the search is greedy rather than global: from the track in the deck, score
every record still available and take the best. A globally optimal ordering
would score better on paper and be worse in practice, because it can only be
reached by playing every track in a fixed order - the moment the DJ deviates,
the plan is void. Greedy gives an order where every step is individually the
right call, which is also what a DJ actually does at the decks.
"""
import json

import numpy as np

from crateapp.mixing import mix_points, score_transition


def energy_level(curve):
    """One number for how big a track is, from its energy curve.

    The mean, not the peak: peak is nearly 1.0 for anything with a drop in it
    and so cannot order tracks against each other.
    """
    if not curve:
        return None
    return round(float(np.mean(np.asarray(curve, dtype="float32"))), 4)


def load_pool(con, crate=None, limit=None):
    """Every track that can be placed in a set, with everything needed to score it.

    A track with no tempo or key cannot be beatmatched or key-matched, so it is
    not a candidate however good it is - it is excluded here rather than
    scored at zero, which would leave it cluttering the results.
    """
    sql = ("SELECT t.id, t.filename, t.path, t.bpm, t.camelot, t.duration_sec, "
           "       an.moments, an.energy, an.vocal, e.vector "
           "FROM tracks t "
           "JOIN analysis an ON an.track_id = t.id "
           "LEFT JOIN embeddings e ON e.track_id = t.id ")
    args = []
    if crate:
        sql += ("JOIN assignments a ON a.track_id = t.id "
                "JOIN crates c ON c.id = a.crate_id AND c.name = ? ")
        args.append(crate)
    sql += ("WHERE t.analysed_at IS NOT NULL AND t.missing = 0 "
            "  AND t.bpm IS NOT NULL AND t.camelot IS NOT NULL "
            "ORDER BY t.id")
    if limit:
        sql += " LIMIT ?"
        args.append(int(limit))

    pool = []
    for r in con.execute(sql, args).fetchall():
        vocal = json.loads(r["vocal"] or "{}")
        curve = json.loads(r["energy"] or "[]")
        pool.append({
            "id": r["id"],
            "filename": r["filename"],
            "path": r["path"],
            "bpm": r["bpm"],
            "camelot": r["camelot"],
            "duration": r["duration_sec"],
            "moments": json.loads(r["moments"] or "[]"),
            "energy": energy_level(curve),
            "vocal_ratio": vocal.get("vocal_ratio"),
            "vec": (np.frombuffer(r["vector"], dtype="float32").copy()
                    if r["vector"] else None),
        })
    return pool


# Measured across 438 analysed tracks of the real library: byte-identical
# copies score exactly 1.0000, the same song from a different rip or with a
# different filename scores 0.9958-0.9960, and the closest pair of genuinely
# different records - two tracks in the same style, by different artists -
# scores 0.9785. The cut goes in that gap. Raising it above 0.9958 would let
# same-song pairs through; lowering it below 0.9785 would start deleting real
# tracks from the DJ's library, which is far worse than leaving a duplicate in.
DUPLICATE_COSINE = 0.99


def dedupe(pool, threshold=DUPLICATE_COSINE):
    """Collapse copies of the same recording down to one entry.

    A DJ library accumulates the same track many times over - the working
    folder plus an artist/album tree plus 'Track (1).aiff'. Left in, every
    track's own copy ranks as its best possible next track (same key, same
    tempo, cosine 1.0), which is both the top of every list and completely
    unplayable.

    Filenames cannot decide this: the copies frequently differ in name, and
    two genuinely different records occasionally share one. The embedding can,
    because it is computed from the audio.
    """
    vecs = [t for t in pool if t.get("vec") is not None]
    if len(vecs) < 2:
        return list(pool)

    m = np.stack([t["vec"] for t in vecs]).astype("float32")
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sim = m @ m.T

    # Of a group of copies, keep the one a DJ would actually reach for: the
    # shallowest path, which is the flat working folder rather than a nested
    # artist/album tree, breaking ties on the shorter name.
    def rank(t):
        return (t["path"].count("/"), len(t["path"]))

    keep, dropped = [], set()
    for i, t in enumerate(vecs):
        if i in dropped:
            continue
        group = [t] + [vecs[j] for j in np.nonzero(sim[i] >= threshold)[0]
                       if j != i and j not in dropped]
        for j in np.nonzero(sim[i] >= threshold)[0]:
            if j != i:
                dropped.add(int(j))
        winner = min(group, key=rank)
        winner = dict(winner)
        winner["duplicates"] = len(group) - 1
        keep.append(winner)

    keep.extend(t for t in pool if t.get("vec") is None)
    keep.sort(key=lambda t: t["id"])
    return keep


def _public(t):
    """The track without its embedding - a 1280-float vector has no business
    crossing into JSON, and nothing downstream reads it."""
    return {k: v for k, v in t.items() if k != "vec"}


def _step(a, b, mode, arc):
    """Score, explain and place one transition from `a` into `b`."""
    r = score_transition(a, b, mode=mode, want=arc)
    r["mix"] = mix_points(a["moments"], a["duration"], b["moments"], a["bpm"])
    return r


def neighbours(pool, seed_id, mode="balanced", arc="steady", limit=25):
    """Everything that mixes out of `seed_id`, best first."""
    by_id = {t["id"]: t for t in pool}
    seed = by_id.get(seed_id)
    if seed is None:
        raise ValueError(f"track {seed_id} is not in the pool")

    out = []
    for t in pool:
        if t["id"] == seed_id:
            continue
        r = _step(seed, t, mode, arc)
        out.append({"track": _public(t), "score": r["score"],
                    "reasons": r["reasons"], "mix": r["mix"],
                    "stretch_pct": r["stretch_pct"]})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:limit]


def build_set(pool, seed_id, length=8, mode="balanced", arc="steady"):
    """A playable order starting from `seed_id`.

    Returns the tracks in order, each carrying the transition that got it
    there, plus the weakest link - the transition a DJ should practise or
    swap out.
    """
    by_id = {t["id"]: t for t in pool}
    if seed_id not in by_id:
        raise ValueError(f"track {seed_id} is not in the pool")

    chosen = [by_id[seed_id]]
    used = {seed_id}
    steps = [None]

    while len(chosen) < length:
        current = chosen[-1]
        best, best_r = None, None
        for t in pool:
            if t["id"] in used:
                continue
            r = _step(current, t, mode, arc)
            if best_r is None or r["score"] > best_r["score"]:
                best, best_r = t, r
        if best is None:
            break                    # pool exhausted - a short set is honest
        chosen.append(best)
        steps.append(best_r)
        used.add(best["id"])

    tracks = []
    for i, (t, r) in enumerate(zip(chosen, steps)):
        d = _public(t)
        d["position"] = i
        d["from_previous"] = r
        tracks.append(d)

    scored = [(i, s["score"]) for i, s in enumerate(steps) if s]
    weakest = None
    if scored:
        pos, score = min(scored, key=lambda x: x[1])
        weakest = {"position": pos, "score": score,
                   "reasons": steps[pos]["reasons"]}

    return {
        "tracks": tracks,
        "mode": mode,
        "arc": arc,
        "average_score": (round(sum(s for _, s in scored) / len(scored), 4)
                          if scored else None),
        "weakest": weakest,
    }
