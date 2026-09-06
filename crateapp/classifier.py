"""Nearest-centroid classification against the DJ's own crates.

Thresholds come from measurements on this library: tracks in a crate sit at
0.73+ cosine similarity to its centroid, while a genre never played sits at
0.34-0.48. UNKNOWN_FLOOR sits in that gap. MARGIN separates a clear winner
from a near-tie; it is deliberately generous because a wrong-but-confident
answer is worse than an honest "not sure".
"""
import json
from pathlib import Path

import numpy as np

UNKNOWN_FLOOR = 0.60
MARGIN = 0.02


class Classifier:
    def __init__(self, model_path):
        data = json.loads(Path(model_path).read_text())
        self.names = sorted(data["crates"])
        mat = np.array([data["crates"][n]["centroid"] for n in self.names],
                       dtype=np.float32)
        self.centroids = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    def crate_names(self):
        return list(self.names)

    def classify(self, vec):
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = self.centroids @ v
        order = np.argsort(sims)[::-1]
        best, second = int(order[0]), int(order[1]) if len(order) > 1 else None
        top = float(sims[best])
        margin = top - float(sims[second]) if second is not None else 1.0

        if top < UNKNOWN_FLOOR:
            band, crate = "unknown", None
        elif margin < MARGIN:
            band, crate = "uncertain", self.names[best]
        else:
            band, crate = "confident", self.names[best]

        return {"crate": crate, "similarity": round(top, 4),
                "margin": round(margin, 4), "band": band,
                "runner_up": self.names[second] if second is not None else None}
