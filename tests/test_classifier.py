import json
import numpy as np
import pytest
from crateapp.classifier import Classifier


@pytest.fixture
def model(tmp_path):
    """Two crates pointing in clearly different directions in embedding space."""
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "crates": {"House": {"n": 50, "centroid": a.tolist()},
                   "Dubstep": {"n": 40, "centroid": b.tolist()}}}))
    return Classifier(p)


def test_a_track_near_a_centroid_is_confident(model):
    v = np.zeros(8, dtype=np.float32); v[0] = 1.0
    out = model.classify(v)
    assert out["crate"] == "House" and out["band"] == "confident"


def test_a_track_between_two_crates_is_uncertain(model):
    # v[1]=0.985 (not the brief's 0.97): after L2-normalization, x=0.97 yields
    # a cosine margin of ~0.0215, which is *above* MARGIN=0.02 and misclassifies
    # as confident. The margin only drops below 0.02 for x >~ 0.972. 0.985 gives
    # a comfortable, unambiguous near-tie (margin ~0.011) that exercises the
    # intended "uncertain" path without touching UNKNOWN_FLOOR/MARGIN themselves.
    v = np.zeros(8, dtype=np.float32); v[0] = 1.0; v[1] = 0.985
    out = model.classify(v)
    assert out["band"] == "uncertain"
    assert out["runner_up"] in ("House", "Dubstep")


def test_a_track_far_from_everything_is_unknown(model):
    """A genre never played before must not be forced into an existing crate."""
    v = np.zeros(8, dtype=np.float32); v[7] = 1.0
    out = model.classify(v)
    assert out["band"] == "unknown"


def test_an_uncertain_track_still_gets_a_crate(model):
    """Spec section 4: uncertain tracks are sorted anyway, just flagged."""
    v = np.zeros(8, dtype=np.float32); v[0] = 1.0; v[1] = 0.985
    assert model.classify(v)["crate"] is not None


def test_an_unknown_track_gets_no_crate(model):
    v = np.zeros(8, dtype=np.float32); v[7] = 1.0
    assert model.classify(v)["crate"] is None


def test_crate_names_are_listed(model):
    assert sorted(model.crate_names()) == ["Dubstep", "House"]


def test_the_real_shipped_model_loads():
    """Structural invariants of the real model file.

    Deliberately NOT a frozen list of crate names: the DJ creates crates as
    they work (they have added 'rock' and 'more chill' since), so pinning the
    names made this test fail on their normal use of the app rather than on
    any defect.

    What must hold is that every crate the trained layer can predict also has
    a centroid - otherwise classify() can name a crate it has no distance
    measurement for, and the "is this like anything I own" check silently
    reads from the wrong row."""
    c = Classifier("crate_model.json")
    assert c.crate_names(), "the shipped model has no crates"
    assert c.trained(), "the shipped model has no trained layer"
    missing = set(c.linear["classes"]) - set(c.crate_names())
    assert not missing, f"predictable crates with no centroid: {missing}"
    assert c.linear["w"].shape[0] == len(c.linear["classes"])
    assert c.linear["w"].shape[1] == c.centroids.shape[1]


# ------------------------------------------- crates the DJ treats as alike

def _trained(tmp_path, classes, weights=None):
    """A model file with a hand-built linear layer, so the banding rules can
    be tested without fitting anything."""
    import numpy as np
    n = len(classes)
    # Gentle weights on purpose: a steep layer turns any input into a
    # near-certain answer, which would make these banding tests pass for the
    # wrong reason. Each test asserts the probability split it relies on.
    w = np.eye(n, 8, dtype=np.float32) * 2.0
    return {
        "crates": {c: {"n": 20, "centroid": np.eye(n, 8)[i].tolist()}
                   for i, c in enumerate(classes)},
        "linear": {"classes": list(classes), "w": w.tolist(),
                   "b": [0.0] * n, "mean": [0.0] * 8, "scale": [1.0] * 8},
    }


def test_a_split_between_house_and_tech_is_still_filed(tmp_path):
    """The DJ says a house/tech mix-up is not worth their time, so a track
    torn between exactly those two should be filed, not queued."""
    import numpy as np
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_trained(tmp_path, ["house", "pop", "tech"])))
    m = Classifier(p)
    v = np.zeros(8, dtype=np.float32); v[0] = 0.52; v[2] = 0.48
    out = m.classify(v)
    by = {s["crate"]: s["p"] for s in out["scores"]}
    assert abs(by["house"] - by["tech"]) < 0.1, "fixture must be a near-tie"
    assert out["band"] == "confident"
    assert out["crate"] in ("house", "tech")
    assert out["p_family"] > out["p"], "family mass must exceed the single crate"


def test_a_split_across_families_goes_to_review(tmp_path):
    """The same near-tie between crates that are NOT alike must be queued -
    this is the rap-filed-as-pop case the DJ complained about."""
    import numpy as np
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_trained(tmp_path, ["house", "pop", "tech"])))
    m = Classifier(p)
    v = np.zeros(8, dtype=np.float32); v[0] = 0.52; v[1] = 0.48
    out = m.classify(v)
    by = {s["crate"]: s["p"] for s in out["scores"]}
    assert abs(by["house"] - by["pop"]) < 0.1, "fixture must be a near-tie"
    assert out["band"] == "uncertain"


def test_a_crate_in_no_family_is_judged_on_its_own(tmp_path):
    import numpy as np
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_trained(tmp_path, ["house", "pop", "tech"])))
    out = Classifier(p).classify(np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype="float32"))
    assert out["family"] is None
    assert out["p_family"] == out["p"]


# ------------------------------------------------------------- acapellas

def test_a_confident_acapella_goes_to_vocals():
    """An isolated vocal has no drum pattern, bassline or production style
    left, so a genre model trained on full mixes guesses from timbre alone.
    Routing them out before the genre question is a stated requirement - and
    before this rule 20 acapellas had been auto-filed as pop, house and rap."""
    from crateapp.classifier import acapella_verdict
    v = {"is_acapella": True, "needs_review": False, "score": 1.0}
    assert acapella_verdict(v, ["house", "vocals"]) == "vocals"


def test_a_full_mix_is_left_to_the_genre_model():
    from crateapp.classifier import acapella_verdict
    assert acapella_verdict({"is_acapella": False}, ["house", "vocals"]) is None
    assert acapella_verdict(None, ["house", "vocals"]) is None


def test_a_track_merely_flagged_for_review_is_not_overridden():
    """2-of-3 signals means 'unsure', and flag-rather-than-guess cuts both
    ways: an unsure acapella call must not override the genre model either."""
    from crateapp.classifier import acapella_verdict
    v = {"is_acapella": False, "needs_review": True, "score": 0.5}
    assert acapella_verdict(v, ["house", "vocals"]) is None


def test_no_vocals_crate_means_no_override():
    """Never invent a crate the DJ does not have."""
    from crateapp.classifier import acapella_verdict
    assert acapella_verdict({"is_acapella": True}, ["house", "tech"]) is None
