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
    """The shipped model's crate names ARE the app's crate names - a mismatch
    means classify() files tracks into crates the UI never shows. These are
    the DJ's own folder names, not display labels of our choosing."""
    c = Classifier("crate_model.json")
    assert sorted(c.crate_names()) == [
        "DUBSTEP", "UKG", "afro", "house", "pop", "rap", "tech", "vocals"]
