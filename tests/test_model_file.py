import json

from crateapp.classifier import Classifier
from crateapp.model_file import read_model, user_model_path, write_model


def test_new_user_starts_with_no_model(tmp_path):
    bundled = tmp_path / "repo_model.json"
    bundled.write_text(json.dumps({"crates": {"house": {"n": 1, "centroid": [1.0]}}}))
    model = user_model_path(tmp_path / "crate", bundled)
    assert model == tmp_path / "crate" / "crate_model.json"
    assert not model.exists()                 # someone else's crates are not imposed


def test_existing_library_is_seeded_from_the_bundled_model(tmp_path):
    bundled = tmp_path / "repo_model.json"
    bundled.write_text(json.dumps({"crates": {}}))
    (tmp_path / "crate").mkdir()
    (tmp_path / "crate" / "library.db").write_bytes(b"")
    model = user_model_path(tmp_path / "crate", bundled)
    assert model.read_text() == bundled.read_text()
    bundled.write_text(json.dumps({"crates": {"changed": {}}}))
    user_model_path(tmp_path / "crate", bundled)           # never overwritten once there
    assert json.loads(model.read_text()) == {"crates": {}}


def test_truncated_model_reads_as_empty_and_does_not_crash(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"crates": {"hou')
    assert read_model(p) == {}
    assert Classifier(p).crate_names() == []
    write_model(p, {"crates": {}})
    assert read_model(p) == {"crates": {}}
    assert [f.name for f in tmp_path.iterdir()] == ["m.json"]   # no temp files left
