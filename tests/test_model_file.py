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


def test_both_launchers_use_the_per_user_model():
    # The desktop .app launcher once still passed the repo's own
    # crate_model.json, so training from the app overwrote the bundled model.
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for name in ("crate.command", "desktop-app-launcher.sh"):
        text = (root / name).read_text()
        assert "user_model_path" in text, name
        assert "serve(connect(db), 'crate_model.json'" not in text, name
    launcher = (root / "desktop-app-launcher.sh").read_text()
    assert launcher.index('mkdir -p "$HOME/.crate"') < launcher.index("nohup")
