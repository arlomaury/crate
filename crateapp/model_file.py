"""Where the trained model lives, and writing it safely.

The model is per user: ~/.crate/crate_model.json, next to the library
database it was trained from. The crate_model.json in the repository is the
author's own trained model; it is never written to, so pulling updates never
conflicts with your training, and a new user starts from their own crates
rather than someone else's.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path


def user_model_path(crate_dir, bundled=None):
    """Return the model path for this user, seeding it on first launch.

    Seeds from the bundled model only for a library that already existed
    before this change (the author's own install, where that file IS their
    training). A brand-new library starts with no model: nothing is filed
    until the DJ has filed a few tracks of their own.
    """
    crate_dir = Path(crate_dir).expanduser()
    crate_dir.mkdir(parents=True, exist_ok=True)
    model = crate_dir / "crate_model.json"
    library = crate_dir / "library.db"
    if not model.exists() and bundled and Path(bundled).is_file() and library.exists():
        try:
            json.loads(Path(bundled).read_text())
            shutil.copyfile(bundled, model)
        except (OSError, ValueError):
            pass
    return model


def read_model(path):
    """The model document, or {} if it is missing or unreadable (a crash
    mid-write on an older version could leave it truncated)."""
    try:
        doc = json.loads(Path(path).read_text())
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def write_model(path, doc):
    """Write through a temp file and rename, so a reader never sees half a
    file and a crash never leaves a truncated one."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".crate_model.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
