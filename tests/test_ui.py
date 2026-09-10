"""JavaScript behaviour that shipped broken once, guarded from here on.

These run the real app.js under node with a minimal DOM stub. They exist
because the browser is not always reachable from an automated session, and
because `node --check` only proves the file parses - it says nothing about
whether the panel quietly rebuilds itself sixty times a minute.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "crateapp" / "static" / "app.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is not installed")


def run(script):
    return subprocess.run(["node", str(ROOT / "tests" / "ui" / script), str(APP)],
                          capture_output=True, text=True, timeout=60)


def test_app_js_parses():
    r = subprocess.run(["node", "--check", str(APP)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_panel_does_not_rebuild_itself_on_every_poll():
    """The reported bug: the per-genre numbers "reset and go back every once
    in a while". poll() calls renderPanel every 600ms during a run, and the
    genre column was rebuilt each time - collapsing every bar to zero and
    re-animating it, and resetting the crate picker sitting underneath.

    The same run also checks the guard is not too aggressive: a different
    track, or the same track once confirmed, must still rebuild.
    """
    r = run("panel_does_not_rebuild.js")
    assert r.returncode == 0, r.stdout + r.stderr
