"""The local API must only answer Crate's own page, not other websites open in
the same browser. See Handler._guard in crateapp/server.py."""
import json
import threading
import urllib.error
import urllib.request

import pytest
from http.server import ThreadingHTTPServer

from crateapp.db import connect
from crateapp.server import make_app


@pytest.fixture
def base(tmp_path):
    con = connect(tmp_path / "l.db")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_app(con, tmp_path / "m.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def status(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.headers


def test_own_page_is_allowed(base):
    assert status(base + "/api/state")[0] == 200
    assert status(base + "/api/state", headers={"Host": "localhost:8420"})[0] == 200


def test_same_origin_post_is_allowed(base):
    code, _ = status(base + "/api/lock", data=json.dumps({"crate": "x"}).encode(),
                     headers={"Content-Type": "application/json", "Origin": base})
    assert code == 404          # reached the handler: no such crate


def test_dns_rebinding_host_is_refused(base):
    assert status(base + "/api/state", headers={"Host": "evil.example:8420"})[0] == 403


def test_foreign_origin_is_refused(base):
    code, _ = status(base + "/api/scan", data=json.dumps({"folder": "/"}).encode(),
                     headers={"Content-Type": "application/json",
                              "Origin": "https://evil.example"})
    assert code == 403


def test_post_without_json_content_type_is_refused(base):
    # A cross-site <form> or no-preflight fetch can only send text/plain or
    # form encodings. Those must never reach a handler that writes to disk.
    for ctype in ("text/plain", "application/x-www-form-urlencoded"):
        code, _ = status(base + "/api/export",
                         data=json.dumps({"dest": "/tmp/x"}).encode(),
                         headers={"Content-Type": ctype})
        assert code == 415


def test_security_headers_present(base):
    _, headers = status(base + "/api/state")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    csp = headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "object-src 'none'" in csp


def test_oversized_body_refused(base):
    code, _ = status(base + "/api/lock", data=b"{}",
                     headers={"Content-Type": "application/json", "Content-Length": "50000000"})
    assert code == 413


def test_non_finite_preview_position_refused(base):
    assert status(base + "/api/preview/1?at=inf")[0] == 400
    assert status(base + "/api/preview/1?at=nan")[0] == 400


@pytest.mark.parametrize("path,payload", [
    ("/api/scan", {"folder": 5}),
    ("/api/correct", {"track_id": 1, "to_crate": {"a": 1}}),
    ("/api/correct", {"track_id": "1", "to_crate": "House"}),
    ("/api/lock", {"crate": ["x"]}),
    ("/api/remove", {"track_id": {"x": 1}}),
    ("/api/set", {"seed_id": 1, "length": None, "mode": {}}),
    ("/api/set", {"seed_id": 1, "length": 10 ** 9}),
    ("/api/next", {"track_id": 1, "crate": ["x"]}),
    ("/api/export", {"dest": 7}),
    ("/api/export", {"dest": "/", "kind": "rekordbox"}),          # a folder, not a file
])
def test_bad_field_types_are_400_not_500(base, path, payload):
    code, _ = status(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    assert code == 400, (path, payload, code)


def test_huge_track_id_is_400(base):
    assert status(base + "/api/track/99999999999999999999")[0] == 400


def test_other_localhost_ports_are_refused(base):
    code, _ = status(base + "/api/lock", data=json.dumps({"crate": "x"}).encode(),
                     headers={"Content-Type": "application/json", "Origin": "http://localhost:3000"})
    assert code == 403


def test_huge_preview_position_is_not_a_500(base):
    assert status(base + "/api/preview/1?at=1e308")[0] == 404     # clamped; then no such track


def test_rekordbox_export_never_overwrites_other_files(base, tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"RIFF....WAVE")
    other_xml = tmp_path / "notes.xml"
    other_xml.write_text("<notes/>")
    for dest in (str(song), str(other_xml), str(tmp_path), "relative/rekordbox.xml"):
        code, _ = status(base + "/api/export", data=json.dumps({"kind": "rekordbox", "dest": dest}).encode(),
                         headers={"Content-Type": "application/json"})
        assert code == 400, dest
    assert song.read_bytes() == b"RIFF....WAVE" and other_xml.read_text() == "<notes/>"
    out = tmp_path / "out" / "rekordbox.xml"
    for _ in range(2):                                   # re-exporting over our own file is fine
        req = urllib.request.Request(base + "/api/export",
                                     data=json.dumps({"kind": "rekordbox", "dest": str(out)}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["dest"] == str(out.resolve())
    assert b"<DJ_PLAYLISTS" in out.read_bytes()
