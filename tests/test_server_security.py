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
