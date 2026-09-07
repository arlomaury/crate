"""SQLite catalogue. The database is the product: crates are rows, not folders."""
import sqlite3
import threading
from pathlib import Path

# Shared by crateapp.server (the HTTP request thread) and crateapp.runner
# (the background analysis/classification thread) around every access to the
# one sqlite3 connection this module hands out. check_same_thread=False below
# only lifts sqlite3's own same-thread check - it does not make the
# connection safe for concurrent use from two threads, which is exactly what
# a run in progress plus an incoming request now is. Acquire this per unit of
# work (one query, one track's writes), never for a whole request or run.
LOCK = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    filename TEXT,
    size INTEGER,
    mtime REAL,
    analysed_at TEXT,
    bpm REAL, camelot TEXT, duration_sec REAL,
    standard_genre TEXT,
    missing INTEGER DEFAULT 0,
    error TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
    track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis (
    track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    moments TEXT, energy TEXT, vocal TEXT, genre_preds TEXT
);
CREATE TABLE IF NOT EXISTS crates (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS assignments (
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    crate_id INTEGER NOT NULL REFERENCES crates(id) ON DELETE CASCADE,
    source TEXT NOT NULL,               -- 'auto' | 'human'
    confidence REAL,
    band TEXT,                          -- 'confident' | 'uncertain'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (track_id, crate_id)
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    from_crate INTEGER,                 -- NULL for an also-add
    to_crate INTEGER NOT NULL,
    was_error INTEGER NOT NULL,         -- always 1 for a move
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE INDEX IF NOT EXISTS idx_assignments_crate ON assignments(crate_id);
CREATE INDEX IF NOT EXISTS idx_tracks_path ON tracks(path);
"""


def connect(path):
    """Open the catalogue, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the connection this returns is handed to the
    # HTTP server (crateapp.server), whose request-handling thread is not
    # the thread that called connect(). Access from that one server thread
    # is still effectively serial (HTTPServer handles one request at a
    # time), so this does not introduce concurrent use of the connection -
    # it only lifts sqlite3's same-thread check so a single-threaded HTTP
    # server can be handed a connection built during setup.
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    con.commit()
    return con
