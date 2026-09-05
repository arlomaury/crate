# Crate — local sorting app design

**Date:** 2026-09-04
**Status:** design, awaiting review. No code written yet.

---

## 1. What this is

A local web app that takes any number of audio files and sorts them into the
DJ's own crates, learning from corrections as it goes.

Today `analyze.py` is a one-shot command-line script pointed at one folder. This
turns it into something that runs continuously, remembers what it has already
done, holds a library of any size, and gets better at *this specific DJ's*
categories every time they correct it.

### Success criteria

1. Point it at a folder once. Never pick that folder again.
2. Add 10,000 more tracks later; only the new ones get analysed.
3. Tracks it isn't confident about go somewhere visible, never silently misfiled.
4. Correcting a track measurably improves how similar tracks are sorted afterwards.
5. The real music folder is never modified.

### Explicit non-goal: "always right"

The app will not always be right, and must never present itself as if it were.
Genre is a human judgement that professional sources actively disagree on —
measured during validation on this library:

- Beatport treats **UK Garage / Bassline** as one genre; Discogs splits it into
  three (UK Garage, Speed Garage, Bassline).
- `Yosemite` is **UK Garage/Speed Garage** as the Original Mix and **Melodic
  House & Techno** as the Extended Mix — same source, same track, different tag.
- `Kettama – Picanya 2400` carries three conflicting tags across platforms
  (Dance/Pop, Deep House, Electronica).

There is no single correct answer to recover. The design goal is therefore
**"right by this DJ's standard, and honest when unsure"** — which is learnable —
not universal correctness, which is not.

---

## 2. What already exists and gets reused

All of this is built and validated; the app wraps it rather than replacing it.

| Component | Status |
|---|---|
| `analyze.py` — tempo, key, structure, cue points | Working; ~3.3 s/track on M4 Pro |
| Discogs-EffNet embedding + 400-label genre head | Working, models installed |
| Reference-centroid classification | Validated: **80% leave-one-out** across 5 crates, 130 tracks |
| Afro House taxonomy-gap signal | Validated: 70% recall, ~18% FP |
| Garage/tech-house conflict signal | Validated: 4/5 confirmed catches, ~6% FP |
| Acapella detection | Validated: **27/27**, 0/130 false positives |
| Viewer HTML | Renders correctly, cue markers verified against energy curve |
| Rekordbox XML export | Generates; **import round-trip still unverified** |

---

## 3. Core decision: taste first, standard genres second

Measured on the same 130 ground-truth tracks:

| Classifier | Accuracy |
|---|---|
| Discogs-400 standard genre head | 28% exact label |
| Personal reference centroids | **80% leave-one-out** |

The DJ's own categories win decisively, because the question that matters is
"which of *my* crates is this?" and not "which of Discogs' 400 labels is this?"
The standard taxonomy also cannot express Afro House, Riddim, or Bass House at
all, and splits garage in a way this DJ does not.

**Therefore:**

- **Primary classifier:** nearest personal crate centroid.
- **Secondary:** the Discogs head, kept for exactly two jobs — cold start (before
  any crate exists) and *naming* tracks that match no crate.

Both run off the same single embedding pass, so keeping both costs no extra
analysis time.

---

## 4. Classification pipeline

Per track, once:

```
audio file
  └─ analyze.py  ──> tempo, key, structure, cue points, acapella check
  └─ 16 kHz mono ──> Discogs-EffNet embedding  (ONE forward pass)
                       ├─ 400-label genre head ──> standard genre name
                       └─ mean-pooled vector   ──> cosine vs every crate centroid
                                                     ↓
                                          best crate + similarity
                                                     ↓
                                     ┌───────────────┴───────────────┐
                              sim >= confident                 sim < confident
                                     ↓                                ↓
                            file into that crate              Unsorted queue
                                                        (+ standard genre as a hint)
```

### Open-set rejection

Nearest-centroid always returns something, so similarity must be thresholded or
unfamiliar music gets force-filed. Measured separations:

| Comparison | Similarity |
|---|---|
| Afrohouse → Afrohouse centroid (same crate) | 0.725 – 0.928 |
| Rap → Afrohouse centroid (distant genre) | **0.338 – 0.481** |
| Tech House → Afrohouse centroid (adjacent genre) | 0.666 – 0.923 |

Genuinely unfamiliar music sits far away; adjacent genres overlap heavily. So
rejection reliably catches **"a genre never played before"** — the actual
requirement — but will not separate near neighbours, which is left to the human.

Three bands:

| Band | Behaviour |
|---|---|
| **Confident** — clear best match, well above floor | Auto-filed into that crate |
| **Uncertain** — matches a crate weakly, or two crates nearly tie | Provisionally filed, *and* shown in the review queue marked unconfirmed — never presented as a settled answer |
| **Unknown** — below the floor for every crate | Held in Unsorted, labelled with its standard genre |

Thresholds start deliberately conservative — over-asking rather than
over-assuming — and get calibrated against real data (see §10).

Existing `needs_review` signals continue to feed this queue: low model
confidence, the Afro House taxonomy gap, the garage/tech-house conflict, and
possible-acapella.

---

## 5. The learning loop

This is the heart of the app.

1. A track lands in **Unsorted** (or is flagged as uncertain).
2. The DJ sees it with everything known about it: standard genre guess, BPM, key,
   nearest crates and their similarities, and playable audio.
3. They assign it — to an existing crate, **or to a brand-new crate created right
   there in the queue**, without leaving the flow.
4. That assignment is stored as ground truth and the crate's centroid is
   recomputed to include it.
5. Every subsequent track is classified against the improved centroid.

A crate becomes usable at roughly 5–10 tracks; Afro House reached 70% recall from
23. Corrections are permanent and cumulative — the app is strictly better at the
DJ's taste every week than the week before.

### First run — bootstrapping from folders that already exist

The DJ has already hand-sorted five folders (`RAP`, `TECH HOUSE`, `AFROHOUSE`,
`UKG`, `DUBSTEP`). On first run the app offers to import any such folder as a
crate, with every track in it as confirmed ground truth.

This matters: it means the app is useful on day one rather than after weeks of
correcting. Those five folders are exactly the 130 tracks that produced the 80%
leave-one-out figure, so the accuracy is known in advance for this user.

For a DJ with **no** sorted folders, the cold start is the Discogs head alone:
everything lands in Unsorted labelled with a standard genre, cluster suggestion
proposes crates, and the taste model takes over as crates fill.

### Cluster suggestion

When several Unsorted tracks are mutually similar and share a standard-genre
label, the app proposes a crate rather than waiting to be told:

> *"12 tracks don't match any crate you have, but they sound like each other.
> The genre model calls most of them Drum n Bass. Make a crate?"*

This is how genres the DJ has never played get discovered instead of forced into
existing crates.

---

## 6. Crates are virtual

Sorting moves nothing. A crate is a set of references in the database.

Consequences, all good:
- The real music folder is **never written to**. Originals cannot be lost, moved,
  or scattered by a wrong guess.
- Re-sorting is instant — no file I/O.
- A track can belong to several crates at once.
- A mistake costs one click, not a file move to reverse.

### Opt-in export to real folders

A deliberate button, never automatic. Creates real folders of real audio files
using **APFS copy-on-write clones** (`cp -c`).

Verified on this machine: cloning a 64 MB track produced a byte-identical file
with **no change in free disk space** (643 GB before and after). Real browsable
folders, originals untouched, at effectively zero disk cost. Falls back to a
normal copy on non-APFS volumes, warning about size first.

### Rekordbox export

Retained as today: genre, BPM, key, and a memory cue at each detected moment.
**Note:** the XML import round-trip has never been tested. If `POSITION_MARK`
cues fail to import, the suspects are the `Type` attribute or needing `Num="0"`
and up rather than `-1`. This is a one-line fix requiring no re-analysis.

---

## 7. Architecture

```
┌──────────────────────────────────────────────┐
│  Browser UI  (localhost)                     │
│  crates · unsorted queue · review · exports  │
└───────────────────┬──────────────────────────┘
                    │ HTTP (JSON)
┌───────────────────┴──────────────────────────┐
│  Local server (Python, stdlib http.server)   │
│   ├─ library scanner   (finds new files)     │
│   ├─ analysis worker   (background, 1 track  │
│   │                     at a time, resumable)│
│   ├─ classifier        (centroids + head)    │
│   └─ exporters         (folders, Rekordbox)  │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────┴──────────────────────────┐
│  SQLite  (~/.crate/library.db)               │
│  tracks · analysis cache · embeddings ·      │
│  crates · assignments · corrections · config │
└──────────────────────────────────────────────┘
```

**Why the browser never touches a file picker:** the server runs locally with
real filesystem access and reads folders by absolute path. Folders are configured
once and stored in the database. This is what killed the previous Electron
attempt (`dialog.showOpenDialog` IPC never wired up) — the architecture removes
the problem rather than working around it.

### Data model

```sql
tracks       (id, path, filename, size, mtime, hash,
              analysed_at, bpm, key, camelot, duration, ...)
embeddings   (track_id, vector BLOB)          -- 1280 floats, the expensive bit
analysis     (track_id, moments JSON, energy JSON, vocal JSON, genre_preds JSON)
crates       (id, name, created_at, is_gap_genre)
assignments  (track_id, crate_id, source: 'auto'|'human', confidence, created_at)
corrections  (track_id, from_crate, to_crate, created_at)   -- audit trail
config       (key, value)                      -- watched folders, thresholds
```

`path + mtime + size` identifies a track. Unchanged files are never re-analysed;
this is what makes "any amount of songs" tractable — a 10,000-track library is
~9 hours *once*, then incremental forever.

Embeddings are stored because centroids must be recomputed on every correction,
and re-deriving them would mean re-reading all the audio.

### Launcher

A double-clickable `Crate.command` (or small `.app` shim) that starts the server
if it isn't running and opens the browser to it. No terminal command to remember.
Idempotent — clicking twice focuses the existing instance rather than starting a
second one.

---

## 8. Error handling

| Case | Behaviour |
|---|---|
| Undecodable / corrupt file | Recorded with the error, skipped, surfaced in a "couldn't read" list. Never halts a run. |
| Track shorter than 5 s | Skipped, as today |
| Genre models missing | Analysis still runs (tempo/key/structure); classification disabled with a clear banner |
| Reference file / embedding dimension mismatch | Signal disables itself rather than producing nonsense (already implemented) |
| Server already running | Launcher focuses the existing tab |
| Analysis interrupted | Resumes; completed tracks are already committed |
| Music folder moved or unplugged | Tracks marked missing, not deleted; crates preserved |

---

## 9. Testing

- **Regression:** the 130-track ground-truth set and 27-acapella set must not
  degrade — 80% LOO crate accuracy, 27/27 acapellas, 0 false positives.
- **Learning loop:** assigning tracks to a new crate must measurably improve
  classification of held-out tracks of that genre.
- **Rejection:** tracks from a genre with no crate must land in Unsorted, not be
  force-filed. Rap-vs-Afrohouse separation (0.34–0.48 vs 0.73+) is the fixture.
- **Idempotence:** re-scanning an unchanged folder analyses zero tracks.
- **Non-destructiveness:** an automated check asserts the source folder's file
  list and mtimes are unchanged after a full sort.
- **Export:** cloned files must be byte-identical to their originals.

---

## 10. Open items

1. **Rejection threshold needs calibration.** Distant-genre numbers exist
   (0.34–0.48); the floor should be set against music genuinely outside this
   library. Starts conservative until then.
2. **Rekordbox round-trip unverified** (§6). Should be tested before the full
   library run, since a fix would otherwise mean re-exporting everything.
3. **Multi-crate membership** — allowed by the schema; the UI for it is
   deliberately deferred until it's actually wanted.
4. **Half-time/double-time BPM** disambiguation remains unsolved, as today.

---

## 11. Deliberately not in v1

- No folder watching / background daemon — scanning is on demand.
- No cloud, no accounts, no network calls beyond the one-time model download.
- No audio playback beyond what the existing viewer does.
- No mobile UI.
