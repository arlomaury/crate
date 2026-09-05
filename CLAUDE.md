# Crate — project context

_Claude Code reads this file automatically at the start of every session.
This is the complete project state. Read it before changing anything._

---


This project was started in a Claude chat session. Everything below is the full
context: what exists, what is verified, what is not, and what to do next.
Read this before changing anything.

---

## 1. Who this is for and what the end goal is

A working DJ with a library of roughly **930 tracks** on a Mac, playing on
**Pioneer CDJs**. The pipeline is:

```
audio files → Rekordbox → USB export → CDJ playback
```

Genres in the library: Dubstep, UK Garage, DNB, House, Deep House, Afrohouse,
Trap, Riddim, and more. A large share of the library is **bootleg edits, dubs,
acapellas and SoundCloud rips with little or no embedded metadata**, which is
the entire reason this tool exists.

**The stated priority is accuracy over speed.** This is non-negotiable and has
already killed several earlier attempts. A tool that runs fast but assigns wrong
genres or wrong BPMs is worse than useless, because wrong tags on 930 tracks
have to be found and removed by hand. When in doubt, the tool must **flag rather
than guess**. Preserve this principle in any change you make.

### Dead ends already hit — do not repeat these

| Approach | Why it failed |
|---|---|
| Genre keyword matching on filenames/titles | Words like "bass", "house", "rock" appear in titles across unrelated genres. Too many false positives. **The current tool never reads filenames to decide genre.** |
| Downloaded standalone HTML files calling APIs | Local HTML files cannot call external APIs or do web search. Architectural dead end, cost several iterations. |
| Browser drag-and-drop folder APIs | Fail in sandboxed / downloaded HTML. |
| Browser-based BPM waveform analysis | Inaccurate. Rekordbox's own BPM analysis was treated as the gold standard instead. |
| An Electron app ("BPM Sorter Pro") | Folder picker button never worked; needed `dialog.showOpenDialog` IPC wiring. Superseded by this tool — do not revive unless asked. |

Also relevant: the library was consolidated into a **flat `AllSongs` folder on
the Desktop** using Terminal `find` commands, because nested folders complicate
Rekordbox import. Assume `~/Desktop/AllSongs` is the input path.

---

## 2. What was built

A Python command-line analyser plus a standalone HTML viewer. It determines,
**from the audio signal only**: tempo, key, genre, whether a file is an
acapella, and where the structural moments (drops, breakdowns, buildups) are.

### Files in this handoff

| File | Purpose |
|---|---|
| `analyze.py` | The whole analysis engine. ~600 lines. Entry point. |
| `viewer_template.html` | Viewer UI. `analyze.py` injects results into the `/*__CRATE_DATA__*/[]` placeholder to produce `viewer.html`. |
| `setup.sh` | Creates venv, installs `essentia-tensorflow`, downloads model weights. |
| `README.md` | End-user documentation. |
| `sample_results.json` | **Real output from 12 of the user's actual tracks.** Use this as a regression fixture. |

### Pipeline

```
audio file
  ├─ MonoLoader @44.1kHz ──┬─ RhythmExtractor2013(multifeature) → BPM, beat grid, confidence
  │                        ├─ KeyExtractor → key/scale → Camelot notation
  │                        ├─ LoudnessEBUR128, Danceability, DynamicComplexity, OnsetRate
  │                        └─ frame_features() → per-frame RMS, 5 band energies,
  │                             spectral centroid, flux
  │                                  ↓
  │                             to_bars()  — collapse frames into BARS using the beat grid
  │                                  ↓
  │                             find_moments() → cue points, snapped to downbeats
  │                                  ↓
  │                             vocal_profile() → acapella decision
  │
  └─ MonoLoader @16kHz ──── Discogs-EffNet embedding → genre_discogs400 head → 400 labels
                                                    └→ voice_instrumental head
```

### Key design decisions and why (do not undo these casually)

**Everything works in bars, not seconds.** Dance music is built in 4- and 8-bar
blocks. A cue point that is not on a downbeat is useless on a CDJ. All moment
timestamps are snapped to downbeats.

**Sub-bass energy is the primary structural signal.** In dance music the kick
and sub drop out entirely during a breakdown and slam back in on the one. That
gap is the structure. `find_moments()` keys off normalised sub-band (20–120 Hz)
energy combined with overall RMS.

**Acapellas are routed to "Vocals" BEFORE genre is considered.** This was an
explicit user request and it is also technically correct: genre models are
trained on full mixes. Given an isolated vocal there is no drum pattern, no
bassline and no production style left to analyse, so the model guesses from
vocal timbre alone — confidently and often wrongly. Detection uses four
independent DSP signals (near-zero sub ratio, vocal-range dominance, low onset
rate, thin low end vs mids). **3 of 4 → Vocals. 2 of 4 → flagged for review, not
decided.** The viewer displays which signals fired so the reasoning is
inspectable.

**Discogs400 taxonomy, not GTZAN.** GTZAN-trained models know ten broad genres
and no electronic subgenres — they cannot distinguish UK Garage from Deep House,
which is the exact distinction this library needs. Discogs400 can. Do not swap
in a simpler model.

**Conservative by design.** Genre confidence below 0.15 sets `needs_review`
rather than filing the track. Low beat-tracking confidence sets it too. Top-5
genre candidates are always retained so the second guess is visible.

### Outputs

- `viewer.html` — self-contained, no server needed. Track list; clicking a track
  shows its energy timeline with labelled cue markers, plus every measured
  value. Filters for category / needs-review / vocals-only. Optional "Load audio
  folder" button uses `<input webkitdirectory>` to enable playback with
  click-a-cue-to-seek (this avoids the `file://` audio loading restriction).
- `rekordbox.xml` — import via Rekordbox → File → Import Collection. Carries
  genre, BPM, key, and a `POSITION_MARK` memory cue at every detected moment.
- `library.csv` — one row per track.
- `results.json` — everything including raw energy curves.

---

## 3. Verification status — read this carefully

### Verified working (tested on the user's real audio files)

- Tempo detection. On a file named `Molly (Shankz Remix) - Extended 133 BPM`
  it returned **132.91** — near-exact against known ground truth.
- Tempo cross-checks against genre convention: Rockwell UKG edit → 140.0,
  Sex On Fire (WESH, tech house) → 127.9, KnightBlock Sweet Escape Dub → 135.0.
  All correct for their styles.
- Key extraction and Camelot conversion.
- **Structural moment detection.** Strongest validation obtained: on
  `Slat - Tavatli` it found intro → buildup at **bar 9** → drop at **bar 17** →
  breakdown at **bar 33** → outro. Bars 9/17/33 are exact 8- and 16-bar phrase
  boundaries. It is locating real musical phrasing, not noise. Similar
  phrase-aligned results (bars 16/24/32) on the Ashdunn track.
- All 12 files analysed end to end, 97 cue points, no crashes.
- `rekordbox.xml`, `library.csv`, `results.json`, `viewer.html` all generate.
- Viewer JS passes `node --check`; data injection verified (12 tracks, 97
  moments, 12 energy curves parsed back out of the generated HTML).
- `setup.sh` passes `bash -n`.
- PyPI confirmed: `essentia-tensorflow` ships **macOS arm64 wheels for
  cp39–cp314**, so it installs on Apple Silicon.

### NOT verified — must be tested on the Mac

- **Genre classification has never actually run.** The chat sandbox blocks
  `essentia.upf.edu` (HTTP 403), so the Discogs-EffNet weights could not be
  downloaded. Every genre code path is written but unexecuted. In
  `sample_results.json` every track shows `"category": "Unclassified"` for this
  reason — that is the expected no-model fallback, not a bug.
- **Acapella detection has never fired on a true positive.** None of the 12 test
  files is an acapella, so only the negative case is exercised (all 12 correctly
  returned `is_acapella: false`, sub ratios 0.61–0.73, well clear of the 0.16
  threshold). **The thresholds in `vocal_profile()` are reasoned estimates, not
  empirically tuned.** They will need calibration against real acapellas.
- `setup.sh` has never been executed. Syntax-checked only.
- The viewer has never been rendered in a browser. No screenshot review was
  possible.
- M4A/WMA decoding depends on ffmpeg being present; untested.

---

## 4. Ground truth data for validation

Before the tool was built, the 12 test tracks were researched manually via
SoundCloud genre tags, Bandcamp, Beatport and label context. **Use this as a
first accuracy check** — run the tool with the genre model installed and compare:

| File | Researched genre | Source of truth | Detected BPM |
|---|---|---|---|
| `Portugal. The Man - Feel It Still (Charlie Shell Edit)` | House | SoundCloud genre tag | 130.0 |
| `BVNQUET - Murder On The Dancefloor Dub` | UK Garage | Bandcamp tags "electronic ukg" | 138.0 |
| `Ashdunn - DARE Dub` | UK Garage (speed garage) | Garage Shared label; SC tag only says "Dance & EDM" | 141.8 |
| `Nice For What (Jay Slow & Zenon Edit)` | House / Deep House | SoundCloud genre tag | 119.0 |
| `Rockwell - Somebody's Watching Me (Boucho UKG Edit)` | UK Garage | stated in title | 140.0 |
| `Kings Of Leon - Sex On Fire (WESH Remix)` | Tech House | uploader `#TechHouse` tag | 127.9 |
| `Molly (Shankz Remix)` | Deep House | SoundCloud genre tag | 132.9 |
| `KnightBlock - Sweet Escape Dub` | UK Garage | YouTube title "[Free UK Garage Download]" | 135.0 |
| `Slat - Tavatli` | **unknown** | no match found anywhere; filename looks mislabelled ("Tavatli" is normally an artist, a techno/house producer) | 125.9 |
| `I Just Can't Get Enough (remix)` | **unknown** | no remixer credit in filename; dozens of versions exist across DNB/dubstep/tech house | 134.9 |
| `Gnarls Barkley - Crazy (GRAM Remix)` | **unknown** | artist page found, no genre tag retrievable | 125.0 |
| `YOUR LOVE FINAL MASTER` | **unknown** | title far too generic, no artist to anchor a search | 130.0 |

Note the pattern: metadata lookup succeeded on 8/12 and failed on exactly the
tracks with generic titles or missing artist credits. **Those 4 unknowns are the
best possible test cases for the audio model** — if audio analysis can classify
the tracks that metadata lookup could not, that is the tool justifying its
existence.

---

## 5. Outstanding tasks, in priority order

### Task 1 — Get genre classification actually running (blocking)

```bash
./setup.sh                       # downloads models to ~/.crate/models
source .venv/bin/activate
python analyze.py ~/Desktop/AllSongs --limit 20 -o ~/Desktop/crate_test
```

Confirm `GenreModel.ok` is True and that `category` is no longer
"Unclassified". If the Essentia TensorFlow node names have changed upstream,
the `input=`/`output=` strings in `GenreModel.__init__` are the thing to fix —
check current docs at `essentia.upf.edu/models.html`.

Then re-run the 12 tracks in `sample_results.json` and diff against the ground
truth table in section 4.

### Task 2 — The 300-track validation the user asked for

The user explicitly requested: *test with 300 songs from every genre and
cross-reference against public Spotify playlists or lookups to confirm
correctness.* **This was not done in chat** — the chat environment has no way to
obtain 300 licensed audio files and could not run the genre model at all. It is
a real and appropriate task for Claude Code on the Mac, where the user's own
library is present.

Suggested method:

1. Sample ~300 tracks from `~/Desktop/AllSongs`, stratified so every genre in
   the library is represented — do not just take the first 300 alphabetically.
2. Build ground truth **per track** the same way section 4 did: SoundCloud
   genre tag, Beatport, Bandcamp, Discogs, label context. Where sources
   conflict, record the conflict rather than picking one.
3. Run `analyze.py` over the sample.
4. Score it. Report **per genre**, not just overall — the interesting question
   is not "what is the accuracy" but "which genres does it confuse". Expect
   UK Garage vs Speed Garage vs Bassline, and Dubstep vs Riddim, to be the hard
   pairs. Build a confusion matrix.
5. Treat "model said X with confidence 0.4, truth was Y" differently from
   "model said X with confidence 0.05" — the second is the tool correctly
   signalling uncertainty and should not count as a plain failure.
6. **Use the results to tune the `needs_review` confidence threshold.** It is
   currently 0.15, chosen by reasoning, not measurement. The right value is
   whatever cleanly separates correct from incorrect predictions in this data.

Caveat to keep in mind while doing this: Spotify does not expose a per-track
genre on the track page — genre lives in artist-level metadata and internal
categories. Discogs, Beatport and SoundCloud tags are better ground truth for
this library than Spotify.

### Task 3 — Calibrate acapella detection against real acapellas

The user's library contains acapellas; none were in the 12-file test set. Gather
20–30 known acapellas plus 20–30 known full mixes and tune the four thresholds
in `vocal_profile()` (`sub_ratio < 0.16`, `vocal_ratio > 0.46`,
`onset_rate < 2.6`, `sub < mid * 0.55`). Report false-positive and
false-negative rates separately — a full mix wrongly filed as Vocals is worse
than an acapella left flagged for review.

### Task 4 — Render and review the viewer

Open a generated `viewer.html` in a real browser. Check the canvas energy curve
draws, markers position correctly against the timeline, the "Load audio folder"
playback path works, and the layout holds at narrow widths.

### Task 5 — Verify the Rekordbox import round-trip

Import a generated `rekordbox.xml` into Rekordbox and confirm genre, BPM, key
and **memory cues** all land correctly, and that the cues survive a USB export
and appear on a CDJ. This is the one step that cannot be simulated and it is the
whole point of the project. If `POSITION_MARK` cues do not import, the likely
fixes are the `Type` attribute or needing `Num="0"` and up rather than `-1`.

### Task 6 — Full library run

Once the above pass: `python analyze.py ~/Desktop/AllSongs --resume -o ~/Desktop/crate_output`.
Roughly 8–12 s/track on Apple Silicon → about 2–3 hours for 930 tracks.
`--resume` makes interruption safe. Then work the `needs_review` queue by ear in
the viewer.

---

## 6. Possible improvements (not requested, do not build unasked)

- Half-time/double-time disambiguation. Dubstep at 140 vs 70 can be read either
  way; the current confidence score usually hints at it but does not resolve it.
- Write genre back into file tags via `mutagen` as an alternative to XML import.
- Sub-section detection inside long uniform sections. Currently a track whose
  energy never drops out gets one "Main section" marker and nothing else — the
  Molly track showed a 3-minute gap between markers for this reason. That is
  correct-but-sparse behaviour, arguably fine.
- Mashup handling: bootlegs blend two source genres and the model returns one
  label. Surfacing "this track's top-2 are close, it may be a hybrid" could be
  more honest than a single answer.

---

## 7. Environment notes

- Python 3.9–3.14 required (`essentia-tensorflow` wheel range).
- `brew install ffmpeg` for M4A/WMA support.
- Models live in `~/.crate/models`, overridable with `CRATE_MODELS`.
- The chat sandbox that produced this had `essentia-tensorflow 2.1b6.dev1389`
  and numpy 2.4.4 on Python 3.12; that combination is known to work for
  everything except the model downloads.
