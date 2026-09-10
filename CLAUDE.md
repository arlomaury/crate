# Crate — project context

_Claude Code reads this file automatically at the start of every session.
This is the complete project state. Read it before changing anything._

---

## 1. Who this is for and what the end goal is

A working DJ with a library of roughly **1,600 catalogued files (about 900
distinct recordings — the library is full of duplicate copies)** on a Mac,
playing on Pioneer CDJs. The pipeline is:

```
audio files → Crate (sort + cue) → Rekordbox XML → USB export → CDJ playback
```

Genres in the library: Dubstep, UK Garage, DNB, House, Tech House, Deep House,
Afrohouse, Trap, Riddim, Rap, Pop. A large share is **bootleg edits, dubs,
acapellas and SoundCloud rips with little or no embedded metadata**, which is
the entire reason this tool exists.

**The stated priority is accuracy over speed.** Non-negotiable. Wrong tags on
900 tracks have to be found and removed by hand, so the tool must **flag rather
than guess**. Preserve this principle in any change you make.

### Error severity is not uniform — this is the DJ's own ruling

> "a song labeled as house but is actually tech house isn't as big of a deal as
> something labeled as something else that is not similar, which is what
> happened with pop and rap and dubstep"

So **house ↔ tech house mix-ups are acceptable**; anything crossing into pop,
rap, DUBSTEP, vocals, afro or UKG is not. `classifier.NEAR_FAMILIES` encodes
this. afro and UKG were explicitly offered as additional "close enough" crates
and **declined** — do not add them without asking again.

### Dead ends already hit — do not repeat these

| Approach | Why it failed |
|---|---|
| **Training the classifier on its own output** | `rebuild_centroids` selected every assignment regardless of `source`, reasoning that "a confidently auto-filed track is still evidence". It is not — it is the model's own opinion, and feeding it back is self-reinforcing. DUBSTEP reached 40 reference tracks of which **2** were the DJ's; rap 137 of which **1**. The crates filled with music that did not belong. **Only `source='human'` rows may train.** |
| **Measuring accuracy without deduplicating first** | This library holds the same recording many times. An identical copy in the training fold while its twin is scored in the test fold is leakage. It reported **90.7%** when the honest figure was **76.1%** — a 15-point illusion. `eval_genre.py` now dedupes before cross-validation; never remove that. |
| Nearest-centroid classification | Assumes each genre is a spherical blob around its mean. Measured 72.1% against logistic regression's 76.1%, and its precision was the visible failure: DUBSTEP 0.800, UK Garage 0.385. Kept only as an out-of-distribution signal (see §3). |
| Discogs400 head output as classifier features | 71.7% alone, no gain concatenated with the embedding. That head is itself a linear layer on the same embedding, so it carries strictly less information. |
| Tempo features, PCA (32–256 dims), RBF SVM | +0.2% (noise), worse at every PCA size, 73.5% respectively. |
| **Groove / beat-synchronous rhythm features** | The EDM subgenre literature (arXiv:2110.08862) says rhythm features dominate this task, so a 52-dim beat-synchronous band-energy profile was built (`groove_experiment.py`). It carries real signal alone — 62.9% on house-vs-tech against a 54.8% baseline — but adds **nothing** on top of the embedding (74.9% → 74.2%). The embedding already knows what groove would tell it. |
| **Discogs labels as a targeted house/tech signal** | The taxonomy has explicit House / Tech House labels and uses them well (says "Tech House" → right 47 of 53 times). A single House-minus-TechHouse axis appeared to hit 77.7% — but that threshold was fitted and scored on the same data. Under repeated cross-validation it is 74.1% ±0.5 against the embedding's 73.5% ±1.1: the same, within noise. |
| **Collecting more labels to fix accuracy** | Measured learning curve: 130 labels → 67.7%, 364 → 76.5%, 520 → 76.9%. It has plateaued; +43% more labels bought 0.4 points. Labelling more helps only the *individually starved* crates (afro at 20, UKG at 25), not overall accuracy. |
| `class_weight="balanced"` | Scores higher overall (77.0%) but buys recall on small crates by giving up precision on them (afro 1.000 → 0.619, DUBSTEP 1.000 → 0.958). Wrong given the severity ruling above. |
| Genre keyword matching on filenames/titles | "bass", "house", "rock" appear across unrelated genres. **The tool never reads filenames to decide genre.** |
| Downloaded standalone HTML calling APIs | Local HTML cannot call external APIs. Architectural dead end. |
| Browser drag-and-drop folder APIs; browser BPM waveform analysis | Fail in sandboxed HTML; inaccurate. Rekordbox's own BPM analysis is the gold standard. |
| An Electron app ("BPM Sorter Pro") | Folder picker never worked. Superseded — do not revive. |
| Serving AIFF to the browser | **Chrome cannot decode AIFF** (`EncodingError`), and AIFF is 669 of ~1,400 files. Converted to WAV server-side instead (§4). |

---

## 2. What exists

| Path | Purpose |
|---|---|
| `analyze.py` | Analysis engine: tempo, key, structure, acapella detection, embeddings. Also a standalone CLI. |
| `crateapp/` | The app: local server, sorting pipeline, classifier, set builder, UI. |
| `eval_genre.py` | **Compares classifier methods under stratified k-fold CV and prints the calibration tables the thresholds come from.** Run this before changing the classifier. |
| `tune_genre.py` | Sweeps regularisation/PCA/class weights, and measures the house-vs-tech ceiling. |
| `groove_experiment.py` | Extracts beat-synchronous rhythm features (cached at `~/.crate/groove.npz`) and tests whether groove separates house from tech house. It does not — see the dead-ends table. |
| `house_vs_tech.py` | Tests every available signal on the one distinction that is left, under repeated CV. |
| `retrain.py` | Resets ground truth from the DJ's Rekordbox export and retrains. Run after re-exporting `rek.xml`. |
| `crate_model.json` | Trained weights + centroids + provenance. |
| `crate.command` | Double-click launcher. `~/Desktop/Crate.app` wraps it. |
| `viewer_template.html` | The original standalone viewer (superseded by the app, still works). |

**Ground truth lives in `~/Documents/rek.xml`** — the DJ's own Rekordbox
playlists. Genre playlists: `tech`, `house`, `pop`, `vocals`, `afro`, `rap`,
`UKG`, `DUBSTEP` (plus `TECH HOUSE`/`AFROHOUSE`/`RAP` which map onto the
lowercase ones). `MAIN`, `PARTY`, `REMIX` are **set lists, not genres**;
`AllSongs` and `Contents` are containers. Both groups are excluded — filing by
them would teach the model that "tracks I play at parties" is a sound.

Library database: `~/.crate/library.db`. Models: `~/.crate/models`. Converted
audio cache: `~/.crate/cache`.

---

## 3. The classifier — how it decides

Two signals answering two different questions.

**Which crate** — multinomial logistic regression over the 1280-dim
Discogs-EffNet embedding, `C=0.001` (heavy regularisation: ~450 distinct
labelled recordings against 1280 dimensions is badly over-parameterised, and
the sweep shows accuracy climbing all the way down to C=0.001). Trained with
scikit-learn; **inference is numpy-only** — the weights are exported into
`crate_model.json`, so predicting never needs sklearn.

**Is this like anything the DJ owns** — a softmax sums to 1, so it reports a
crate even for music unlike anything in the library. Cosine distance to the
nearest crate centroid answers this instead, and the centroids are kept for
exactly that. **This is what lets the app spot a genuinely new sound** rather
than forcing every track into an existing crate — an explicit user
requirement.

**Confidence is measured over the family, not the crate.** A track split 0.45
house / 0.40 tech is 0.85 sure of the family and merely unsure which half — so
it is filed. Split 0.45 house / 0.40 pop it is 0.45 sure of anything — so it
goes to review. One threshold does both jobs.

Measured on 452 distinct labelled recordings, 5-fold stratified CV:

```
nearest centroid (was shipped)   72.1%
logistic regression, tuned       76.1%

precision, then -> now:   DUBSTEP 0.800 -> 1.000
                          UK Garage 0.385 -> 0.778
                          rap       0.757 -> 0.944
```

**house vs tech house is 69% of all remaining error** (81 of 117), between two
crates holding 280 of the 452 labels. Merging them takes the same model to
**90.5%** — so the rest of the taxonomy is close to solved and this one
distinction is nearly all that is left.

It is now well established that this pair is close to unlearnable from audio
here. Three unrelated signal families were measured on it in isolation, under
repeated cross-validation:

```
always guess the bigger crate     54.8%
embedding (what ships)            73.5%  +/-1.1
Discogs taxonomy labels           74.1%  +/-0.5
groove / beat-synchronous rhythm  60.4%  +/-1.1
all three together                73.7%  +/-0.5
```

Three independent views of the audio agree on roughly 74%, and combining them
adds nothing — which points at the labels rather than the method. The
published literature agrees: tech house is *defined* as a blend of techno
elements with progressive-house harmonies and grooves, and academic EDM
subgenre classifiers report 48–59% on comparable tasks. The DJ's own filing
here follows Beatport's tagging, which is a convention rather than a property
of the sound.

**Merging them is an open product question — ask, do not decide.** Anyone
planning to "fix" this pair with a better model should read the dead-ends
table first and expect to be disappointed.

Threshold trade-off (family-summed probability), and why 0.65:

```
threshold   filed   to sort   errors that MATTER
    0.55     89%      11%          6.2%
    0.65     81%      19%          3.8%   <- chosen, the knee
    0.75     71%      29%          3.4%
```

**A correction or a confirmation retrains the model.** Confirming promotes the
provisional `auto` row to `source='human'`, so agreeing with the model teaches
it as much as correcting it — which matters a lot at only ~450 labels.
Retraining is debounced ~3s off the request path (`server._Retrainer`): a fit
takes ~1s and holds the DB lock, so doing it inline stalled every click.

---

## 4. Design decisions worth not undoing

**Everything works in bars, not seconds.** A cue point not on a downbeat is
useless on a CDJ. All moment timestamps snap to downbeats.

**Sub-bass energy is the primary structural signal.** The kick and sub drop out
entirely during a breakdown and slam back on the one; that gap is the structure.

**The mix-in target is the first DROP, never "main section".** A `main section`
marker sits at bar 0 on nearly every track and only means "the track starts".
Targeting it made 88% of tracks read as intro-less while the real drop sat 40
bars later.

**The lead-in is 16 bars, not 32 seconds.** A DJ counts phrases; 16 bars is 30s
at 128 BPM and 22s at 175.

**Acapellas route to "Vocals" BEFORE genre.** Genre models are trained on full
mixes; given an isolated vocal there is no drum pattern or bassline left, so
the model guesses from timbre alone — confidently and wrongly. Detection is
2-of-3 DSP signals, calibrated against 27 real acapellas and 130 full mixes
(27/27 caught, 0 false positives).

**Duplicates are collapsed for training and for the set builder, never on
disk.** Cut at cosine 0.99 — measured gap between same-song pairs (0.9958) and
the closest genuinely distinct pair (0.9785). Nearly half the library is
duplicate copies; left in, every track's own twin ranked as its best possible
next track.

**Audio is served as short WAV segments or full WAV conversions, addressed by
track id.** Never by path — the browser can name a row the analyser created,
not a file to read. AIFF→WAV is lossless (both are uncompressed PCM).

**The server is a `ThreadingHTTPServer`.** Streaming a 56MB file on the
single-threaded one froze the whole UI.

**Never render a number the API did not return**, and never conflate two facts
on one line. The panel showed "tech · confident · margin 0.023" for a track the
model scored highest as *house* — where it is filed and what the model thinks
are separate rows now.

---

## 5. Verification status

### Measured / verified

- Tempo, key, Camelot conversion. Structural moments land on 8- and 16-bar
  phrase boundaries on real tracks.
- Classifier accuracy, per-crate precision/recall, confusion matrices, and
  every threshold — see §3. Re-runnable: `python eval_genre.py`.
- Acapella detection: 27/27 acapellas, 0/130 false positives.
- Full library analysed: 1,632 of 1,636 (the 4 are files no longer on disk).
- Rekordbox XML round-trip: **the DJ confirmed cue points import and work.**
  Export currently produces 1,620 tracks, 8 playlists, 28,704 cue points,
  genre + key on every track.
- Source audio is never modified — verified across full cycles.
- No XSS: crate and track names live-tested with `<script>` and `onerror`
  payloads.
- AIFF/WAV/M4A → WAV segments and full conversions decode in Chrome.
- 197 tests pass (`python -m pytest tests/ -q`).

### NOT verified

- **Whether the newly re-filed crates match the DJ's ear.** The numbers say
  3.8% cross-family error; only they can confirm.
- Audible playback of the crossfade preview and the track player — verified up
  to the speaker (segments decode, ramps schedule, AudioContext runs), but the
  automation browser is backgrounded and Chrome defers media there.
- The move/add/confirm controls have not been eyeballed in a browser; Chrome
  began refusing `127.0.0.1:8420` in the automation tab while curl got 200.
  Endpoints and served assets were verified instead.
- `setup.sh` has never been executed end to end (syntax-checked only).
- Half-time/double-time BPM disambiguation.

---

## 6. Outstanding, in priority order

1. **Work "Worth a second look" first** (49 tracks). These are filed
   confidently but the two independent signals disagree, and measured, 25% of
   them are genuinely misfiled against 2.9% elsewhere. Shortest list, highest
   yield — and the only way these errors ever surface, since the model is sure.
2. **Then the close calls** (~106). Every **Correct** is one click and becomes
   training data.
3. **Decide the house/tech question** (§3). Merging is worth ~14 accuracy
   points; it is their taxonomy, not ours. Note the evidence says no model
   will fix it.
4. **Export and load onto a USB / CDJ** with the re-sorted crates.
5. More labels **only for afro (20) and UKG (25)** — those two crates are
   individually starved (afro recall 0.30, UKG 0.64). Overall accuracy has
   plateaued, so labelling anything else is not worth the evening.
6. An untested idea worth one experiment if the above is exhausted: swap
   Discogs-EffNet for **MuQ** (`pip install muq`, PyTorch, CPU-capable, needs
   24kHz audio), the current state of the art on music genre benchmarks. It
   would mean re-embedding the library and a ~2GB dependency, and it is
   unproven on *this* distinction — expect it to help the taxonomy generally
   rather than rescue house-vs-tech.
7. `scan()` still holds the DB lock for a whole folder scan.

## 7. Possible improvements (not requested — do not build unasked)

- Half-time/double-time disambiguation.
- Write genre into file tags via `mutagen` as an alternative to XML import.
- Sub-section detection inside long uniform sections.
- Mashup handling: surfacing "this track's top-2 are close, it may be a hybrid".
- A dedicated house-vs-tech submodel, if the DJ supplies more labels for it.

## 8. Environment

- Python 3.9–3.14 (`essentia-tensorflow` wheel range). Currently 3.14.
- `./setup.sh` creates the venv and installs `essentia-tensorflow`, `numpy`,
  `scikit-learn`, then downloads model weights.
- `brew install ffmpeg` is **not** required — essentia decodes everything,
  including the AIFF→WAV conversion for browser playback.
- Models in `~/.crate/models`, overridable with `CRATE_MODELS`.
