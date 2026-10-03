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
| **Letting the voice model decide an acapella** | `voice_instrumental` answers "is there singing on this", not "is this an isolated vocal", so every sung pop record scores above 0.85 on it. It was allowed to declare an acapella at a 1-of-3 DSP score — on the reasoning that 0.33 was "the same half-convinced state 0.5 meant on the old 4-signal scale", which is wrong: 0.33 of three is ONE, where 0.5 of four was two. It filed 28 full mixes as acapellas (Mr. Brightside, Viva La Vida, Starships) **and cleared their review flag**, so they never surfaced. It may confirm a 2-of-3 verdict; it may never create one. |
| **Learning a genre rule from acapellas** | `vocals` is a FORMAT, not a genre. The DJ files their own `_vocals_split` stems there, so counting those as evidence taught the artist rule "Lil Yachty → vocals" and it filed his full tracks alongside the stems. Both `tag_crate` and `who_made_it` now exclude acapellas from their evidence — the acapella rule owns that question and runs first. |
| **Training the classifier on its own output** | `rebuild_centroids` selected every assignment regardless of `source`, reasoning that "a confidently auto-filed track is still evidence". It is not — it is the model's own opinion, and feeding it back is self-reinforcing. DUBSTEP reached 40 reference tracks of which **2** were the DJ's; rap 137 of which **1**. The crates filled with music that did not belong. **Only `source='human'` rows may train.** |
| **Measuring accuracy without deduplicating first** | This library holds the same recording many times. An identical copy in the training fold while its twin is scored in the test fold is leakage. It reported **90.7%** when the honest figure was **76.1%** — a 15-point illusion. `eval_genre.py` now dedupes before cross-validation; never remove that. |
| Nearest-centroid classification | Assumes each genre is a spherical blob around its mean. Measured 72.1% against logistic regression's 76.1%, and its precision was the visible failure: DUBSTEP 0.800, UK Garage 0.385. Kept only as an out-of-distribution signal (see §3). |
| Discogs400 head output as classifier features | 71.7% alone, no gain concatenated with the embedding. That head is itself a linear layer on the same embedding, so it carries strictly less information. |
| Tempo features, PCA (32–256 dims), RBF SVM | +0.2% (noise), worse at every PCA size, 73.5% respectively. |
| **Groove / beat-synchronous rhythm features** | The EDM subgenre literature (arXiv:2110.08862) says rhythm features dominate this task, so a 52-dim beat-synchronous band-energy profile was built (`groove_experiment.py`). It carries real signal alone — 62.9% on house-vs-tech against a 54.8% baseline — but adds **nothing** on top of the embedding (74.9% → 74.2%). The embedding already knows what groove would tell it. |
| **Discogs labels as a targeted house/tech signal** | The taxonomy has explicit House / Tech House labels and uses them well (says "Tech House" → right 47 of 53 times). A single House-minus-TechHouse axis appeared to hit 77.7% — but that threshold was fitted and scored on the same data. Under repeated cross-validation it is 74.1% ±0.5 against the embedding's 73.5% ±1.1: the same, within noise. |
| **Voting between the rules instead of a priority chain** | Measured: where the tag and the audio model disagree (129 cases) the tag is right **87%** and the audio **9%**; where the tag and the person rule disagree (19 cases) the tag is right 14 to 2. Even when the person rule sides with the audio against the tag, the tag is still right 5 of 6. The chain is already optimal — no ensemble beats it. The disagreement is still worth surfacing, which is what `disputed` does. |
| **Propagating data between identical copies** | The library holds 322 groups of byte-identical recordings, so it looked like free information. It is not: only **8** untagged tracks have a tagged twin, and only **1** group sits in two different crates. |
| **Recovering genres from rek.xml that the files lack** | Rekordbox knows a genre for 897 filenames, but of the 576 library tracks with no embedded tag it covers **3**. Those tracks are SoundCloud rips with no metadata anywhere. |
| **MFCCs** | The classic genre feature, and the one every tutorial starts with — so it was measured rather than waved away (`mfcc_experiment.py`, 60 numbers: mean, std and mean-delta of 20 coefficients). Alone: **55.8%** against the embedding's 72.2%. Concatenated with the embedding: **72.3%**, which is +0.1 and inside the noise. The embedding is a convolutional network trained on millions of tracks for this exact task; MFCCs describe timbre, which it already subsumes. Worth knowing: MFCCs alone reach 70.7% on house-vs-tech, nearly the embedding's 72.1% — both are poor at it, which is more evidence that pair is not an audio problem. |
| **A two-stage / hierarchical classifier** | Decide the family first (dance vs pop vs rap vs vocals), then which member. 71.9% against the flat model's 72.2%. The first stage is easy and the second stage is the same hard problem, so nothing is gained by splitting them. |
| **Semi-supervised learning on the unlabelled tracks** | 1,037 unlabelled tracks sit there, so label spreading over the embedding graph looked free. **65.2%** against 72.2% — substantially worse. The way those tracks cluster does not line up with the DJ's genre boundaries, so propagating along it drags labels across them. (Note this is not the self-training feedback loop; that was a separate and worse bug.) |
| **kNN instead of the linear model** | Worse at every k: 62.0% at k=1, 66.8% at k=5, 67.3% at k=10, against logistic regression's 72.2%. Genre is not locally clustered in this embedding. |
| **Record label / album as a filing key** | Label: 662 tracks have one and **zero** of them lack a genre tag — the same Beatport purchases, no new information. Album: 87.9% accurate but covers 10%, and **0** of the untagged tracks have one. |
| **Pooling the embedding differently** | The stored vector is `emb.mean(axis=0)` over the whole track, which averages the intro and outro in with the groove — so mean+std, percentiles, max, and embedding *only the drop section* were all tried (`pooling_experiment.py`). The shipped mean wins or ties every one: mean 76.5%, percentiles 76.6% (3× the dimensions), mean+std 76.2%, drop-only **73.3%** — worse. |
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
| `pooling_experiment.py` | Tests whether a different reduction of the per-frame embeddings beats the mean. It does not — see the dead-ends table. Cache at `~/.crate/pooling.npz`. |
| `retrain.py` | Resets ground truth from the DJ's Rekordbox export and retrains. Run after re-exporting `rek.xml`. |
| — | Removing a track (`crates.remove_track`, `POST /api/remove`) drops the DJ's *record* of it and **never the file**. Two clicks in the UI; there is a test asserting the bytes on disk survive. |
| `crate_model.json` | The author's trained weights + centroids + provenance, kept for reference and **never written**. The live model is `~/.crate/crate_model.json` (`crateapp/model_file.py`), seeded from this file once for an existing library; a new user starts with none. |
| `crate.command` | Double-click launcher. `~/Desktop/Crate.app` wraps it. |
| `viewer_template.html` | The original standalone viewer (superseded by the app, still works). |

**Ground truth lives in `~/Documents/rek.xml`** — the DJ's own Rekordbox
playlists. Genre playlists: `tech`, `house`, `pop`, `vocals`, `afro`, `rap`,
`UKG`, `DUBSTEP` (plus `TECH HOUSE`/`AFROHOUSE`/`RAP` which map onto the
lowercase ones). `MAIN`, `PARTY`, `REMIX` are **set lists, not genres**;
`AllSongs` and `Contents` are containers. Both groups are excluded — filing by
them would teach the model that "tracks I play at parties" is a sound.

Library database: `~/.crate/library.db`. Trained crate model: `~/.crate/crate_model.json`. Models: `~/.crate/models`. Converted
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

### §3a. The decision order, and why it is mostly not audio

**acapella → genre tag → who made it → audio model.** Measured held out on
530 distinct labelled recordings:

```
audio model alone                        75.1%
acapella -> person -> audio              77.4%
acapella -> tag -> person -> audio       93.2%   <- ships
   ...on house vs tech house             97.6%   (audio alone: 76.3%)
   ...vocals                    precision 100%, recall 96.8%

per crate: house .970  tech .939  pop .985  DUBSTEP 1.000  vocals 1.000
           UKG .880   afro .895  rap .815   rock .778     (precision)
```

**The genre tag written into the file decides 88% of tracks.** Whoever sold or
ripped the track wrote "Tech House" or "UK Garage / Bassline" into it, and the
DJ files by that tag - which is why the audio never recovered the house/tech
split and why the tag recovers it at 98%. Present on 95% of their labelled
tracks (only 32% of the unfiled ones, which is the current limit).

`crates.tag_crate` learns the mapping from the DJ's own filing - **nothing is
hardcoded**, so "Tech House" means whatever crate they put such tracks in. A
tag must be seen twice and point one way at least half the time.

Two subtleties, both load-bearing:

- Tag purity **excludes acapellas**, because the acapella rule runs first and
  they never reach the tag rule. "Hip-Hop/Rap" covers 39 rap tracks, 12 pop
  and 32 isolated vocals: 46% pure counting the vocals, 73.5% without them.
  That is the difference between the tag being unusable and unlocking 51
  tracks.
- Read the genre from `MetadataReader` index **4**. Index 5 is the track
  NUMBER, and reading it produces a confident mapping from the tag "1" to a
  crate. This happened.

**Who made it** is the fallback where there is no usable tag: the remixer
named in the filename first ("(Gorgon City Remix)" is Gorgon City's record,
not Diplo's), then the artist tag, and only where the DJ's filing for that
person is unanimous. 88.7% accurate over 30% of tracks.

**None of this is the filename-keyword dead end.** That guessed a genre from
words in a *title*. These read the file's own metadata and ask "where do I
file tracks like this", learning the answer from the DJ. Never reintroduce
genre-from-words.

Where any rule disagrees with the audio model the track is marked `disputed`
and surfaces in "Worth a second look" — none of them is perfect, and the
disagreement costs nothing because both numbers are computed anyway.

**Coverage, not accuracy, is the remaining limit.** 96% of hand-filed tracks
carry a genre tag but only 32% of the unfiled ones do; the rest are
SoundCloud rips that fall through to the audio model at ~77%. Every tag the
DJ teaches (two tracks is enough) unlocks all the others carrying it — e.g.
"minimal / deep tech" is on 34 unfiled tracks and has never been filed once.

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
planning to "fix" this pair with a better *audio* model should read the
dead-ends table first and expect to be disappointed.

**But the pair is not unfixable — it just is not an audio problem.** See §3a.

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

**A crate can be locked to what the DJ put in it.** `crates.set_locked` —
"the song" is one of these. Nothing is ever filed into a locked crate, and,
just as important, **nothing inside it teaches any rule**: without that, a
crate holding one favourite track would teach "this artist belongs here" and
the rules would file that artist's whole catalogue in beside it. It is also
kept out of the model entirely — not a predictable class, and not a centroid,
since a one-track mean would otherwise sit in the middle of the "is this like
anything I own" check. The DJ can still file into it by hand; locking
constrains the tool, not them.

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
- 287 tests pass (`python -m pytest tests/ -q`).
- `scan()` holds the DB lock only for its reads and writes, not while walking
  the folder or reading tags, so the UI stays responsive during a scan.

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

1. **Teach the tags** — the review queue now leads with "Teach a tag"
   (`crates.tag_gaps`). Filing ONE track from each of 8 tags settles ~84
   others, because the tag rule then applies to all of them. Highest return
   on the DJ's attention by a wide margin.
2. **Work "Worth a second look"** (~129 tracks). These are filed
   confidently but the two independent signals disagree, and measured, 25% of
   them are genuinely misfiled against 2.9% elsewhere. Shortest list, highest
   yield — and the only way these errors ever surface, since the model is sure.
3. **Then the close calls** (~84). Every **Correct** is one click and becomes
   training data.
4. **Export and load onto a USB / CDJ** with the re-sorted crates.
5. More labels **only for afro (20) and UKG (25)** — those two crates are
   individually starved (afro recall 0.30, UKG 0.64). Overall *audio* accuracy
   has plateaued, so labelling for its own sake is not worth the evening —
   but every track filed by a NEW artist extends the artist rule's coverage,
   which is worth a great deal (§3a).
6. An untested idea worth one experiment if the above is exhausted: swap
   Discogs-EffNet for **MuQ** (`pip install muq`, PyTorch, CPU-capable, needs
   24kHz audio), the current state of the art on music genre benchmarks. It
   would mean re-embedding the library and a ~2GB dependency, and it is
   unproven on *this* distinction — expect it to help the taxonomy generally
   rather than rescue house-vs-tech.

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
