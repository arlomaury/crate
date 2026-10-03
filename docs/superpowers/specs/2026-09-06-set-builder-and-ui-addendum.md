# Crate — addendum: UI, live sorting view, and the set builder

**Date:** 2026-09-06
**Status:** design, awaiting approval. No code written yet.
**Extends:** `2026-09-04-crate-app-design.md` (unchanged except where stated)

---

## 0. Correction to the base spec: what a move destroys

A move currently deletes **every** assignment row for a track. So a track the DJ
deliberately also-added to Party loses that membership the moment an unrelated
correction moves it from House to Tech House.

**Decision: a move replaces only the placement the model made.** It deletes the
`source='auto'` assignment and nothing else. Human also-adds are the DJ's own
statements about a track and are never collateral damage of correcting a
different crate.

Two consequences that follow, both fixed with it:

- An also-add must clear the track from the **Uncertain** view. Acting on a track
  is acting on it; it should not keep reappearing as unresolved.
- Re-classifying a track must replace its previous `source='auto'` row rather
  than accumulating stale ones.

---

## 1. Visual design

The look is Apple's: content first, chrome receding, one accent colour, generous
whitespace, restrained motion. **Dark throughout — the app is black, not white.**
This is not a theme toggle; it is the design. A DJ works in dark rooms and beside
Rekordbox, which is itself dark.

- **Ground:** true near-black, not charcoal grey. Panels sit slightly above it as
  translucent layers, separated by hairlines rather than boxes.
- **Type:** system font stack (`-apple-system`/SF), a strict scale, tight
  headings, comfortable body leading. Light text on black must be a shade below
  pure white or it vibrates.
- **Colour:** desaturated neutrals carrying a faint accent hue, one true accent
  used sparingly. Semantic colours (uncertain, unknown) are separate from the
  accent and never compete with it.
- **Motion:** short, eased, purposeful. Things that change state animate; nothing
  animates for decoration.

No light mode. Every surface is designed once, for black.

---

## 2. The live sorting view

One panel shows the track currently being decided. **The panel is always
present — only its motion is conditional.** It never appears or disappears,
because a panel that comes and goes makes the whole layout jump each time
sorting starts or stops.

Two states:

- **Active** — sorting is running. Everything below animates: the curve draws,
  the bars race, cue markers land, tracks cross-dissolve.
- **At rest** — nothing to sort. The panel holds the **last track it decided**
  as a still frame: its finished curve, its crate bars settled, its verdict.
  Dimmed slightly, motion stopped. Nothing spins, pulses, or idles.

Holding the last result is deliberate: it is informative rather than decorative,
so a DJ who looks over after a long import sees where it finished instead of an
empty box. Before anything has ever been sorted, it shows a single quiet line and
a short invitation to add a folder — the same shape, at rest.

**It shows the real computation, never invented reasoning.** Every number on
screen is one the classifier actually used:

1. the track's energy curve drawing in as it is analysed;
2. its measured BPM and Camelot key;
3. a bar per crate showing **actual cosine similarity** to that crate's centroid;
4. the verdict, with the number that decided it —
   *"House 0.81 · Tech House 0.79 · margin 0.02 → uncertain, flagged for you."*

The value here is trust: the DJ sees *why* something was filed, and sees the
near-misses. A track that lands 0.81/0.79 is visibly a coin-flip, and that is
worth knowing. Fabricated "thinking" narration is explicitly out of scope — it
would undermine the one thing this panel is for.

**It should be genuinely nice to watch.** This panel is the app's one ambient
moment — something a DJ may leave running over a long import — so it earns real
visual care:

- The energy curve draws left to right as the track is read, sub-bass and full
  band as two layered traces rather than a plain bar chart.
- Crate similarity bars ease outward and settle, so near-ties are visible as a
  photo-finish rather than a static list.
- Cue markers land on the curve as they are detected, each briefly pulsing.
- Between tracks, the panel cross-dissolves rather than cutting.

**Calm colour, not a light show.** Deep desaturated blues, teals and slate
against the black ground, with warmth reserved for meaning — an amber only for
"uncertain", nothing saturated unless it is telling you something. Low contrast
between decorative elements, high contrast only for the numbers that matter.
Nothing flashes, strobes, or competes with the numbers for attention.

---

## 3. The set builder

A second half of the app: given the library, find tracks that mix well together
and say **where** to mix them.

### 3.1 Why this can beat existing tools

Commercial harmonic-mixing tools (Mixed In Key, rekordbox, Beatport) match on
**key and tempo only**. Two tracks can be 8A at 128 BPM and still sound wrong
together. We hold a 1,280-dimension embedding per track, so we can also score
whether two records *inhabit the same sonic world*. That signal is unavailable to
key-and-tempo tools and is the reason this can suggest pairings they cannot.

Second advantage: we know where every drop, breakdown, buildup and outro is —
11,291 cue points across the library. So the output is not "play B after A" but
"mix out of A at 5:12, bring B in at 0:32."

### 3.1a Mix points: an exact point AND a usable window

Every transition returns both:

- **the recommended point** — a single best in/out pair, snapped to a downbeat,
  e.g. *out of A at 5:12.4, into B at 0:32.1*;
- **the window it sits in** — the range over which the transition still works,
  e.g. *out anywhere 4:56–5:28*, because that whole span is A's outro and phrasing
  repeats every 8 bars.

A single point is too rigid — DJs ride a transition by feel and rarely hit an
exact second. A range alone is too vague to act on quickly. Both together say
"start here, and you have this much room."

Windows are derived from structure, not invented: the outro window runs from the
last breakdown or energy fall to the end; the intro window runs from the start to
the first drop. Both are snapped to downbeats and quantised to 8-bar phrases, so
every offered point is phrase-aligned and lands where a DJ would actually cut.

### 3.2 Compatibility score

Five signals, combined into one score with the reasons kept visible:

| Signal | Rule | Basis |
|---|---|---|
| **Tempo** | ±6% is the practical pitch range; ±3 BPM is comfortable. Half/double-time counts as compatible (140 ↔ 70). | measured DJ practice |
| **Harmonic** | Camelot: same code = perfect; ±1 same letter = perfect fifth; same number, opposite letter = relative major/minor; +2 = energy lift. | Camelot wheel |
| **Timbral** | cosine similarity between track embeddings | ours, unique |
| **Vocal clash** | two vocal-dominant tracks back to back fight; penalise using `vocal_ratio` | ours |
| **Energy flow** | loudness, danceability and energy curve, so a set moves deliberately | ours |

**Tempo and key are not independent.** Beyond about ±6% a track shifts roughly a
semitone — seven Camelot positions (the wheel is a circle of fifths) — so a large tempo stretch silently breaks the
harmonic match unless key-lock is on. The score accounts for this rather than
treating the two as separate checks.

Every suggestion carries its reasons in plain words: *"same key (8A), +1.2 BPM,
sounds close, energy lifts slightly."* A score with no explanation is not
trustworthy enough to act on.

### 3.3 Two modes

- **Build a set.** Give it a length, a starting point (track and/or crate), and an
  energy shape (build / peak / cool-down). It returns an ordered setlist with a
  mix point for every transition.
- **What goes next.** Pick any track, get ranked candidates with reasons and mix
  points. This is the co-pilot mode — the one likely used most while preparing.

Both run off the same scoring engine.

### 3.4 Controls

- **Safe ↔ adventurous slider**, default centre. Safe favours tight tempo,
  textbook Camelot moves and high timbral similarity — transitions that cannot
  really fail. Adventurous widens tempo tolerance, allows energy-boost key jumps
  and lower timbral similarity, and reaches across crates.
- **Crate scope**, default the whole library, optionally restricted ("UKG only",
  "House + Tech House"). Whole-library is the default because cross-crate
  pairings are exactly what the timbral signal is good at finding.

### 3.5 Transition preview

A suggestion can be auditioned: the last ~20 seconds of track A crossfaded into
track B at the proposed cue. The server streams the two files with HTTP range
requests; the browser crossfades them via the Web Audio API.

This is what separates a list of theoretical suggestions from something worth
acting on. Audio is streamed from the original files and never transcoded,
converted, or modified — quality is exactly the source file's.

---

## 4. Build order

1. Finish the core sorting app (base spec tasks 6–10), with the Apple visual
   language and the live sorting view built in rather than retrofitted.
2. Then the set builder as its own phase: scoring engine → API → UI → preview.

The set builder depends on a populated catalogue and real crates, so it inherits
a finished foundation this way instead of being wired into a half-built app.

---

## 5. Deliberately out of scope

- No fabricated reasoning text anywhere in the sorting view.
- No transcoding, format conversion, or writing to source audio — ever.
- No auto-publishing of sets anywhere; a set is a local artefact.
- No beatgrid editing or waveform scrubbing; Rekordbox owns that.
