#!/usr/bin/env python3
"""
Crate - audio analysis for DJ libraries.

Reads every audio file in a folder and works out, from the sound itself:
  tempo, key, genre, whether it is an acapella, and where the drops are.

Nothing here reads the filename to decide a genre.
"""

import os, sys, json, csv, math, time, argparse, hashlib
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import essentia
import essentia.standard as es
essentia.log.warningActive = False

AUDIO_EXT = {".wav", ".mp3", ".aiff", ".aif", ".flac", ".m4a", ".ogg", ".wma"}

MODEL_DIR = Path(os.environ.get("CRATE_MODELS", Path.home() / ".crate" / "models"))

# Camelot wheel - DJs mix by this, not by "F# minor".
CAMELOT = {
    ("B",  "major"): "1B",  ("F#", "major"): "2B",  ("Db", "major"): "3B",
    ("Ab", "major"): "4B",  ("Eb", "major"): "5B",  ("Bb", "major"): "6B",
    ("F",  "major"): "7B",  ("C",  "major"): "8B",  ("G",  "major"): "9B",
    ("D",  "major"): "10B", ("A",  "major"): "11B", ("E",  "major"): "12B",
    ("Ab", "minor"): "1A",  ("Eb", "minor"): "2A",  ("Bb", "minor"): "3A",
    ("F",  "minor"): "4A",  ("C",  "minor"): "5A",  ("G",  "minor"): "6A",
    ("D",  "minor"): "7A",  ("A",  "minor"): "8A",  ("E",  "minor"): "9A",
    ("B",  "minor"): "10A", ("F#", "minor"): "11A", ("Db", "minor"): "12A",
}
ENHARMONIC = {"C#": "Db", "D#": "Eb", "G#": "Ab", "A#": "Bb", "Gb": "F#",
              "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}

# Frequency bands, in Hz. Sub is the one that matters most for drop detection:
# in dance music the kick and sub-bass drop out entirely during a breakdown
# and slam back in on the one. That gap is the structure.
BANDS = {"sub": (20, 120), "lowmid": (120, 500), "mid": (500, 2000),
         "high": (2000, 8000), "air": (8000, 20000)}

FRAME, HOP, SR = 2048, 1024, 44100


# ----------------------------------------------------------------- utilities

def camelot(key, scale):
    key = ENHARMONIC.get(key, key)
    return CAMELOT.get((key, scale), "")


def norm01(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def slope(y):
    """Trend of a short series, scaled so it is comparable between tracks."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0]) * len(y)


# ------------------------------------------------------------ frame features

def frame_features(audio):
    """Per-frame energy in each band, plus brightness and onset strength."""
    w = es.Windowing(type="hann")
    spec = es.Spectrum()
    centroid = es.Centroid(range=SR / 2)
    rms_algo = es.RMS()
    flux = es.Flux()

    freqs = np.fft.rfftfreq(FRAME, 1.0 / SR)
    idx = {n: np.where((freqs >= lo) & (freqs < hi))[0] for n, (lo, hi) in BANDS.items()}

    out = {n: [] for n in BANDS}
    out.update(rms=[], centroid=[], flux=[])

    for fr in es.FrameGenerator(audio, frameSize=FRAME, hopSize=HOP, startFromZero=True):
        s = spec(w(fr))
        out["rms"].append(rms_algo(fr))
        out["centroid"].append(centroid(s))
        out["flux"].append(flux(s))
        for n, ix in idx.items():
            out[n].append(float(np.sqrt(np.mean(s[ix] ** 2))) if ix.size else 0.0)

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def to_bars(feat, beats, times):
    """
    Collapse frames into bars.

    Everything downstream works in bars rather than seconds, because dance music
    is built in 4-bar and 8-bar blocks and a cue point that is not on a downbeat
    is no use on a CDJ.
    """
    if len(beats) < 8:
        return None, None

    downbeats = beats[::4]
    bars = []
    for i in range(len(downbeats) - 1):
        m = (times >= downbeats[i]) & (times < downbeats[i + 1])
        if not m.any():
            continue
        bars.append({k: float(np.mean(v[m])) for k, v in feat.items()})

    if len(bars) < 8:
        return None, None

    agg = {k: np.array([b[k] for b in bars]) for k in bars[0]}
    return agg, downbeats[:len(bars)]


# --------------------------------------------------------- structure / moments

def find_moments(bars, downbeats, dur):
    """
    Locate the points a DJ actually cares about: where the bass drops out,
    where it comes back, where the track is building.

    Returns timestamps already snapped to downbeats.
    """
    if bars is None:
        return [], {}

    sub = norm01(bars["sub"])
    rms = norm01(bars["rms"])
    bright = norm01(bars["centroid"])
    n = len(sub)

    # "Full" = kick and sub are present. This is the single most reliable
    # structural signal in four-to-the-floor and garage alike.
    full = (sub > 0.45) & (rms > 0.40)

    # Smooth out single-bar dropouts so one quiet bar mid-drop doesn't split it.
    for i in range(1, n - 1):
        if full[i - 1] and full[i + 1]:
            full[i] = True

    # Contiguous runs of full-energy and of low-energy.
    runs, start = [], 0
    for i in range(1, n + 1):
        if i == n or full[i] != full[start]:
            runs.append((start, i - 1, bool(full[start])))
            start = i

    moments = []
    sections = []
    drops = [r for r in runs if r[2] and (r[1] - r[0] + 1) >= 4]

    for k, (a, b, _) in enumerate(drops):
        # A drop only counts if energy actually lifted into it.
        pre = rms[max(0, a - 4):a]
        lift = rms[a] - (float(np.mean(pre)) if pre.size else 0.0)
        kind = "drop" if (k > 0 or a > 3) and lift > 0.12 else "main section"
        moments.append({
            "type": kind, "bar": int(a), "time": float(downbeats[a]),
            "confidence": round(float(min(1.0, 0.45 + lift * 2.2)), 2),
            "label": f"Drop {k + 1}" if kind == "drop" else "Main section",
        })
        sections.append({"type": "drop", "start": float(downbeats[a]),
                         "end": float(downbeats[min(b + 1, n - 1)])})

    for a, b, is_full in runs:
        length = b - a + 1
        if is_full or length < 4:
            continue
        if a == 0:
            kind, label = "intro", "Intro"
        elif b >= n - 2:
            kind, label = "outro", "Outro"
        else:
            kind, label = "breakdown", "Breakdown"
        moments.append({
            "type": kind, "bar": int(a), "time": float(downbeats[a]),
            "confidence": 0.8, "label": label,
        })
        sections.append({"type": kind, "start": float(downbeats[a]),
                         "end": float(downbeats[min(b + 1, n - 1)])})

    # Buildups: the bars before a drop where energy and brightness climb
    # together. Snare rolls and risers both show up this way.
    for m in [x for x in moments if x["type"] == "drop"]:
        a = m["bar"]
        w0 = max(0, a - 8)
        if a - w0 < 4:
            continue
        e_tr, b_tr = slope(rms[w0:a]), slope(bright[w0:a])
        if e_tr > 0.25 or b_tr > 0.30:
            moments.append({
                "type": "buildup", "bar": int(w0), "time": float(downbeats[w0]),
                "confidence": round(float(min(1.0, 0.4 + (e_tr + b_tr) / 2)), 2),
                "label": "Buildup",
            })

    moments.sort(key=lambda x: x["time"])

    # Merge anything landing within a bar of a previous marker.
    bar_sec = (downbeats[1] - downbeats[0]) if len(downbeats) > 1 else 2.0
    merged = []
    for m in moments:
        if merged and abs(m["time"] - merged[-1]["time"]) < bar_sec * 0.9:
            if m["confidence"] > merged[-1]["confidence"]:
                merged[-1] = m
            continue
        merged.append(m)

    # Number each kind in playing order, so labels read "Drop 1, Drop 2" and
    # "Buildup 1, Buildup 2". Numbering happens here, after merging, rather
    # than where moments are created: the old code numbered drops by their
    # index in the drops list, so a track whose first high-energy section was
    # classed "Main section" started counting at Drop 2 and never had a Drop 1.
    seen = {}
    for m in merged:
        if m["type"] in ("intro", "outro", "main section"):
            continue
        seen[m["type"]] = seen.get(m["type"], 0) + 1
        m["index"] = seen[m["type"]]
        m["label"] = f"{m['type'].capitalize()} {seen[m['type']]}"

    # Rekordbox has eight hot cue slots and a DJ wants all eight usable. Where
    # a track has less detected structure than that, top up on 32-bar phrase
    # boundaries - real mix points in 4/4 dance music. Labelled "Phrase N" so
    # detected structure is never disguised as something it isn't.
    if len(merged) < 8 and downbeats is not None and len(downbeats) >= 16:
        bar_sec = (downbeats[1] - downbeats[0]) if len(downbeats) > 1 else 2.0
        extra = []
        # Prefer wide, obviously-musical spacing; fall back to tighter phrase
        # boundaries only when a track is too short to yield eight that way.
        # 32, 16 and 8 are all real phrase lengths in 4/4 dance music.
        # (bar step, how far a new cue must stay from an existing one, in bars).
        # Later passes relax both, so a short track can still fill eight slots.
        for step, clear in ((32, 3.5), (16, 3.5), (8, 2.0), (4, 1.5)):
            if len(merged) + len(extra) >= 8:
                break
            for b in range(0, len(downbeats), step):
                if len(merged) + len(extra) >= 8:
                    break
                t = float(downbeats[b])
                near = [m["time"] for m in merged] + [e["time"] for e in extra]
                if all(abs(t - u) > bar_sec * clear for u in near):
                    extra.append({"type": "phrase", "bar": int(b), "time": t,
                                  "confidence": 0.5, "label": "Phrase"})
        merged = sorted(merged + extra, key=lambda x: x["time"])
        n_ph = 0
        for m in merged:
            if m["type"] == "phrase":
                n_ph += 1
                m["index"] = n_ph
                m["label"] = f"Phrase {n_ph}"

    stats = {
        "bars_analysed": int(n),
        "drop_count": sum(1 for m in merged if m["type"] == "drop"),
        "breakdown_count": sum(1 for m in merged if m["type"] == "breakdown"),
        "energy_curve": [round(float(v), 3) for v in rms],
        "sub_curve": [round(float(v), 3) for v in sub],
        "bar_times": [round(float(t), 2) for t in downbeats],
    }
    return merged, stats


# ----------------------------------------------------------- vocal / acapella

def vocal_profile(bars, feat, dur):
    """
    Decide whether this file is an acapella.

    An acapella has vocal-range energy but almost no kick and no sub. That
    combination is distinctive enough to detect without a trained model, which
    matters because genre models are trained on full mixes and produce nonsense
    when handed an isolated vocal.

    Thresholds calibrated 2026-09-04 against 27 real acapellas (stem-splits and
    a labelled acapella from the library) and 130 known full mixes:

        signal                 acapellas   full mixes
        very_little_sub          26/27        0/130
        vocal_range_dominant     27/27        0/130
        sparse_low_end           25/27        0/130

    Each is close to a perfect separator on its own, with a wide empty gap
    between the populations (sub ratio tops out at 0.172 on acapellas and
    starts at 0.511 on full mixes), so 2-of-3 detects every acapella in the
    sample without misfiling a single full mix.

    A fourth signal, `onset_rate < 2.6`, was removed: measured across the same
    157 tracks it fired on 0/27 acapellas and 1/130 full mixes - real vocal
    takes are full of consonant onsets (2.95-4.98/s), overlapping full mixes
    (2.43-8.04/s) almost entirely. It could only ever push a full mix toward
    "acapella", never identify one.
    """
    if bars is None:
        band = {k: float(np.mean(feat[k])) for k in BANDS}
    else:
        band = {k: float(np.mean(bars[k])) for k in BANDS}

    total = sum(band.values()) + 1e-9
    sub_r = band["sub"] / total
    voc_r = (band["mid"] + band["lowmid"]) / total

    # Score each acapella indicator separately so the reason is inspectable.
    ev = {
        "very_little_sub": sub_r < 0.25,
        "vocal_range_dominant": voc_r > 0.46,
        "sparse_low_end": band["sub"] < band["mid"] * 0.55,
    }
    score = sum(ev.values()) / len(ev)

    return {
        "is_acapella": score >= 0.66,       # 2 of 3
        "needs_review": 0.33 <= score < 0.66,  # exactly 1 of 3
        "score": round(score, 2),
        "evidence": ev,
        "sub_ratio": round(sub_r, 3),
        "vocal_ratio": round(voc_r, 3),
    }


# ------------------------------------------------------------- genre (models)

class GenreModel:
    """
    Discogs-EffNet embeddings feeding a 400-label Discogs genre/style head.

    The Discogs taxonomy is the right one here: it distinguishes UK Garage from
    Deep House from Speed Garage, which the common GTZAN-trained models
    (10 broad genres, no electronic subgenres) simply cannot do.

    Some genres the DJ actually uses - Afro House confirmed so far - are not
    among the 400 classes this head outputs at all, so they can never appear
    as `category` no matter how correct the audio read is. See
    `taxonomy_gap_signal()` below for the embedding-similarity workaround,
    and build_reference.py for how its reference data is built/refreshed.
    """

    EMB = "discogs-effnet-bs64-1.pb"
    HEAD = "genre_discogs400-discogs-effnet-1.pb"
    META = "genre_discogs400-discogs-effnet-1.json"
    VOICE = "voice_instrumental-discogs-effnet-1.pb"
    REFERENCE = Path(__file__).resolve().parent / "genre_reference.json"

    def __init__(self, model_dir=MODEL_DIR):
        self.dir = Path(model_dir)
        self.ok = False
        self.voice_ok = False
        self.ref_ok = False
        emb, head, meta = self.dir / self.EMB, self.dir / self.HEAD, self.dir / self.META
        if not (emb.exists() and head.exists() and meta.exists()):
            return
        try:
            self.embed = es.TensorflowPredictEffnetDiscogs(
                graphFilename=str(emb), output="PartitionedCall:1")
            self.head = es.TensorflowPredict2D(
                graphFilename=str(head), input="serving_default_model_Placeholder",
                output="PartitionedCall:0")
            self.labels = json.loads(meta.read_text())["classes"]
            self.ok = True
        except Exception as e:
            print(f"  ! genre model failed to load: {e}", file=sys.stderr)
            return
        v = self.dir / self.VOICE
        if v.exists():
            try:
                self.voice = es.TensorflowPredict2D(
                    graphFilename=str(v), output="model/Softmax")
                self.voice_ok = True
            except Exception:
                pass
        if self.REFERENCE.exists():
            try:
                ref = json.loads(self.REFERENCE.read_text())
                folders = ref["folders"]
                slugs = sorted(folders.keys())
                centroids = np.array([folders[s]["centroid"] for s in slugs],
                                      dtype=np.float32)
                self.ref_slugs = slugs
                self.ref_meta = folders
                self.ref_centroids = centroids / (
                    np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
                self.ref_ok = True
            except Exception as e:
                print(f"  ! {self.REFERENCE.name} failed to load: {e}", file=sys.stderr)

    def taxonomy_gap_signal(self, mean_emb, predicted_category):
        """
        Is this track's embedding nearest to a reference playlist folder whose
        genre the Discogs-400 taxonomy has no label for at all (currently:
        Afro House), *and* did the genre model's own top prediction land on a
        label that folder's real tracks actually produce?

        Validated 2026-09-04 via leave-one-out nearest-centroid classification
        on a 130-track, 5-folder sample: 70% recall on Afro House, Tech House
        the main false-positive source (~18%). Real but imperfect, so this is
        exposed as data for `needs_review` - it never overrides `category`.
        """
        if not self.ref_ok or mean_emb.shape[0] != self.ref_centroids.shape[1]:
            # Dimension mismatch means genre_reference.json was built against
            # a different embedding model - stale reference data, not a bug
            # to crash on. Silently skip the signal rather than guess.
            return None
        v = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        sims = self.ref_centroids @ v
        i = int(np.argmax(sims))
        slug = self.ref_slugs[i]
        meta = self.ref_meta[slug]
        if not meta["is_gap"]:
            return None
        if predicted_category not in meta["compatible_categories"]:
            return None
        return {
            "nearest_reference_genre": slug,
            "display_name": meta["display_name"],
            "similarity": round(float(sims[i]), 3),
        }

    # Deliberately the exact rule that was measured, not a generalised one:
    # "Bassline" sits in both the UKG and TECH_HOUSE folders' observed label
    # sets, so a purely data-driven version of this check would be murkier
    # than the thing actually validated.
    GARAGE_LABELS = {"Bassline", "UK Garage", "Speed Garage"}

    def garage_conflict_signal(self, mean_emb, predicted_category):
        """
        The model calls some polished commercial tech house "Bassline" with
        *high* confidence - confirmed against Beatport tags on Dom Dolla,
        FISHER, Bob Sinclar, The Blessed Madonna and Green Velvet. The
        low-confidence review flag never catches these, precisely because the
        model is sure.

        So: the label says garage, but is the embedding actually closer to the
        tech house reference folder than to the garage one? Validated
        leave-one-out on the reference folders 2026-09-04 - catches 2/3 of
        the in-folder mislabels and 4/5 of the Beatport-confirmed cases, while
        wrongly flagging 1 of 18 genuine garage tracks (~6%).
        """
        if not self.ref_ok or mean_emb.shape[0] != self.ref_centroids.shape[1]:
            return None
        if predicted_category not in self.GARAGE_LABELS:
            return None
        if "TECH_HOUSE" not in self.ref_meta or "UKG" not in self.ref_meta:
            return None
        v = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
        sims = self.ref_centroids @ v
        tech = float(sims[self.ref_slugs.index("TECH_HOUSE")])
        garage = float(sims[self.ref_slugs.index("UKG")])
        if tech <= garage:
            return None
        return {
            "predicted": predicted_category,
            "tech_house_similarity": round(tech, 3),
            "garage_similarity": round(garage, 3),
        }

    def predict(self, audio16k, top=5):
        emb = self.embed(audio16k)
        act = self.head(emb).mean(axis=0)
        order = np.argsort(act)[::-1][:top]
        preds = [{"label": self.labels[i].split("---")[-1],
                  "parent": self.labels[i].split("---")[0],
                  "full": self.labels[i],
                  "confidence": round(float(act[i]), 4)} for i in order]
        voice = None
        if self.voice_ok:
            vp = self.voice(emb).mean(axis=0)
            voice = {"instrumental": round(float(vp[0]), 3),
                     "voice": round(float(vp[1]), 3)}
        return preds, voice, emb


# --------------------------------------------------------------- main analysis

def embedding_path(emb_dir, track_path):
    """
    Where a track's embedding is cached. Keyed by a hash of the resolved path
    so filenames of any length or character set map to a safe flat filename.
    """
    key = hashlib.sha1(str(Path(track_path).resolve()).encode("utf-8")).hexdigest()
    return Path(emb_dir) / f"{key}.npy"


def analyse(path, gm, verbose=True, emb_dir=None):
    t0 = time.time()
    p = Path(path)
    res = {"file": p.name, "path": str(p.resolve()),
           "size_mb": round(p.stat().st_size / 1e6, 1), "errors": []}

    try:
        audio = es.MonoLoader(filename=str(p), sampleRate=SR)()
    except Exception as e:
        res["errors"].append(f"could not decode: {e}")
        return res

    dur = len(audio) / SR
    res["duration_sec"] = round(dur, 2)
    res["duration"] = f"{int(dur // 60)}:{int(dur % 60):02d}"
    if dur < 5:
        res["errors"].append("file shorter than 5s, skipped")
        return res

    # --- tempo -------------------------------------------------------------
    try:
        bpm, beats, bconf, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
        res["bpm"] = round(float(bpm), 2)
        # multifeature confidence runs 0-5.32; rescale and be honest about it.
        res["bpm_confidence"] = round(min(1.0, float(bconf) / 3.5), 2)
        res["bpm_reliable"] = bool(bconf > 1.0)
        res["beat_count"] = len(beats)
    except Exception as e:
        beats = np.array([])
        res["errors"].append(f"tempo: {e}")

    # --- key ---------------------------------------------------------------
    try:
        k, sc, ks = es.KeyExtractor()(audio)
        res.update(key=f"{k} {sc}", camelot=camelot(k, sc),
                   key_confidence=round(float(ks), 2))
    except Exception as e:
        res["errors"].append(f"key: {e}")

    # --- level / feel ------------------------------------------------------
    try:
        res["loudness_lufs"] = round(float(es.LoudnessEBUR128()(
            np.array([audio, audio]).T.astype(np.float32))[2]), 1)
    except Exception:
        pass
    try:
        res["danceability"] = round(float(es.Danceability()(audio)[0]), 2)
    except Exception:
        pass
    try:
        res["dynamic_complexity"] = round(float(es.DynamicComplexity()(audio)[0]), 2)
    except Exception:
        pass
    try:
        onset_rate = float(es.OnsetRate()(audio)[1])
        res["onset_rate"] = round(onset_rate, 2)
    except Exception:
        onset_rate = 3.0

    # --- structure ---------------------------------------------------------
    feat = frame_features(audio)
    times = np.arange(len(feat["rms"])) * HOP / SR
    bars, downbeats = to_bars(feat, beats, times)
    moments, stats = find_moments(bars, downbeats, dur)
    res["moments"] = moments
    res["structure"] = stats

    tot = sum(float(np.mean(feat[b])) for b in BANDS) + 1e-9
    res["tonal_balance"] = {b: round(float(np.mean(feat[b])) / tot, 3) for b in BANDS}
    res["brightness_hz"] = round(float(np.mean(feat["centroid"])), 1)

    # --- vocals ------------------------------------------------------------
    voc = vocal_profile(bars, feat, dur)
    res["vocal_analysis"] = voc

    # --- genre -------------------------------------------------------------
    if gm and gm.ok:
        try:
            a16 = es.MonoLoader(filename=str(p), sampleRate=16000,
                                resampleQuality=4)()
            preds, voice, emb = gm.predict(a16)
            res["genre_predictions"] = preds
            if voice:
                res["voice_model"] = voice
                # The voice model may CONFIRM the DSP evidence; it may never
                # promote it.
                #
                # It answers "is there singing on this", not "is this an
                # isolated vocal" - so every sung pop record scores above 0.85
                # on it. This once ran at `score >= 0.33`, on the reasoning
                # that 0.33 was "the same half-convinced state 0.5 meant on
                # the old 4-signal scale". It is not: 0.33 of three signals is
                # ONE, where 0.5 of four was two. So a single DSP hint plus
                # audible singing was enough to declare an acapella, and it
                # declared 28 of them - Mr. Brightside, Viva La Vida,
                # A Thousand Miles, Starships. Those average a 0.425 sub-bass
                # ratio; real acapellas here run 0.03-0.15.
                #
                # It also cleared needs_review, so the mistakes never surfaced
                # for the DJ to catch. vocal_profile already flags a 1-of-3
                # score for review, and that is the correct outcome.
                if voice["voice"] > 0.85 and voc["score"] >= 0.66:
                    voc["evidence"]["voice_model_agrees"] = True
            mean_emb = emb.mean(axis=0)
            # Cache the embedding rather than discarding it. Recomputing means
            # re-reading the audio, and the crate-matching in the app needs to
            # recompute centroids on every correction. Kept out of results.json
            # on purpose - that payload gets injected wholesale into viewer.html.
            if emb_dir is not None:
                try:
                    dest = embedding_path(emb_dir, p)
                    if not dest.exists():
                        np.save(dest, mean_emb.astype(np.float32))
                except Exception as e:
                    res["errors"].append(f"embedding cache: {e}")
            gap = gm.taxonomy_gap_signal(mean_emb, preds[0]["label"])
            if gap:
                res["taxonomy_gap_signal"] = gap
            conflict = gm.garage_conflict_signal(mean_emb, preds[0]["label"])
            if conflict:
                res["garage_conflict_signal"] = conflict
        except Exception as e:
            res["errors"].append(f"genre: {e}")
    else:
        res["genre_predictions"] = []

    # --- final category ----------------------------------------------------
    # Acapellas are routed to Vocals before genre is even considered, because
    # a genre label on an isolated vocal is meaningless.
    if voc["is_acapella"]:
        res["category"] = "Vocals"
        res["category_confidence"] = round(voc["score"], 2)
        res["category_basis"] = "acapella detected from audio (no drums/sub)"
    elif res.get("genre_predictions"):
        top = res["genre_predictions"][0]
        res["category"] = top["label"]
        res["category_parent"] = top["parent"]
        res["category_confidence"] = top["confidence"]
        res["category_basis"] = "Discogs-EffNet genre model"
        if top["confidence"] < 0.15:
            res["needs_review"] = True
            res["review_reason"] = "genre model confidence below 0.15"
    else:
        res["category"] = "Unclassified"
        res["category_confidence"] = 0.0
        res["category_basis"] = "no genre model available"
        res["needs_review"] = True
        res["review_reason"] = "genre model not installed - run setup.sh"

    # Some genres (Afro House confirmed so far) have no label in the genre
    # model's own taxonomy at all - see GenreModel.taxonomy_gap_signal(). When
    # it fires, flag for review instead of silently overriding category -
    # it's ~70% precision/recall, a real signal, not a certain one.
    gap = res.get("taxonomy_gap_signal")
    if gap:
        res["needs_review"] = True
        res["review_reason"] = (
            f"possible {gap['display_name']} (embedding similarity {gap['similarity']}) "
            f"- Discogs taxonomy has no {gap['display_name']} label, model said "
            f"'{res['category']}' instead; verify by ear")

    # The model over-calls "Bassline" on commercial tech house, confidently
    # enough that the confidence threshold above never catches it - see
    # GenreModel.garage_conflict_signal().
    conflict = res.get("garage_conflict_signal")
    if conflict:
        res["needs_review"] = True
        reason = (
            f"'{conflict['predicted']}' looks wrong - this sounds closer to tech "
            f"house ({conflict['tech_house_similarity']}) than to garage "
            f"({conflict['garage_similarity']}); verify by ear")
        res["review_reason"] = (f"{res['review_reason']}; {reason}"
                                if res.get("review_reason") else reason)

    if voc["needs_review"]:
        res["needs_review"] = True
        res["review_reason"] = "possible acapella, check manually"
    if not res.get("bpm_reliable", True):
        res["needs_review"] = True
        res["review_reason"] = (res.get("review_reason", "") +
                                "; low tempo confidence").strip("; ")

    res["analysis_sec"] = round(time.time() - t0, 1)
    if verbose:
        flag = "  REVIEW" if res.get("needs_review") else ""
        print(f"  {res.get('bpm', 0):6.1f} bpm  {res.get('camelot', '--'):>3}  "
              f"{len(moments):2d} cues  {res['category'][:26]:26}{flag}")
    return res


# ------------------------------------------------------------------- exports

# Colours Rekordbox itself uses, read out of a real exported collection, so
# imported cues look native rather than arbitrary. Keyed to moment type and
# matched to the viewer's palette: drops orange, breakdowns blue, builds amber.
CUE_COLOURS = {
    "drop":         (224, 100, 27),
    "breakdown":    (69, 172, 219),
    "buildup":      (255, 140, 0),
    "main section": (48, 210, 110),
    "intro":        (48, 210, 110),
    "outro":        (48, 210, 110),
}
DEFAULT_CUE_COLOUR = (48, 210, 110)


def grid_start(moments, bpm):
    """Where the beat grid's first downbeat sits, in seconds, for Rekordbox's
    TEMPO Inizio - or None when there is nothing to anchor it to.

    Writing Inizio="0.000" told Rekordbox that beat 1 falls exactly at the
    start of the file, which is almost never true: the grid came out shifted
    by up to half a beat, so quantize and beat sync snapped to the wrong
    place. Every moment is snapped to one of the analyser's downbeats, so the
    earliest one fixes the phase; stepping back whole bars from it gives the
    first downbeat in the track. With no moment the TEMPO element is left
    out and Rekordbox analyses the grid itself.
    """
    times = [float(m["time"]) for m in (moments or []) if m.get("time") is not None]
    if not bpm or bpm <= 0 or not times:
        return None
    t0 = min(times)
    bar = 4 * 60.0 / float(bpm)
    return round(t0 % bar, 3)


def tempo_element(moments, bpm, indent="      "):
    """The TEMPO line for a track, or "" to let Rekordbox build the grid."""
    start = grid_start(moments, bpm)
    if start is None:
        return ""
    return (f'{indent}<TEMPO Inizio="{start:.3f}" Bpm="{bpm}" '
            f'Metro="4/4" Battito="1"/>')


def write_rekordbox_xml(tracks, out):
    """
    rekordbox.xml with genre, tempo, key and a cue at every detected moment.

    Each moment is written twice, on purpose:

      * as a memory cue (Num="-1") - unlimited in number, so nothing is lost
        on a track with more than eight moments;
      * as a hot cue (Num="0".."7") with an RGB colour - because that is what
        Rekordbox actually shows on the waveform. Every one of the 41 cues in
        this DJ's own exported collection was a hot cue with a colour; memory
        cues alone did not display, which is what made the first import look
        empty.

    Cue times land on downbeats, so they stay usable on a CDJ.
    """
    from xml.sax.saxutils import escape
    from urllib.parse import quote

    def esc(s):
        # escape() alone only handles &, < and > - not the double quote,
        # so a title containing '"' would otherwise produce a malformed
        # attribute value that Rekordbox rejects outright.
        return escape(s or "", {'"': "&quot;"})

    L = ['<?xml version="1.0" encoding="UTF-8"?>', '<DJ_PLAYLISTS Version="1.0.0">',
         '  <PRODUCT Name="Crate" Version="1.0" Company="local"/>',
         f'  <COLLECTION Entries="{len(tracks)}">']

    emitted = []
    for i, t in enumerate(tracks, 1):
        if t.get("errors") and "bpm" not in t:
            continue
        emitted.append(i)
        loc = "file://localhost" + quote(t["path"])
        name = esc(Path(t["file"]).stem)
        L.append(
            f'    <TRACK TrackID="{i}" Name="{name}" Kind="Audio File" '
            f'Location="{loc}" AverageBpm="{t.get("bpm", 0)}" '
            f'Tonality="{esc(t.get("camelot", ""))}" '
            f'Genre="{esc(t.get("category", ""))}" '
            f'TotalTime="{int(t.get("duration_sec", 0))}" '
            f'Comments="{esc(t.get("category_basis", ""))}">')
        tempo = tempo_element(t.get("moments", []), t.get("bpm"))
        if tempo:
            L.append(tempo)
        for n, m in enumerate(t.get("moments", [])):
            label = esc(m["label"])
            start = f'{m["time"]:.3f}'
            L.append(f'      <POSITION_MARK Name="{label}" Type="0" '
                     f'Start="{start}" Num="-1"/>')
            if n < 8:   # A-H, the only hot cue slots Rekordbox has
                r, g, b = CUE_COLOURS.get(m.get("type", ""), DEFAULT_CUE_COLOUR)
                L.append(f'      <POSITION_MARK Name="{label}" Type="0" '
                         f'Start="{start}" Num="{n}" '
                         f'Red="{r}" Green="{g}" Blue="{b}"/>')
        L.append('    </TRACK>')

    # Rekordbox's xml panel browses the PLAYLIST tree, not the collection. An
    # empty ROOT node therefore renders as nothing to expand, which is exactly
    # what an import of this file used to look like - the tracks were all
    # present in COLLECTION, but with no playlist there was no way to reach
    # them. Real exports nest Type="1" playlist nodes that reference tracks by
    # TrackID, so emit one named after the output folder.
    playlist = esc(Path(out).resolve().parent.name or "Crate")
    L += ['  </COLLECTION>', '  <PLAYLISTS>',
          '    <NODE Type="0" Name="ROOT" Count="1">',
          f'      <NODE Name="{playlist}" Type="1" KeyType="0" '
          f'Entries="{len(emitted)}">']
    L += [f'        <TRACK Key="{i}"/>' for i in emitted]
    L += ['      </NODE>', '    </NODE>',
          '  </PLAYLISTS>', '</DJ_PLAYLISTS>']
    Path(out).write_text("\n".join(L), encoding="utf-8")


def write_csv(tracks, out):
    cols = ["file", "category", "category_confidence", "bpm", "bpm_confidence",
            "key", "camelot", "duration", "loudness_lufs", "danceability",
            "onset_rate", "brightness_hz", "drop_count", "breakdown_count",
            "is_acapella", "needs_review", "review_reason", "analysis_sec"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in tracks:
            r = dict(t)
            r["drop_count"] = t.get("structure", {}).get("drop_count", 0)
            r["breakdown_count"] = t.get("structure", {}).get("breakdown_count", 0)
            r["is_acapella"] = t.get("vocal_analysis", {}).get("is_acapella", False)
            w.writerow(r)


def write_viewer(tracks, out, template):
    html = Path(template).read_text(encoding="utf-8")
    payload = json.dumps(tracks, ensure_ascii=False)
    Path(out).write_text(html.replace("/*__CRATE_DATA__*/[]", payload),
                         encoding="utf-8")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Analyse a folder of audio for tempo, key, genre and structure.")
    ap.add_argument("folder", help="folder of audio files (searched recursively)")
    ap.add_argument("-o", "--out", default="crate_output", help="output folder")
    ap.add_argument("--models", default=str(MODEL_DIR), help="model folder")
    ap.add_argument("--limit", type=int, help="stop after N files (for testing)")
    ap.add_argument("--resume", action="store_true",
                    help="skip files already in results.json")
    args = ap.parse_args()

    src = Path(args.folder).expanduser()
    if not src.is_dir():
        sys.exit(f"Not a folder: {src}")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    emb_dir = out / "embeddings"
    emb_dir.mkdir(exist_ok=True)

    files = sorted(p for p in src.rglob("*")
                   if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("._"))

    done = {}
    if args.resume and results_path.exists():
        done = {t["path"]: t for t in json.loads(results_path.read_text())}
        files = [f for f in files if str(f.resolve()) not in done]
        print(f"Resuming - {len(done)} already analysed.")

    if args.limit:
        files = files[:args.limit]
    if not files and not done:
        sys.exit(f"No audio files found in {src}")

    gm = GenreModel(args.models)
    if not gm.ok:
        print("\n  Genre model not found. Tempo, key, structure and acapella")
        print("  detection will still run. For genre, run: ./setup.sh\n")

    print(f"Analysing {len(files)} files -> {out}\n")
    tracks = list(done.values())
    t0 = time.time()

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name[:52]}")
        try:
            tracks.append(analyse(f, gm, emb_dir=emb_dir))
        except KeyboardInterrupt:
            print("\nInterrupted - saving what is done so far.")
            break
        except Exception as e:
            print(f"  ! failed: {e}")
            tracks.append({"file": f.name, "path": str(f.resolve()),
                           "errors": [str(e)], "category": "Failed",
                           "needs_review": True})
        if i % 25 == 0:
            results_path.write_text(json.dumps(tracks, indent=1))

    results_path.write_text(json.dumps(tracks, indent=1))
    write_csv(tracks, out / "library.csv")
    write_rekordbox_xml(tracks, out / "rekordbox.xml")

    tpl = Path(__file__).parent / "viewer_template.html"
    if tpl.exists():
        write_viewer(tracks, out / "viewer.html", tpl)

    ok = [t for t in tracks if not t.get("errors")]
    rev = [t for t in tracks if t.get("needs_review")]
    aca = [t for t in tracks if t.get("vocal_analysis", {}).get("is_acapella")]
    cues = sum(len(t.get("moments", [])) for t in tracks)

    print(f"\n{'-' * 58}")
    print(f"  {len(ok)}/{len(tracks)} analysed in {(time.time() - t0) / 60:.1f} min")
    print(f"  {cues} cue points found")
    print(f"  {len(aca)} acapellas routed to Vocals")
    print(f"  {len(rev)} flagged for review")
    print(f"\n  Open {out / 'viewer.html'} to browse")
    print(f"  Import {out / 'rekordbox.xml'} into Rekordbox")


if __name__ == "__main__":
    main()
