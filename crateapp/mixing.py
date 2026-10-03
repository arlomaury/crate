"""How well two tracks mix, and where to mix them.

Five signals. Four are standard DJ practice; the fifth is the one commercial
tools cannot do.

  tempo     +/-6% is the practical pitch range on a CDJ; +/-3 BPM is
            comfortable. Half and double time count (a 140 mixes with a 70).
  harmonic  Camelot: same code, +/-1 on the wheel, or the relative
            major/minor. +2 is an energy lift.
  timbral   cosine distance between the tracks' embeddings - do these two
            records actually inhabit the same world. Mixed In Key, rekordbox
            and Beatport match on key and tempo only, so they cannot tell
            that two 8A/128 records sound wrong together. This can.
  vocals    two vocal-dominant tracks fight each other over a transition.
  energy    so a set moves deliberately rather than flatlining.

Tempo and key are NOT independent. Pushing tempo about 6% shifts pitch by a
semitone - seven positions on the Camelot wheel, which is a circle of fifths -
so a large stretch silently breaks a harmonic match unless key lock is on. `harmonic_score`
therefore takes the tempo change into account rather than scoring the printed
keys in isolation.
"""
import json
import math

# Practical limits, from DJ practice rather than taste.
PITCH_LIMIT = 0.06        # +/-6% before time-stretch artefacts get audible
COMFORTABLE_BPM = 3.0     # within this, no real work to beatmatch
SEMITONE_PCT = 0.0595     # 6% tempo ~ one semitone ~ seven Camelot positions


def parse_camelot(code):
    """'8A' -> (8, 'A'). Returns None for anything unparseable."""
    if not code or len(code) < 2:
        return None
    num, letter = code[:-1], code[-1].upper()
    if letter not in ("A", "B") or not num.isdigit():
        return None
    n = int(num)
    return (n, letter) if 1 <= n <= 12 else None


def tempo_score(bpm_a, bpm_b):
    """(score 0-1, stretch as a signed fraction, note).

    Also considers half and double time, because a 140 dubstep track and a 70
    hip hop track share a pulse even though the printed numbers do not.
    """
    if not bpm_a or not bpm_b:
        return 0.0, 0.0, "unknown tempo"

    best = None
    for factor, label in ((1.0, ""), (2.0, " (double time)"), (0.5, " (half time)")):
        target = bpm_b * factor
        stretch = (target - bpm_a) / bpm_a
        if best is None or abs(stretch) < abs(best[0]):
            best = (stretch, label)
    stretch, label = best
    delta = abs(stretch)

    if delta > PITCH_LIMIT:
        return 0.0, stretch, f"{delta*100:.1f}% tempo change - beyond the pitch range"
    if abs(bpm_a - bpm_b * (1.0 if not label else (2.0 if "double" in label else 0.5))) <= COMFORTABLE_BPM:
        return 1.0, stretch, f"tempo matches{label}"
    # linear falloff from comfortable to the 6% ceiling
    score = 1.0 - (delta / PITCH_LIMIT) * 0.65
    return max(0.0, score), stretch, f"{delta*100:.1f}% tempo change{label}"


def harmonic_score(cam_a, cam_b, stretch=0.0):
    """(score 0-1, note). `stretch` is the tempo change from tempo_score.

    A tempo stretch moves the outgoing track's key. Ignoring that is the
    classic mistake: two tracks that are 'in key' on paper clash once one has
    been pulled 5% to match the other.
    """
    a, b = parse_camelot(cam_a), parse_camelot(cam_b)
    if not a or not b:
        return 0.0, "unknown key"

    # A stretch shifts the played key. The Camelot wheel is a circle of
    # fifths, so one semitone up is SEVEN places round it (8A, A minor, up a
    # semitone is B-flat minor, 3A) and one down is five. It is not two:
    # two places is a whole tone along the circle, i.e. two fifths.
    semitones = int(round(stretch / SEMITONE_PCT))
    shift = (7 * semitones) % 12
    num_a = ((a[0] - 1 + shift) % 12) + 1
    letter_a = a[1]
    num_b, letter_b = b

    same_letter = letter_a == letter_b
    dist = min((num_a - num_b) % 12, (num_b - num_a) % 12)

    if same_letter and dist == 0:
        note = "same key" if not shift else "same key once pitched"
        return 1.0, note
    if not same_letter and dist == 0:
        return 0.95, "relative major/minor"
    if same_letter and dist == 1:
        return 0.9, "one step on the wheel"
    if same_letter and dist == 2:
        return 0.62, "energy lift (+2)"
    if not same_letter and dist == 1:
        return 0.45, "loosely related"
    return 0.1, "keys clash"


def timbral_score(vec_a, vec_b):
    """Cosine similarity of the two embeddings, rescaled to 0-1.

    In practice tracks sit between about 0.5 and 0.95 against each other, so
    the raw cosine is stretched across that band - otherwise every pair looks
    similar and the signal does no work.
    """
    if vec_a is None or vec_b is None:
        return 0.0, "unknown sound"
    import numpy as np
    a = np.asarray(vec_a, dtype="float32"); b = np.asarray(vec_b, dtype="float32")
    cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
    score = max(0.0, min(1.0, (cos - 0.5) / 0.45))
    if score > 0.8:
        note = "sounds very close"
    elif score > 0.55:
        note = "sounds compatible"
    elif score > 0.3:
        note = "different feel"
    else:
        note = "sounds unrelated"
    return score, note


def vocal_score(voc_a, voc_b):
    """Two vocal-dominant tracks fight over a transition. Penalise that only;
    an instrumental into anything is fine."""
    if voc_a is None or voc_b is None:
        return 1.0, ""
    if voc_a > 0.5 and voc_b > 0.5:
        return 0.25, "both vocal-led - they will clash"
    if voc_a > 0.5 and voc_b > 0.42:
        return 0.6, "vocals may overlap"
    return 1.0, ""


def energy_score(a, b, want="steady"):
    """Reward the direction the DJ asked for: steady, build, or cool down."""
    if a is None or b is None:
        return 0.7, ""
    delta = b - a
    if want == "build":
        return (min(1.0, 0.6 + delta * 4), "lifts") if delta > 0 else (0.35, "drops when you wanted a lift")
    if want == "cool":
        return (min(1.0, 0.6 - delta * 4), "eases off") if delta < 0 else (0.35, "lifts when you wanted a cool down")
    return (1.0 - min(1.0, abs(delta) * 3), "holds the level")


# The slider: safe favours what cannot fail, adventurous tolerates more risk
# for more interesting pairings.
WEIGHTS = {
    "safe":         {"tempo": 0.34, "harmonic": 0.30, "timbral": 0.24, "vocal": 0.07, "energy": 0.05},
    "balanced":     {"tempo": 0.28, "harmonic": 0.24, "timbral": 0.30, "vocal": 0.08, "energy": 0.10},
    "adventurous":  {"tempo": 0.20, "harmonic": 0.16, "timbral": 0.36, "vocal": 0.10, "energy": 0.18},
}


def score_transition(a, b, mode="balanced", want="steady"):
    """Score mixing OUT of track `a` and INTO track `b`.

    `a` and `b` are dicts with bpm, camelot, vec, vocal_ratio, energy.
    Returns a dict with the overall score and every reason behind it - a score
    with no explanation is not trustworthy enough to act on.
    """
    w = WEIGHTS.get(mode, WEIGHTS["balanced"])
    t, stretch, t_note = tempo_score(a.get("bpm"), b.get("bpm"))
    h, h_note = harmonic_score(a.get("camelot"), b.get("camelot"), stretch)
    s, s_note = timbral_score(a.get("vec"), b.get("vec"))
    v, v_note = vocal_score(a.get("vocal_ratio"), b.get("vocal_ratio"))
    e, e_note = energy_score(a.get("energy"), b.get("energy"), want)

    total = (t * w["tempo"] + h * w["harmonic"] + s * w["timbral"]
             + v * w["vocal"] + e * w["energy"])

    # A transition nobody can beatmatch is not a transition, whatever else it
    # has going for it.
    if t == 0.0:
        total *= 0.25

    reasons = [n for n in (t_note, h_note, s_note, v_note, e_note) if n]
    return {
        "score": round(total, 4),
        "tempo": round(t, 3), "harmonic": round(h, 3), "timbral": round(s, 3),
        "vocal": round(v, 3), "energy": round(e, 3),
        "stretch_pct": round(stretch * 100, 2),
        "reasons": reasons,
    }


def bars_to_seconds(bars, bpm):
    """Bars are the unit that matters; seconds are what a player displays."""
    if not bpm:
        return None
    return bars * 4 * 60.0 / bpm


def mix_points(moments_a, dur_a, moments_b, bpm_a=None):
    """Where to mix out of A and into B: an exact point AND the window it sits in.

    A single point is too rigid - a DJ rides a transition by feel and rarely
    hits an exact second. A range alone is too vague to act on. Both together
    say "start here, and you have this much room".

    Windows come from real structure: the outro window runs from the last
    energy fall to the end, the intro window from the start to the first drop.

    The lead-in is 16 bars, not a fixed number of seconds - at 128 BPM that is
    30s and at 175 it is 22s, and the phrase is what the DJ is counting. It is
    measured at A's tempo because the incoming record is pitched to match the
    deck already playing.

    Plenty of edits and bootlegs drop within the first bar. That is not a
    narrow window to aim at, it is no window at all, and saying so is more
    use than returning a range of a third of a second.
    """
    def last_of(ms, kinds):
        hits = [m for m in ms if m.get("type") in kinds]
        return hits[-1] if hits else None

    def first_of(ms, kinds):
        for m in ms:
            if m.get("type") in kinds:
                return m
        return None

    lead_in = bars_to_seconds(16, bpm_a) or 32.0
    min_room = bars_to_seconds(4, bpm_a) or 8.0

    out_from, out_to, out_at = None, dur_a, None
    tail = last_of(moments_a or [], {"outro", "breakdown"})
    if tail:
        out_from = tail["time"]
        out_at = tail["time"]
    elif dur_a:
        out_from = max(0.0, dur_a * 0.75)
        out_at = out_from

    in_from, in_to, in_at, in_note = 0.0, None, None, ""
    # The target is the first DROP. A "main section" marker sitting at bar 0
    # only means "the track starts" - almost every track has one, and treating
    # it as the landing point makes every mix-in look like it has no intro
    # while the real drop sits 16 or 40 bars later.
    head = first_of(moments_b or [], {"drop"})
    if head is None:
        head = next((m for m in (moments_b or [])
                     if m.get("type") == "main section" and (m.get("bar") or 0) >= 4),
                    None)
    if head:
        in_to = head["time"]
        in_at = max(0.0, head["time"] - lead_in)
        if in_to < min_room:
            in_note = "no intro - drop it on the one"
        elif in_to < lead_in:
            in_note = f"short intro - only {in_to:.0f}s before the drop"
    else:
        in_to = lead_in
        in_at = 0.0
        in_note = "no drop detected - mix in from the top"

    return {
        "out_at": None if out_at is None else round(out_at, 2),
        "out_window": [None if out_from is None else round(out_from, 2),
                       None if out_to is None else round(out_to, 2)],
        "in_at": None if in_at is None else round(in_at, 2),
        "in_window": [round(in_from, 2), None if in_to is None else round(in_to, 2)],
        "in_note": in_note,
        "lead_in_bars": 16,
    }
