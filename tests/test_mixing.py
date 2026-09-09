import numpy as np

from crateapp.mixing import (energy_score, harmonic_score, mix_points,
                             parse_camelot, score_transition, tempo_score,
                             timbral_score, vocal_score)


# ---------------------------------------------------------------- camelot

def test_parse_camelot():
    assert parse_camelot("8A") == (8, "A")
    assert parse_camelot("12B") == (12, "B")
    assert parse_camelot("") is None
    assert parse_camelot("13A") is None      # the wheel only goes to 12
    assert parse_camelot("8C") is None


# ------------------------------------------------------------------ tempo

def test_identical_tempo_is_perfect():
    s, stretch, _ = tempo_score(128.0, 128.0)
    assert s == 1.0 and abs(stretch) < 1e-9


def test_a_few_bpm_apart_is_still_perfect():
    """Within about 3 BPM there is no real beatmatching work."""
    assert tempo_score(128.0, 130.0)[0] == 1.0


def test_beyond_the_pitch_range_scores_zero():
    """+/-6% is where time-stretch artefacts become audible."""
    s, _, note = tempo_score(128.0, 145.0)
    assert s == 0.0 and "beyond" in note


def test_a_140_and_a_70_track_share_a_pulse():
    """A 140 dubstep track and a 70 hip hop track are mixable. The label
    describes what happens to the INCOMING track: coming out of 140 into a
    70, the incoming record is played at double time."""
    s, _, note = tempo_score(140.0, 70.0)
    assert s == 1.0 and "double time" in note


def test_the_same_pair_the_other_way_round():
    """Out of 70 into 140, the incoming record is played at half time."""
    s, _, note = tempo_score(70.0, 140.0)
    assert s == 1.0 and "half time" in note


def test_unknown_tempo_is_not_guessed():
    assert tempo_score(None, 128.0)[0] == 0.0


# --------------------------------------------------------------- harmonic

def test_same_key_is_perfect():
    assert harmonic_score("8A", "8A")[0] == 1.0


def test_relative_major_minor_is_near_perfect():
    s, note = harmonic_score("8A", "8B")
    assert s > 0.9 and "relative" in note


def test_one_step_on_the_wheel_is_strong():
    assert harmonic_score("8A", "9A")[0] == 0.9
    assert harmonic_score("8A", "7A")[0] == 0.9


def test_the_wheel_wraps():
    """12A and 1A are neighbours, not eleven steps apart."""
    assert harmonic_score("12A", "1A")[0] == 0.9


def test_distant_keys_clash():
    assert harmonic_score("8A", "2A")[0] < 0.2


def test_a_big_tempo_stretch_breaks_a_paper_key_match():
    """Two tracks 'in key' on paper clash once one is pulled several percent -
    6% is about a semitone, which is two positions on the wheel."""
    on_paper = harmonic_score("8A", "8A", stretch=0.0)[0]
    stretched = harmonic_score("8A", "8A", stretch=0.06)[0]
    assert on_paper == 1.0
    assert stretched < on_paper


# --------------------------------------------------------------- timbral

def test_identical_embeddings_score_top():
    v = np.random.rand(64).astype("float32")
    assert timbral_score(v, v)[0] == 1.0


def test_orthogonal_embeddings_score_bottom():
    a = np.zeros(8, dtype="float32"); a[0] = 1
    b = np.zeros(8, dtype="float32"); b[1] = 1
    assert timbral_score(a, b)[0] == 0.0


def test_missing_embedding_is_not_guessed():
    assert timbral_score(None, np.ones(4, dtype="float32"))[0] == 0.0


# ---------------------------------------------------------------- vocals

def test_two_vocal_tracks_are_penalised():
    s, note = vocal_score(0.8, 0.8)
    assert s < 0.4 and "clash" in note


def test_instrumental_into_vocal_is_fine():
    assert vocal_score(0.2, 0.8)[0] == 1.0


# ---------------------------------------------------------------- energy

def test_build_rewards_a_lift():
    up = energy_score(0.4, 0.6, want="build")[0]
    down = energy_score(0.6, 0.4, want="build")[0]
    assert up > down


def test_cool_down_rewards_a_drop():
    assert energy_score(0.6, 0.4, want="cool")[0] > energy_score(0.4, 0.6, want="cool")[0]


def test_steady_rewards_holding_the_level():
    assert energy_score(0.5, 0.5)[0] > energy_score(0.5, 0.9)[0]


# ------------------------------------------------------------ transitions

def _t(bpm, cam, vec, voc=0.2, en=0.5):
    return {"bpm": bpm, "camelot": cam, "vec": vec, "vocal_ratio": voc, "energy": en}


def test_a_perfect_transition_scores_high():
    v = np.random.rand(64).astype("float32")
    r = score_transition(_t(128, "8A", v), _t(128, "8A", v))
    assert r["score"] > 0.9
    assert any("same key" in x for x in r["reasons"])


def test_an_unmixable_tempo_is_heavily_penalised():
    """Whatever else it has going for it, nobody can beatmatch it."""
    v = np.random.rand(64).astype("float32")
    good = score_transition(_t(128, "8A", v), _t(128, "8A", v))
    bad = score_transition(_t(128, "8A", v), _t(175, "8A", v))
    assert bad["score"] < good["score"] * 0.4


def test_every_score_carries_its_reasons():
    """A score with no explanation is not trustworthy enough to act on."""
    v = np.random.rand(64).astype("float32")
    r = score_transition(_t(128, "8A", v), _t(130, "9A", v))
    assert r["reasons"] and all(isinstance(x, str) for x in r["reasons"])


def test_adventurous_weights_sound_over_key():
    """Same pair, different appetite: adventurous leans on how it sounds."""
    a = np.zeros(8, dtype="float32"); a[0] = 1
    close = a.copy()
    safe = score_transition(_t(128, "8A", a), _t(128, "2A", close), mode="safe")
    adv = score_transition(_t(128, "8A", a), _t(128, "2A", close), mode="adventurous")
    assert adv["score"] > safe["score"]


# --------------------------------------------------------------- mix points

def test_mix_points_give_a_point_and_a_window():
    a = [{"type": "drop", "time": 60.0}, {"type": "breakdown", "time": 240.0}]
    b = [{"type": "drop", "time": 64.0}]
    p = mix_points(a, 300.0, b)
    assert p["out_at"] == 240.0
    assert p["out_window"] == [240.0, 300.0]
    assert p["in_window"][1] == 64.0
    assert p["in_at"] <= 64.0


def test_mix_points_cope_with_no_structure():
    """A track with nothing detected still needs a usable answer."""
    p = mix_points([], 300.0, [])
    assert p["out_at"] is not None and p["in_at"] is not None


def test_the_lead_in_is_bars_not_seconds():
    """16 bars is 30s at 128 BPM and 22s at 175. A DJ counts the phrase, so
    the lead-in has to follow tempo rather than sit at a fixed number."""
    b = [{"type": "drop", "time": 120.0}]
    slow = mix_points([], 300.0, b, bpm_a=128.0)["in_at"]
    fast = mix_points([], 300.0, b, bpm_a=175.0)["in_at"]
    assert slow == round(120.0 - 30.0, 2)
    assert fast > slow          # less real time in 16 bars, so come in later


def test_a_track_that_drops_immediately_says_so():
    """Plenty of edits drop in the first bar. That is not a narrow window to
    aim at, it is no window, and the answer should say that."""
    p = mix_points([], 300.0, [{"type": "drop", "time": 0.4}], bpm_a=128.0)
    assert "no intro" in p["in_note"]
    assert p["in_at"] == 0.0


def test_a_short_intro_is_flagged_with_its_length():
    p = mix_points([], 300.0, [{"type": "drop", "time": 16.0}], bpm_a=128.0)
    assert "short intro" in p["in_note"] and "16s" in p["in_note"]


def test_a_main_section_at_bar_zero_is_not_the_mix_in_target():
    """Nearly every analysed track carries a 'main section' marker in its
    first bar - it means 'the track starts', not 'land here'. Taking it as
    the target made every mix-in read 'no intro' while the real drop sat
    forty bars later. Measured on the library: 88% of tracks looked
    intro-less because of this."""
    b = [{"type": "main section", "bar": 0, "time": 0.4},
         {"type": "drop", "bar": 40, "time": 73.7}]
    p = mix_points([], 300.0, b, bpm_a=132.0)
    assert p["in_window"][1] == 73.7
    assert p["in_at"] > 0.0
    assert p["in_note"] == ""


def test_a_late_main_section_still_works_when_there_is_no_drop():
    b = [{"type": "main section", "bar": 16, "time": 30.0}]
    assert mix_points([], 300.0, b, bpm_a=128.0)["in_window"][1] == 30.0


def test_a_roomy_intro_needs_no_warning():
    p = mix_points([], 300.0, [{"type": "drop", "time": 90.0}], bpm_a=128.0)
    assert p["in_note"] == ""
