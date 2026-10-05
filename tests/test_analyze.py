"""Guards on analyze.py's decision rules.

These assert on the source rather than by running the pipeline: the rules
here are cheap one-line thresholds whose *value* is the whole point, and
running them for real needs TensorFlow, model weights and a decoded audio
file. A threshold that silently drifts is exactly the failure this catches.
"""
from pathlib import Path




# ---------------------------------------------- the voice model's proper role

def test_the_voice_model_cannot_promote_a_weak_acapella_score():
    """It answers "is there singing on this", not "is this an isolated
    vocal" - so every sung pop record scores above 0.85 on it.

    Allowing it to decide at a 1-of-3 DSP score filed 28 full mixes as
    acapellas (Mr. Brightside, Viva La Vida, A Thousand Miles), AND cleared
    their review flag so nobody ever saw them. A single DSP hint plus audible
    singing is a track to look at, not a verdict.
    """
    import re
    src = (Path(__file__).resolve().parent.parent / "analyze.py").read_text()
    m = re.search(r'voice\["voice"\] > 0\.85 and voc\["score"\] >= ([\d.]+)', src)
    assert m, "the voice-model guard has moved or gone"
    assert float(m.group(1)) >= 0.66, (
        "the voice model must only confirm a 2-of-3 DSP verdict, never create one")
    # And it must not be able to clear the review flag.
    guard = src[m.start():m.start() + 400]
    assert 'voc["needs_review"] = False' not in guard
    assert 'voc["is_acapella"] = True' not in guard


# ------------------------------------------------- inputs with nothing in them

def _wav(path, samples, sr=44100):
    import wave

    import numpy as np
    with wave.open(str(path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def test_silent_audio_is_flagged_not_given_a_tempo(tmp_path):
    """A silent 20s file came back as 738 BPM, key 8A, with eight phrases,
    and nothing flagged it."""
    import numpy as np
    import pytest
    analyze = pytest.importorskip("analyze")
    f = tmp_path / "blank.wav"
    _wav(f, np.zeros(44100 * 20))
    r = analyze.analyse(f, None, verbose=False)
    assert r.get("bpm") is None and r.get("camelot") is None and not r.get("moments")
    assert any("silent" in e for e in r["errors"])


def test_a_real_beat_still_analyses(tmp_path):
    import numpy as np
    import pytest
    analyze = pytest.importorskip("analyze")
    sr, secs = 44100, 30
    sig = np.zeros(sr * secs)
    for b in np.arange(0, secs, 60 / 128):
        i = int(b * sr); k = np.arange(min(int(0.12 * sr), len(sig) - i))
        sig[i:i + len(k)] += np.sin(2 * np.pi * 55 * k / sr) * np.exp(-k / (0.05 * sr))
    f = tmp_path / "beat.wav"
    _wav(f, sig / np.abs(sig).max() * 20000)
    r = analyze.analyse(f, None, verbose=False)
    assert r["errors"] == [] and abs(r["bpm"] - 128) < 1


def test_an_impossible_tempo_is_left_blank_but_the_track_is_still_usable(tmp_path, monkeypatch):
    """Over 250 or under 40 BPM means no steady beat was found. It must not
    be an error (an acapella with no beat must still reach the vocals
    rule), and it must not be passed on as a number."""
    import numpy as np
    import pytest
    analyze = pytest.importorskip("analyze")

    class FakeRhythm:
        def __init__(self, **kw):
            pass

        def __call__(self, audio):
            return 738.3, np.array([0.1, 0.2]), 4.0, None, None
    monkeypatch.setattr(analyze.es, "RhythmExtractor2013", FakeRhythm)
    sr = 44100
    f = tmp_path / "voice.wav"
    _wav(f, 8000 * np.sin(2 * np.pi * 220 * np.arange(sr * 20) / sr))
    r = analyze.analyse(f, None, verbose=False)
    assert r["bpm"] is None and r["bpm_reliable"] is False and "no steady beat" in r["bpm_note"]
    assert r["errors"] == []
    xml = tmp_path / "x.xml"
    analyze.write_rekordbox_xml([dict(r, category="vocals")], xml)
    assert 'AverageBpm="0"' in xml.read_text() and "None" not in xml.read_text()
