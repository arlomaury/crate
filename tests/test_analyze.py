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
