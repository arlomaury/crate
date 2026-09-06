"""Regression test for the XML-attribute quote-escaping bug in
analyze.write_rekordbox_xml - xml.sax.saxutils.escape() alone does not
escape a literal double quote, which produces malformed XML inside a
double-quoted attribute value and makes Rekordbox reject the import
outright."""
import xml.etree.ElementTree as ET

from analyze import write_rekordbox_xml


def test_rekordbox_xml_escapes_quotes_and_ampersands(tmp_path):
    tracks = [{
        "path": str(tmp_path / 'Say "Hi" & Bye.wav'),
        "file": 'Say "Hi" & Bye.wav',
        "bpm": 128.0,
        "camelot": "8A",
        "category": 'Rock & "Roll"',
        "duration_sec": 200,
        "category_basis": "test",
        "moments": [{"label": 'Drop "1" & 2', "time": 1.0, "type": "drop"}],
    }]
    out = tmp_path / "rekordbox.xml"
    write_rekordbox_xml(tracks, out)

    # The real bug: this raises xml.etree.ElementTree.ParseError if the
    # quote isn't escaped, because the attribute value closes early.
    root = ET.parse(out).getroot()

    track = root.find("COLLECTION/TRACK")
    assert track.get("Genre") == 'Rock & "Roll"'
    assert track.get("Name") == 'Say "Hi" & Bye'

    marks = root.findall("COLLECTION/TRACK/POSITION_MARK")
    assert any(m.get("Name") == 'Drop "1" & 2' for m in marks)
