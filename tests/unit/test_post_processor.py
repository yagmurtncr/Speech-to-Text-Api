from post_processor import merge_segments


def test_empty_returns_empty():
    assert merge_segments([]) == []


def test_single_speaker_merges_in_time_order():
    segs = [
        {"start": 1.0, "end": 2.0, "text": "world", "speaker": "s1", "lang": "tr"},
        {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "s1", "lang": "tr"},
    ]
    out = merge_segments(segs)
    assert len(out) == 1
    assert out[0]["text"] == "hello world"
    assert out[0]["start"] == 0.0 and out[0]["end"] == 2.0


def test_multiple_speakers_split_per_speaker():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "s1", "lang": "tr"},
        {"start": 1.0, "end": 2.0, "text": "yo", "speaker": "s2", "lang": "tr"},
    ]
    out = merge_segments(segs)
    assert len(out) == 2
    by_spk = {o["speaker"]: o["text"] for o in out}
    assert by_spk["s1"] == "hi"
    assert by_spk["s2"] == "yo"
    assert out[0]["start"] <= out[1]["start"]
