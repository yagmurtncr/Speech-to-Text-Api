# test_diarize_alone.py
import os

from whisperx.diarize import DiarizationPipeline

wav = r"voices\small_talk.wav"
tok = os.getenv("HUGGINGFACE_TOKEN")
pipe = DiarizationPipeline(use_auth_token=tok, device=os.getenv("WHISPER_DEVICE","cpu"))

# önce dar aralık
res = pipe(wav, min_speakers=int(os.getenv("MIN_SPEAKERS","2")),
                max_speakers=int(os.getenv("MAX_SPEAKERS","2")))
print("Deneme-1 segment:", len(res.get("segments", [])))

# hala 0 ise geniş aralık
if len(res.get("segments", [])) == 0:
    res = pipe(wav, min_speakers=1, max_speakers=5)
    print("Deneme-2 segment:", len(res.get("segments", [])))

# örnek ilk 5 segmenti yaz
for s in res.get("segments", [])[:5]:
    print(round(float(s["start"]),2), "->", round(float(s["end"]),2), s["speaker"])
