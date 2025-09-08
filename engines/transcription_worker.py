# ============================================================ 
#  ANA GÖREVLER:
#   1️ Motor Seçimi: nvidia | whisperx | whisperx_nemo
#   2️ Parametre Yönlendirme: Her motora doğru parametreleri ilet  
#   3️ Ortak İşlemler: WAV dönüştürme, duygu analizi, özetleme
#   4️ Sonuç Standardizasyonu: Tüm motorların çıktısını aynı formata getir
#   5️ Güvenlik: Hata durumunda sistem çökmesin
#
#  MOTORLAR :
#    nvidia: Pure NVIDIA (Parakeet ASR + Sortformer Diarization)
#    whisperx_nemo: Hibrit (WhisperX ASR + NVIDIA Diarization)  
#    whisperx: Professional (WhisperX ASR + WhisperX Diarization)
# ============================================================

# [1] Standart kütüphaneler
import sys, os, json, re, time, glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from contextlib import suppress

# [2] Proje kökü (engines.* importları için)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# [3] Ortam/performans
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("MKL_NUM_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.getenv("NUMEXPR_NUM_THREADS", "1"))
os.environ.setdefault("KMP_AFFINITY", "disabled")

# [4] NumPy ad güvenliği
import numpy as _np
if not hasattr(_np, "NaN"): _np.NaN = _np.nan
if not hasattr(_np, "Inf"): _np.Inf = _np.inf

# [5] Dış bağımlılıklar
import ffmpeg
try:
    from engines.nvidia_parakeet import transcribe_nvidia_parakeet as _run_parakeet
except Exception:
    _run_parakeet = None

try:
    import langid
except Exception:
    langid = None

try:
    # wrapper: WhisperX ASR + NVIDIA diar
    from engines.whisperx_whit_nemo import transcribe_large as _run_wx_nemo
except Exception:
    _run_wx_nemo = None

# [6] Geçici dosya/log tercihleri
KEEP_TMP_WAVS   = os.getenv("KEEP_TMP_WAVS", "false").lower() == "true"
KEEP_WORKER_LOGS= os.getenv("KEEP_WORKER_LOGS", "false").lower() == "true"

# [7] Yardımcılar
_H_TEXT_PAT = re.compile(r"text='([^']*)'")

def _log(path: str, msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except:
        pass

def _clean_hypothesis_text(s: Optional[str]) -> str:
    if not s: return ""
    m = _H_TEXT_PAT.search(s)
    return (m.group(1) if m else s).strip()

def _is_gibberish(text: str) -> bool:
    t = (text or '').strip()
    if len(t) < 2: return True
    letters = sum(ch.isalpha() for ch in t)
    return (letters / max(1, len(t))) < 0.35

def _merge_adjacent_segments(segs: List[dict], max_gap: float = 0.35) -> List[dict]:
    """Aynı speaker + dil + kısa boşluk → birleştir."""
    out: List[dict] = []
    for s in segs:
        if not out:
            out.append(dict(s)); continue
        prev = out[-1]
        if (s.get("speaker") == prev.get("speaker")
            and (s.get("lang") or "unknown") == (prev.get("lang") or "unknown")
            and (float(s.get("start", 0.0)) - float(prev.get("end", 0.0))) <= max_gap):
            prev["end"]  = float(s.get("end", prev["end"]))
            prev["text"] = (prev.get("text","") + " " + s.get("text","")).strip()
        else:
            out.append(dict(s))
    return out

def _freeze_segment_fields(segment: dict) -> dict:
    """Eksik alanları tamamla ve tipleri standardize et."""
    return {
        "speaker": segment.get("speaker", "speaker01"),
        "start":   round(float(segment.get("start", 0.0)), 3),
        "end":     round(float(segment.get("end",   0.0)), 3),
        "text":    _clean_hypothesis_text(str(segment.get("text", "") or "")),
        "lang":    (segment.get("lang") or "unknown").strip().lower(),
        "emotion_pred": segment.get("emotion_pred", segment.get("emotion", "neutral")),
        "emotion":      segment.get("emotion",      segment.get("emotion_pred", "neutral")),
    }

def _ensure_text_field(segs: List[dict]) -> List[dict]:
    """Her segmentte 'text' alanı olduğundan emin ol (yoksa boş string ekle)."""
    safe = []
    for s in segs or []:
        if "text" not in s or s.get("text") is None:
            s = {**s, "text": ""}
        elif not isinstance(s.get("text"), str):
            s = {**s, "text": str(s.get("text"))}
        safe.append(s)
    return safe

def _fallback_summary_from_text(text: str, max_chars: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_dot = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return cut[:last_dot+1] if last_dot >= 200 else cut + "..."

def _ensure_wav16k_mono(src_path: str, out_path: Optional[str] = None, max_seconds: int = 0) -> str:
    base, _ = os.path.splitext(src_path)
    out_path = out_path or f"{base}_16k.wav"
    try:
        stream = ffmpeg.input(src_path)
        kwargs = dict(ac=1, ar=16000, sample_fmt="s16", format="wav", loglevel="error")
        stream = stream.output(out_path, t=max_seconds, **kwargs) if max_seconds > 0 else stream.output(out_path, **kwargs)
        stream.overwrite_output().run()
        return out_path
    except ffmpeg.Error:
        return src_path

def _import_whisperx_transcriber():
    """transcribe_large(...) fonksiyonunu dinamik yükle."""
    import importlib
    ROOT = Path(__file__).resolve().parents[1]
    candidates = [
        ("engines.transcribe_large_multil", ROOT / "engines" / "transcribe_large_multil.py"),
        ("transcribe_large_multil",        ROOT / "transcribe_large_multil.py"),
        ("engines.transcribe_large_multi", ROOT / "engines" / "transcribe_large_multi.py"),
    ]
    for modname, _ in candidates:
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "transcribe_large"):
                return m.transcribe_large
        except:
            continue
    raise ModuleNotFoundError("WhisperX transcriber modülü bulunamadı.")

def _refine_langs_with_langid(segs: List[dict], whitelist: Optional[set] = None) -> List[dict]:
    if not langid:
        return segs
    wl = set(whitelist or [])
    MIN_SCORE = float(os.getenv("LANGID_MIN_SCORE", "0.96"))

    def clean_txt(t: str) -> str:
        t = (t or "").strip()
        t = re.sub(r"[^\w\u00C0-\u024FğüşöçıİĞÜŞÖÇ]+", " ", t, flags=re.UNICODE)
        return t.strip()

    for s in segs:
        raw = s.get("text") or ""
        txt = clean_txt(raw)
        if len(txt) < 4:
            continue
        try:
            code, score = langid.classify(txt)
            code = (code or "").lower()
            if wl and code not in wl:
                continue
            prev = (s.get("lang") or "unknown").lower()
            if prev in ("", "unknown", None):
                if score >= MIN_SCORE:
                    s["lang"] = code
                continue
            if code != prev and score >= 0.99 and len(txt) >= 20:
                s["lang"] = code
        except Exception:
            pass
    return segs


# [8] ANA WORKER FONKSİYONU - Transkripsiyon Koordinatörü  
def transcribe_and_enrich(file_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """       
    İŞLEM AKIŞI:
    ============
    1️ PARAMETRE HAZIRLIĞI: options parse + log başlatma
    2️ DRY MOD KONTROLÜ: Test modu ise dummy data dön
    3️ WAV DÖNÜŞTÜRME: ffmpeg ile 16kHz mono standardizasyonu  
    4️ MOTOR SEÇİMİ: engine parametresine göre doğru motoru çağır
         nvidia: transcribe_nvidia_parakeet()
         whisperx_nemo: _run_wx_nemo()  
         whisperx: transcribe_large() (default)
    5️ DUYGU ANALİZİ: EmotionJSONAnalyzer (opsiyonel)
    6️ STANDARDIZASYON: _freeze_segment_fields()
    7️ DİL İYİLEŞTİRME: _refine_langs_with_langid()
    8️ SEGMENT BİRLEŞTİRME: _merge_adjacent_segments()
    9️ ÖZETLEME: Summarization (opsiyonel)
     TEMİZLİK: Geçici dosyaları sil
        
    Kullanım: api.py → transcription_service.py → transcribe_and_enrich() → motor seçimi
    """
    # 1️⃣ PARAMETRE HAZIRLIĞI ve LOG BAŞLATMA
    options = options or {}  # None güvenliği
    dry = options.get("dry", False)                           # Test modu aktif mi?
    disable_emotion = options.get("disable_emotion", False)   # Duygu analizi kapalı mı?
    disable_summary = options.get("disable_summary", False)   # Özetleme kapalı mı?  
    max_seconds = int(options.get("max_seconds", 0))          # İşlem süre sınırı (0=sınırsız)
    
    # Log dosyası oluştur
    log_path = os.path.splitext(file_path)[0] + ".worker.log"
    _log(log_path, f"🚀 START file={file_path} | opts={options}")

    # [8.A] DRY mod
    if dry:
        dummy = {
            "language": "tr",
            "segments": [
                {"speaker": "speaker01", "start": 0.0, "end": 2.5, "text": "Merhaba dünya.", "lang": "tr"},
                {"speaker": "speaker01", "start": 2.5, "end": 5.0, "text": "Bu bir testtir.", "lang": "tr"},
            ],
            "summary": ""
        }
        frozen = [_freeze_segment_fields(s) for s in dummy["segments"]]
        dummy["segments"] = frozen
        dummy["summary"] = _fallback_summary_from_text(" ".join(s.get("text","") for s in frozen))
        return dummy

    # [8.B] WAV hazırlığı + motor seçimi
    engine = (options.get("engine") or os.getenv("DEFAULT_ENGINE", "whisperx")).lower()
    lang_hint = options.get("language") or options.get("language_hint") or options.get("output_language")
    safe_path = _ensure_wav16k_mono(file_path, max_seconds=max_seconds)

    # [8.C] Transkripsiyon - 3 Motor Koordinatörü
    print(f"[WORKER] 🎯 Seçilen motor: {engine}")
    
    # Ortak parametreler
    common_params = {
        "audio_path": safe_path,
        "device": options.get("device") or os.getenv("WHISPER_DEVICE", "cpu"),
        "compute_type": options.get("compute_type"),
        "min_spk": int(options.get("min_spk", options.get("max_speakers", 2))),
        "max_spk": int(options.get("max_spk", options.get("max_speakers", 2))),
        "hf_token": options.get("hf_token") or os.getenv("HUGGINGFACE_TOKEN"),
        "language_hint": lang_hint
    }
    
    if engine == "nvidia":
        #  Pure NVIDIA: Parakeet ASR + NVIDIA Diarization
        if _run_parakeet is None:
            raise ModuleNotFoundError("NVIDIA Parakeet engine eksik.")
        print(f"[WORKER]  Pure NVIDIA engine başlatılıyor...")
        final_result = _run_parakeet(safe_path, device=common_params["device"], hf_token=common_params["hf_token"])

    elif engine == "whisperx_nemo":
        #  Hibrit: WhisperX ASR + NVIDIA Diarization 
        if _run_wx_nemo is None:
            raise ModuleNotFoundError("WhisperX + NeMo engine eksik.")
        print(f"[WORKER]  WhisperX + NVIDIA Diarization başlatılıyor...")
        final_result = _run_wx_nemo(**common_params)

    elif engine == "whisperx":
        #  WhisperX: WhisperX ASR + WhisperX Diarization
        transcriber = _import_whisperx_transcriber()
        print(f"[WORKER]  WhisperX + Built-in Diarization başlatılıyor...")
        final_result = transcriber(**common_params)
        
    else:
        #  Fallback: Basit WhisperX (diarization yok)
        transcriber = _import_whisperx_transcriber()
        print(f"[WORKER]  Fallback: Temel WhisperX (diarization yok)")
        final_result = transcriber(safe_path, language_hint=lang_hint)

    # [8.D] 'text' alanını garantiye al
    segs = _ensure_text_field(final_result.get("segments") or [])

    # [8.E] Duygu analizi (opsiyonel, çökmez)
    if not disable_emotion and segs:
        try:
            from emotion_detection import EmotionJSONAnalyzer
            analyzer = EmotionJSONAnalyzer(device=int(os.getenv("EMOTION_DEVICE", "-1")))
            segs = analyzer.analyze_segments(segs) or segs
            for s in segs:
                s.pop("emotion_dist", None)
        except Exception as e:
            _log(log_path, f"EMOTION hata: {e}")

    # [8.F] Normalize + temizlik + birleştirme
    frozen = [_freeze_segment_fields(s) for s in segs]
    if os.getenv("USE_LANGID", "true").lower() == "true" and (lang_hint or "").lower() not in {"en", "english"}:
        frozen = _refine_langs_with_langid(frozen)
    frozen = [s for s in frozen if (s.get("text","").strip() and not _is_gibberish(s.get("text","")))]
    frozen = _merge_adjacent_segments(frozen, float(os.getenv("MERGE_GAP", "0.35")))

    # [8.G] Çeviri modu/dil ipucu İngilizce ise dilleri sabitle
    if (lang_hint or "").lower() in {"en", "english"}:
        for s in frozen: s["lang"] = "en"
        final_result["language"] = "en"

    # [8.H] Üst dil: segmentlerden türet
    langs = {(s.get("lang") or "").lower() for s in frozen if (s.get("lang") or "").lower() not in ("", "unknown")}
    final_result["language"] = ("multilanguage" if len(langs) >= 2 else (next(iter(langs)) if langs else "unknown"))

    # [8.I] Model/engine alanları
    model = final_result.get("model") or engine
    for s in frozen: s["model"] = model
    final_result.update({"segments": frozen, "engine": final_result.get("engine", engine), "model": model})

    # [8.J] Özetleme (opsiyonel)
    if not disable_summary:
        full_text = " ".join(s.get("text","") for s in frozen).strip()
        summary = (final_result.get("summary") or "") if isinstance(final_result.get("summary"), str) else ""
        if not summary and full_text:
            try:
                from summarization import get_summary
                summary = get_summary(full_text)
            except Exception:
                summary = _fallback_summary_from_text(full_text)
        final_result["summary"] = summary

    return final_result

# [9] Dosyaya yazma + temizlik
def transcribe_and_enrich_and_dump(file_path: str, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str, float]:
    log_path = os.path.splitext(file_path)[0] + ".worker.log"
    try:
        res = transcribe_and_enrich(file_path, options)
        duration = max((float(s.get("end", 0.0)) for s in (res.get("segments") or [])), default=0.0)
        out_path = os.path.splitext(file_path)[0] + "_final.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        _log(log_path, f"DUMP OK -> {out_path} duration={duration:.2f}s")
        return res, out_path, duration
    except Exception as e:
        _log(log_path, f"FATAL: {e}")
        raise
    finally:
        base = os.path.splitext(file_path)[0]
        if not KEEP_TMP_WAVS:
            for cand in glob.glob(base + "_16k*.wav"):
                with suppress(Exception): os.remove(cand)
        if not KEEP_WORKER_LOGS:
            log_file = base + ".worker.log"
            if os.path.exists(log_file):
                with suppress(Exception): os.remove(log_file)

# [10] Alt süreç giriş noktası
def run_in_subprocess(file_path: str, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str, float]:
    return transcribe_and_enrich_and_dump(file_path, options or {})
