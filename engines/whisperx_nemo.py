# ============================================================
#  İŞLEM AKIŞI:
#   1. WhisperX ile ASR + word alignment
#   2. NVIDIA Sortformer ile diarization  
#   3. Word-to-segment speaker assignment
#   4. Final segment formatting
# ============================================================

import numpy as _np

if not hasattr(_np, "NaN"): _np.NaN = _np.nan
if not hasattr(_np, "Inf"): _np.Inf = _np.inf

import json
import os
from typing import Any, Dict, List, Optional

import torch
import whisperx
from dotenv import load_dotenv

load_dotenv()

# Ayarlar
SR = 16000
DISABLE_ALIGN = (os.getenv("DISABLE_ALIGN", "0") == "1")
DISABLE_DIAR = (os.getenv("DISABLE_DIARIZATION", "0") == "1")
ROUGH_BATCH = int(os.getenv("WHX_ROUGH_BATCH", "8"))
SUB_BATCH = int(os.getenv("WHX_SUB_BATCH", "4"))

torch.set_grad_enabled(False)
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    torch.set_num_interop_threads(int(os.getenv("TORCH_INTEROP_THREADS", "1")))
except Exception:
    pass

def overlap(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))

def normalize_speakers(segments):
    first_seen = {}
    for s in segments:
        spk = s.get("speaker", "SPEAKER_00")
        if spk not in first_seen:
            first_seen[spk] = s.get("start", 0.0)
    ordered = sorted(first_seen.items(), key=lambda kv: kv[1])
    mapping = {k: f"SPEAKER_{i+1:02d}" for i, (k, _) in enumerate(ordered) if k != "SPEAKER_00"}
    for s in segments:
        if s["speaker"] != "SPEAKER_00":
            s["speaker"] = mapping.get(s["speaker"], s["speaker"])

def majority_label_for_sentence(sent, words, min_ratio=0.10):
    dur = max(1e-6, sent["end"] - sent["start"])
    tally = {}
    if words is None:
        words = []
    for w in words:
        ov = overlap(sent["start"], sent["end"], w["start"], w["end"])
        if ov > 0.0:
            spk = w.get("speaker", "SPEAKER_00")
            tally[spk] = tally.get(spk, 0.0) + ov
    if not tally:
        return "SPEAKER_00", None, 0.0
    best_spk, best_ov = max(tally.items(), key=lambda kv: kv[1])
    ratio = best_ov / dur
    return (best_spk if ratio >= min_ratio else "SPEAKER_00"), best_spk, ratio

def get_nvidia_diar_segments(audio_path, device, min_spk=2, max_spk=2):
    """NVIDIA diarization kullanarak segment listesi döndür"""
    try:
        from engines.nvidia_parakeet import _diarize, _load_diar
        print("[NVIDIA-DIAR] NVIDIA diarization model yükleniyor...")
        
        diar_model = _load_diar(device)
        diar_segments = _diarize(audio_path, diar_model)
        
        if diar_segments:
            print(f"[NVIDIA-DIAR] ✓ NVIDIA diarization başarılı! {len(diar_segments)} segment")
            # Merge same speaker blocks
            merged = []
            diar_segments = sorted(diar_segments, key=lambda d: float(d.get("start", 0.0)))
            for d in diar_segments:
                if not merged:
                    merged.append(dict(d))
                    continue
                prev = merged[-1]
                if (d.get("speaker") == prev.get("speaker") and 
                    float(d.get("start", 0.0)) <= float(prev.get("end", 0.0)) + 0.05):
                    prev["end"] = max(float(prev["end"]), float(d.get("end", 0.0)))
                else:
                    merged.append(dict(d))
            return merged
        else:
            print("[NVIDIA-DIAR] ✗ NVIDIA diarization boş sonuç")
            return []
    except Exception as e:
        print(f"[NVIDIA-DIAR] ✗ NVIDIA diarization hatası: {e}")
        return []

def assign_speakers_to_words(words, diar_segments):
    """Kelimeler ve diarization segmentlerine göre konuşmacı ata"""
    if not words or not diar_segments:
        return [{**w, "speaker": "SPEAKER_01"} for w in words]
    
    words_with_spk = []
    for w in words:
        w_start, w_end = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        best_spk, best_overlap = "SPEAKER_00", 0.0
        
        for d in diar_segments:
            d_start, d_end = float(d["start"]), float(d["end"])
            ov = max(0.0, min(w_end, d_end) - max(w_start, d_start))
            if ov > best_overlap:
                best_overlap, best_spk = ov, d["speaker"]
        
        words_with_spk.append({**w, "speaker": best_spk})
    
    return words_with_spk

def transcribe_large(audio_path, device=None, compute_type=None, min_spk=2, max_spk=2, hf_token=None, language_hint=None):
    """
    ANA FONKSİYON - Hibrit WhisperX + NVIDIA Pipeline
    İŞLEM AKIŞI - HİBRİT STRATEJİ:
    ==============================
    1️ WHISPERX ASR: En iyi açık kaynak ASR ile transcription
    2️ WORD ALIGNMENT: WhisperX'in mükemmel kelime hizalanması
    3️ NVIDIA DİARİZATİON: Sortformer ile güçlü konuşmacı ayırma  
    4️ HİBRİT BİRLEŞTİRME: İki teknolojinin sonuçlarını merge et
    5️ SPEAKER ASSIGNMENT: Kelime → cümle konuşmacı ataması
    6️ FİNAL STANDARDİZASYON: Çıktı formatını normalize et
    
    Kullanım: transcription_worker.py → _run_wx_nemo() → transcribe_large()
    """
    # 1️ PARAMETRE HAZIRLIĞI ve KONFIGÜRASYON
    device = device or os.getenv("WHISPER_DEVICE", "cpu")                    # Cihaz seçimi (CPU/CUDA)
    compute_type = compute_type or ("int8" if device == "cpu" else "float16") # CPU→int8, GPU→float16
    translate_to_en = str(language_hint or "").lower() in {"en", "english"}  # İngilizce çeviri modu

    print(f"[WXN-ENGINE] WhisperX + NVIDIA başlatılıyor... (device={device})")

    # WhisperX ASR
    try:
        model = whisperx.load_model("large-v3", device, compute_type=compute_type,
                                   task="translate" if translate_to_en else "transcribe")
    except Exception as e:
        if "CUDA" in str(e):
            device = "cpu"
            compute_type = "int8"
            print("[WXN-ENGINE] CUDA hatası, CPU'ya geçiliyor")
            model = whisperx.load_model("large-v3", device, compute_type=compute_type,
                                       task="translate" if translate_to_en else "transcribe")
        else:
            raise

    audio = whisperx.load_audio(audio_path)
    rough = model.transcribe(audio, batch_size=ROUGH_BATCH, chunk_size=15)
    
    final_sents, all_words, langs = [], [], []

    # Segment processing
    for seg in rough.get("segments", []):
        s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        chunk = audio[int(s*SR):int(e*SR)]
        sub = model.transcribe(chunk, batch_size=SUB_BATCH, chunk_size=max(0.01, e-s))
        lang = sub.get("language", "unknown")
        langs.append(lang)
        sents = sub.get("segments", [])

        # Word alignment
        if not DISABLE_ALIGN:
            try:
                lang_code = lang if (isinstance(lang, str) and len(lang) == 2) else "en"
                align_model, meta = whisperx.load_align_model(language_code=lang_code, device=device)
                aligned = whisperx.align(sents, align_model, meta, chunk, device=device, return_char_alignments=False)
                sents = aligned.get("segments", [])
                for w in aligned.get("word_segments", []):
                    if w.get("word"):
                        all_words.append({
                            "start": float(w.get("start", s)) + s,
                            "end": float(w.get("end", e)) + s,
                            "word": w.get("word"),
                            "text": w.get("word"),  # Compatibility
                            "lang": lang,
                        })
            except Exception as e:
                print(f"[WXN-ALIGN] Alignment hatası: {e}")

        # Collect sentences
        for ss in sents:
            final_sents.append({
                "speaker": "SPEAKER_00",
                "start": float(ss.get("start", 0.0)) + s,
                "end": float(ss.get("end", 0.0)) + s,
                "text": ss.get("text", ""),
                "lang": lang,
            })

    print(f"[WXN-ENGINE] ASR tamamlandı. {len(final_sents)} cümle, {len(all_words)} kelime")

    # NVIDIA Diarization
    diar_segments = []
    words_with_spk = None
    
    if not DISABLE_DIAR:
        diar_segments = get_nvidia_diar_segments(audio_path, device, min_spk, max_spk)
        
        if diar_segments and all_words:
            words_with_spk = assign_speakers_to_words(all_words, diar_segments)
            print(f"[WXN-DIAR] ✓ {len(words_with_spk)} kelimeye konuşmacı atandı")
        else:
            print("[WXN-DIAR] ⚠ Diarization başarısız, tek konuşmacı varsayılıyor")
    
    # Fallback
    if words_with_spk is None:
        words_with_spk = [{**w, "speaker": "SPEAKER_01"} for w in all_words]

    # Speaker assignment to sentences
    print("[WXN-ENGINE] Cümlelere konuşmacı ataması yapılıyor...")
    for s in final_sents:
        s["speaker"] = majority_label_for_sentence(s, words_with_spk)[0]

    # Normalize speakers
    normalize_speakers(final_sents)
    for s in final_sents:
        s["speaker"] = s["speaker"].replace("SPEAKER_", "speaker")

    # Prepare output
    out_segments = [{
        "speaker": s["speaker"],
        "start": s["start"],
        "end": s["end"],
        "text": s.get("text", ""),
        "lang": s["lang"],
    } for s in final_sents]

    from collections import Counter
    lang_majority = Counter(langs).most_common(1)[0][0] if langs else "unknown"
    
    # Speaker dağılımı raporu
    speaker_counts = Counter(s["speaker"] for s in out_segments)
    print(f"[WXN-ENGINE] ✓ Tamamlandı! Konuşmacı dağılımı: {dict(speaker_counts)}")

    return {
        "engine": "whisperx_multil",
        "model": "large-v3",
        "device": device,
        "compute_type": compute_type,
        "language": lang_majority,
        "summary": None,
        "segments": out_segments,
    }

if __name__ == "__main__":
    load_dotenv()
    AUDIO_FILE = os.getenv("AUDIO_FILE", "voices\\multil.mp4")
    res = transcribe_large(AUDIO_FILE)
    with open("hypotheses_multil_diarized_raw.json", "w", encoding="utf-8") as f:
        json.dump(res["segments"], f, ensure_ascii=False, indent=2)
    print(f"[OK] hypotheses_multil_diarized_raw.json yazıldı | {len(res['segments'])} segment")