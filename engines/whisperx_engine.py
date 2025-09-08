# ============================================================
#  WHISPERX ENGINE - CLEAN & SIMPLE
# ============================================================
#  ✅ WhisperX ASR (large-v3) - Konuşma tanıma
#  ✅ Pyannote Diarization - Konuşmacı ayrımı  
#  ✅ Cümle Segmentasyonu - Her cümle ayrı segment
# ============================================================

import os, json, re
from typing import Optional
from dotenv import load_dotenv
import torch
import whisperx
from collections import Counter

# NumPy uyumluluk düzeltmesi
import numpy as _np
if not hasattr(_np, "NaN"): _np.NaN = _np.nan
if not hasattr(_np, "Inf"): _np.Inf = _np.inf

load_dotenv()

# Konfigürasyon
SR = 16000
DISABLE_ALIGN = True  # Word alignment kapalı (hata veriyor)
DISABLE_DIAR = (os.getenv("DISABLE_DIARIZATION", "0") == "1")
ROUGH_BATCH = int(os.getenv("WHX_ROUGH_BATCH", "8"))
SUB_BATCH = int(os.getenv("WHX_SUB_BATCH", "4"))

# PyTorch optimizasyonu
torch.set_grad_enabled(False)
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    torch.set_num_interop_threads(int(os.getenv("TORCH_INTEROP_THREADS", "1")))
except:
    pass

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def overlap(a1, a2, b1, b2):
    """İki zaman aralığının kesişim süresini hesapla"""
    return max(0.0, min(a2, b2) - max(a1, b1))

def normalize_speakers(segments):
    """Konuşmacıları kronolojik sıraya göre düzenle (ilk konuşan = SPEAKER_01)"""
    # Her konuşmacının ilk görülme zamanı
    first_seen = {}
    for s in segments:
        spk = s.get("speaker", "SPEAKER_00")
        if spk not in first_seen:
            first_seen[spk] = s.get("start", 0.0)
    
    # Zaman sırasına göre mapping
    ordered = sorted(first_seen.items(), key=lambda kv: kv[1])
    mapping = {k: f"SPEAKER_{i+1:02d}" for i, (k, _) in enumerate(ordered) if k != "SPEAKER_00"}
    
    # Güncelle
    for s in segments:
        if s["speaker"] != "SPEAKER_00":
            s["speaker"] = mapping.get(s["speaker"], s["speaker"])

def assign_speaker_by_overlap(sentence, words_with_speakers):
    """Cümleye konuşmacı ata (kelime overlap'ine göre)"""
    if not words_with_speakers:
        return "SPEAKER_00"
    
    duration = max(1e-6, sentence["end"] - sentence["start"])
    speaker_times = {}
    
    # Her kelime için overlap hesapla
    for w in words_with_speakers:
        overlap_time = overlap(sentence["start"], sentence["end"], w["start"], w["end"])
        if overlap_time > 0:
            spk = w.get("speaker", "SPEAKER_00")
            speaker_times[spk] = speaker_times.get(spk, 0.0) + overlap_time
    
    if not speaker_times:
        return "SPEAKER_00"
    
    # En çok overlap eden konuşmacı
    best_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
    confidence = speaker_times[best_speaker] / duration
    
    return best_speaker if confidence >= 0.10 else "SPEAKER_00"

def split_text_to_sentences(text):
    """Metni cümlelere böl"""
    # Noktalama ile böl
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Uzun metinleri mecburi böl (10+ kelime)
    if len(sentences) <= 1 and len(text.split()) > 10:
        words = text.split()
        mid = len(words) // 2
        sentences = [' '.join(words[:mid]), ' '.join(words[mid:])]
    
    return sentences if sentences else [text]

# ============================================================
# ANA FONKSİYON
# ============================================================

def transcribe_large(audio_path, device=None, compute_type=None, min_spk=2, max_spk=2, hf_token=None, language_hint=None):
    """
    WhisperX + Pyannote ile konuşma tanıma ve konuşmacı ayrımı
    
    1. WhisperX ASR - Konuşma tanıma
    2. Cümle segmentasyonu 
    3. Pyannote diarization - Konuşmacı ayrımı
    4. Speaker atama ve normalize
    """
    
    # Parametre hazırlığı
    device = device or os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = compute_type or ("int8" if device == "cpu" else "float16")
    hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN", "")
    translate_to_en = str(language_hint or "").lower() in {"en", "english"}
    
    print(f"[WHISPERX] ASR başlatılıyor... (device={device})")
    print(f"[WHISPERX] Diarization: {'Aktif' if not DISABLE_DIAR and hf_token else 'Kapalı'}")
    
    # WhisperX model yükle
    model = whisperx.load_model("large-v3", device, compute_type=compute_type,
                               task="translate" if translate_to_en else "transcribe")
    audio = whisperx.load_audio(audio_path)
    
    # Kaba transkripsiyon
    rough = model.transcribe(audio, batch_size=ROUGH_BATCH, chunk_size=15)
    final_sentences = []
    all_words = []
    languages = []
    
    # Her segmenti işle
    for seg in rough.get("segments", []):
        s, e = seg["start"], seg["end"]
        chunk = audio[int(s*SR):int(e*SR)]
        
        # Detaylı transkripsiyon
        detailed = model.transcribe(chunk, batch_size=SUB_BATCH, chunk_size=max(0.01, e-s))
        lang = detailed.get("language", "unknown")
        languages.append(lang)
        segments = detailed.get("segments", [])
        words = []
        
        # Word alignment (eğer aktifse)
        if not DISABLE_ALIGN:
            try:
                align_model, meta = whisperx.load_align_model(lang, device)
                aligned = whisperx.align(segments, align_model, meta, chunk, device)
                segments = aligned.get("segments", [])
                
                # Kelimeleri topla
                for segment in segments:
                    segment_words = segment.get("words", [])
                    if isinstance(segment_words, list):
                        words.extend(segment_words)
            except:
                words = []
        
        # Kelimeleri kaydet
        for w in words:
            if isinstance(w, dict):
                all_words.append({
                    "start": float(w.get("start", 0.0)) + s,
                    "end": float(w.get("end", 0.0)) + s,
                    "text": w.get("text") or w.get("word", ""),
                    "lang": lang
                })
        
        # Segmentleri cümlelere böl
        for segment in segments:
            text = segment.get("text", "").strip()
            seg_start = segment["start"] + s
            seg_end = segment["end"] + s
            seg_duration = seg_end - seg_start
            
            # Cümlelere böl
            sentences = split_text_to_sentences(text)
            
            if len(sentences) <= 1:
                # Tek cümle
                final_sentences.append({
                    "speaker": "SPEAKER_00",
                    "start": seg_start,
                    "end": seg_end,
                    "text": text,
                    "lang": lang
                })
            else:
                # Çoklu cümle - zamanı böl
                total_chars = sum(len(sent) for sent in sentences)
                current_time = seg_start
                
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                    
                    # Zaman hesapla
                    char_ratio = len(sentence) / max(total_chars, 1)
                    sent_duration = seg_duration * char_ratio
                    sent_end = current_time + sent_duration
                    
                    if i == len(sentences) - 1:  # Son cümle
                        sent_end = seg_end
                    
                    final_sentences.append({
                        "speaker": "SPEAKER_00",
                        "start": current_time,
                        "end": sent_end,
                        "text": sentence.strip(),
                        "lang": lang
                    })
                    
                    current_time = sent_end
    
    print(f"[WHISPERX] ASR tamamlandı: {len(final_sentences)} cümle, {len(all_words)} kelime")
    
    # Diarization (konuşmacı ayrımı)
    words_with_speakers = None
    
    if not DISABLE_DIAR and hf_token:
        try:
            from pyannote.audio import Pipeline
            diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", 
                                                  use_auth_token=hf_token).to(torch.device(device))
            
            diar_result = diarization(audio_path, min_speakers=min_spk, max_speakers=max_spk)
            
            # Diarization sonuçlarını parse et
            if hasattr(diar_result, 'itertracks'):
                timeline = []
                for turn, _, speaker in diar_result.itertracks(yield_label=True):
                    timeline.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': str(speaker)
                    })
                
                # Kelimelere konuşmacı ata
                if all_words and timeline:
                    words_with_speakers = []
                    for word in all_words:
                        best_speaker = "SPEAKER_00"
                        max_overlap = 0.0
                        
                        for spk_seg in timeline:
                            overlap_time = overlap(word["start"], word["end"], 
                                                 spk_seg['start'], spk_seg['end'])
                            if overlap_time > max_overlap:
                                max_overlap = overlap_time
                                best_speaker = spk_seg['speaker']
                        
                        words_with_speakers.append({**word, "speaker": best_speaker})
                
                print(f"[WHISPERX] Diarization tamamlandı: {len(timeline)} segment")
        except Exception as e:
            print(f"[WHISPERX] Diarization hatası: {e}")
    
    # Cümlelere konuşmacı ata
    print(f"[WHISPERX] {len(final_sentences)} cümleye konuşmacı atanıyor...")
    
    if words_with_speakers:
        # Kelime bazlı atama
        for sentence in final_sentences:
            assigned_speaker = assign_speaker_by_overlap(sentence, words_with_speakers)
            if assigned_speaker == "SPEAKER_00":
                # Fallback: Zaman bazlı
                total_duration = max(s.get("end", 0.0) for s in final_sentences)
                mid_point = total_duration / 2.0
                assigned_speaker = "SPEAKER_01" if sentence["start"] < mid_point else "SPEAKER_02"
            sentence["speaker"] = assigned_speaker
    else:
        # Zaman bazlı atama
        total_duration = max(s.get("end", 0.0) for s in final_sentences)
        mid_point = total_duration / 2.0
        for sentence in final_sentences:
            sentence["speaker"] = "SPEAKER_01" if sentence["start"] < mid_point else "SPEAKER_02"
    
    # Konuşmacıları normalize et
    normalize_speakers(final_sentences)
    
    # SPEAKER_XX -> speakerXX formatına çevir
    for s in final_sentences:
        s["speaker"] = s["speaker"].replace("SPEAKER_", "speaker")
    
    # Çıktı formatı
    segments = [{
        "speaker": s["speaker"],
        "start": s["start"],
        "end": s["end"],
        "text": s["text"],
        "lang": s["lang"]
    } for s in final_sentences]
    
    # Sonuç
    majority_lang = Counter(languages).most_common(1)[0][0] if languages else "unknown"
    speaker_dist = Counter(s["speaker"] for s in segments)
    
    print(f"[WHISPERX] ✓ Tamamlandı! {dict(speaker_dist)}")
    
    return {
        "engine": "whisperx",
        "model": "large-v3",
        "device": device,
        "compute_type": compute_type,
        "language": majority_lang,
        "summary": None,
        "segments": segments
    }

# Test için
if __name__ == "__main__":
    load_dotenv()
    audio_file = os.getenv("AUDIO_FILE", "sample.wav")
    result = transcribe_large(audio_file)
    
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=2)
    
    print(f"[OK] output.json yazıldı | {len(result['segments'])} segment")