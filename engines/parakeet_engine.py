# ------------------------------------------------------------
# AKIŞ: 1) Audio → WAV 16kHz mono dönüştür
#       2) NVIDIA Sortformer ile diarization (konuşmacı segmentleri al)
#       3) Her segment için NVIDIA Parakeet ile ASR yap
#       4) Sonuçları birleştirip standardize et
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import suppress  # Hata yakalama için güvenli context manager
from typing import Any, Dict, List, Optional, Tuple

import torch  # NVIDIA model'ları için
from dotenv import load_dotenv  # Çevre değişkenlerini .env dosyasından yükle

# NVIDIA NeMo kütüphanesi - Konuşma tanıma ve diarization için
from nemo.collections.asr.models import (
    ASRModel,  # Ana ASR (Automatic Speech Recognition) model sınıfı
)
from pydub import AudioSegment  # Ses dosyası manipülasyonu için

try:
    # Diarization için Sortformer modelini yükle (farklı import yolları denenerek)
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError:
    # Eski NeMo sürümlerinde farklı yolda olabilir
    from nemo.collections.asr.models.label_models import SortformerEncLabelModel

load_dotenv()  # .env dosyasından çevre değişkenlerini yükle

# ============================================================
# GLOBAL AYARLAR - .env dosyasından veya varsayılan değerler
# ============================================================
ASR_HF_ID = os.getenv("ASR_HF_ID", "nvidia/parakeet-tdt-0.6b-v3")      # NVIDIA ASR model ID'si (Hugging Face Hub'dan)
DIAR_ID = os.getenv("DIAR_MODEL", "nvidia/diar_sortformer_4spk-v1")     # NVIDIA diarization model ID'si (4 konuşmacıya kadar)
PADDING_SEC = float(os.getenv("SEG_PADDING_SEC", "0.20"))               # Her segmente eklenecek padding süresi (saniye) - kesilme önleme
TMP_SR = int(os.getenv("TMP_SR", "16000"))                              # Geçici WAV dosyaları için sample rate (16kHz NVIDIA standart)
BATCH_SIZE = int(os.getenv("ASR_BATCH", "8"))                           # ASR batch boyutu (aynı anda kaç segment işle)
LANG_DETECT = os.getenv("NEMO_LANG_DETECT", "1") == "1"                 # Dil tespit özelliğini aç/kapat

# ============================================================
# YARDIMCI FONKSİYONLAR - Konuşmacı normalizasyon ve segment birleştirme
# ============================================================

def _norm_spk(speaker: str) -> str:
    """
    Konuşmacı etiketlerini normalize et: "SPEAKER_1" → "speaker01"     
    Kullanım sırası: _diarize() → _norm_spk() → normalize edilmiş speaker etiketi
    """
    if not speaker:  # Boş string veya None ise
        return "speaker01"  # Varsayılan konuşmacı
    
    s = speaker.upper().replace("SPEAKER_", "")  # "SPEAKER_1" → "1"
    m = re.search(r"(\d+)", s)  # İlk sayıyı bul (regex ile)
    return f"speaker{int(m.group(1)):02d}" if m else "speaker01"  # "1" → "speaker01"

def _merge(segs: List[Dict[str, Any]], gap: float = 0.30) -> List[Dict[str, Any]]:
    """
    Aynı konuşmacının ardışık segmentlerini birleştir
    Args:
        segs: Diarization segmentleri [{"start":0, "end":2, "speaker":"spk1"}, ...]
        gap: Birleştirme eşiği (saniye) - bu süreden az boşluk varsa birleştir
    Algoritma:
        1) Segmentleri başlama zamanına göre sırala
        2) Her segment için: eğer önceki segment aynı konuşmacı + gap süresi içinde ise birleştir
        3) Aksi halde yeni segment olarak ekle
        
    Kullanım sırası: _diarize() → _merge() → transcribe_nvidia_parakeet()
    """
    if not segs:  # Boş liste ise
        return []
    
    segs = sorted(segs, key=lambda x: x["start"])  # Zamansal sıralama (start zamanına göre)
    out = [segs[0].copy()]  # İlk segmenti ekle
    
    for s in segs[1:]:  # Kalan segmentleri tek tek kontrol et
        last = out[-1]  # Listedeki son segment
        
        # KOŞUL: Aynı konuşmacı VE gap süresi içinde ise
        if s["speaker"] == last["speaker"] and (s["start"] - last["end"]) <= gap:
            # BİRLEŞTİR: Son segmentin end zamanını uzat
            last["end"] = max(last["end"], s["end"])
        else:
            # YENİ SEGMENT: Ayrı konuşmacı veya çok uzak zaman
            out.append(s.copy())
    
    return out

# ============================================================
# SES DOSYASI İŞLEME FONKSİYONLARI
# ============================================================

def _ensure_wav16k_mono(src_path: str) -> Tuple[str, AudioSegment]:
    """
    Ses dosyasını NVIDIA standartlarına uygun hale getir: 16kHz, mono, WAV
    İşlem adımları:
        1) Önce convert_audio.py kullanmayı dene (varsa)
        2) Yoksa pydub ile manuel dönüştür:
           - Sample rate: 16000 Hz (NVIDIA standart)
           - Kanal: Mono (1 kanal)
           - Bit derinliği: 16-bit (2 byte) 
    Kullanım sırası: transcribe_nvidia_parakeet() → _ensure_wav16k_mono() → WAV dosyası hazır
    """
    # Önce özel converter'ı dene (convert_audio.py)
    with suppress(Exception):
        from convert_audio import AudioConverter
        wav_path = AudioConverter().convert_to_wav(src_path)
        return wav_path, AudioSegment.from_wav(wav_path)
    
    # Fallback: pydub ile manuel dönüşüm
    audio = AudioSegment.from_file(src_path)  # Her türlü ses dosyasını oku
    wav = audio.set_frame_rate(TMP_SR).set_channels(1).set_sample_width(2)  # 16kHz, mono, 16-bit
    outp = os.path.splitext(src_path)[0] + "_tmp16k.wav"  # Çıkış dosya adı
    wav.export(outp, format="wav")  # WAV olarak kaydet
    return outp, AudioSegment.from_wav(outp)  # Dosya yolu ve AudioSegment döndür

# ============================================================
# NVIDIA MODEL YÜKLEME FONKSİYONLARI
# ============================================================

def _load_asr(device: str, hf_token: str = None) -> Tuple[ASRModel, str]:
    """
    NVIDIA Parakeet ASR modelini yükle ve GPU/CPU'ya taşı
    NOT: Authentication ana fonksiyonda yapıldı, burada sadece model yükle
    """
    print(f"[NVIDIA-ASR] ASR Model yükleniyor: {ASR_HF_ID}")
    try:
        m = ASRModel.from_pretrained(ASR_HF_ID).to(device)
        print("[NVIDIA-ASR] ✅ ASR Model başarıyla yüklendi!")
    except Exception as e:
        print(f"[NVIDIA-ASR] ❌ ASR Model yükleme hatası: {e}")
        raise
    
    m.eval()  # Değerlendirme moduna al (training modları kapat)
    return m, ASR_HF_ID  # Model ve ID'sini döndür

def _load_diar(device: str, hf_token: str = None) -> SortformerEncLabelModel:
    """
    NVIDIA Sortformer diarization modelini yükle 
    NOT: Authentication ana fonksiyonda yapıldı, burada sadece model yükle
    """
    print(f"[NVIDIA-DIAR] Diarization Model yükleniyor: {DIAR_ID}")
    
    try:
        d = SortformerEncLabelModel.from_pretrained(DIAR_ID).to(device)
        print("[NVIDIA-DIAR] ✅ Diarization model başarıyla yüklendi!")
    except Exception as e:
        print(f"[NVIDIA-DIAR] ❌ Diarization model yükleme hatası: {e}")
        raise
    
    d.eval()  # Değerlendirme moduna al
    return d


# DİARİZATİON VE DİL TESPİTİ FONKSİYONLARI
def _diarize(wav_path: str, diar_model: SortformerEncLabelModel) -> List[Dict[str, Any]]:
    """
    NVIDIA Sortformer ile konuşmacı ayırma işlemi yap 
    İşlem adımları:
        1) Model.diarize() çalıştır (ses dosyasını işle)
        2) Çıktı formatını normalize et (list/string/nested list)
        3) Her satırı parse et ve zaman damgalarını çıkart
        4) Speaker etiketlerini normalize et ("speaker01" formatına)
        
    Kullanım sırası: _load_diar() → _diarize() → segment listesi → _merge()
    """
    # NVIDIA Sortformer modelini çalıştır
    res = diar_model.diarize(audio=wav_path, batch_size=1)  # batch_size=1: tek dosya işle
    
    # Çıktı formatını normalize et - farklı NeMo sürümleri farklı format döndürebilir
    lines: List[str] = []
    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
        # Nested list format: [[line1, line2, ...]]
        lines = res[0]
    elif isinstance(res, list):
        # Direct list format: [line1, line2, ...]
        lines = res
    elif isinstance(res, str):
        # String format: "line1\nline2\n..."
        lines = res.strip().splitlines()

    # Her satırı parse et ve segment oluştur
    out: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            p = ln.strip().split()  # Boşluklara göre ayır
            
            if len(p) == 3:
                # Format 1: "start end speaker" (basit format)
                start, end, spk = float(p[0]), float(p[1]), p[2]
                
            elif len(p) >= 8 and p[0].upper() == "SPEAKER":
                # Format 2: RTTM format - "SPEAKER file_name channel start_time duration ..."  
                start, dur = float(p[3]), float(p[4])  # 3. index: başlangıç, 4. index: süre
                end = start + dur  # Bitiş = başlangıç + süre
                spk = p[7]  # 7. index: konuşmacı etiketi
                
            else:
                continue  # Tanınmayan format, atla
            
            # Segment oluştur ve normalize edilmiş speaker etiketi ekle
            out.append({"start": start, "end": end, "speaker": _norm_spk(spk)})
            
        except Exception:
            # Parse hatası durumunda o satırı atla
            pass
    
    return out

def _guess_lang(text: str) -> str:
    """
    Metindeki dili otomatik tespit et
    Returns:
        Dil kodu (ISO 639-1): "tr", "en", "fr" vs. | "unknown" tespit edilemezse
    Algoritma:
        1) LANG_DETECT flag'i kapalıysa → "unknown" dön
        2) Text boş veya sadece boşluksa → "unknown" dön  
        3) Önce langid kütüphanesini dene 
        4) langid yoksa langdetect kütüphanesini dene  
        5) Her ikisi de yoksa → "unknown" dön  
    Kullanılan kütüphaneler:
        - langid: Daha hafif, hızlı
        - langdetect: Google'ın dil tespit kütüphanesi
    Kullanım sırası: transcribe_nvidia_parakeet() → _guess_lang() → segment["lang"] = dil_kodu
    """
    # Dil tespiti kapalı mı veya metin boş mu?
    if not LANG_DETECT or not (text or "").strip():
        return "unknown"
    
    # Yöntem 1: langid kütüphanesi (önerilen)
    with suppress(Exception):
        import langid  # pip install langid
        lid, _ = langid.classify(text)  # (dil_kodu, güven_skoru) döndürür
        return lid or "unknown"
    
    # Yöntem 2: langdetect kütüphanesi (fallback)
    with suppress(Exception):
        from langdetect import detect  # pip install langdetect
        return detect(text) or "unknown"
    
    # Her iki kütüphane de yoksa
    return "unknown"

# ============================================================
# ANA FONKSİYON - PURE NVIDIA TRANSKRİPSİYON PİPELINE
# ============================================================

def transcribe_nvidia_parakeet(audio_path: str, device: Optional[str] = None, hf_token: Optional[str] = None) -> Dict[str, Any]:
    """     
    İŞLEM AKIŞI - ADIM ADIM:
    ========================
    1️ CİHAZ SEÇİMİ: CUDA var mı → GPU, yoksa → CPU
    2️ SES HAZIRLIĞI: Audio → 16kHz mono WAV dönüştür
    3️ MODEL YÜKLEME: NVIDIA ASR + Diarization modellerini yükle
    4️ DİARİZATİON: Konuşmacı segmentlerini belirle
    5️ SEGMENT BİRLEŞTİRME: Yakın segmentleri merge et
    6️ PADDING: Her segmente 0.2s kenar boşluğu ekle  
    7️ ASR İŞLEME: Her segment için ayrı ASR yap
    8️ DİL TESPİTİ: Segment metinlerinden dili tahmin et
    9️ SONUÇ BİRLEŞTİRME: Tüm segmentleri standardize et
    10 TEMİZLİK: Geçici dosyaları sil
    
    Kullanım sırası: transcription_worker.py → transcribe_nvidia_parakeet() → sonuç dict
    """
    # 1️ CİHAZ SEÇİMİ - Otomatik GPU/CPU tespiti
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN", "")
    print(f"[NVIDIA-ENGINE] NVIDIA Parakeet başlatılıyor... (device={device})")
    print(f"[NVIDIA-ENGINE] HF Token: {'✓' if hf_token else '✗'} (uzunluk: {len(hf_token)})")
    
    # 🔑 GLOBAL TOKEN AUTHENTICATION - NeMo için kritik!
    if hf_token:
        print("[NVIDIA-ENGINE] 🔑 Global HF authentication başlatılıyor...")
        
        # ÖNEMLİ: Önce tüm çakışan environment variables'ı temizle!
        conflicting_vars = [
            "HFF_TOKEN",          # Bu çifte F'li olan çakışma yapıyor!
            "HUGGING_FACE_TOKEN", 
            "HUGGINGFACE_API_TOKEN", 
            "HF_API_TOKEN"
        ]
        for var in conflicting_vars:
            if var in os.environ:
                print(f"[NVIDIA-ENGINE] 🧹 Çakışan token temizleniyor: {var}")
                del os.environ[var]
        
        # Token validation
        if len(hf_token) < 10 or not hf_token.startswith(('hf_', 'hfa_')):
            print(f"[NVIDIA-ENGINE] ❌ Geçersiz token formatı! Token: {hf_token[:10]}...")
            print("[NVIDIA-ENGINE] Doğru format: hf_xxxxxx veya hfa_xxxxxx")
            raise ValueError("Geçersiz HuggingFace token formatı")
        
        # Doğru environment variables set et
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
        
        print("[NVIDIA-ENGINE] ✅ Environment variables set edildi")
        
        # Global HuggingFace login for session-wide authentication
        try:
            from huggingface_hub import HfApi, login, logout
            
            # Önce logout yap (eski session'ları temizle)
            try:
                logout()
            except:
                pass
                
            # Yeni login yap
            login(token=hf_token, add_to_git_credential=False, write_permission=False)
            
            # Test connection
            api = HfApi(token=hf_token)
            user_info = api.whoami()
            print(f"[NVIDIA-ENGINE] ✅ HF Login başarılı: {user_info.get('name', 'User')}")
            
        except Exception as e:
            print(f"[NVIDIA-ENGINE] ❌ CRITICAL HF Login hatası: {e}")
            raise  # Bu sefer exception'ı yukarı at
    else:
        print("[NVIDIA-ENGINE] ❌ HF Token yok - NVIDIA modelleri başarısız olabilir!")

    # 2️ SES HAZIRLIĞI - WAV formatına dönüştür  
    wav_path, audio_full = _ensure_wav16k_mono(audio_path)  # 16kHz mono WAV oluştur
    wav_dur = len(audio_full) / 1000.0  # Toplam süre (saniye)
    base_dir = os.path.dirname(os.path.abspath(wav_path))  # Geçici dosyalar için dizin

    # 3️ MODEL YÜKLEME - ASR ve Diarization modelleri (authentication zaten yapıldı)
    asr, used_model = _load_asr(device)  # Parakeet ASR modelini yükle
    diar = _load_diar(device)  # Sortformer diarization modelini yükle

    # 4️ DİARİZATİON - Konuşmacı ayırma işlemi
    print("[NVIDIA-DIAR] Diarization işlemi başlatılıyor...")
    raw = _diarize(wav_path, diar) or [{"start": 0.0, "end": wav_dur, "speaker": "speaker01"}]  # Fallback: tek konuşmacı
    print(f"[NVIDIA-DIAR] {len(raw)} diarization segment bulundu")
    
    # 5️ SEGMENT BİRLEŞTİRME - Yakın segmentleri merge et
    merged = _merge(raw, gap=0.30)  # 0.3 saniyeden az boşluk varsa birleştir
    print(f"[NVIDIA-DIAR] Segmentler birleştirildi: {len(merged)} final segment")

    # 6️ PADDING - Her segmente kenar boşluğu ekle (kesilme önleme)
    for s in merged:
        s["start"] = max(0.0, s["start"] - PADDING_SEC)          # Başlangıcı 0.2s öne al (minimum 0)
        s["end"]   = min(wav_dur, s["end"] + PADDING_SEC)        # Bitişi 0.2s geriye al (maksimum toplam süre)

    # 7️ ASR İŞLEME - Her segment için ayrı transkripsiyon
    print(f"[NVIDIA-ASR] {len(merged)} segment için ASR başlatılıyor...")
    tmp_paths, metas = [], []  # Geçici dosya yolları ve metadata
    
    # Her segment için geçici WAV dosyası oluştur
    for s in merged:
        # Segmenti kes (milisaniye cinsinden)
        clip = audio_full[int(s["start"]*1000): int(s["end"]*1000)]
        
        # Geçici dosya adı oluştur
        tmp = os.path.join(base_dir, f"nemo_seg_{uuid.uuid4().hex[:8]}.wav")
        
        # Segment'i WAV olarak kaydet (16kHz, mono, 16-bit)
        clip.set_frame_rate(TMP_SR).set_channels(1).set_sample_width(2).export(tmp, format="wav")
        
        tmp_paths.append(tmp)    # Dosya yolu listesine ekle
        metas.append(s)          # Metadata listesine ekle (speaker, start, end)

    # ASR model'ini çalıştır (batch processing)
    try:
        try:
            # Yeni NeMo API: paths2audio_files parametresi
            hyps = asr.transcribe(paths2audio_files=tmp_paths, batch_size=BATCH_SIZE)
        except TypeError:
            # Eski NeMo API: direkt liste verme  
            hyps = asr.transcribe(tmp_paths)
        print(f"[NVIDIA-ASR] ✓ ASR tamamlandı, {len(hyps)} hypothesis")
    finally:
        # 10 TEMİZLİK - Geçici dosyaları sil
        for p in tmp_paths:
            with suppress(Exception):
                os.remove(p)  # Segment WAV dosyalarını sil
                
        # Ana WAV dosyasını da sil (orijinalden farklıysa)
        with suppress(Exception):
            if os.path.abspath(wav_path) != os.path.abspath(audio_path):
                os.remove(wav_path)

    # 8️ DİL TESPİTİ ve 9️ SONUÇ BİRLEŞTİRME
    segments: List[Dict[str, Any]] = []
    langs: List[str] = []  # Tespit edilen diller
    
    # Her ASR sonucu ile metadata'yı birleştir
    for h, meta in zip(hyps, metas):  # h: hypothesis, meta: segment bilgisi
        
        # Text çıkarma (NeMo model tipine göre değişebilir)
        text = h.text if hasattr(h, "text") else (h or "")
        if text is None: text = ""
        text = str(text).strip()  # String'e çevir ve trim et
        
        # Dil tespiti
        lang = _guess_lang(text)  # langid/langdetect ile dil tespit et
        if lang != "unknown":
            langs.append(lang)  # Bilinen dilleri topla
        
        # Final segment oluştur
        segments.append({
            "speaker": meta["speaker"],      # Diarization'dan gelen konuşmacı
            "start": float(meta["start"]),   # Segment başlangıcı (saniye)
            "end": float(meta["end"]),       # Segment bitişi (saniye)  
            "text": text,                    # ASR'dan gelen metin
            "lang": lang                     # Tespit edilen dil
        })

    # Çoğunluk dili belirle
    from collections import Counter
    majority_lang = Counter(langs).most_common(1)[0][0] if langs else "unknown"

    # Speaker dağılımı raporu (debug için)
    speaker_counts = Counter(s["speaker"] for s in segments)
    print(f"[NVIDIA-ENGINE] ✓ Tamamlandı! Konuşmacı dağılımı: {dict(speaker_counts)}")

    # Final sonuç oluştur
    return {
        "engine": "nvidia",                # Engine identifier
        "model": used_model,               # Model ID (nvidia/parakeet-tdt-0.6b-v3)  
        "language": majority_lang,         # Tespit edilen ana dil
        "summary": None,                   # Bu engine'de özet yok
        "segments": segments,              # Tüm transkripsiyon segmentleri
    }
