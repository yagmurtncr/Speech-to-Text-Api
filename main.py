from __future__ import annotations

import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from dotenv import load_dotenv
from engines.transcribe_large_multil import transcribe_large  # Geniş boyutlu transkripsiyon
from tqdm import tqdm

# Proje içi modüller (ayrı dosyalardan)
from convert_audio import convert_to_wav  # Ses dosyasını WAV formatına çevirir
from emotion_detection import EmotionJSONAnalyzer  # Duygu analiz modeli
from kafka_producer import send_media_event  # Kafka'ya event gönderimi
from save_to_mongo import save_converted_file_bulk, save_media, save_segments  # MongoDB işlemleri
from services.speaker_service import (
    SpeakerDiarizationService,  # (Eğer varsa) konuşmacı ayrımı servisi
)

# Sistem ve Ortam Ayarları

# CPU thread sayısını sınırla – performans ve stabilite için
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQ")

# Gürültücü log'ları bastır (hatalar hariç)
for name in ["uvicorn", "uvicorn.access", "transformers", "ctranslate2", "pyannote", "speechbrain", "elasticsearch"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# Uyarı mesajlarını bastır
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", module="speechbrain")
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# .env dosyasını yükle 
load_dotenv()

# Klasör Yapısı ve Giriş Dosyaları
input_dir = os.path.abspath("cv-corpus-22.0-delta-2025-06-20/en/clips")  # Orijinal ses dosyalarının olduğu klasör
output_dir = os.path.abspath("converted")  # Dönüştürülmüş dosyalar burada tutulacak
os.makedirs(output_dir, exist_ok=True)

# Desteklenen ses dosyası uzantıları
SUPPORTED_EXTENSIONS = (".mp3", ".mp4", ".wav", ".webm", ".m4a")

# Maksimum 100 ses dosyasını listele
media_files = [f for f in os.listdir(input_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)][:100]
print(f"{len(media_files)} dosya bulundu. Dönüştürme başlıyor...\n")

# Yardımcı Fonksiyonlar

# 'text=' ile başlayan metni regex ile temizle (segment içinden)
_H_TEXT_PAT = re.compile(r"text='([^']*)'")

def _clean_hypothesis_text(s: str) -> str:
    """
    Segmentteki metni regex ile temizler. Örn: text='hello world' → hello world
    """
    if not s:
        return ""
    m = _H_TEXT_PAT.search(s)
    return (m.group(1) if m else s).strip()

def _now_iso() -> str:
    """
    UTC formatında şu anki zamanı ISO string olarak döner.
    Kafka timestamp gibi alanlarda kullanılır.
    """
    return datetime.now(timezone.utc).isoformat()

# Ana İşlem Fonksiyonu (Her bir dosya için)
def process_file(filename):
    try:
        # 1. Giriş / çıkış dosya yolları
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_dir, output_filename)

        # 2. Dosya daha önce dönüştürülmüşse atla
        if os.path.exists(output_path):
            return None

        # 3. WAV formatına dönüştür
        convert_to_wav(input_path, output_path)

        # 4. Transkripsiyon (Whisper modeli veya benzeri)
        result = transcribe_large(output_path)

        # 5. Duygu Analizi (segment bazında)
        segments = result.get("segments", []) or []
        try:
            analyzer = EmotionJSONAnalyzer(device=None)  # CPU kullanarak analiz et
            segments = analyzer.analyze_segments(segments)  # Her segmente `emotion_pred` ekler
            for s in segments:
                s.pop("emotion_dist", None)  # Detaylı duygu dağılımı kaldırılır
        except Exception as ex:
            print(f"[EMOTION WARN] analyzer çalıştırılamadı: {ex}")

        # 6. Segmentleri normalize et (sadece gerekli alanlar alınır)
        normalized_segments = []
        for seg in segments:
            clean_text = _clean_hypothesis_text(seg.get("text") or "")
            normalized_segments.append({
                "speaker": seg.get("speaker", "speaker01"),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": clean_text,
                "lang": seg.get("lang", "unknown"),
                "emotion_pred": seg.get("emotion_pred", "neutral"),
            })
        result["segments"] = normalized_segments

        # 7. JSON olarak çıktı dosyası oluştur
        json_output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_final.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 8. MongoDB'ye kaydet
        duration = max([s["end"] for s in result["segments"]], default=0.0)
        media_id = save_media(
            filename=filename,
            duration=duration,
            language=result.get("language"),
            summary=result.get("summary"),
        )
        save_segments(media_id, result.get("segments", []))

        # 9. Kafka Event gönderimi
        try:
            send_media_event(
                media_id=str(media_id),
                filename=filename,
                status="processed",
                duration=duration,
                language=result.get("language"),
                summary=result.get("summary"),
                segments_count=len(result.get("segments") or []),
                timestamp=_now_iso(),
            )
            print(f" Kafka event gönderildi: {filename}")
        except Exception as e:
            print(f" Kafka event gönderilemedi ({filename}): {e}")

        # 10. Sonuçları geri döndür (toplu meta için)
        return {
            "original_filename": filename,
            "original_extension": os.path.splitext(filename)[1].lower(),
            "converted_filename": output_filename,
            "converted_at": datetime.utcnow().isoformat(),
            "source_dir": input_dir,
            "target_dir": output_dir,
            "mongo_id": str(media_id),
        }

    except Exception as e:
        print(f"- Hata oluştu ({filename}): {e}")
        return "error"
    
# Ana Fonksiyon: Toplu İşlem (Paralel)

def main():
    print(" Toplu ses dosyası işleme başlatılıyor...")

    converted_info_list = []  # Meta kayıt listesi
    success, failed = 0, 0

    # Paralel işleme için thread havuzu (4 dosya aynı anda)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in tqdm(executor.map(process_file, media_files), total=len(media_files), unit="dosya"):
            if result == "error":
                failed += 1
            elif result:
                converted_info_list.append(result)
                success += 1

    # Tüm meta verileri MongoDB’ye topluca kaydet
    if converted_info_list:
        save_converted_file_bulk(converted_info_list)

    # Sonuç özeti
    print("\n" + "="*50)
    print(" Dönüştürme Özeti:")
    print(f"+ Başarılı: {success}")
    print(f"- Hatalı: {failed}")
    print(f" Toplam: {len(media_files)}")
    print("="*50)
    print(" Medya, segment ve meta bilgiler MongoDB'ye kaydedildi.")
    print(" Kafka eventleri gönderildi.")
    print(" Elasticsearch indexleri hazır.")

if __name__ == "__main__":
    main()
