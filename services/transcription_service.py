# ------------------------------------------------------------
# services/transcription_service.py
# Amaç: Upload + ağır iş + Mongo/ES/Kafka adımlarını yönetmek.
# Kullanım sırası:
# 1) upload_and_start(file) -> tmp'e yazar, job kaydı oluşturur, arka plan işi başlatır.
# 2) _run_transcription(...) -> gerçek ağır işlem; bittiğinde Mongo/ES/Kafka yazıp jobs/RAM günceller.
# ------------------------------------------------------------

from __future__ import annotations  #  İleriye dönük type hint'ler (Python 3.10 öncesi kolaylığı)
import os, uuid, shutil, asyncio, json, time  #  OS/dosya, UUID, kopyalama, async, JSON, zaman
from datetime import timezone, datetime        #  ISO timestamp için
from typing import Any, List, Dict             #  Tip ipuçları
from fastapi import UploadFile                 #  FastAPI upload tipi
from concurrent.futures import ProcessPoolExecutor  #  CPU-bound işleri paralel çalıştırmak için

from engines.transcription_worker import run_in_subprocess  #  Asıl ağır işi yapan alt süreç çağrısı
from save_to_mongo import save_media, save_segments         #  MongoDB yazıcı yardımcıları
from save_to_elastic import save_to_elasticsearch           #  Elasticsearch indexleme helper'ı
from kafka_producer import send_media_event                 #  Kafka'ya event atmak için

from .storage_service import TMP_DIR, set_processing, set_completed, set_error  #  Job durum cache + yollar


# ------------------------------------------------------------
# Ortam ayarları, path düzeltmeleri
# ------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()  #  .env dosyasını belleğe yükler; os.getenv(...) çağrıları buradan değer okur

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #  Bu dosyanın bulunduğu klasör
ROOT_DIR = os.path.dirname(BASE_DIR)                   #  Proje kök klasörü (services/.. → kök)

import sys
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)                       #  Kök klasörü import yoluna ekle (alt süreçler için de)
os.environ.setdefault("PYTHONPATH", ROOT_DIR)          #  ProcessPoolExecutor çocuklarının modülleri bulabilmesi için

# ------------------------------------------------------------
# 0) Genel konfigürasyon ve havuz ayarları
# ------------------------------------------------------------
HEAVY_WORKERS = int(os.getenv("HEAVY_WORKERS", "2"))        #  Aynı anda kaç ağır iş koşturulabilir
HEAVY_TIMEOUT = int(os.getenv("HEAVY_TIMEOUT", "1800"))     #  Tek iş için üst zaman sınırı (saniye)
INDEX_TO_ES = (os.getenv("INDEX_TO_ES") or "false").lower() == "true"  #  ES indexleme açık mı?

# Semaphore: Aynı anda çalışan iş sayısını sınırlar
_sem = asyncio.Semaphore(HEAVY_WORKERS)                     #  Koşan async görevleri üstten sayısal sınırla
# Process havuzu: CPU-bound işleri paralel işlemeye yarar
_pool = ProcessPoolExecutor(max_workers=HEAVY_WORKERS)      #  Ağır işleri ayrı süreçlerde çalıştır

# ------------------------------------------------------------
# 1) Yardımcılar
# ------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  #  ISO-8601 formatında (UTC) zaman string'i üretir

def _attach_ids(final_result: dict, media_id: str) -> List[dict]:
    """1) Segmentlere kimlik bilgisi ekle (media_id, segment_id, index, parent_id)"""
    raw = final_result.get("segments", []) or []    #  Segment listesi yoksa boş liste kabul et
    enriched, prev = [], None                       #  prev = bir önceki segment'in id'si
    for i, seg in enumerate(raw, start=1):
        sid = f"{media_id}:{i:06d}"                 #  6 haneli sıralı id (sıralama kolaylığı için)
        seg = dict(seg or {})
        seg["media_id"] = media_id
        seg["segment_id"] = sid
        seg["index"] = i
        seg["parent_id"] = prev
        prev = sid
        enriched.append(seg)
    final_result["segments"] = enriched
    return enriched

def _normalize_segments(segs: List[dict]) -> List[dict]:
    """2) Segment alanlarını normalize et (tipler, boş metinler, süre yuvarlama vb.)."""
    out = []
    for s in segs or []:
        s = dict(s or {})
        # metin güvenliği
        txt = s.get("text", "")
        if txt is None:
            txt = ""
        elif not isinstance(txt, str):
            txt = str(txt)
        s["text"] = txt.strip()

        # speaker default
        spk = s.get("speaker") or "speaker01"
        s["speaker"] = str(spk)

        # zamanları float'a yuvarla
        try:
            s["start"] = round(float(s.get("start", 0.0)), 3)
        except Exception:
            s["start"] = 0.0
        try:
            s["end"] = round(float(s.get("end", 0.0)), 3)
        except Exception:
            s["end"] = s["start"]

        # model/lang field'ları normalize
        s["model"] = str(s.get("model") or "unknown")
        lang = (s.get("lang") or "unknown").strip().lower()
        s["lang"] = lang if lang else "unknown"

        # emotion alanlarını tekleştir
        emo = s.get("emotion") or s.get("emotion_pred") or "neutral"
        s["emotion"] = str(emo)
        s["emotion_pred"] = str(emo)

        out.append(s)
    return out

def _fallback_summary(text: str, max_chars: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_dot = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return cut[:last_dot+1] if last_dot >= 200 else cut + "..."

# ---- SCHEMA GUARD: every segment must have 'text' as str ----
from typing import Any

def _ensure_text_field_on_segments(segments: List[dict]) -> List[dict]:
    safe = []
    for s in segments or []:
        # Eğer kelime hizalamasından 'word' geldiyse text'e köprüle
        if "text" not in s and "word" in s:
            s = {**s, "text": s.get("word", "")}
        # text alanını garanti et ve str yap
        txt = s.get("text", "")
        if txt is None:
            txt = ""
        elif not isinstance(txt, str):
            txt = str(txt)
        s = {**s, "text": txt}
        safe.append(s)
    return safe

def _force_schema(res: Dict[str, Any]) -> Dict[str, Any]:
    res = dict(res or {})
    res["segments"] = _ensure_text_field_on_segments(res.get("segments") or [])
    # summary/language/engine/model alanlarını da güvene al
    if not isinstance(res.get("summary"), str):
        res["summary"] = "" if res.get("summary") is None else str(res["summary"])
    for k in ("language", "engine", "model"):
        if k in res and res[k] is not None and not isinstance(res[k], str):
            res[k] = str(res[k])
    return res

# ------------------------------------------------------------
# 2) Upload → İş başlat
# ------------------------------------------------------------
async def upload_and_start(file: UploadFile, options: dict | None = None) -> dict:
    """
    1) tmp'e yaz
    2) job=processing
    3) arka plan işi başlat
    4) media_id döndür
    """
    media_id = str(uuid.uuid4())                                  # İşi benzersiz kimlik ile takip edeceğiz
    ext = os.path.splitext(file.filename)[-1]                     # Orijinal uzantıyı koru (.wav/.mp3 ...)
    tmp_path = os.path.join(TMP_DIR, f"tmp_{media_id}{ext}")      # Geçici klasörde hedef yol

    # (1) Yüklenen dosyayı diske yaz
    with open(tmp_path, "wb") as buf:                             # Büyük dosyalar için stream kopyalama
        shutil.copyfileobj(file.file, buf)

    # (2) RAM + Mongo job başlat
    set_processing(media_id, tmp_path, file.filename)             # UI polling "processing" görecek

    # (3) Ağır işlem arka planda başlasın
    asyncio.create_task(_run_transcription(media_id, tmp_path, file.filename, options or {}))  # Fire-and-forget

    # (4) Medya ID’yi döndür
    return {"media_id": media_id, "status": "processing"}         # UI bu ID ile /results/{id} poll eder

# ------------------------------------------------------------
# 3) Ağır işlem hattı (arka planda çalışır)
# ------------------------------------------------------------
async def _run_transcription(media_id: str, file_path: str, original_name: str, options: dict):
    """
    1) alt süreçte transkripsiyon çalıştır
    2) segmentleri normalize et + id ata
    3) dil/özet işle
    4) Mongo ve Elasticsearch’e kaydet
    5) Kafka'ya event at
    6) RAM/Mongo job'ı tamamlandı olarak işaretle
    """
    t0 = time.time()
    try:
        # (0) Opsiyonları hazırla
        engine = (options.get("engine") or os.getenv("DEFAULT_ENGINE", "whisperx")).lower()
        opts = dict(options or {})
        opts["engine"] = engine

        # (1) Ağır işi ayrı süreçte çalıştır
        loop = asyncio.get_event_loop()
        fut = loop.run_in_executor(_pool, run_in_subprocess, file_path, opts)

        try:
            # Soft-timeout: bekleme süresi dolarsa yine de işin bitmesini bekleyip finalize edeceğiz
            # Ama bu blokta timeout atarsa bir alt bloğa geçer ve finalize öncesi hard-wait yapılır
            final_result, json_path, duration = await asyncio.wait_for(
                asyncio.shield(fut), timeout=HEAVY_TIMEOUT
            )
        except asyncio.TimeoutError:
            # Soft-timeout oldu; erken dönme. İş gerçekten bitsin ki finalize adımları çalışsın.
            print("[WORKER] soft-timeout: converting to hard wait; will finalize before completing")
            # İşin bitmesini bekle (hata fırlatırsa burada yakalanır)
            final_result, json_path, duration = await asyncio.shield(fut)

        # Şema kalkanı: her segmentte 'text' en azından boş string olsun
        final_result = _force_schema(final_result)

        # (2) Segment ID ve bağlam bilgileri
        segs = _attach_ids(final_result, media_id) # Her segmente kimlik sahası eklenir
        segs = _normalize_segments(segs)
      

        # (3) Dil tahmini ve override
        engine_lang = (final_result.get("language") or "").strip().lower()  #  Engine'in raporladığı dil
        out_lang = ((options or {}).get("language") or (options or {}).get("output_language") or "").strip().lower()
        #  UI/opsiyon dil geçersiz kılma (override). Örn: İngilizceye çeviri gibi

        if out_lang:
            final_result["language"] = out_lang              # Dil zorla çıktı dili yapılır
            if out_lang.startswith("en"):                    # Dil override "en" ise segmentlerin lang'ı da uyumlansın
                for s in segs:
                    s["lang"] = "en"
        else:
            # Engine "unknown" dönerse segmentlerden çoğunluk dilini çıkar
            langs = {s.get("lang") for s in segs if s.get("lang") and s.get("lang") != "unknown"}
            if not engine_lang or engine_lang == "unknown":
                final_result["language"] = "multilanguage" if len(langs) > 1 else (next(iter(langs)) if langs else "unknown")
            else:
                final_result["language"] = engine_lang

        # (3b) Özet yoksa oluştur
        if not (final_result.get("summary") or "").strip():
            # Basit fallback: Tüm metinleri birleştir, 600 karakter civarında kes
            txt = " ".join(s.get("text", "") for s in segs if s.get("text"))
            final_result["summary"] = _fallback_summary(txt)

        # (4) MongoDB'ye kaydet
        # Duration'ı segmentlerden hesapla
        duration = max((float(s.get("end", 0.0)) for s in segs), default=0.0) if segs else 0.0
        
        mongo_media_id = save_media(
            filename=original_name,
            duration=duration,
            language=final_result.get("language"),
            summary=final_result.get("summary"),
            media_id=media_id,
            model=final_result.get("model", engine)
        )

        # (5) Segmentleri kaydet (Mongo)
        save_segments(media_id, segs)

        # (6) Elasticsearch indexleme (opsiyonel)
        if INDEX_TO_ES:
            # Media üst kaydını indexle
            media_doc = {
                "media_id": media_id,
                "file_name": original_name,
                "engine": engine,
                "language": final_result.get("language"),
                "summary": final_result.get("summary"),
                "duration": duration,
                "created_at": _utcnow_iso(),
            }
            save_to_elasticsearch("media", media_doc, doc_id=media_id)
            
            # Her segmenti ayrı ayrı indexle
            for seg in segs:
                seg_doc = {
                    "media_id": media_id,
                    "segment_id": seg.get("segment_id"),
                    "segment_index": seg.get("index"),
                    "parent_id": seg.get("parent_id"),
                    "speaker": seg.get("speaker"),
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", ""),
                    "lang": seg.get("lang", "unknown"),
                    "emotion": seg.get("emotion", "neutral"),
                    "created_at": _utcnow_iso(),
                }
                if seg.get("segment_id"):
                    save_to_elasticsearch("segments", seg_doc, doc_id=seg["segment_id"])

        # (7) Kafka'ya event (opsiyonel)
        try:
            send_media_event({
                "media_id": media_id,
                "file_name": original_name,
                "engine": engine,
                "language": final_result.get("language"),
                "summary": final_result.get("summary"),
                "segments_count": len(segs),
                "created_at": _utcnow_iso(),
            })
        except Exception:
            pass  # Kafka opsiyonel; hata durumunda işi düşürmeyelim

        # (8) RAM + Mongo jobs güncelleme
        set_completed(media_id, json_path, mongo_media_id)       # UI'da /results/{id} → "completed" + JSON hazır
        print(f"[WORKER] done {media_id} segs={len(segs)} dur={duration:.2f}s in {time.time()-t0:.1f}s")

        # (9) Geçici dosyayı siler
        # Temizlik bazı kurulumlarda storage_service içinde yapılabilir; burada ekstra silme yapılmıyor.
        #  Eğer burada silmek istersen: try: os.remove(file_path) except: pass

    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=2)                        # Kısa stacktrace
        set_error(media_id, f"{e.__class__.__name__}: {e}")       # Job durumunu "error" yap ve mesajı kaydet
        print("[WORKER][ERROR]", tb)                              # Log'a hatayı yaz (debug kolaylığı)
