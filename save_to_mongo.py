# save_to_mongo.py
# ============================================================
# MongoDB yardımcıları:
# - media belgesi ekle (opsiyonel media_id alanı ile)
# - segments belgelerini idempotent ve toplu ekle (upsert)
# - UTC zaman damgası kullan
# - Girişte init edilmemişse init_collections() çağır
# ============================================================

from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

from pymongo import UpdateOne
from pymongo.results import BulkWriteResult

# Koleksiyon referansları ve init fonksiyonu
from db import media_col, segments_col, init_collections

log = logging.getLogger("save_to_mongo")
logging.basicConfig(level=logging.INFO)

# Import sırasında koleksiyonlar yoksa oluşturmayı dene
if media_col is None or segments_col is None:
    init_collections()


def _utcnow() -> datetime:
    """UTC timezone-aware datetime (Mongo tarihlerinde tek tiplik için)."""
    return datetime.now(timezone.utc)


def save_media(
    filename: str,
    duration: float,
    language: Optional[str],
    summary: Optional[str],
    media_id: Optional[str] = None,  # API'den gelen UUID (cross-ref için)
    model: Optional[str] = None,
):
    """
    'media' koleksiyonuna bir medya kaydı ekle.
    Dönüş: Mongo ObjectId (inserted_id)
    """
    if media_col is None:
        raise RuntimeError("media_col None; db.py init_collections kontrol edin.")

    doc = {
        "filename": filename,
        "model": model,
        "duration": float(duration or 0.0),
        "language": language,
        "summary": summary or None,
        "created_at": _utcnow(),
    }
    if media_id:
        doc["media_id"] = media_id  # ES/Kafka/segments ile aynı UUID

    res = media_col.insert_one(doc)
    inserted = res.inserted_id
    log.info(f"Mongo media kaydedildi: {inserted}")
    return inserted


def save_segments(media_id: str, segments: List[Dict[str, Any]],model: Optional[str] = None) -> int:
    """
    'segments' koleksiyonuna toplu ekleme (idempotent).
    - Her kayda media_id garanti eklenir.
    - segment_id yoksa media_id + segment_index'ten türetilir.
    - Upsert kullanıldığı için tekrar çalıştırmalarda duplicate hatası alınmaz;
      mevcut kayıtlar güncellenir, yeni kayıtlar eklenir.
    Dönüş: yeni eklenen kayıt sayısı (upserted_count).
    """
    if not segments:
        return 0
    if segments_col is None:
        raise RuntimeError("segments_col None; db.py init_collections kontrol edin.")

    now = _utcnow()
    ops: List[UpdateOne] = []

    for s in segments:
        seg_index = s.get("segment_index")
        seg_id = (
            s.get("segment_id")
            or (f"{media_id}:{int(seg_index):06d}" if seg_index is not None else None)
        )

        doc = {
            "media_id":   s.get("media_id") or media_id,
            "model": s.get("model", model),
            "segment_id": seg_id,
            "segment_index": int(seg_index) if seg_index is not None else None,
            "parent_id":  s.get("parent_id"),
            "speaker":    s.get("speaker", "speaker01"),
            "start":      float(s.get("start", 0.0)),
            "end":        float(s.get("end", 0.0)),
            "text":       s.get("text", ""),
            "lang":       s.get("lang", "unknown"),
            "emotion":    s.get("emotion") or s.get("emotion_pred") or "neutral",
            "created_at": now,
        }

        # segment_id anahtarımız; varsa ona göre upsert yap
        if seg_id:
            ops.append(
                UpdateOne(
                    {"segment_id": seg_id},
                    {"$set": doc},
                    upsert=True,
                )
            )
        else:
            # segment_id üretilemediyse (uç durum): media_id + zaman + speaker ile yaklaştırmalı anahtar
            fallback_key = {
                "media_id": doc["media_id"],
                "start": doc["start"],
                "end": doc["end"],
                "speaker": doc["speaker"],
            }
            ops.append(UpdateOne(fallback_key, {"$set": doc}, upsert=True))

    if not ops:
        return 0

    result: BulkWriteResult = segments_col.bulk_write(ops, ordered=False)
    # upserted_count -> gerçekten yeni eklenenler
    inserted = int(getattr(result, "upserted_count", 0) or 0)
    modified = int(getattr(result, "modified_count", 0) or 0)
    log.info(f"Mongo segments: upserted={inserted}, modified={modified}, total_ops={len(ops)}")
    return inserted
