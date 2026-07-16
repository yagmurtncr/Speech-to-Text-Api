# ---------------------------------------------------------
# kafka_producer.py
# Kafka'ya event göndermek için yardımcı modül
# - Tekil / toplu event gönderme
# - Bağlantı testi
# ---------------------------------------------------------

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()

# ---------------------------------------------------------
# LOG AYARLARI
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# KONFİG
# ---------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9093")
TOPIC_NAME = os.getenv("KAFKA_TOPIC", "media_processed")

# Tek producer kullan (reconnect gerektiğinde yeniden oluşturulur)
_PRODUCER: Optional[KafkaProducer] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------
# PRODUCER OLUŞTURMA
# ---------------------------------------------------------
def get_kafka_producer() -> Optional[KafkaProducer]:
    """Kafka producer oluşturur; cache'ler. Hata varsa None döner."""
    global _PRODUCER
    if _PRODUCER is not None:
        return _PRODUCER
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            acks="all",
            retries=3,
        )
        # Bağlantı testi
        producer.metrics()
        logger.info("+ Kafka producer bağlantısı başarılı")
        _PRODUCER = producer
        return _PRODUCER
    except Exception as e:
        logger.error(f"- Kafka producer bağlantı hatası: {e}")
        return None


# ---------------------------------------------------------
# TEK EVENT
# ---------------------------------------------------------
def send_media_event(
    *,
    media_id: str,
    filename: str,
    status: str,
    duration: float,
    language: Optional[str] = None,
    summary: Optional[str] = None,
    segments_count: Optional[int] = None,
    timestamp: Optional[str] = None,
    model: Optional[str] = None
) -> bool:
    """
    Medya işlendiğinde Kafka'ya tek event gönder.
    """
    producer = get_kafka_producer()
    if not producer:
        logger.error("Kafka producer oluşturulamadı")
        return False

    event: Dict[str, Any] = {
        "media_id": str(media_id),
        "model": model or "",
        "filename": filename,
        "status": status,
        "duration": float(duration),
        "timestamp": timestamp or _now_iso(),
        "language": language or "",
        "summary": (summary or "")[:1000],  # güvenli kısaltma
        "segments_count": int(segments_count or 0),
    }

    try:
        future = producer.send(TOPIC_NAME, event)
        meta = future.get(timeout=10)
        logger.info(f"Kafka event gönderildi: topic={meta.topic} part={meta.partition} off={meta.offset} payload={event}")
        producer.flush()
        return True
    except Exception as e:
        logger.error(f"- Kafka event gönderme hatası: {e}")
        return False


# ---------------------------------------------------------
# TOPLU EVENT
# ---------------------------------------------------------
def send_bulk_events(events_list) -> bool:
    producer = get_kafka_producer()
    if not producer:
        logger.error("Kafka producer oluşturulamadı")
        return False

    ok = 0
    for ev in events_list:
        try:
            future = producer.send(TOPIC_NAME, ev)
            future.get(timeout=5)
            ok += 1
        except Exception as e:
            logger.error(f"- Event gönderilemedi: {e}")
    producer.flush()
    logger.info(f"+ Toplam {ok}/{len(events_list)} event başarıyla gönderildi")
    return ok == len(events_list)


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
def test_kafka_connection() -> bool:
    p = get_kafka_producer()
    if not p:
        return False
    try:
        future = p.send(TOPIC_NAME, {"test": True, "ts": _now_iso()})
        future.get(timeout=10)
        p.flush()
        logger.info("+ Kafka bağlantı testi başarılı")
        return True
    except Exception as e:
        logger.error(f"- Kafka bağlantı testi başarısız: {e}")
        return False


if __name__ == "__main__":
    if test_kafka_connection():
        send_media_event(
            media_id="test-123",
            filename="test.mp4",
            status="test",
            duration=10.0,
            language="tr",
            summary="Test özeti",
            segments_count=5,
        )
    else:
        print("- Kafka bağlantısı kurulamadı")
