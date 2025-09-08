# save_to_elastic.py
# ============================================================
# Elasticsearch yardımcıları:
# - Ortam değişkenlerinden bağlantı bilgisi (API key / basic / anon)
# - Tekil client (lazy init)
# - Index yoksa oluştur (media, segments)
# - Belge indexleme helper'ı
# Not: ES kapalıysa client None döner; çağıran taraf bunu kontrol etmeli.
# ============================================================

import os
import logging
from typing import Optional
from elasticsearch import Elasticsearch
from packaging.version import Version  # ES versiyon kıyaslaması için

log = logging.getLogger("save_to_elastic")

# ------------------------------------------------------------
# Bağlantı ayarları (ENV)
# ------------------------------------------------------------
ES_URL = os.getenv("ELASTIC_URL") or os.getenv("ELASTICSEARCH_URL") or "http://localhost:9200"
ES_USER = os.getenv("ELASTIC_USER") or os.getenv("ELASTIC_USERNAME")
ES_PASS = os.getenv("ELASTIC_PASSWORD")
ES_API_KEY = os.getenv("ELASTIC_API_KEY")                 # "<id>:<api_key>" biçimi
ES_TLS_VERIFY = (os.getenv("ELASTIC_TLS_VERIFY") or "false").lower() == "true"
ES_TIMEOUT = int(os.getenv("ELASTIC_TIMEOUT") or "30")

# Tekil client (cache)
_client: Optional[Elasticsearch] = None

def get_elasticsearch_client() -> Optional[Elasticsearch]:
    """
    Elasticsearch client'ı üret (veya cache'den ver).
    Tercih sırası: API Key > Basic Auth > Anon
    Başarılıysa /info çağırarak versiyonu logla.
    """
    global _client
    if _client is not None:
        return _client

    try:
        if ES_API_KEY:
            _client = Elasticsearch(
                ES_URL,
                api_key=ES_API_KEY,
                verify_certs=ES_TLS_VERIFY,
                request_timeout=ES_TIMEOUT,
            )
        elif ES_USER and ES_PASS:
            _client = Elasticsearch(
                ES_URL,
                basic_auth=(ES_USER, ES_PASS),
                verify_certs=ES_TLS_VERIFY,
                request_timeout=ES_TIMEOUT,
            )
        else:
            _client = Elasticsearch(
                ES_URL,
                verify_certs=ES_TLS_VERIFY,
                request_timeout=ES_TIMEOUT,
            )

        info = _client.info()
        ver = Version(info["version"]["number"])
        log.info(f"+ Elasticsearch bağlandı (v{ver})")
        return _client

    except Exception as e:
        log.error(f"- Elasticsearch bağlantı hatası: {e}")
        _client = None
        return None

def _ensure_index(es: Elasticsearch, name: str, body: dict) -> bool:
    """
    Index var mı? Yoksa verilen mapping/settings ile oluştur.
    """
    try:
        if not es.indices.exists(index=name):
            es.indices.create(
                index=name,
                mappings=body.get("mappings", {}),
                settings=body.get("settings", {}),
            )
            log.info(f"+ Index oluşturuldu: {name}")
        else:
            log.info(f"  Index zaten var: {name}")
        return True
    except Exception as e:
        log.error(f"- Index oluşturma hatası ({name}): {e}")
        return False

def create_indices() -> bool:
    """
    'media' ve 'segments' indexlerini garanti altına al.
    """
    es = get_elasticsearch_client()
    if not es:
        return False

    media_mapping = {   
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "media_id":   {"type": "keyword"},  # UUID 
                "model": {"type": "keyword"},
                "filename":   {"type": "keyword"},
                "duration":   {"type": "double"},
                "language":   {"type": "keyword"},
                "summary":    {"type": "text"},
                "created_at": {"type": "date"},
            }
        },
    }

    segments_mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "media_id":      {"type": "keyword"},
                "model": {"type": "keyword"},
                "segment_id":    {"type": "keyword"},
                "segment_index": {"type": "integer"},
                "parent_id":     {"type": "keyword"},
                "speaker":       {"type": "keyword"},
                "start":         {"type": "double"},
                "end":           {"type": "double"},
                "text":          {"type": "text"},
                "lang":          {"type": "keyword"},
                "emotion":       {"type": "keyword"},
                "created_at":    {"type": "date"},
            }
        },
    }

    ok1 = _ensure_index(es, "media", media_mapping)
    ok2 = _ensure_index(es, "segments", segments_mapping)
    return ok1 and ok2

def save_to_elasticsearch(index_name: str, document: dict, doc_id: Optional[str] = None) -> bool:
    """
    Belge indexleme yardımcı fonksiyonu.
    - doc_id verilirse aynı id ile overwrite (idempotent)
    """
    es = get_elasticsearch_client()
    if not es:
        return False

    try:
        es.index(index=index_name, id=doc_id, document=document)
        return True
    except Exception as e:
        log.error(f"- ES indexleme hatası ({index_name}): {e}")
        return False
