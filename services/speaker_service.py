# services/speaker_service.py
# Amaç: Konuşmacıları listeleme ve yeniden adlandırma akışı.
# Kullanım sırası:
# 1) get_speakers(media_id) -> dağılımı getirir.
# 2) rename_speakers(media_id, mapping, update_json, reindex_es) ->
#    Mongo güncelle, istenirse sonuç JSON'u güncelle, istenirse ES'yi yeniden yaz.

from __future__ import annotations

from datetime import datetime
from typing import Dict

from db import segments_col
from save_to_elastic import save_to_elasticsearch

from .storage_service import refresh_result_json_speakers


def _utcnow_iso():
    from datetime import timezone
    return datetime.now(tz=timezone.utc).isoformat() # UTC zaman damgasını ISO 8601 formatında üretir.

async def get_speakers(media_id: str):
    pipeline = [
        {"$match": {"media_id": media_id}}, # media_id alanı verilen değere eşit olanları akışa sokar.
        {"$group": {"_id": "$speaker", # Aynı speaker değerine sahip tüm dökümanlar tek bir grupta toplanır.
                     "count": {"$sum": 1}, # Konuşmacı başına segment sayısı.
                       "first_text": {"$first": "$text"}, # Gruplama sırasında o gruptaki ilk dökümanın text alanını alır.
                         "langs": {"$addToSet": "$lang"}}}, # Gruptaki tüm dökümanlardan lang değerlerini tekrarsız bir dizi olarak toplar.
        {"$project": {"_id": 0, #Mongo’nun otomatik _id alanını çıktıdan kaldırır.
                       "speaker": "$_id", # Kullanıcı id'si yerine speaker dönmek için.
                         "count": 1, # 1 → geçir, 0 → gizle 
                           "example_text": "$first_text", 
                             "langs": 1}},
        {"$sort": {"speaker": 1}} # Sonuçları speaker alanına göre artan sıralar (A→Z).
    ]
    # segments_col: MongoDB’deki segments koleksiyonu (her satır bir konuşma segmenti).
    speakers = list(segments_col.aggregate(pipeline)) # aggregate: lazy, parça parça çeker.
    total = sum(s["count"] for s in speakers) if speakers else 0 # Tüm konuşmacıların count değerlerini toplayıp toplam segment sayısı
    return {"media_id": media_id, "distinct": len(speakers), "total_segments": total, "speakers": speakers} # distinct: Benzersiz konuşmacı sayısı

async def rename_speakers(media_id: str, mapping: Dict[str, str], update_json: bool, reindex_es: bool): # mapping: {eski_speaker: yeni_speaker} eşlemesi.
    mapping = {k: v for k, v in mapping.items() if k and v and k != v}
    # k (eski ad) truthy olmalı → None, "" (boş string) gibi boş/None değerleri ele.
    # v (yeni ad) truthy olmalı → yine boş/None değerleri ele.
    # k != v → no-op (aynı isme yeniden adlandırma) eşlemelerini ele.
    per_key, total_modified = {}, 0
    # per_key: Her eski speaker için özet tutulacak bir sözlük.
    # total_modified: Tüm mapping boyunca toplam kaç segment değiştiğinin sayacı.

    for old, new in mapping.items(): # “rename yapacak bir şey var mı?”
        before = segments_col.count_documents({"media_id": media_id, "speaker": old})
        if before == 0:
            per_key[old] = {"before": 0, "modified": 0, #“Eski isim” yok, hiç belge güncellenmedi.
                            "after": segments_col.count_documents({"media_id": media_id, "speaker": new})}
            continue

        res = segments_col.update_many(
            {"media_id": media_id, "speaker": old},
            [{"$set": {"speaker_raw": {"$ifNull": ["$speaker_raw", "$speaker"]}, "speaker": new}}]
            # speaker_raw: Eğer yoksa ($ifNull) mevcut speaker değerini yedekler; varsa ellemiyor.
            # speaker: Artık yeni isim (new).
        )
        modified = getattr(res, "modified_count", before)
        total_modified += modified
        after = segments_col.count_documents({"media_id": media_id, "speaker": new})
        per_key[old] = {"before": before, "modified": modified, "after": after}
        # before: Eski isimli kayıt sayısı (başlangıç),
        # modified: O turda gerçekten değişen kayıt sayısı,
        # after: Yeni isimdeki güncel toplam.

    json_refreshed = refresh_result_json_speakers(media_id, mapping) if update_json else False

    es_reindexed = False # ?
    if reindex_es:
        try:
            for seg in segments_col.find({"media_id": media_id}):
                doc = {
                    "media_id": seg.get("media_id"), # Yoksa none döner.
                    "segment_id": seg.get("segment_id"),
                    "segment_index": seg.get("segment_index"),
                    "parent_id": seg.get("parent_id"),
                    "speaker": seg.get("speaker"),
                    "start": float(seg.get("start", 0.0)), # Eğer start yoksa 0.0 varsayılır.
                    "end": float(seg.get("end", 0.0)),
                    "text": seg.get("text", ""),
                    "lang": seg.get("lang", "unknown"),
                    "emotion": seg.get("emotion", "neutral"),
                    "created_at": seg.get("created_at").isoformat() if seg.get("created_at") else _utcnow_iso(),
                }
                if doc["segment_id"]:
                    save_to_elasticsearch("segments", doc, doc_id=doc["segment_id"])
            es_reindexed = True
        except Exception as e:
            print(f"[ES][WARN] reindex failed: {e}")

    agg_after = list(segments_col.aggregate([
        {"$match": {"media_id": media_id}},
        {"$group": {"_id": "$speaker", "count": {"$sum": 1}}},
        # Speaker alanına göre grupla.
        # Her belgeyi 1 sayarak konuşmacı başına segment adedi (count) çıkar.
        {"$project": {"_id": 0, "speaker": "$_id", "count": 1}},
        # Çıktı şekilldendirme aşaması: 
        # "_id": 0 → _id alanını çıkar (gizle).
        # "speaker": "$_id" → Yeni bir alan oluşturur: speaker = önceki belgedeki _id.
        # (MongoDB’de “rename” doğrudan yoktur; yeni alanı eskiye eşit yapıp eskisini gizleyerek yeniden adlandırma etkisi yaratırız.)
        {"$sort": {"speaker": 1}} # Sonucu speaker adına göre artan sırada döndürür (A→Z).
    ]))

    return {
        "media_id": media_id,
        "modified_total": total_modified,   # toplam değişen segment adedi
        "per_key": per_key,                 # her eski->yeni için özet
        "json_refreshed": json_refreshed,   # sonuç JSON'u güncellendi mi?
        "es_reindexed": es_reindexed,       # ES yeniden yazıldı mı?
        "speakers_after": agg_after         # son dağılım
    }
